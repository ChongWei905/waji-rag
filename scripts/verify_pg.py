from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from waji_rag.pg_index import DatabaseOptions, redact_database_url  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Verify PostgreSQL, pgvector, and pg_trgm capabilities.")
    parser.add_argument("--database-url", help="PostgreSQL URL. Defaults to WAJI_DATABASE_URL or local Docker default.")
    parser.add_argument("--connect-timeout", type=int, default=10, help="Connection timeout in seconds. Defaults to 10.")
    parser.add_argument(
        "--no-create-extension",
        action="store_true",
        help="Only check installed extensions instead of running CREATE EXTENSION IF NOT EXISTS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PostgreSQL capability checks."""

    args = build_parser().parse_args(argv)
    database = DatabaseOptions.from_env(args.database_url)
    started_at = time.time()
    try:
        payload = verify_postgres(
            database.database_url,
            connect_timeout=args.connect_timeout,
            create_extensions=not args.no_create_extension,
        )
        payload["database_url"] = redact_database_url(database.database_url)
        payload["elapsed_ms"] = int((time.time() - started_at) * 1000)
        payload["status"] = "ok"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - verification script should report all failures.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "database_url": redact_database_url(database.database_url),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": int((time.time() - started_at) * 1000),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


def verify_postgres(database_url: str, *, connect_timeout: int, create_extensions: bool) -> dict[str, Any]:
    """Verify the PostgreSQL features required by this project."""

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("psycopg is not installed. Run: pip install -e .") from exc

    with psycopg.connect(database_url, connect_timeout=connect_timeout) as conn:
        with conn.cursor() as cur:
            server = fetch_server_info(cur)
            if create_extensions:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                conn.commit()
            extensions = fetch_extensions(cur)
            missing = [name for name in ("vector", "pg_trgm") if name not in extensions]
            if missing:
                raise RuntimeError(f"missing PostgreSQL extensions: {', '.join(missing)}")
            vector_result = verify_pgvector(cur)
            trigram_result = verify_pg_trgm(cur)
            temp_write_result = verify_temp_write(cur)
    return {
        "server": server,
        "extensions": extensions,
        "checks": {
            "connect": "ok",
            "create_or_check_extensions": "ok",
            "pgvector_cosine_search": vector_result,
            "pg_trgm_similarity": trigram_result,
            "temp_table_write_read": temp_write_result,
        },
    }


def fetch_server_info(cur: Any) -> dict[str, object]:
    """Return basic PostgreSQL server details."""

    cur.execute(
        """
        SELECT
            version(),
            current_database(),
            current_user,
            inet_server_addr()::text,
            inet_server_port()
        """
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("failed to read PostgreSQL server info")
    return {
        "version": row[0],
        "database": row[1],
        "user": row[2],
        "host": row[3],
        "port": row[4],
    }


def fetch_extensions(cur: Any) -> dict[str, str]:
    """Return installed extension versions required by this project."""

    cur.execute(
        """
        SELECT extname, extversion
        FROM pg_extension
        WHERE extname = ANY(%s::text[])
        ORDER BY extname
        """,
        (["vector", "pg_trgm"],),
    )
    return {str(name): str(version) for name, version in cur.fetchall()}


def verify_pgvector(cur: Any) -> dict[str, object]:
    """Verify vector type insert and cosine-distance ordering."""

    cur.execute(
        """
        CREATE TEMP TABLE waji_verify_vector (
            id bigserial PRIMARY KEY,
            label text NOT NULL,
            embedding vector(3) NOT NULL
        ) ON COMMIT DROP
        """
    )
    cur.execute(
        """
        INSERT INTO waji_verify_vector(label, embedding)
        VALUES
            ('same', '[1,0,0]'::vector),
            ('close', '[0.9,0.1,0]'::vector),
            ('far', '[0,1,0]'::vector)
        """
    )
    cur.execute(
        """
        SELECT label, ROUND((embedding <=> '[1,0,0]'::vector)::numeric, 6)::text AS distance
        FROM waji_verify_vector
        ORDER BY embedding <=> '[1,0,0]'::vector ASC
        LIMIT 3
        """
    )
    rows = [{"label": str(label), "distance": str(distance)} for label, distance in cur.fetchall()]
    if not rows or rows[0]["label"] != "same":
        raise RuntimeError(f"pgvector ordering check failed: {rows}")
    return {"status": "ok", "nearest": rows}


def verify_pg_trgm(cur: Any) -> dict[str, object]:
    """Verify pg_trgm similarity function availability."""

    cur.execute("SELECT similarity(%s, %s)", ("fan belt abnormal noise", "fan belt noise"))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("pg_trgm similarity returned no row")
    score = float(row[0])
    if score <= 0:
        raise RuntimeError(f"pg_trgm similarity score should be positive, got {score}")
    return {"status": "ok", "similarity": round(score, 6)}


def verify_temp_write(cur: Any) -> dict[str, object]:
    """Verify basic write/read permissions through a temporary table."""

    cur.execute("CREATE TEMP TABLE waji_verify_temp(id integer PRIMARY KEY, content text) ON COMMIT DROP")
    cur.execute("INSERT INTO waji_verify_temp(id, content) VALUES (1, %s)", ("ok",))
    cur.execute("SELECT content FROM waji_verify_temp WHERE id = 1")
    row = cur.fetchone()
    if row is None or row[0] != "ok":
        raise RuntimeError("temporary table write/read check failed")
    return {"status": "ok"}


if __name__ == "__main__":
    raise SystemExit(main())
