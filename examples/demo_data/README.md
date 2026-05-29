# Waji RAG Demo Data

这是一套 Mac 本地调试用的小样本数据，不需要正式售后数据。

目录结构：

```text
examples/demo_data/
├── work_orders/
│   ├── WO-DEMO-FAN-BELT-001.txt
│   ├── WO-DEMO-TRAVEL-001.txt
│   └── WO-DEMO-AIRCON-001.txt
└── manuals/
    └── XE215G/
        ├── 典型故障解析/
        │   ├── 动力系统故障解析/
        │   ├── 液压系统故障解析/
        │   └── 空调系统故障解析/
        └── 机器故障代码解析/
```

建议验证问题：

```text
用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，相应故障需要更换备件的详细信息（备件的编号及名称，备件编码，备件数量）
```

预期关键结果：

- 能召回历史工单 `WO-DEMO-FAN-BELT-001`；
- 能召回手册 `故障现象：风扇皮带异响.html`；
- 备件候选中应出现 `XE215G 风扇皮带`、备件编码 `310900001`、数量 `1`；
- 答案应说明备件来自历史工单，未经过当前设备适配校验。

Mac 本地前端验证：

```bash
docker compose up -d postgres

.venv/bin/python -m waji_rag.cli serve --host 127.0.0.1 --port 8767
```

然后打开：

```text
http://127.0.0.1:8767
```

页面默认已经填好这套 demo 数据目录。点击：

```text
一键跑全流程
```

即可完成：

```text
配置预览
→ 初始化 PG
→ 解析工单和手册 HTML
→ HTML 转 Markdown 并入库
→ 检索证据
→ 生成答案
→ 展示阶段日志、召回证据、备件候选和原始 JSON
```

如果想用命令行对照排查，也可以运行：

```bash
docker compose up -d postgres

.venv/bin/python -m waji_rag.cli ingest-db \
  --work-order-dir examples/demo_data/work_orders \
  --manual-dir examples/demo_data/manuals \
  --reset \
  --report-json tmp/demo_outputs/demo_ingest_report.json \
  --debug

.venv/bin/python -m waji_rag.cli ask-db \
  --query "用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，相应故障需要更换备件的详细信息（备件的编号及名称，备件编码，备件数量）" \
  --top-k 1 \
  --out-json tmp/demo_outputs/demo_bm25_top1_ask.json \
  --debug
```
