# 电商售后服务智能体

李省春小组的 Python 课堂项目。当前保留老师发放的“项目起始模板”代码骨架，电商任务卡和模拟资料已放在同一工程中。课程目前进行到第 05 单元；订单工具、知识检索和 Agent 流程留待小组按每日任务实现。

## 在 Trae 中打开

将 `D:\class\ecommerce-after-sales-agent` 作为**文件夹**打开。工程使用 Python 3.11，虚拟环境在本目录的 `.venv`；编辑器会通过 `.vscode/settings.json` 指向它。首次配置后，每次打开这个文件夹即可继续开发。

## 启动与检查

在项目根目录的 PowerShell 终端执行：

```powershell
uv sync
uv run python main.py
uv run python -m unittest discover -s tests -v
uv run python 验证数据.py
```

默认 `.env` 使用离线 `mock` 模式，不需要 API Key。课程要求真实模型时，先执行 `uv pip install --python .venv/Scripts/python.exe -r requirements-deepseek.txt`，再在本机 `.env` 填写自己的 `DEEPSEEK_API_KEY` 并把 `LLM_PROVIDER` 改成 `deepseek`；真实模式使用 `.venv/Scripts/python.exe main.py` 启动。真实调用会产生 API 用量。`.env` 和 `.venv` 不会提交到 GitHub。

## 工程内容

- `main.py`、`app/`、`tests/`：老师的 Python 起始骨架；目前只有统一问答入口和 Mock 测试。
- `02-电商售后服务智能体-任务卡.docx`：项目范围、五人分工、阶段成果和验收要求。
- `模拟数据/records.json`：30 条虚构订单记录。
- `模拟数据/knowledge/`：15 份带 `DOC-02` 编号的售后知识文档。
- `模拟数据/sample_questions.json`：起始验收问题。
- `验证数据.py`：老师提供的数据检查脚本。

所有业务数据均为课程模拟数据，不接真实电商系统。当前程序只回显 Mock 答复，尚不能判断真实退货资格；后续按任务卡开发只读查询与待人工处理草稿。

## 来源

代码取自 `unit1.zip/项目起始模板`；电商资料取自 `unit1.zip/自选项目任务包/02-电商售后服务智能体`。`unit5.zip` 的工单与报销课堂示例属于另一个演示场景，未并入本工程。
