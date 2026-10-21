# 电商售后服务智能体

李省春小组的 Python 课堂项目。在老师发放的“项目起始模板”代码骨架之上，已按电商任务卡实现第 02—08 单元：多轮对话与流式输出、Prompt 模板与同题对比、订单工具调用闭环、工具注册表与调度器、售后文本向量化与相似度排序、向量库选型与 Chroma 空集合建库、售后资料批量写入与项目内检索。所有数据均为课程虚构样例，只读，不接真实电商系统。

## 在 Trae 中打开

将 `D:\class\ecommerce-after-sales-agent` 作为**文件夹**打开。工程使用 Python 3.11，虚拟环境在本目录的 `.venv`；编辑器会通过 `.vscode/settings.json` 指向它。首次配置后，每次打开这个文件夹即可继续开发。

## 启动与检查

在项目根目录的 PowerShell 终端执行：

```powershell
uv sync --extra embedding --extra chroma
uv run python main.py
uv run python -m unittest discover -s tests -v
uv run python 验证数据.py
uv run python scripts/generate_evidence_unit06.py
uv run python scripts/generate_evidence_unit07.py
uv run python scripts/generate_evidence_unit08.py
```

`--extra embedding`（单元06 向量模型）与 `--extra chroma`（单元07、08 向量库）都是可选依赖。只跑单元 02—05 时 `uv sync` 即可，其余命令不受影响。

单元06 的中文向量模型约 180MB，放在 `models/`，已用 `.gitignore` 排除、不随仓库分发；换电脑时在 `.env` 里设置 `EMBEDDING_MODEL_DIR` 指向课程材料的 `models/` 目录即可，仍保持离线加载、不联网下载。

单元07 的 Chroma 数据库落在 `.chroma_unit7/`（运行产物，同样不进仓库）。课堂验收要求的顺序是：

```powershell
uv run python setup_chroma.py      # 建空集合，记下版本/目录/集合名/ID/元数据/0 条记录
uv run python inspect_chroma.py    # 只读查看
uv run python setup_chroma.py      # 再跑一次，核对集合 ID 不变
```

单元08 沿用同一个集合定义（同名、同元数据、同向量模型），但把库文件写到 `.chroma_unit8/`，以保住单元07「0 条记录」的验收现场；想直接写进单元07 那个库时，在 `.env` 里设 `CHROMA_DB_DIR=.chroma_unit7`。写入命令是交互式的 `/ingest`（见下），流程为：读 15 篇资料 → 记录导入前数量 → 批量 upsert → 重新连接核对记录数与集合 ID → 项目内 Top 3 检索。重复运行时稳定 ID（`项目标识:DOC-02-05`）让写入变成更新，记录数不会增长。

默认 `.env` 使用离线 `mock` 模式，不需要 API Key。课程要求真实模型时，先执行 `uv pip install --python .venv/Scripts/python.exe -r requirements-deepseek.txt`，再在本机 `.env` 填写自己的 `DEEPSEEK_API_KEY` 并把 `LLM_PROVIDER` 改成 `deepseek`；真实模式使用 `.venv/Scripts/python.exe main.py` 启动。真实调用会产生 API 用量。`.env`、`.venv`、`models/`、`.chroma_unit7/`、`.chroma_unit8/` 不会提交到 GitHub。

## 交互命令

```text
<直接输入问题>   工具闭环模式：调用订单/退货/工单/政策工具后再回答
/agent <问题>    同上，并强制打印本次工具调用轨迹
/chat <问题>     单元02 流式模式：不使用工具，逐段显示回答
/compare [问题]  单元03：同一份订单数据 + 同一个问题，对比差/好两套提示词
/missing         单元04：演示工具名与参数校验（缺参、空白值、多余参数、未知工具）
/missingloop     单元04：模型漏参 -> 工具报错 -> 自动补全重试的完整证据链
/registry        单元05：打印注册表（工具、用途、参数、只读标记）
/dispatch        单元05：不访问模型，直接验证调度：成功 + 未知工具 + 缺参 + 处理函数异常
/log             单元05：本次会话的调用日志（轮次、状态、耗时、异常提示）
/trace           查看最近一次工具调用轨迹
/embed [问题]    单元06：本地中文模型向量化 + 余弦相似度排序 + 边界核对 + 误判风险
/vector          单元07：记录字段图 + 精确遍历检索 + 元数据过滤 + 项目隔离
/chroma          单元07：真实 Chroma 建空集合 + 只读查看 + 重复运行核对集合 ID
/ingest          单元08：本组资料批量写入 + 重新连接核对 + 项目内 Top 3 检索
/demo            跑通 3 个正常问题 + 2 个越权问题
/history /clear /help /exit
```

`/embed` 用 `模拟数据/sample_questions.json` 里 6 条电商售后样例问题做候选，把自拟问题和它们一起转成 512 维向量后排序，输出前三条，并附空文本／维度不一致的实测拦截结果。它**不做问答、不读知识库**，只比较“像不像”。

`/vector` 把 15 份售后知识文档 + 6 条样例问题共 21 条写入内存集合，展示记录字段图、精确遍历 Top-K、按 `doc_type` 过滤与 `tenant_id` 项目隔离。它是**全量遍历的精确基线**，后面量 Recall@K 时拿它当标准答案；不是真正的向量数据库。

`/ingest` 走真实 Chroma：把 `模拟数据/knowledge/` 的 15 篇售后文档用单元06 同一个模型转成 512 维向量后批量 upsert，重新连接核对记录数与集合 ID，再按 `project_id` 过滤做 Top 3 检索（同时给出 l2 距离与余弦相似度）。它**只做检索，不生成业务答案**——距离/相似度说明的是"像不像"，不是正确答案。

## 工程内容

- `main.py`：统一入口，各单元的命令开关。
- `app/conversation.py`：会话对象，保存/裁剪/读取消息。
- `app/llm.py`：离线 MockLLM（含工具调用与政策查询模拟）、模型工厂与故障降级。
- `app/deepseek.py`：真实模型客户端（stream + complete/tools）。
- `app/prompts.py`：差/好两套提示词模板与变量渲染。
- `app/prompt_lab.py`：单元03 同题同数据对比。
- `app/tools.py`：四个受控工具 handler；从注册表派生 Schema 以兼容单元04。
- `app/registry.py`：单元05 工具注册表、调度器、调用日志；新增工具只在这里加 `register()`。
- `app/tool_loop.py`：工具调用闭环（`answer_with_registry` 与 unit04 兼容入口）。
- `app/data_store.py`：只读订单仓库（`模拟数据/records.json`，可导出同源 CSV）。
- `app/embedding.py`：单元06 向量化，`embed_many` / `embed` / `cosine_similarity`。
- `app/vector_lab.py`：单元06 课堂实验，文本清单、排序、边界核对、误判风险。
- `app/vector_store.py`：单元07 记录字段图与精确检索基线（`VectorRecord` / `InMemoryVectorStore`）。
- `setup_chroma.py` / `inspect_chroma.py`：单元07 真实 Chroma 建空集合与只读查看。
- `app/ingest.py`：单元08 资料准备，稳定 ID、来源与项目标识。
- `app/chroma_store.py`：单元08 真实写入与检索，集合定义以这里为唯一来源。
- `tests/`：单元01—08 单元测试，129 项（单元06 需 `embedding`、单元07/08 需 `chroma`）。
- `docs/课堂验收证据.md`：单元02—05 的真实运行记录。
- `docs/单元06验收证据.md`：单元06 的真实运行记录（由脚本生成，分数与报错均为实测）。
- `docs/单元07验收证据.md`：单元07 的真实运行记录（建库两次 + 只读查看，脚本抓取）。
- `docs/单元07选型决策卡.md`：向量库选型决策卡（数据规模、元数据字段、选型依据与重评条件）。
- `docs/单元08验收证据.md`：单元08 的真实运行记录（全新库首次导入 + 真实库重复运行两次，脚本抓取）。
- `docs/单元05当堂日报.md`：组长日报，完成标准核对与分工登记。
- `02-电商售后服务智能体-任务卡.docx`：项目范围、五人分工、阶段成果和验收要求。
- `模拟数据/records.json`：30 条虚构订单记录。
- `模拟数据/knowledge/`：15 份带 `DOC-02` 编号的售后知识文档。
- `模拟数据/sample_questions.json`：起始验收问题。
- `验证数据.py`：老师提供的数据检查脚本。

## 四个受控工具（全部只读）

| 工具 | 入参 | 用途 |
| --- | --- | --- |
| `query_order` | `order_id` | 只读核对订单事实 |
| `check_return_eligibility` | `order_id`, `reason` | 按样例政策给出退货结论与依据，争议转人工 |
| `draft_manual_ticket` | `order_id`, `issue` | 越权与纠纷的出路：只出草稿，不退款不改地址 |
| `lookup_policy_doc` | `doc_id` | 读取样例政策原文，支撑【政策依据】引用 |

## 小组任务与日报

从 [项目管理/README.md](项目管理/README.md) 查看第 01—08 单元的五人分工、提交清单、验收标准和每日记录。任务安排与实际完成情况分开记录；完成后在对应日期的日报中填写本人结果和证据链接，由组长审核。

## 来源

代码取自 `unit1.zip/项目起始模板`；电商资料取自 `unit1.zip/自选项目任务包/02-电商售后服务智能体`。`unit5.zip` 的工单与报销课堂示例属于另一个演示场景，未并入本工程。`unit6`、`unit7`、`unit8` 的做法已按电商售后语料改造后并入（语料换成 `模拟数据/sample_questions.json` 与 `模拟数据/knowledge/`，参考实现里的工单／报销语料一律不用，集合名、元数据与项目标识也一并改造）。
