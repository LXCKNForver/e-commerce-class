# 电商售后服务智能体

李省春小组的 Python 课堂项目。在老师发放的“项目起始模板”代码骨架之上，已按电商任务卡实现第 02—14 单元：多轮对话与流式输出、Prompt 模板与同题对比、订单工具调用闭环、工具注册表与调度器、售后文本向量化与相似度排序、向量库选型与 Chroma 空集合建库、售后资料批量写入与项目内检索、可复用检索服务（Top-K／来源过滤／批量）、RAG 问答闭环（无证据拒答／编号上下文／真实生成／引用核验）、检索侧优化（重叠分块与本地重排）、多格式知识库解析、RAG 评估与监控（两类指标 + SQLite 日志），以及 ReAct Agent（模型提动作、Python 执行检索、观察回填对话、max_steps 控制停止）。所有数据均为课程虚构样例，只读，不接真实电商系统。

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
uv run python scripts/generate_evidence_unit09.py
uv run python scripts/generate_evidence_unit10.py
uv run python scripts/generate_evidence_unit11.py
uv run python scripts/generate_evidence_unit12.py
uv run python scripts/generate_evidence_unit13.py
uv run python scripts/generate_evidence_unit14.py
```

`--extra embedding`（单元06 向量模型）与 `--extra chroma`（单元07—10 向量库）都是可选依赖。只跑单元 02—05 时 `uv sync` 即可，其余命令不受影响。

单元06 的中文向量模型约 180MB，放在 `models/`，已用 `.gitignore` 排除、不随仓库分发；换电脑时在 `.env` 里设置 `EMBEDDING_MODEL_DIR` 指向课程材料的 `models/` 目录即可，仍保持离线加载、不联网下载。

单元07 的 Chroma 数据库落在 `.chroma_unit7/`（运行产物，同样不进仓库）。课堂验收要求的顺序是：

```powershell
uv run python setup_chroma.py      # 建空集合，记下版本/目录/集合名/ID/元数据/0 条记录
uv run python inspect_chroma.py    # 只读查看
uv run python setup_chroma.py      # 再跑一次，核对集合 ID 不变
```

单元08 沿用同一个集合定义（同名、同元数据、同向量模型），但把库文件写到 `.chroma_unit8/`，以保住单元07「0 条记录」的验收现场；想直接写进单元07 那个库时，在 `.env` 里设 `CHROMA_DB_DIR=.chroma_unit7`。写入命令是交互式的 `/ingest`（见下），流程为：读 15 篇资料 → 记录导入前数量 → 批量 upsert → 重新连接核对记录数与集合 ID → 项目内 Top 3 检索。重复运行时稳定 ID（`项目标识:DOC-02-05`）让写入变成更新，记录数不会增长。

单元09 的检索服务再用一个独立库 `.chroma_unit9/`——任务卡要求「数据库保存在本单元代码目录，不复制第八单元数据库」，所以三个单元的库目录互不干扰；需要改路径时用 `CHROMA_DB_DIR`（单元08）、`CHROMA_UNIT9_DB_DIR`（单元09）与 `CHROMA_UNIT10_DB_DIR`（单元10）。查询命令是 `/retrieve`：Top-3/Top-5、来源过滤、跨项目返回 0 条、两个问题的批量检索，以及一次空问题报错，全部一条命令跑完。

单元10 的 RAG 问答再用一个独立库 `.chroma_unit10/`（任务卡要求「数据库留在本单元目录」）。它完全复用单元09 的检索服务，只多了编排：检索 → 无证据拒答 → 编号上下文 → 真实 DeepSeek 生成 → 引用核验，命令是 `/rag [问题]`。**无证据时压根不调用模型**（`/rag` 会打印「模型累计调用次数：0」），这需要 `.env` 里配好 `DEEPSEEK_API_KEY` 且 `LLM_PROVIDER=deepseek`；密钥本身就留在本机，不进输出也不进仓库。

单元11 在单元10 的问答流程上做优化，用独立库 `.chroma_unit11/`，库里两个集合并存：整篇 `after_sale_whole_docs`（15 条）与子块 `after_sale_chunks`（45 块）。命令是 `/chunk [问题]`：80 字符窗口 + 20 字符重叠切分 → 整篇 Top3 与分块 Top5 并列 → 本地 BGE Cross-Encoder 重排取前三 → 两版回答与引用对照。**两版共用同一个模型与同一份提示词**，只有候选来源不同（靠 `RAGService.answer_from_hits()` 锁死生成环节），所以差异只能归因于检索。本地重排权重 `bge-reranker-base` 约 1.1GB，不进仓库；把课程材料里的权重目录填进 `.env` 的 `RERANK_MODEL_DIR`（留空则读 `models/bge-reranker-base`），否则 `/chunk` 会在重排环节前提示不可用。

单元12 换了个方向：不问答，只解决「三种格式怎么变成同一种记录」。资料在 `模拟数据/单元12项目资料/`，两组各 3 篇 Markdown + 1 份 PDF + 1 份 CSV（本组 7 条、对照项目 7 条）。命令是 `/parse [项目]`（`ecom` 默认、`travel`）：Markdown 整篇一条（`document`）、PDF 每页一条（`page`）、CSV 每行一条（`row`，行号从 2 起），全部带上 `project_id` / `source` / `location` 后写进独立库 `.chroma_unit12/` 的**同一个集合**；再从数据库回读 PDF 第 1 页与 CSV 第 2 行，重跑验证条数不增长，最后切到另一个项目——库内 14 条，但按 `project_id` 数仍然各 7 条。**两个项目写在同一集合里，隔离靠元数据过滤，不靠分库。** PDF 解析要装 `pypdf`：`uv sync --extra parse`。

单元13 给前面所有单元装上"尺子"：用独立库 `.chroma_unit13/` 装载本组资料，对根目录 `evaluation_cases.json` 里的 3 道评估题跑完整闭环——**检索与回答复用单元09/10**（不重写检索），本单元只加两类打分与落库。命令是 `/eval`（跑完 3 题）、`/eval 1`（先跑 1 题确认，对应任务卡的 `main.py --limit 1`）、`/eval summary`（汇总均值与反馈）、`/eval feedback 日志ID good|bad`（给单题打标记）。两类指标必须分开看：`Precision@3` / `Recall@3` 是**规则算的**（比对人工标注的来源文件，零模型调用；Recall 按去重来源算，重复片段不虚增覆盖率）;`Context Precision` / `Context Recall` / `Faithfulness` 是 **DeepSeek 当评审给的模型判断**。日志落在本地 `rag_monitor.sqlite3` 的 `rag_logs` 表（九列：ID、时间、题号、问题、Precision、Recall、RAGAS、耗时、反馈），会话结束后可复盘。RAGAS 评审要装 `ragas`：`uv sync --extra evaluation`（须 0.4.3 及以上，0.3.x 与新版 langchain-community 冲突）。

单元14 让前面攒下来的能力第一次在同一个循环里协作：复用单元12 的本组资料与单元09 的检索服务，但这次**轮次由模型决定**——每轮它只输出一个 JSON 动作（search_knowledge / finish），Python 校验白名单后真正去检索，再把结果以「工具观察：」回填对话，让模型据此决定是否再查一轮。命令是 `/act [问题]`（单事项）、`/act double`（双事项，看第二轮 query 怎么被改写）、`/act limit`（`max_steps=2` 重跑，看未完成的停止状态）、`/act demo`（按任务卡顺序跑完三段并做轨迹对比）。资料装进独立库 `.chroma_unit14/` 的集合 `after_sale_agent_knowledge`。**限制轮次只改变停在哪一轮**：耗尽时 `stop_reason=max_steps`、答案为空，且不会再偷偷多问一次模型；那时的观察片段不是答案。`LLM_PROVIDER=mock` 时会换成离线脚本策略（动作按“逐项检索、检索完汇总”写死），方便零成本看轨迹形状——但它的答复由真实检索片段拼成，不编造政策。

默认 `.env` 使用离线 `mock` 模式，不需要 API Key。课程要求真实模型时，先执行 `uv pip install --python .venv/Scripts/python.exe -r requirements-deepseek.txt`，再在本机 `.env` 填写自己的 `DEEPSEEK_API_KEY` 并把 `LLM_PROVIDER` 改成 `deepseek`；真实模式使用 `.venv/Scripts/python.exe main.py` 启动。真实调用会产生 API 用量。`.env`、`.venv`、`models/`、`.chroma_unit7/`～`.chroma_unit13/`、`rag_monitor.sqlite3` 不会提交到 GitHub。

依赖按单元分组，装齐用：`uv sync --extra deepseek --extra embedding --extra chroma --extra parse --extra evaluation`。**注意 `uv sync` 会按 extras 重算依赖**，少写一个 extra 就会把对应的包卸掉（例如不带 `deepseek` 会卸掉 `openai`，真实模型模式当场失效）。

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
/retrieve        单元09：检索服务 Top-3/Top-5 + 来源过滤 + 项目隔离 + 批量查询
/rag [问题]      单元10：RAG 问答——拒答分支、编号上下文、真实生成、引用核验
/chunk [问题]    单元11：80字符分块召回Top5 → 本地BGE重排取前三 → 两版回答对照
/parse [项目]    单元12：md/pdf/csv 解析 → 7条入库 → 回读PDF页与CSV行 → 重跑 → 换项目
                项目填 ecom（默认）或 travel
/eval [参数]    单元13：评估闭环——检索/回答/两类指标/SQLite 日志/反馈/汇总
                不带参数跑完 3 题；/eval 1 只跑 1 题；/eval summary 看汇总；
                /eval feedback 2 bad 给第 2 条日志打反馈
/act [参数]     单元14：ReAct——模型提动作，Python 执行真实检索
                不带参数用单事项问题；/act 你的问题 自定义问题；
                /act double 跑双事项问题（观察第二轮检索）；
                /act limit 用 max_steps 2 重跑（看未完成的停止状态）；
                /act demo 按任务卡顺序跑完三段并做轨迹对比
/demo            跑通 3 个正常问题 + 2 个越权问题
/history /clear /help /exit
```

`/embed` 用 `模拟数据/sample_questions.json` 里 6 条电商售后样例问题做候选，把自拟问题和它们一起转成 512 维向量后排序，输出前三条，并附空文本／维度不一致的实测拦截结果。它**不做问答、不读知识库**，只比较“像不像”。

`/vector` 把 15 份售后知识文档 + 6 条样例问题共 21 条写入内存集合，展示记录字段图、精确遍历 Top-K、按 `doc_type` 过滤与 `tenant_id` 项目隔离。它是**全量遍历的精确基线**，后面量 Recall@K 时拿它当标准答案；不是真正的向量数据库。

`/ingest` 走真实 Chroma：把 `模拟数据/knowledge/` 的 15 篇售后文档用单元06 同一个模型转成 512 维向量后批量 upsert，重新连接核对记录数与集合 ID，再按 `project_id` 过滤做 Top 3 检索（同时给出 l2 距离与余弦相似度）。它**只做检索，不生成业务答案**——距离/相似度说明的是"像不像"，不是正确答案。

`/retrieve` 是单元09 的检索服务入口：`RetrievalService.search()` 与 `search_many()` 返回统一的 `SearchResult`（原文、来源、项目标识、资料类型、距离、余弦），单条接口直接委托批量接口，两个入口的排序与过滤规则只有一份实现。`project_id` 是必选过滤条件、留空直接报错；`source` 为可选的精确过滤。**命中不足 K 条时实际返回条数会少于 K**，这是过滤生效的正常表现。

`/rag` 是单元10 的 RAG 闭环：`RAGService.answer()` 先调检索，**没有证据时直接返回拒答话术、一次模型都不调用**（用 `RecordingLLM` 计数器证明调用次数为 0）；有证据时把每条编上 `[1][2][3]` 编号，编号与来源文件一一对应，再连同问题交给真实 DeepSeek；最后抽取回答里的方括号编号，**只保留"引用了且编号合法"的来源，出现一个越界编号就整条判为不可核验**。注意这条核验只能证明编号存在且合法，证明不了答案每句话都忠于原文。

`/chunk` 是单元11 的优化对照：把 15 篇资料按 80 字符窗口、20 字符重叠切成 45 块（子块 ID 带起止位置，重复导入只更新不增长），同一问题分别走「整篇 Top3」和「分块 Top5 → BGE 重排取前三」。BGE 输出的是相关性 logit（越大越相关），与 Chroma 的平方 L2 距离方向相反、量纲不同，不能混算也不能当概率。**重排只改顺序、不增删候选**，候选池之外的证据一条都找不回来——直接证据能不能进候选，靠的是分块而不是重排。

`/parse` 是单元12 的解析入口：`parse_file()` 按后缀分三支（`.md` 整篇 / `.pdf` 逐页 / `.csv` 逐行），出口统一成 `KnowledgeRecord`，上层完全不知道资料原来是什么格式。稳定 ID = `项目:文件名:位置类型:位置`，所以重跑是更新而不是追加。回读用 `read_record()`（`collection.get(ids=...)`），证明原文与元数据真的落了盘。**坏文件不阻断整批**，逐个记录失败原因；但整批解析出 0 条会直接报错，不会静默写一个空批次。

`/eval` 是单元13 的评估入口：一题走四步——检索（复用单元09 `RetrievalService`）→ 回答（复用单元10 `RAGService`）→ 打分（`retrieval_scores` 规则算 + `ragas_scores` 让 DeepSeek 当评审）→ 落库（`LogStore.record()` 写进 `rag_logs`）。**Precision@3 按条数计、Recall@3 按去重来源计**，所以同一份 CSV 的三行全命中也不会让覆盖率虚增。来源在比较前统一取路径最后一段，因为题集标注文件名、库里存相对路径。**评审调用失败写进 `error` 如实记录，绝不写假分数**；汇总时这些行不进均值。反馈只打标记、不改动已落库分数。

`/act` 是单元14 的 Agent 入口：每轮模型给出一个 JSON 动作，`run()` 先校验动作在白名单里（`delete_collection` 这种提议没有对应的执行分支），再交给单元09 的检索服务真正去查（`where project_id` 在 Chroma 端过滤），命中结果回填进对话成为下一轮的上下文。**双事项问题的第 2 轮**最能说明问题：它的 query 是模型看完第一轮观察后改写的，不是把原话拆开查两遍。`max_steps` 耗尽时回报 `max_steps` 而不是假装答完——这是完成任务的比例问题，不是答案质量问题。

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
- `app/chroma_store.py`：单元08 真实写入与检索（`query_many` 批量查询），集合定义以这里为唯一来源。
- `app/retriever.py`：单元09 检索服务（`SearchResult` / `RetrievalService` / `render_results`）。
- `app/rag.py`：单元10 RAG 编排（`RAGService` / `RAGResult` / `RecordingLLM`）。
- `app/rag_prompts.py`：单元10 RAG 提示词，与单元03 订单问答提示词分开维护。
- `app/chunking.py`：单元11 重叠分块，80 字符窗口 + 20 字符重叠，起止位置进 ID。
- `app/rerank.py`：单元11 本地 BGE Cross-Encoder 重排（`rerank_hits` / `load_reranker` / `rerank_ready`）。
- `app/rag_opt.py`：单元11 两版对照编排（整篇 Top3 vs 分块 Top5→重排前三）。
- `app/loaders.py`：单元12 三格式解析（`KnowledgeRecord` / `parse_file` / `load_materials`）。
- `app/parse_lab.py`：单元12 流水线编排（解析 → 入库 → 回读 → 重跑 → 换项目）。
- `app/evaluation.py`：单元13 两类指标（规则 `retrieval_scores` + RAGAS `ragas_scores`）。
- `app/monitoring.py`：单元13 SQLite 日志（`LogStore`：记录 / 反馈 / 汇总 / 读表）。
- `app/eval_lab.py`：单元13 闭环编排（解析入库 → 检索回答 → 打分 → 落库 → 反馈 → 汇总）。
- `app/agent.py`：单元14 ReAct 循环（`ReActAgent` / `AgentStep` / `AgentResult`、DeepSeek 与离线脚本两种决策模型）。
- `app/agent_lab.py`：单元14 流水线编排（装资料 → 单/双/限轮次三个场景 → 轨迹对比 → 渲染）。
- `tests/`：单元01—14 单元测试，291 项（单元06 需 `embedding`、单元07—11 需 `chroma`、单元12 另需 `parse`、单元13 另需 `evaluation`）。
- `docs/课堂验收证据.md`：单元02—05 的真实运行记录。
- `docs/单元06验收证据.md`：单元06 的真实运行记录（由脚本生成，分数与报错均为实测）。
- `docs/单元07验收证据.md`：单元07 的真实运行记录（建库两次 + 只读查看，脚本抓取）。
- `docs/单元07选型决策卡.md`：向量库选型决策卡（数据规模、元数据字段、选型依据与重评条件）。
- `docs/单元08验收证据.md`：单元08 的真实运行记录（全新库首次导入 + 真实库重复运行两次，脚本抓取）。
- `docs/单元09验收证据.md`：单元09 的真实运行记录（Top-3/Top-5、来源过滤、跨项目空、批量、边界，脚本抓取）。
- `docs/单元10验收证据.md`：单元10 的真实运行记录（拒答分支调用数 0、编号上下文、真实回答与引用核验，脚本抓取）。
- `docs/单元11验收证据.md`：单元11 的真实运行记录（分块观察、两版召回、BGE 重排、直接证据定位、两版回答，脚本抓取）。
- `docs/单元12验收证据.md`：单元12 的真实运行记录（解析清单、入库与回读、重跑与换项目、按项目计数、七类边界，脚本抓取）。
- `docs/单元13验收证据.md`：单元13 的真实运行记录（题集、单题与三题闭环、rag_logs 表、反馈与汇总、指标说明、低分样例分析、边界，脚本抓取）。
- `docs/单元14验收证据.md`：单元14 的真实运行记录（准备三件套、单事项、双事项、限轮次、轨迹对比、五条理由、六类边界，脚本抓取）。
- `docs/单元05当堂日报.md`：组长日报，完成标准核对与分工登记。
- `02-电商售后服务智能体-任务卡.docx`：项目范围、五人分工、阶段成果和验收要求。
- `模拟数据/records.json`：30 条虚构订单记录。
- `模拟数据/knowledge/`：15 份带 `DOC-02` 编号的售后知识文档。
- `模拟数据/sample_questions.json`：起始验收问题。
- `模拟数据/单元12项目资料/`：课程实践项目数据包的两组资料（本组 3 篇 md + 1 份 PDF + 1 份 CSV；对照项目同结构），共解析 14 条。
- `模拟数据/单元13评估题/`：课程为两个项目组提供的评估题原件各一份（本组那一份同时放在项目根 `evaluation_cases.json` 作为激活题集）。
- `验证数据.py`：老师提供的数据检查脚本。

## 四个受控工具（全部只读）

| 工具 | 入参 | 用途 |
| --- | --- | --- |
| `query_order` | `order_id` | 只读核对订单事实 |
| `check_return_eligibility` | `order_id`, `reason` | 按样例政策给出退货结论与依据，争议转人工 |
| `draft_manual_ticket` | `order_id`, `issue` | 越权与纠纷的出路：只出草稿，不退款不改地址 |
| `lookup_policy_doc` | `doc_id` | 读取样例政策原文，支撑【政策依据】引用 |

## 小组任务与日报

从 [项目管理/README.md](项目管理/README.md) 查看第 01—14 单元的五人分工、提交清单、验收标准和每日记录。任务安排与实际完成情况分开记录；完成后在对应日期的日报中填写本人结果和证据链接，由组长审核。

## 来源

代码取自 `unit1.zip/项目起始模板`；电商资料取自 `unit1.zip/自选项目任务包/02-电商售后服务智能体`。`unit5.zip` 的工单与报销课堂示例属于另一个演示场景，未并入本工程。`unit6`、`unit7`、`unit8`、`unit9`、`unit10`、`unit11`、`unit12`、`unit13`、`unit14` 的做法已按电商售后语料改造后并入（语料换成 `模拟数据/sample_questions.json` 与 `模拟数据/knowledge/`，参考实现里的工单／报销语料一律不用，集合名、元数据、项目标识、检索问题与真实业务问题也一并改造）。
