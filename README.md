# 电商售后服务智能体

李省春小组的 Python 课堂项目。在老师发放的“项目起始模板”代码骨架之上，已按电商任务卡实现第 02—16 单元：多轮对话与流式输出、Prompt 模板与同题对比、订单工具调用闭环、工具注册表与调度器、售后文本向量化与相似度排序、向量库选型与 Chroma 空集合建库、售后资料批量写入与项目内检索、可复用检索服务（Top-K／来源过滤／批量）、RAG 问答闭环（无证据拒答／编号上下文／真实生成／引用核验）、检索侧优化（重叠分块与本地重排）、多格式知识库解析、RAG 评估与监控（两类指标 + SQLite 日志），ReAct Agent（模型提动作、Python 执行检索、观察回填对话、max_steps 控制停止）与 Agent 工具集（三项只读工具 + 白名单 + 结构化查询 + 计算工具），以及 LangChain 核心（Runnable + Prompt + ChatDeepSeek + StrOutputParser 组合成链，同题对照旧 Agent）。所有数据均为课程虚构样例，只读，不接真实电商系统。

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
uv run python scripts/generate_evidence_unit15.py
uv run python scripts/generate_evidence_unit16.py
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

单元15 把「一个检索工具」换成「一套工具集」：同一份本组 CSV 有两条去路——逐行进 Chroma 靠**相似度**被找到，整表进 SQLite 靠**编号**精确命中（`app/faq_db.py`，只读 URI + 固定参数化 SQL `SELECT ... WHERE question_id = ?`）。三项只读工具按 `name` / `description` / `parameters` / `execute` 四要素注册进 `ToolRegistry`，**模型看得见的清单就是程序肯执行的清单**：`search_knowledge`（Chroma）、`lookup_faq`（SQLite）、`compute_return_deadline`（Python 日期加法）。命令是 `/tools [问题]`（单工具）、`/tools combo`（组合问题，看两个工具连着调）、`/tools calc`（计算题）、`/tools reject`（未知工具与非法参数被拒）、`/tools demo`（按任务卡顺序跑完四段）。**计算值不能由模型猜**：`window_days` 必须与资料原文里的「N天」一致（`window_days=30` 会被拒，因为 `DOC-02-01` 里没有这个写法），日期加法交给 `datetime`。未知工具、多字段、少字段、格式错、注入串全部在 `authorize_tool()` 里拦下——被拒的工具一次数据库访问都不会发生。向量库 `.chroma_unit15/`、FAQ 库 `.sqlite_unit15/faq.db`。

单元16 不新增业务能力，只把上面这些能力**串成一条看得见的链**：`RunnableLambda` 包住单元09 的 `search()`（检索逻辑一个字没改），`RunnablePassthrough.assign` 一路**加字段**——`hits` → `context` / `sources` → `answer`，所以一次 `invoke` 就把问题、命中片段、证据文本、来源和答案全带回来；`ChatPromptTemplate` 把「资料为空就只许回答资料不足」写进 system；`ChatDeepSeek` 是唯一调用模型的地方；`StrOutputParser` 只做 `AIMessage → str` 的类型转换。命令是 `/chain [问题]`（组合链一次跑）、`/chain steps`（逐段打印六段的输入输出）、`/chain compare`（同题对照单元15 ToolAgent）、`/chain empty`（资料不足）、`/chain bad`（无效输入）、`/chain demo`（按任务卡四节课跑完）。**对照比的是硬事实不是措辞**：两边都引用 `DOC-02-01-普通商品退货规则.md`、都得出「7天」就算一致。证据门槛（Top1 余弦相似度 ≥ 0.55）是在本组 7 条资料上实测标定的演示值，**不是正确率也不是通用阈值**，换资料必须重新标定。原有 Agent 与工具集一行没删，新链不覆盖旧实现；离线模式把第⑤段换成摘抄模型，其余五段代码一行不动。向量库 `.chroma_unit16/`。LangChain 依赖要装：`uv sync --extra langchain`。

单元17 把单元15 注册表里那三项只读工具**原样包成 LangChain 的 `BaseTool`**，交给 `langchain.agents.create_agent` 生成的执行器去跑循环——工具还是那套白名单能力（一个字没改），变的是「循环不再手写」：`ToolCallLog` 把每次工具调用的输入、输出/失败原因、耗时都记下来，所以工具出错能定位到第几步；`max_steps` 是**模型决策次数**上限，触顶就收尾且不产出答案（不会拿框架那句 "Model call limits exceeded" 当回答）。命令是 `/lcagent [问题]`（最小场景一次跑）、`/lcagent combo`（连续调用两个工具）、`/lcagent error`（工具报错定位到第几步）、`/lcagent limit`（步数上限触顶）、`/lcagent reject`（清单外的工具提了也执行不了）、`/lcagent compare`（同题对照单元15 手写的循环）、`/lcagent tools`（执行器配置与工具接线）、`/lcagent demo`（按任务卡四节课跑完）。**工具清单里没有的名字执行器根本不认**：模型只能挑名字、填参数，路径、SQL、代码一概碰不到；清单外的那一步在我们自己的日志里是 **0 条**。资料与库沿用单元15（`.chroma_unit15/` + `.sqlite_unit15/`），本单元不另建库。

单元18 换的是**谁来编排**：单元16 是一条直线、单元17 让模型决定下一步，本单元把「谁先做、走哪条分支」写死在程序里——`route_question()` 用三条纯规则（`EC-FAQ-NN` 编号 / 日期+时限词 / 售后业务词）判定分支，**路由不消耗模型调用**，同样的问题永远走同一条分支；`collect_evidence()` 按分支只查其中一类资料（按编号查 SQLite / 向量检索 / 检索原文 + Python 算日期），出口统一成 `evidence` / `sources` / `has_evidence`；`RunnableBranch` 判两道条件——先看是不是 `unknown`（三条规则都没命中 → 安全兜底），再看有没有证据（低于相似度门槛 → 资料不足），两者都**不检索或检索了也不调模型**，模型只在「摘要」「润色」这两步说话。每步的输入字段、输出字段、耗时与说明都记在 `StepLog` 里（`/flow steps` 可看），所以能分清是取证据、摘要还是润色的责任。命令是 `/flow [问题]`（知识分支）、`/flow faq`、`/flow deadline`、`/flow unknown`（兜底，0 次模型调用）、`/flow empty`（有分支没证据，0 次模型调用）、`/flow routes`（流程图 + 分支规则表）、`/flow steps`、`/flow bad`、`/flow demo`（按任务卡四节课跑完）。**四条分支各配一条本组真实问题**——走不到的分支等于没实现；兜底话术只说明「没接住」并转人工，不猜、不编。资料与工具仍是单元12/15 那一套（`.chroma_unit15/` + `.sqlite_unit15/`），本单元不另建库。

单元19 给上面那个执行器加上**记忆**，顺序固定为 **读 → 答 → 写**：`load()` 先按会话回忆（完整历史 / 摘要 / 向量长期记忆），并把「刚才那项」用规则解析成具体编号；`run_agent()` 再**重新**调工具取事实——记忆只解决「哪一项」，不解决「答案是什么」；只有正常完成的轮次才 `save()` 回记忆，失败轮次不写。三种模式的差别在**存活范围**：完整历史与摘要只在内存里（进程一退出就没了），向量长期记忆落盘到 `.chroma_unit19_memory/`（换个进程仍可召回，证据脚本用**两个独立 Python 进程**验证）。会话隔离是双保险——内存键带 `项目:会话` 前缀，长期记忆按 `source=session:<id>` 在库端过滤。对话记忆库与业务知识库严格分开：混在一起，检索证据时就会把上一轮说过的话当成制度原文引用。命令是 `/memory [问题]`（接着问就是下一轮）、`/memory buffer｜summary｜vector｜hybrid`、`/memory flow`、`/memory modes`、`/memory restart`、`/memory isolation`、`/memory new 会话名`、`/memory bad`、`/memory demo`。

单元20 换的是**先出计划、再逐步执行**：单元17 让模型边想边做、单元18 把顺序与分支写死在程序里，本单元先让模型把目标拆成一张 1~3 步的 JSON 计划——`make_plan()` 只产出「工具名 + 参数 + 这一步解决什么」，路径、SQL、代码一概进不了 `PlanStep`；再由 Python 按 cursor 逐步调用单元15 注册表里的真实工具（`execute_plan()`），每成功一步记一条观察（第几步、工具、参数、真实返回值、耗时）。失败时不猜、也不无限重试：把「第几步、什么错、已经拿到了什么」交回规划器，**只允许改一次**（`MAX_REPLANS = 1`），改完从第 1 步重跑，之前成功的观察留作审计证据；修订后仍失败就抛 `PlanAborted`——状态 aborted、答复为空，**连最后一次总结模型都不调用**，「没解决」绝不落成「完成」。两件事严格分开：`BAD` 是格式错（异常分支 → 触发重规划），`EC-FAQ-99`（格式合法但库里没有）只是「未找到」，是正常观察、不触发重规划——任务卡的故障提示点名的正是这一条。执行状态只在本次运行内保存（`TaskPlan` 在内存里，不落库），与单元19 的长期记忆正好相反。命令是 `/plan [目标]`（最小场景，单步）、`/plan multi`（三步：查答复 → 检索规则 → 算截止日期）、`/plan replan`（参数格式错 → 重规划一次 → 跑完）、`/plan abort`（修订后仍失败 → 终止并报错）、`/plan missing`（编号合法但查不到 → 未找到）、`/plan flow`（流程图与概念对照）、`/plan bad`、`/plan demo`（按任务卡四节课跑完）。

单元21 换的是**谁来干活、谁来复核**：前几个单元都是一条单人流水线（同一个模型既拆计划又写答复，写完就交付），本单元把它拆成一支队伍——规划、执行、证据核验、答复、评审**五个专职角色**通过**共享黑板**交接，由**协调者** `coordinate()` 调度（协调者不调模型，只做调度与放行判断）。每个角色的输入都是上一步写进黑板的产物：规划只看目标、执行只看计划、证据核验只看观察、答复只看观察 + 证据意见（+ 评审意见）、评审只看目标 + 观察 + 证据 + 草稿；规划与执行直接复用单元20（只给 `execute_plan` 加了 `compose_answer=False`，答复不再由执行器代写）。黑板只存**产物**，每条消息都留发送者、内容与说明，所以「这份答复是怎么来的」能一条条追回去（黑板不落库，只活在内存里，与单元19 的长期记忆相反）。**答复智能体没有放行权限**：它只写草稿，草稿变成答复只能由协调者在评审通过后做。评审不通过时把理由交回答复智能体，**只许修订一次**（`MAX_REVISIONS = 1`），且只改表述、不许改工具事实；修订后仍不通过就**停止自动交付**——答复留空、状态转人工，两版草稿与两次评审意见留在黑板上给人工复核。执行阶段若终止（计划修订后仍失败），证据核验 / 答复 / 评审三个角色**根本不启动**：没有观察就没有草稿，没有草稿就无从评审。命令是 `/agents [目标]`（正常协作，三步计划）、`/agents revise`（评审拒绝 → 修订一次 → 通过 → 交付）、`/agents reject`（两次都不通过 → 停止交付、转人工）、`/agents gap`（证据不足 → 如实说明而不是编）、`/agents abort`（执行终止 → 连草稿都没有）、`/agents roles`（架构图 + 角色输入输出表）、`/agents bad`、`/agents demo`（按任务卡四节课跑完）。

单元22 换的是**协作的形态**：单元21 用共享黑板交接"字段化的产物"，本单元三个角色（规划员 / 答复执行员 / 评审员）同处**一间群聊**——消息按顺序累积，**后发言的人看得到前面所有人的原话**，由 `GroupChatManager` 决定下一位谁说话、什么时候停（管理器自己不调模型）。`max_round` 是**消息条数**硬上限，**含发起者那条**（3 个角色各发言一次要写 4，这是 AutoGen 的口径，也是课堂最容易数错的一格）；成本另算一条线：按 1.6 字符≈1 token 估算，超预算就不再让下一位开口。两条上限互相独立、检查都发生在花钱之前，触发时状态一律是「未交付」。**复核不通过不立刻收尾**：群聊继续下一轮，执行员看到复核意见再改一版，直到有人通过或轮次/成本用完；仍未通过就答复留空、转人工。**执行员写的永远是草稿**，只有评审员 `passed=true` 才变成正式答复；复核输出不是 JSON 或缺布尔字段，一律判**不通过**（"看不懂就当通过"是最危险的默认值）。四种终止原因分开写：`reviewed` / `review_rejected` / `max_round` / `token_budget`。工具在群聊开始前由 Python 调好（群聊里没有工具，模型无法新增事实）。关于 AutoGen：任务卡允许「AutoGen 或**等价轻量实现**」，本组选后者——`ChatAgent` / `GroupChat` / `GroupChatManager` 与 AutoGen 0.2 的三个对象一一对应、行为相同，但不引入 `pyautogen`（它会带进 openai/docker/flaml 一整套），取舍写在 `app/chat_crew.py` 文件头。命令是 `/group`（正常协作）、`/group tight`（只给 2 轮 → 复核没轮到）、`/group budget`（预算花完 → 钱先用完、轮次还有余）、`/group reject`（复核不通过 → 答复留空）、`/group gap`（编号查不到 → 如实说明）、`/group roles`、`/group bad`、`/group demo`。

单元23 换的是**让模型碰数据、但不让它碰执行**：本单元目标写得很直白——Agent 安全读取项目 CSV，完成指定统计并用业务语言解释结果，**不执行任意生成代码**。守住这条的是三道设计：第一，模型两次发言都不碰数据本体——第一次只给它数据字典、字段白名单和 8 个固定算子（`count / sum / mean / max / min / group_count / group_sum / group_mean`），它只能回一个 JSON 口径（`{"metric": ..., "field": ..., "filters": {...}}`），**看不到一行数据、拿不到文件路径**；第二次只给它 Python 已经算好的结果，让它翻译成业务语言。第二，真正的加法全在 `compute()` 的 8 个分支里，模型想塞一段表达式（比如 `sum(x for x in rows)`）会被当成"未知算子"拒掉，`app/data_agent.py` 里没有 `eval(` / `exec(` / `__import__(`（有一条测试用 `inspect.getsource` 盯着）。第三，写完解释还有一道 `check_explanation`：每个带单位的数字都要能在结果里找到出处，找不到就点名列出——模型可以说"物流异常 8 单（25.0%）"，但不能自己造一个 8。**字段白名单的意义是"列存在 ≠ 可以统计"**：`case_id`、`order_id` 在表里真的有，但统计编号没有业务含义、还可能泄露客户信息，一律在统计之前拒绝（统计 0 次，连解释都不调）。**空值与非数字既不静默当 0，也不整行丢弃**：脏样本里跳过 3 行、参与 29 行，均值 162.34 元（当成 0 算只有 147.12），分组字段为空则归入「（空值）」组，行数一条都对得上。每个结果都带四行行数（总 / 筛选后 / 参与 / 跳过）+ 一条含除号的算式 + 前 5 行抽样，可以直接拿计算器复核。命令是 `/data`（最小场景）、`/data schema`（数据字典与白名单）、`/data refund`（数值聚合）、`/data handoff`（带筛选计数）、`/data dirty`（空值与非数字）、`/data forbidden`（越权字段）、`/data missing`（筛选后 0 行）、`/data bad`（守门）、`/data demo`（按任务卡四节课跑完）。

默认 `.env` 使用离线 `mock` 模式，不需要 API Key。课程要求真实模型时，先执行 `uv pip install --python .venv/Scripts/python.exe -r requirements-deepseek.txt`，再在本机 `.env` 填写自己的 `DEEPSEEK_API_KEY` 并把 `LLM_PROVIDER` 改成 `deepseek`；真实模式使用 `.venv/Scripts/python.exe main.py` 启动。真实调用会产生 API 用量。`.env`、`.venv`、`models/`、`.chroma_unit7/`～`.chroma_unit16/`、`.sqlite_unit15/`、`rag_monitor.sqlite3` 不会提交到 GitHub。

依赖按单元分组，装齐用：`uv sync --extra deepseek --extra embedding --extra chroma --extra parse --extra evaluation --extra langchain --extra classic`。**注意 `uv sync` 会按 extras 重算依赖**，少写一个 extra 就会把对应的包卸掉（例如不带 `deepseek` 会卸掉 `openai`，真实模型模式当场失效）。其中 `classic` 只给课程参考实现用（单元17～19 的示例代码 import `langchain_classic` / `langchain_chroma`），**本项目自己的代码不 import 它**，装上只是为了能在本机把参考实现跑起来做对照。

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
/tools [参数]    单元15：Agent 工具集——三项只读工具 + 白名单
                不带参数跑单工具；/tools combo 组合问题；/tools calc 计算题；
                /tools reject 未知工具与非法参数被拒；/tools demo 跑完四段
/chain [参数]    单元16：LangChain LCEL 链——检索 → assign → Prompt → 模型 → Parser
                不带参数用本组真实问题跑一次组合链；/chain 你的问题 自定义；
                /chain steps 逐段打印六段的输入输出；
                /chain compare 同一问题对照单元15 的 ToolAgent（只比对硬事实）；
                /chain empty 资料不足时的处理；/chain bad 无效输入被守门拦下；
                /chain demo 按任务卡四节课顺序跑完一遍
/lcagent [参数]  单元17：LangChain Agent 执行器——同一批工具，循环交给框架跑
                不带参数跑最小业务场景（按编号查标准答复）；/lcagent 你的问题 自定义；
                /lcagent combo 连续调用两个工具；/lcagent error 工具报错定位到第几步；
                /lcagent limit 步数上限触顶；/lcagent reject 清单外的工具；
                /lcagent compare 对照单元15 手写的循环；/lcagent tools 执行器配置；
                /lcagent demo 按任务卡四节课顺序跑完一遍
/flow [参数]     单元18：Chain 编排——路由 → 取证据 → 分支 → 摘要 → 润色
                不带参数跑知识分支；/flow 你的问题 自定义；/flow faq 按编号精确查；
                /flow deadline 查时限再算截止日期；/flow unknown 兜底（0 次模型调用）；
                /flow empty 有分支没证据（0 次模型调用）；/flow routes 流程图与分支规则表；
                /flow steps 逐段打印每步的输入输出；/flow bad 无效输入被守门拦下；
                /flow demo 按任务卡四节课顺序跑完一遍
  /memory [参数]   单元19：对话记忆——读 → 答 → 写
                不带参数问一句（默认问 EC-FAQ-01），接着问就是同一会话的下一轮；
                /memory buffer|summary|vector|hybrid 用指定模式连跑两轮；
                /memory flow 记忆流程与两类库的分工；/memory modes 三种模式对比；
                /memory restart 重启后还能召回什么；/memory isolation 两个会话不串话；
                /memory new 会话名 换一个会话；/memory bad 无效输入被守门拦下；
                /memory demo 按任务卡四节课顺序跑完一遍
  /plan [参数]     单元20：Plan-and-Solve——先出计划，再逐步执行
                不带参数跑最小场景（单步：按编号查标准答复）；/plan 你的目标 自定义；
                /plan multi 三步计划；/plan replan 参数格式错 → 重规划一次 → 跑完；
                /plan abort 修订后仍失败 → 终止并报错；/plan missing 编号合法但查不到；
                /plan flow 流程图与概念对照；/plan bad 无效输入；/plan demo 跑完四节课
  /agents [参数]   单元21：多智能体架构——五角色 + 共享黑板 + 协调者
                不带参数跑正常协作（计划 → 观察 → 证据 → 草稿 → 评审通过 → 交付）；
                /agents 你的目标 自定义；/agents revise 评审拒绝 → 修订一次 → 通过；
                /agents reject 两次都不通过 → 停止自动交付、转人工复核；
                /agents gap 证据不足 → 如实说明；/agents abort 执行终止 → 无草稿；
                /agents roles 架构图与角色表；/agents bad 无效输入；/agents demo 跑完四节课
  /group [参数]    单元22：AutoGen 实战——三角色同处一间群聊（轮次/成本双上限）
                不带参数跑正常协作（证据 → 规划 → 草稿 → 复核通过 → 交付）；
                /group 你的任务 自定义；/group tight 只给 2 轮（复核没轮到）；
                /group budget 预算花完就停；/group reject 复核不通过 → 答复留空；
                /group gap 编号查不到 → 如实说明；/group roles 群聊时序与角色表；
                /group bad 无效输入与越界参数；/group demo 跑完四节课
  /data [参数]     单元23：数据分析 Agent——安全读本组 CSV（白名单 + 8 个固定算子）
                不带参数跑最小场景（各类问题各多少单；口径与解释都由模型写）；
                /data 你的问题 自定义；/data schema 数据字典与字段白名单；
                /data refund 已退款金额合计；/data handoff 需要转人工的工单；
                /data dirty 空值与非数字（跳过并如实报数）；/data forbidden 越权字段；
                /data bad 无效问题与越界文件名；/data demo 跑完四节课
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

`/chain` 是单元16 的组合链入口：链上六段里只有第⑤段是模型，其余五段都是 Python——`RunnableLambda` 取证据、`assign` 加字段、`ChatPromptTemplate` 编消息、`StrOutputParser` 取字符串。**用 `assign` 而不是管道替换**，是因为加字段才能把 `question` / `hits` / `sources` 一路留到最后；换成替换式写法，来源就只剩模型自己写的那句话，无从核对。`inspect_stages=True` 时会逐段 invoke 并记下真实输入输出，课堂上一眼能看出「问题在第②段之后一直都在」。`/chain compare` 跑的是**同一份资料、同一个检索服务**下的两条路径，比对的是来源文件名与「7天」这类硬事实——任务卡写明无需逐字比较措辞。

`/tools` 是单元15 的工具集入口：每轮模型仍只给一个 JSON 动作，但动作从「检索」变成「从注册表里挑一个工具」，`ToolRegistry.call()` 先过白名单与参数格式再进真实业务函数。**组合问题的第 2 轮**最能说明问题：模型先按编号查 SQLite，看到观察后才换 `search_knowledge` 去检索 Chroma——两轮落在两个不同的数据源上。`compute_return_deadline` 是第三类：它不查库，只做日期加法，但天数必须先在资料原文里出现过。拒绝路径返回的 `tool_rejected` 与轮次耗尽的 `max_steps` 是两种不同的停止状态：**前者是提议没被放行（0 次调用），后者是轮次用完（有调用但没答完）**，都不能拿半途的观察当答案。

`/lcagent` 是单元17 的执行器入口：工具还是 `/tools` 那三项，换的是循环不再手写——`build_tools()` 把注册表里的 `ToolSpec` 包成 `BaseTool`（名字、说明、参数全部来自注册表，描述里没有路径、SQL 或代码），`create_agent` 负责「决策 → 执行 → 观察 → 再决策」。**错误定位靠我们自己的 `ToolCallLog`**：框架只把提议发出去，有没有真执行、成没成以日志为准，所以清单外的工具那一步是 0 条、坏参数那一步带着步号和原因回填给模型。`max_steps` 是**模型决策次数**上限（一次决策要么提工具调用、要么给答案），触顶时 `answer` 为空、只留下执行器的上限提示。**工具报错、步数上限、清单外的工具这三个异常场景用离线剧本模型跑**（模型那一栏写着 `scripted-tool-agent`），要的是可复现的异常路径，与真实模型好不好无关——工具、白名单、日志和上限都是真的。

`/flow` 是单元18 的编排入口：**步骤由程序写死，模型只负责「摘要」和「润色」两步**。路由 `route_question()` 是纯函数（三条正则按优先级判：`EC-FAQ-NN` → 日期+时限词 → 售后业务词 → 兜底），所以同样的问题永远同一条分支、且不消耗模型调用；`collect_evidence()` 按分支只查其中一类资料，出口统一成同一个 `EvidencePack`。`RunnableBranch` 判两道条件：`unknown` 先兜底（**不检索、不调模型**），无证据再兜底（检索了但过不了门槛，同样**0 次模型调用**）。**「未知类型」是一条正常分支而不是异常处理**——所以知识分支刻意只收 19 个售后业务词、不收「什么/怎么」，否则任何闲聊都能被它吃掉、兜底永远走不到。每步的输入字段、输出字段、耗时与说明都进 `StepLog`，字段数逐级上涨（只加不改），出问题能分清是取证据、摘要还是润色的责任；摘要单独成一步是为了这个可归因性，合并成一次调用更省但说不清。日期分支里「N天」必须来自资料原文、日期加法由 Python 做（`compute_return_deadline` 会核对原文），日期不合法或资料没写天数一律判无证据、不让模型口算。

`/memory` 是单元19 的入口：**记忆只解决指代，事实永远来自工具**。每轮固定三步——`load()` 回忆（完整历史 / 摘要 / 向量长期记忆，并把「刚才那项」解析成具体编号）、`run_agent()` 作答（工具照查，所以第2轮问「刚才那项」时步骤表里仍有 `lookup_faq`）、`save()` 落记忆（只写正常完成的轮次）。指代解析刻意用规则而不是「让模型自己体会」：解析结果要能打印、能断言、能复现，而模型内部的理解看不见；且先看该轮问句里的编号、取第一个匹配，否则答复末尾的来源编号 `DOC-02-01` 会被当成指代对象。三种模式的差别在**存活范围**而非强弱：完整历史与摘要只活在当前进程（重启即空），向量长期记忆落在 `.chroma_unit19_memory/`，换个进程还能召回——证据脚本为此跑了两个独立进程。会话隔离是双保险：内存键带 `项目:会话` 前缀，长期记忆按 `source=session:<id>` 在库端过滤，不靠「查回来再自己筛」。长期记忆只存「问了什么 + 涉及哪个编号」（答案截断 160 字），离线摘要干脆不抄答案原文——记忆的职责是让下一轮知道在聊哪个对象，不是保存事实。

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
- `app/faq_db.py`：单元15 结构化 FAQ 库（CSV → SQLite，只读 URI + 固定参数化 SQL）。
- `app/toolkit.py`：单元15 三项只读工具 + 白名单校验 + 工具注册表（四要素）。
- `app/tool_agent.py`：单元15 多工具 ReAct 循环（决策 → 放行 → 执行 → 观察 → 再决策）。
- `app/tool_lab.py`：单元15 流水线编排（两份数据源 → 注册 → 四个场景 → 轨迹与清单渲染）。
- `app/lc_chain.py`：单元16 LCEL 链（六段组装、证据门槛、离线/真实模型工厂、逐段记录与渲染）。
- `app/chain_lab.py`：单元16 流水线编排（装库 → 组链 → 四节课场景 → 同题对照 → 渲染）。
- `app/lc_agent.py`：单元17 工具适配 + 执行器 + 步骤日志（`build_tools` / `build_executor` / `run_agent` / `ToolCallLog`、离线剧本模型）。
- `app/lc_agent_lab.py`：单元17 流水线编排（装工具 → 组执行器 → 四节课场景 → 同题对照 → 渲染）。
- `app/flow_chain.py`：单元18 路由 + 取证据 + 条件分支 + 步骤日志（`route_question` / `collect_evidence` / `build_workflow` / `StepLog`、离线编排模型）。
- `app/memory_chain.py`：单元19 三种记忆与会话隔离（`AgentMemoryManager` / `VectorMemory` / `SummaryEngine` / `resolve_reference` / `run_turn`）。
- `app/memory_lab.py`：单元19 流水线编排（装资料与执行器 → 四节课场景 → 三模式对比 → 跨进程召回与隔离 → 逐条验收）。
- `app/flow_lab.py`：单元18 流水线编排（装资料与工具 → 组工作流 → 五条路径逐条验证 → 横向对照表）。
- `app/plan_solver.py`：单元20 规划器与执行器（`make_plan` / `execute_plan` / `TaskPlan` / `StepObservation` / `PlanAborted`、离线剧本模型）。
- `app/plan_lab.py`：单元20 流水线编排（装资料与工具 → 五条路径 → 四节课 → 逐条验收）。
- `app/multi_agent.py`：单元21 五角色与共享黑板（`planner_agent` / `executor_agent` / `analyze_evidence` / `compose_answer` / `review_answer` / `coordinate` / `Blackboard`、离线剧本模型）。
- `app/multi_lab.py`：单元21 流水线编排（装资料与工具 → 五条路径 → 四节课 → 逐条验收）。
- `app/chat_crew.py`：单元22 群聊协作（`ChatAgent` / `GroupChat` / `GroupChatManager` 三个等价 AutoGen 对象、`Collaboration`、成本估算、复核协议、离线剧本模型）。
- `app/chat_lab.py`：单元22 编排（装资料与工具 → 六条路径 → 四节课 → 逐条验收）。
- `app/data_agent.py`：单元23 数据分析（数据字典与字段白名单、`resolve_source` 目录守门、`load_table` / `scan_issues`、8 个固定算子的 `compute`、两次模型调用的分工与提示词、`check_explanation` 数字核验）。
- `app/data_lab.py`：单元23 编排（装本组 CSV → 六条路径 → 四条标准 → 守门 → 概念表）。
- `tests/`：单元01—23 单元测试，750 项（单元06 需 `embedding`、单元07—11 需 `chroma`、单元12 另需 `parse`、单元13 另需 `evaluation`、单元16—23 另需 `langchain`；真实模型用例需显式 `RUN_REAL_MODEL=1`）。
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
- `docs/单元15验收证据.md`：单元15 的真实运行记录（数据源与工具清单、单工具、连续调用、拒绝路径、计算工具与取舍、九类边界，脚本抓取）。
- `docs/单元16验收证据.md`：单元16 的真实运行记录（可复用件与资料来源、六段流转、组合链一次运行、同题对照与资料不足、换模型只换第五段、九类边界，脚本抓取）。
- `docs/单元17验收证据.md`：单元17 的真实运行记录（执行器配置与工具接入、最小场景、连续调用与步骤追踪、工具报错/步数上限/清单外工具、同题对照旧循环、换模型不换工具、九类边界，脚本抓取）。
- `docs/单元18验收证据.md`：单元18 的真实运行记录（资料与工具清单、流程与分支规则、路由逐条验、三条分支各跑一次、每步输入输出、兜底/无证据/无效输入、五条路径对照、换模型不换编排、六类边界，脚本抓取）。
- `docs/单元19验收证据.md`：单元19 的真实运行记录（资料与两类库分工、三轮连续问答、第2轮回忆到的内容、三种模式对比、跨进程召回、会话隔离、无效输入、换模型不换记忆、四条标准逐条验收，脚本抓取）。
- `docs/单元20验收证据.md`：单元20 的真实运行记录（资料与工具、最小场景、多步骤场景、每步输入输出、重规划、终止、未找到、无效输入、四条标准逐条验收，脚本抓取）。
- `docs/单元21验收证据.md`：单元21 的真实运行记录（架构与角色表、真实模型跑一次完整协作、每角色输入输出字段、评审修订、二次拒绝停止交付、证据不足、执行终止、无效输入、三条标准逐条验收，脚本抓取）。
- `docs/单元22验收证据.md`：单元22 的真实运行记录（群聊结构与角色表、真实模型跑一次完整协作、逐位核对每人读到的提示词、轮次上限、成本上限、复核不通过、多轮往复、证据不足、无效任务与越界参数、四条标准逐条验收，脚本抓取）。
- `docs/单元23验收证据.md`：单元23 的真实运行记录（数据字典与白名单、真实模型跑一次完整分析、两次调用各自读到什么、数值与筛选、脏数据、越权字段、筛选后 0 行、三道守门、不执行生成代码的源码核对、四条标准逐条验收，脚本抓取）。
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

从 [项目管理/README.md](项目管理/README.md) 查看第 01—16 单元的五人分工、提交清单、验收标准和每日记录。任务安排与实际完成情况分开记录；完成后在对应日期的日报中填写本人结果和证据链接，由组长审核。

## 来源

代码取自 `unit1.zip/项目起始模板`；电商资料取自 `unit1.zip/自选项目任务包/02-电商售后服务智能体`。`unit5.zip` 的工单与报销课堂示例属于另一个演示场景，未并入本工程。`unit6`、`unit7`、`unit8`、`unit9`、`unit10`、`unit11`、`unit12`、`unit13`、`unit14`、`unit15`、`unit16` 的做法已按电商售后语料改造后并入（语料换成 `模拟数据/sample_questions.json` 与 `模拟数据/knowledge/`，参考实现里的工单／报销语料一律不用，集合名、元数据、项目标识、检索问题与真实业务问题也一并改造）。
