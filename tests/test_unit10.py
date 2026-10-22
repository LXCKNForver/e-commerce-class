"""单元10 验收测试：检索 → 拒答 → 编号上下文 → 生成 → 引用核验。

真实 Chroma 一律写在临时目录，不碰单元08/09/10 各自的库。
模型部分用本地替身，因此**不依赖 DeepSeek 网络与余额**：本单元要验证的是编排与核验规则，
真实调用的那条链路由 `scripts/generate_evidence_unit10.py` 单独留下证据。
缺本地向量模型或 chromadb 时相关用例跳过并说明原因。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.embedding import model_ready
from app.ingest import load_markdown

MODEL_AVAILABLE, MODEL_DETAIL = model_ready()

try:
    import chromadb  # noqa: F401

    from app.chroma_store import (
        DEFAULT_UNIT10_DB_PATH,
        PROJECT_ID,
        ChromaVectorStore,
        resolve_unit10_db_path,
    )
    from app.rag import (
        DEMO_QUESTION,
        EMPTY_PROJECT,
        NO_CITATION_ANSWER,
        NO_EVIDENCE_ANSWER,
        RAGResult,
        RAGService,
        RecordingLLM,
        render_result,
    )
    from app.rag_prompts import SYSTEM_PROMPT, build_messages
    from app.retriever import RetrievalService

    CHROMA_AVAILABLE = True
    CHROMA_DETAIL = ""
except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
    CHROMA_AVAILABLE = False
    CHROMA_DETAIL = str(exc)


class StubLLM:
    """本地替身模型：按脚本返回带编号的回答，用来验证编排与核验规则。"""

    def __init__(self, script: dict[str, str] | None = None, default: str = "依据资料回答 [1]。"):
        self.script = script or {}
        self.default = default
        self.prompts: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages)
        question = messages[-1]["content"].rsplit("\n", 1)[-1]
        for key, value in self.script.items():
            if key in question:
                return value
        return self.default


class BoomLLM:
    """抛出任意异常的模型，用来验证失败时如实上报而不是编答案。"""

    def chat(self, messages: list[dict[str, str]]) -> str:  # noqa: ARG002
        raise RuntimeError("模拟：余额不足")


def require_service(test_case: unittest.TestCase) -> None:
    if not MODEL_AVAILABLE:
        test_case.skipTest(f"本地向量模型不可用：{MODEL_DETAIL}")
    if not CHROMA_AVAILABLE:
        test_case.skipTest(f"缺少 chromadb：{CHROMA_DETAIL}；请执行 uv sync --extra chroma")


def temp_dir(test_case: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="chroma-unit10-"))
    test_case.addCleanup(shutil.rmtree, path, True)
    return path


@unittest.skipUnless(MODEL_AVAILABLE and CHROMA_AVAILABLE, "需要本地向量模型与 chromadb")
class TestUnit10RAG(unittest.TestCase):
    """真实检索 + 替身模型的完整链路。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = Path(tempfile.mkdtemp(prefix="chroma-unit10-cls-"))
        store = ChromaVectorStore(cls._dir)
        cls.record_count = len(load_markdown())
        store.upsert_documents(load_markdown())
        cls.store = store

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, True)

    def service(self, llm, top_k: int = 3) -> RAGService:
        return RAGService(RetrievalService(self.store), RecordingLLM(llm), top_k=top_k)

    # ------------------------------------------------------------ ① 检索复用
    def test_uses_unit09_retrieval_service(self):
        """第1课：本单元不重写检索，直接复用第九单元的 RetrievalService。"""
        hits = self.service(StubLLM()).retrieve(DEMO_QUESTION, PROJECT_ID, 3)
        self.assertEqual(len(hits), 3)
        for index, hit in enumerate(hits, start=1):
            with self.subTest(hit=hit.source):
                self.assertTrue(hit.source.startswith("模拟数据/knowledge/"))
                self.assertEqual(hit.project_id, PROJECT_ID)
                self.assertGreaterEqual(index, 1)

    def test_unit10_has_own_database_path(self):
        """第1课：数据库留在本单元目录，不复制第八/九单元的数据库。"""
        require_service(self)
        self.assertEqual(DEFAULT_UNIT10_DB_PATH.name, ".chroma_unit10")
        self.assertNotEqual(DEFAULT_UNIT10_DB_PATH.name, ".chroma_unit9")
        self.assertEqual(resolve_unit10_db_path(), DEFAULT_UNIT10_DB_PATH)

    # ------------------------------------------------------------ ② 未命中拒答
    def test_no_evidence_refuses_without_calling_model(self):
        """第2课：没有证据时拒答，并且一次模型都不调用。"""
        recorder = RecordingLLM(StubLLM())
        rag = RAGService(RetrievalService(self.store), recorder)
        result = rag.answer(DEMO_QUESTION, EMPTY_PROJECT)

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.hits, [])
        self.assertFalse(result.llm_called)
        self.assertTrue(result.refused)
        self.assertEqual(recorder.call_count, 0)

    def test_blank_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service(StubLLM()).answer("   ", PROJECT_ID)

    def test_blank_project_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service(StubLLM()).answer(DEMO_QUESTION, "")

    # ------------------------------------------------------------ ③ 编号上下文
    def test_context_numbers_every_evidence_once(self):
        """第3课：编号与来源一一对应，模型只能在编号范围内引用。"""
        rag = self.service(StubLLM())
        hits = rag.retrieve(DEMO_QUESTION, PROJECT_ID, 3)
        context = rag.build_context(hits)

        for index, hit in enumerate(hits, start=1):
            with self.subTest(source=hit.source):
                self.assertIn(f"[{index}] 来源：{hit.source}", context)
        self.assertNotIn("[4]", context)
        self.assertNotIn("【当前问题】", context)  # 上下文里不带问题，问题单独拼装

    def test_messages_put_question_and_context_in_user_message(self):
        """提示词里 system 由程序控制，检索资料与问题都在 user 消息。"""
        messages = build_messages(DEMO_QUESTION, "[1] 来源：x\n内容：y")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertNotIn("DEMO_QUESTION", SYSTEM_PROMPT)
        self.assertIn("【检索资料】", messages[1]["content"])
        self.assertIn("【当前问题】", messages[1]["content"])

    def test_prompt_build_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            build_messages("", "上下文")
        with self.assertRaises(ValueError):
            build_messages("问题", "")

    # ------------------------------------------------------------ ④⑤ 生成与核验
    def test_valid_citation_is_mapped_to_real_source(self):
        """第4/5课：引用的编号必须换成真实存在的文件来源。"""
        rag = self.service(StubLLM(default="定制商品不适用普通退货规则 [2]。"))
        result = rag.answer(DEMO_QUESTION, PROJECT_ID)

        self.assertTrue(result.llm_called)
        self.assertEqual(result.cited, [2])
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0], result.hits[1].source)
        self.assertTrue(result.ok)

    def test_out_of_range_citation_is_rejected(self):
        """越界编号说明模型编了不存在的证据，一律判为不可核验。"""
        rag = self.service(StubLLM(default="这是我看出来的 [9]。"))
        result = rag.answer(DEMO_QUESTION, PROJECT_ID)

        self.assertEqual(result.answer, NO_CITATION_ANSWER)
        self.assertEqual(result.sources, [])
        self.assertFalse(result.ok)

    def test_answer_without_any_citation_is_rejected(self):
        rag = self.service(StubLLM(default="定制商品不能退。"))
        result = rag.answer(DEMO_QUESTION, PROJECT_ID)
        self.assertEqual(result.answer, NO_CITATION_ANSWER)
        self.assertEqual(result.sources, [])

    def test_model_saying_no_evidence_is_treated_as_refusal(self):
        rag = self.service(StubLLM(default=NO_EVIDENCE_ANSWER))
        result = rag.answer(DEMO_QUESTION, PROJECT_ID)
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, [])
        self.assertTrue(result.llm_called)  # 是模型说的不知道，不是我们没走生成

    def test_top_k_limits_context_size(self):
        """Top-K 决定模型能看到几条证据，因此也决定了合法编号的上限。"""
        rag_small = self.service(StubLLM(default="依据 [2] 回答。"), top_k=2)
        result = rag_small.answer(DEMO_QUESTION, PROJECT_ID, top_k=2)
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(result.cited, [2])

        rag_wide = self.service(StubLLM(default="依据 [3] 回答。"), top_k=3)
        wide = rag_wide.answer(DEMO_QUESTION, PROJECT_ID, top_k=3)
        self.assertEqual(len(wide.hits), 3)

    def test_model_failure_is_reported_not_faked(self):
        """模型调用失败要如实上报，绝不降级出一个看起来像答案的东西。"""
        result = self.service(BoomLLM()).answer(DEMO_QUESTION, PROJECT_ID)
        self.assertIsNotNone(result.error)
        self.assertIn("模拟：余额不足", result.error)
        self.assertEqual(result.sources, [])
        self.assertFalse(result.ok)

    def test_duplicate_citations_are_deduplicated(self):
        rag = self.service(StubLLM(default="先说 [1]，再补充 [1]，另外 [2]。"))
        result = rag.answer(DEMO_QUESTION, PROJECT_ID)
        self.assertEqual(result.cited, [1, 2])
        self.assertEqual(len(result.sources), 2)


class TestUnit10Helpers(unittest.TestCase):
    """不依赖 Chroma 的纯规则测试：编号抽取、渲染、计数器。"""

    def setUp(self) -> None:
        self.rag = RAGService(retriever=None, llm=None)  # type: ignore[arg-type]

    def test_extract_citations(self):
        self.assertEqual(self.rag.extract_citations("见 [1] 与 [2]。"), {1, 2})
        self.assertEqual(self.rag.extract_citations("没有引用。"), set())
        self.assertEqual(self.rag.extract_citations("[1][1][10]"), {1, 10})

    def test_verify_sources_rejects_out_of_range(self):
        class Hit:
            source = "s"

        self.assertEqual(self.rag.verify_sources("见 [1]。", [Hit(), Hit()]), [1])
        self.assertEqual(self.rag.verify_sources("见 [3]。", [Hit(), Hit()]), [])

    def test_recording_llm_counts_calls(self):
        recorder = RecordingLLM(StubLLM())
        self.assertEqual(recorder.call_count, 0)
        recorder.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(recorder.call_count, 1)

    def test_render_result_marks_refusal_branch(self):
        refused = RAGResult("q", NO_EVIDENCE_ANSWER)
        text = render_result(refused)
        self.assertIn("否（未命中分支）", text)
        self.assertIn("来源：无", text)

    def test_result_defaults_are_empty(self):
        result = RAGResult("q", "a")
        self.assertEqual(result.sources, [])
        self.assertEqual(result.cited, [])
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
