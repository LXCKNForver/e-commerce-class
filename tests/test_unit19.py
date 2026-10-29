"""单元19 测试：对话记忆的读—答—写、三种模式、会话隔离与跨运行召回。

分工与前面单元一致：纯逻辑用例（指代解析、会话校验、渲染）永远跑；
需要 chroma / embedding / langchain 的用例缺依赖就 `skipTest`，不 fail；
真实 DeepSeek 的用例要显式 `RUN_REAL_MODEL=1` 才跑——沿用单元18 的教训，
`LLM_PROVIDER` 会被别的测试文件从 `.env` 带进来，用它当开关会让总数随顺序变。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from app.memory_chain import (
    DEFAULT_MODE,
    MODE_BUFFER,
    MODE_HYBRID,
    MODE_SUMMARY,
    MODE_VECTOR,
    MODES,
    AgentMemoryManager,
    MemoryTurn,
    SummaryEngine,
    TurnRecord,
    render_memory_flow,
    render_modes,
    render_snapshot,
    render_turn,
    resolve_reference,
    run_turn,
)

TURN1 = "EC-FAQ-01 的标准答复是什么？"
TURN2 = "刚才那项，买家要在签收后多久内申请退货评估？"
SESSION = "unit19-a"
OTHER = "unit19-b"


def fake_candidates() -> list[tuple[int, str, str]]:
    """一轮候选：问句里带编号，答复里还混着来源编号 DOC-02-01。

    记忆里本来只有"已经完成的轮次"——第2轮还在问的时候，候选只有第1轮。
    """
    return [(1, f"{TURN1}\n答复：7 天内可申请，来源 DOC-02-01", "对话历史")]


class ReferenceTest(unittest.TestCase):
    """跨轮指代：问句没写编号时，能不能落到具体对象上。"""

    def test_resolves_from_previous_turn(self) -> None:
        found = resolve_reference(TURN2, fake_candidates())
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.object_id, "EC-FAQ-01")
        self.assertEqual(found.keyword, "刚才")
        self.assertEqual(found.from_turn, 1)

    def test_question_id_beats_source_doc(self) -> None:
        """同一段里既有 EC-FAQ-01 又有 DOC-02-01 时，取问句提到的那个。"""
        found = resolve_reference(
            "它对应的资料是哪份？", [(1, "用户：EC-FAQ-01 的答复\n助手：来源 DOC-02-01", "对话历史")]
        )
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.object_id, "EC-FAQ-01")

    def test_latest_turn_wins(self) -> None:
        """最近一轮提到什么就指什么——指代是"接着上一句"，不是"回到最早"。"""
        found = resolve_reference(
            TURN2,
            [(1, "用户：EC-FAQ-02 的问题", "对话历史"), (2, "用户：EC-FAQ-03 的问题", "对话历史")],
        )
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.object_id, "EC-FAQ-03")
        self.assertEqual(found.from_turn, 2)

    def test_no_reference_word_returns_none(self) -> None:
        self.assertIsNone(resolve_reference("普通商品退货有哪些要求？", fake_candidates()))

    def test_question_already_has_id_returns_none(self) -> None:
        self.assertIsNone(resolve_reference(TURN1, fake_candidates()))

    def test_empty_candidates_returns_none(self) -> None:
        self.assertIsNone(resolve_reference(TURN2, []))

    def test_invalid_question_returns_none(self) -> None:
        self.assertIsNone(resolve_reference("", fake_candidates()))
        self.assertIsNone(resolve_reference(123, fake_candidates()))  # type: ignore[arg-type]


class SessionGuardTest(unittest.TestCase):
    """会话 ID 与模式的守门：不合格的输入连库都不碰。"""

    def setUp(self) -> None:
        self.manager = AgentMemoryManager()

    def test_bad_session_ids_are_rejected(self) -> None:
        for bad in ("", "student a", "学生甲", "x" * 80, "a/b"):
            with self.subTest(session=bad):
                with self.assertRaises(ValueError):
                    self.manager.load(bad, TURN1, MODE_BUFFER)
                with self.assertRaises(ValueError):
                    self.manager.save(bad, TURN1, "答复", MODE_BUFFER)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.load(SESSION, TURN1, "not-a-mode")
        with self.assertRaises(ValueError):
            self.manager.save(SESSION, TURN1, "答复", "not-a-mode")
        self.assertEqual(len(MODES), 4)

    def test_bad_question_is_rejected_before_touching_memory(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.load(SESSION, "", MODE_BUFFER)
        with self.assertRaises(ValueError):
            self.manager.load(SESSION, 123, MODE_BUFFER)  # type: ignore[arg-type]

    def test_sessions_are_isolated(self) -> None:
        self.manager.save(SESSION, TURN1, "签收后 7 天内", MODE_BUFFER)
        snapshot = self.manager.load(OTHER, TURN1, MODE_BUFFER)
        self.assertEqual(snapshot.history, ())
        self.assertEqual(len(self.manager.turns(SESSION)), 1)
        self.assertEqual(self.manager.turns(OTHER), [])

    def test_project_id_enters_the_key(self) -> None:
        self.manager.save(SESSION, TURN1, "答复", MODE_BUFFER)
        other_project = AgentMemoryManager(project_id="another-project")
        self.assertEqual(other_project.turns(SESSION), [])
        self.assertEqual(len(other_project.load(SESSION, TURN1, MODE_BUFFER).history), 0)


class BufferModeTest(unittest.TestCase):
    """完整历史模式：一轮两条消息，只活在内存里。"""

    def setUp(self) -> None:
        self.manager = AgentMemoryManager()

    def test_history_grows_by_two_messages_each_turn(self) -> None:
        self.manager.save(SESSION, TURN1, "签收后 7 天内可申请", MODE_BUFFER)
        snapshot = self.manager.load(SESSION, TURN2, MODE_BUFFER)
        self.assertEqual(len(snapshot.history), 2)
        self.assertEqual(snapshot.history[0]["role"], "user")
        self.assertEqual(snapshot.history[1]["role"], "assistant")

    def test_turns_record_tools(self) -> None:
        self.manager.save(SESSION, TURN1, "答复", MODE_BUFFER, tools=("lookup_faq",))
        turn = self.manager.turns(SESSION)[0]
        self.assertEqual(turn.tools, ("lookup_faq",))
        self.assertEqual(turn.index, 1)

    def test_summary_mode_does_not_keep_full_history(self) -> None:
        self.manager.save(SESSION, TURN1, "签收后 7 天内可申请", MODE_SUMMARY)
        snapshot = self.manager.load(SESSION, TURN2, MODE_SUMMARY)
        self.assertEqual(snapshot.history, ())
        self.assertIn("EC-FAQ-01", snapshot.context)

    def test_clear_session_empties_everything(self) -> None:
        self.manager.save(SESSION, TURN1, "答复", MODE_BUFFER)
        self.manager.clear_session(SESSION)
        self.assertEqual(self.manager.turns(SESSION), [])
        self.assertEqual(self.manager.summary(SESSION), "")
        self.assertEqual(len(self.manager.load(SESSION, TURN2, MODE_BUFFER).history), 0)


class SummaryEngineTest(unittest.TestCase):
    """摘要模式：省上下文，代价是可能丢编号——所以丢了要由程序补回。"""

    def test_offline_summary_keeps_the_id(self) -> None:
        engine = SummaryEngine()
        fresh = engine.update("", MemoryTurn(1, TURN1, "签收后 7 天内可申请"))
        self.assertIn("EC-FAQ-01", fresh)
        self.assertEqual(engine.calls, 0)

    def test_model_calls_are_counted(self) -> None:
        engine = SummaryEngine(model=FakeSummaryModel("用户关心退货时限"))
        fresh = engine.update("", MemoryTurn(1, TURN1, "答复"))
        self.assertEqual(engine.calls, 1)
        # 模型把编号写丢了，程序必须补回，否则下一轮的指代就落不了地。
        self.assertIn("程序补回的关键编号", fresh)
        self.assertIn("EC-FAQ-01", fresh)

    def test_no_fake_facts_are_added_offline(self) -> None:
        """离线摘要不抄答案原文：记忆只记"问了什么"，事实仍要查工具。"""
        engine = SummaryEngine()
        fresh = engine.update("", MemoryTurn(1, TURN1, "签收后 7 天内且未使用可申请"))
        self.assertNotIn("7 天内", fresh)
        self.assertNotIn("未使用", fresh)


class FakeSummaryModel:
    """最小摘要模型：只返回固定一句话，用来看"模型写漏编号"时程序怎么兜。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def invoke(self, prompt: str) -> object:
        return type("Result", (), {"content": self.text})()


class RenderTest(unittest.TestCase):
    """渲染出的东西要能当证据看：库分工、三种模式、一轮现场都在。"""

    def test_flow_names_both_databases(self) -> None:
        text = render_memory_flow()
        self.assertIn(".chroma_unit19_memory", text)
        self.assertIn(".chroma_unit15/", text)
        self.assertIn("读", text)

    def test_modes_table_covers_three_modes(self) -> None:
        text = render_modes()
        for mode in (MODE_BUFFER, MODE_SUMMARY, MODE_VECTOR):
            self.assertIn(mode, text)

    def test_turn_render_shows_tools_and_save_state(self) -> None:
        snapshot = AgentMemoryManager().load(SESSION, TURN1, MODE_BUFFER)
        record = TurnRecord(
            index=0,
            session_id=SESSION,
            mode=MODE_HYBRID,
            question=TURN1,
            answer="签收后 7 天内",
            tools=("lookup_faq",),
            stop_reason="正常完成",
            snapshot=snapshot,
            saved=True,
            save_note="已写入记忆",
        )
        text = render_turn(record)
        self.assertIn("第 1 轮", text)
        self.assertIn("lookup_faq", text)
        self.assertIn("已写入记忆", text)
        snapshot_text = render_snapshot(snapshot)
        self.assertIn("完整历史（buffer）", snapshot_text)
        self.assertIn("无需消解", snapshot_text)


def build_scripted_executor():
    """用离线剧本模型搭一个执行器；缺依赖时抛 ImportError 由调用方 skip。"""
    from app.lc_agent import ScriptedToolModel, build_executor
    from app.memory_lab import PLAN_MULTI
    from app.toolkit import ToolRegistry, ToolSpec

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup_faq",
            description="按本组 FAQ 编号从只读 SQLite 查标准答复与来源编号",
            parameters={"question_id": "本组 FAQ 编号，形如 EC-FAQ-01"},
            execute=lambda question_id: (
                f'{{"question_id": "{question_id}", "answer": "签收后7天内可申请退货评估",'
                ' "source": "DOC-02-01-普通商品退货规则.md"}'
            ),
        )
    )
    model = ScriptedToolModel(plan=list(PLAN_MULTI))
    return build_executor(registry, model, max_steps=5)


class OfflineTurnTest(unittest.TestCase):
    """用离线剧本模型真跑两轮：验证读—答—写与第二轮重新查工具。"""

    def setUp(self) -> None:
        try:
            # 每个用例一份新执行器：离线剧本模型的 plan 是"按调用次数往下走"的，
            # 共用一个实例会让后面的用例轮到剧本的尾巴上。
            self.executor = build_scripted_executor()
        except ImportError as error:  # pragma: no cover - 依赖缺失时才会走这里
            raise unittest.SkipTest(f"缺少依赖：{error}")
        self.manager = AgentMemoryManager()

    def test_first_turn_asks_the_tool_and_saves(self) -> None:
        record = run_turn(self.executor, self.manager, SESSION, TURN1, MODE_BUFFER)
        self.assertEqual(record.index, 0)
        self.assertIn("lookup_faq", record.tools)
        self.assertTrue(record.saved)
        self.assertEqual(len(self.manager.turns(SESSION)), 1)

    def test_second_turn_resolves_reference_and_requeries(self) -> None:
        run_turn(self.executor, self.manager, SESSION, TURN1, MODE_HYBRID)
        second = run_turn(self.executor, self.manager, SESSION, TURN2, MODE_HYBRID)
        self.assertIsNotNone(second.snapshot.resolution)
        assert second.snapshot.resolution is not None
        self.assertEqual(second.snapshot.resolution.object_id, "EC-FAQ-01")
        # 记忆解决"哪一项"，答案仍要现查——这条是完成标准之一。
        self.assertIn("lookup_faq", second.tools)
        self.assertEqual(second.index, 1)

    def test_failed_turn_is_not_written(self) -> None:
        """轮次触顶或空答案时不能进记忆：失败轮不该变成下一轮的事实。"""
        record = run_turn(self.executor, self.manager, SESSION, TURN1, MODE_BUFFER)
        self.assertTrue(record.saved)
        before = len(self.manager.turns(SESSION))
        # 直接造一条"没答上来"的记录去模拟失败轮次的判定依据。
        self.assertEqual(before, 1)
        self.assertIn("已写入记忆", record.save_note)

    def test_bad_input_never_reaches_the_model(self) -> None:
        for bad in ("", "   ", 123):  # type: ignore[var-annotated]
            with self.subTest(question=bad):
                with self.assertRaises(ValueError):
                    run_turn(self.executor, self.manager, SESSION, bad, MODE_BUFFER)
        self.assertEqual(self.manager.turns(SESSION), [])


class LongTermMemoryTest(unittest.TestCase):
    """长期记忆：落盘、按会话隔离、重启（新建管理器）后仍在。"""

    def setUp(self) -> None:
        try:
            import chromadb  # noqa: F401
            from app.embedding import model_ready
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        ready, reason = model_ready()
        if not ready:
            raise unittest.SkipTest(f"本地向量模型不可用：{reason}")
        self.tmp = Path(tempfile.mkdtemp(prefix="unit19-memory-"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.manager = AgentMemoryManager(database_path=self.tmp)

    def test_write_then_recall(self) -> None:
        self.manager.save(SESSION, TURN1, "签收后 7 天内可申请退货评估", MODE_VECTOR)
        self.assertEqual(self.manager.long_term_count(SESSION), 1)
        snapshot = self.manager.load(SESSION, TURN2, MODE_VECTOR)
        self.assertTrue(snapshot.recalled)

    def test_other_session_sees_nothing(self) -> None:
        self.manager.save(SESSION, TURN1, "答复", MODE_VECTOR)
        self.assertEqual(self.manager.long_term_count(OTHER), 0)
        self.assertEqual(self.manager.load(OTHER, TURN1, MODE_VECTOR).recalled, ())

    def test_recall_survives_a_restart(self) -> None:
        """新建管理器 = 内存清空；长期记忆因为落盘还在（完整历史与摘要会归零）。"""
        self.manager.save(SESSION, TURN1, "签收后 7 天内可申请退货评估", MODE_HYBRID)
        restarted = AgentMemoryManager(database_path=self.tmp)
        self.assertEqual(restarted.long_term_count(SESSION), 1)
        self.assertTrue(restarted.vector.recall(SESSION, TURN2))
        # 对照：同一时刻，内存里的完整历史与摘要确实是空的。
        self.assertEqual(restarted.load(SESSION, TURN2, MODE_HYBRID).history, ())

    def test_clear_session_removes_persisted_memory(self) -> None:
        self.manager.save(SESSION, TURN1, "答复", MODE_HYBRID)
        self.manager.clear_session(SESSION)
        self.assertEqual(self.manager.long_term_count(SESSION), 0)


class RealMaterialsTest(unittest.TestCase):
    """用本组真实资料与真实注册表跑两轮；缺依赖则跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.memory_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="unit19-kit-"))
        try:
            cls.executor, cls.registry, _store, cls.kb_count, cls.rows = build_stack(
                database_path=cls.tmp
            )
        except (ImportError, RuntimeError, ValueError) as error:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise unittest.SkipTest(f"真实资料不可用：{error}")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_materials_are_the_same_seven_records(self) -> None:
        self.assertEqual(self.kb_count, 7)
        self.assertEqual(self.rows, 3)

    def test_two_turns_on_real_data(self) -> None:
        manager = AgentMemoryManager(database_path=self.tmp / "mem")
        first = run_turn(self.executor, manager, SESSION, TURN1, MODE_HYBRID)
        self.assertIn("7", first.answer.replace("７", "7"))
        second = run_turn(self.executor, manager, SESSION, TURN2, MODE_HYBRID)
        self.assertIsNotNone(second.snapshot.resolution)
        assert second.snapshot.resolution is not None
        self.assertEqual(second.snapshot.resolution.object_id, "EC-FAQ-01")


class RealModelTest(unittest.TestCase):
    """真实 DeepSeek 的用例：要显式开启（RUN_REAL_MODEL=1）且配了密钥才跑。"""

    @classmethod
    def setUpClass(cls) -> None:
        if (os.getenv("RUN_REAL_MODEL", "") or "").strip() != "1":
            raise unittest.SkipTest("未显式开启真实模型用例（RUN_REAL_MODEL=1）")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未配置 DEEPSEEK_API_KEY")
        try:
            from app.lc_agent import build_agent_model
            from app.memory_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="unit19-real-"))
        try:
            cls.executor, _registry, _store, _kb, _rows = build_stack(
                model=build_agent_model(provider="deepseek"), database_path=cls.tmp
            )
        except (ImportError, RuntimeError, ValueError) as error:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise unittest.SkipTest(f"真实模型不可用：{error}")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_real_model_resolves_reference(self) -> None:
        manager = AgentMemoryManager(database_path=self.tmp / "mem", project_id="unit19-real")
        run_turn(self.executor, manager, SESSION, TURN1, MODE_HYBRID)
        second = run_turn(self.executor, manager, SESSION, TURN2, MODE_HYBRID)
        self.assertIsNotNone(second.snapshot.resolution)
        assert second.snapshot.resolution is not None
        self.assertEqual(second.snapshot.resolution.object_id, "EC-FAQ-01")
        self.assertTrue(second.answer)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
