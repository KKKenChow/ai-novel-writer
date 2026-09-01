"""Phase 2/3/4 测试：套话检测 / 台账合并 / 伏笔告警 / 滚动摘要 / 文风指纹 / 导出"""
import json
import os
import pytest
from workflow.novel_workflow import FullNovelWorkflow
from workflow.text_quality import detect_cliches, cliche_report, cliche_avoidance_instruction
from storage.exporters import build_markdown, build_docx, build_epub


class FakeVS:
    def __init__(self):
        self.extra = {}
        self.added = []
    def delete_section(self, t, title): pass
    def add_section(self, t, title, content): self.added.append((t, title, content))
    def update_section(self, t, title, content): pass
    def get_section(self, t, title): return None
    def search_related(self, q, n_results=5): return []
    def get_all_by_type(self, t): return []
    def save_extra_data(self, k, v):
        if v is None:
            self.extra.pop(k, None)
        else:
            self.extra[k] = v
    def delete_extra_field(self, k):
        self.extra.pop(k, None)
    def load_extra_data(self, k=None, default=None):
        return self.extra.get(k, default) if k else self.extra


class FakeAPI:
    MAX_TOKENS_LIMIT = 32768
    model = "fake"
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.prompts = []
    def generate(self, prompt, step="", **kw):
        self.prompts.append(prompt)
        if "full_summary" in prompt:
            return json.dumps({
                "delta": {"characters": [{"name": "林凡", "status": "受伤"}],
                          "timeline": [{"chapter": 1, "event": "进城"}],
                          "foreshadowing": [{"item": "纸条", "planted_chapter": 1, "status": "未回收"}]},
                "full_summary": "全书梗概：主角进城。",
                "recent_summary": "近期摘要：主角进城，遇见师父。"}, ensure_ascii=False)
        return self.outputs.pop(0) if self.outputs else "默认。" * 50


def make_wf(outputs=()):
    return FullNovelWorkflow(FakeAPI(list(outputs)), FakeVS())


# ---------- 2.2 套话黑名单 ----------

def test_detect_cliches():
    text = "他不禁握紧了拳头，嘴角勾起一抹冷笑，空气仿佛凝固了。"
    hits = detect_cliches(text)
    words = [w for w, _ in hits]
    assert "不禁" in words and "嘴角勾起" in words and "空气仿佛凝固" in words


def test_cliche_report_empty_when_clean():
    assert cliche_report("他转身离开了房间，脚步很急。") == ""


def test_cliche_instruction_includes_custom():
    ins = cliche_avoidance_instruction(["邪魅一笑"])
    assert "邪魅一笑" in ins and "不禁" in ins


def test_chapter_generation_injects_cliche_avoidance():
    wf = make_wf(["正文内容。" * 200])
    wf.novel_info["outline"] = "第1章 开始 —— 起始"
    wf.generate_chapter(1, "开始", max_tokens=16000, target_words=1000)
    main_prompt = next(p for p in wf.api.prompts if "正文：" in p)
    assert "避免套话" in main_prompt


def test_cliche_detection_warning_after_generate():
    wf = make_wf(["他不禁握紧了拳头。" * 150])
    wf.novel_info["outline"] = "第1章 开始 —— 起始"
    wf.generate_chapter(1, "开始", max_tokens=16000, target_words=1000)
    assert "套话" in wf.last_scope_warning


# ---------- 3.1/3.2 状态台账 ----------

def test_merge_ledger_characters():
    old = {"characters": [{"name": "林凡", "status": "炼气期"}], "timeline": [], "foreshadowing": []}
    delta = {"characters": [{"name": "林凡", "status": "筑基期"}]}
    merged = FullNovelWorkflow.merge_ledger(old, delta)
    assert merged["characters"][0]["status"] == "筑基期"


def test_merge_ledger_foreshadowing_recycle():
    old = {"characters": [], "timeline": [],
           "foreshadowing": [{"item": "神秘玉佩", "planted_chapter": 1, "status": "未回收"}]}
    delta = {"foreshadowing": [{"item": "神秘玉佩", "status": "已回收"}]}
    merged = FullNovelWorkflow.merge_ledger(old, delta)
    assert merged["foreshadowing"][0]["status"] == "已回收"
    assert merged["foreshadowing"][0]["planted_chapter"] == 1  # 保留旧字段


def test_update_state_ledger_persists():
    wf = make_wf()
    ledger = wf.update_state_ledger(1, "正文内容")
    assert ledger["characters"][0]["name"] == "林凡"
    assert wf.vs.extra["state_ledger"]["foreshadowing"][0]["item"] == "纸条"


def test_update_state_ledger_bad_json_keeps_old():
    old = {"characters": [{"name": "甲", "status": "好"}], "timeline": [], "foreshadowing": []}

    class BadAPI(FakeAPI):
        def generate(self, prompt, step="", **kw):
            self.prompts.append(prompt)
            return "这不是JSON"
    wf = FullNovelWorkflow(BadAPI(), FakeVS())
    wf.vs.extra["state_ledger"] = old
    ledger = wf.update_state_ledger(1, "正文")
    assert ledger["characters"][0]["name"] == "甲"   # 失败保留旧值
    assert wf.vs.extra["ledger_stale"] is True        # B4：失败标记待重建，不再静默


def test_foreshadowing_recovery_warning():
    wf = make_wf()
    wf.vs.extra["state_ledger"] = {"foreshadowing": [
        {"item": "A", "status": "未回收"}, {"item": "B", "status": "未回收"}]}
    assert "伏笔回收率" in wf.foreshadowing_recovery_warning()
    wf.vs.extra["state_ledger"] = {"foreshadowing": [
        {"item": "A", "status": "已回收"}, {"item": "B", "status": "已回收"}]}
    assert wf.foreshadowing_recovery_warning() == ""


# ---------- 台账优化：人工修正层 / 分块配额 / 伏笔排序 / 清空 ----------

def test_apply_ledger_fix_survives_rebuild():
    """人工修正层：标记伏笔已回收后，rebuild 不会被 AI 重建覆盖"""
    wf = make_wf()
    wf.vs.extra["state_ledger"] = {"characters": [], "timeline": [], "foreshadowing": [
        {"item": "纸条", "planted_chapter": 1, "status": "未回收"}]}
    wf.vs.extra["ledger_deltas"] = {"1": {"foreshadowing": [
        {"item": "纸条", "planted_chapter": 1, "status": "未回收"}]}}
    merged = wf.apply_ledger_fix("foreshadowing", "纸条", {"status": "已回收"})
    assert merged["foreshadowing"][0]["status"] == "已回收"
    # 再重建：修正层仍生效
    wf.rebuild_memory_from_deltas()
    assert wf.vs.extra["state_ledger"]["foreshadowing"][0]["status"] == "已回收"


def test_ledger_brief_quotas_and_ordering():
    """A1 分块配额 + A2 活跃度排序 + B1 伏笔排序 + A3 里程碑事件"""
    wf = make_wf()
    wf.vs.extra["state_ledger"] = {
        "characters": [
            {"name": "沉睡甲", "status": "旧状态", "updated_chapter": 1},
            {"name": "活跃乙", "status": "新状态", "updated_chapter": 20},
        ],
        "timeline": [
            {"chapter": 3, "event": "埋纸条伏笔"},
            {"chapter": 20, "event": "最近事件"},
        ],
        "foreshadowing": [
            {"item": "远期", "planted_chapter": 3, "target_chapter": 90, "status": "未回收"},
            {"item": "逾期", "planted_chapter": 3, "target_chapter": 18, "status": "未回收"},
            {"item": "临近", "planted_chapter": 3, "target_chapter": 22, "status": "未回收"},
        ],
    }
    brief = wf._ledger_brief(20, {"phase": "mid_dev"})
    # A2：活跃角色排前面
    assert brief.index("活跃乙") < brief.index("沉睡甲")
    # B1：逾期伏笔最先出现
    assert brief.index("逾期") < brief.index("临近") < brief.index("远期")
    # A3：里程碑事件（埋纸条伏笔的第3章）也注入
    assert "第3章" in brief


def test_clear_memory_deep():
    """C4：deep=False 保留按章快照；deep=True 彻底清空"""
    wf = make_wf()
    wf.vs.extra["ledger_deltas"] = {"1": {"characters": []}}
    wf.vs.extra["state_ledger"] = {"characters": [{"name": "甲"}]}
    wf.vs.extra["rolling_summary_recent"] = "摘要"
    wf.vs.extra["ledger_manual_fixes"] = {"characters": {}}
    wf.clear_memory(deep=False)
    assert "state_ledger" not in wf.vs.extra
    assert "ledger_deltas" in wf.vs.extra          # 快照保留
    wf.clear_memory(deep=True)
    assert "ledger_deltas" not in wf.vs.extra
    assert "ledger_manual_fixes" not in wf.vs.extra


# ---------- 3.3 滚动摘要 ----------

def test_rolling_summary_update():
    wf = make_wf()
    s = wf.update_rolling_summary(1, "正文")
    assert "师父" in s
    assert wf.vs.extra["rolling_summary_recent"] == s


def test_rolling_summary_injected_in_context():
    wf = make_wf()
    wf.vs.extra["rolling_summary"] = "前情：主角已获传承。"
    wf.novel_info["outline"] = "第5章 战斗 —— 大战"
    ctx = wf._build_chapter_context(5, "战斗")
    assert "前情：主角已获传承。" in ctx["context_text"]


# ---------- 2.1 文风指纹 ----------

def test_style_fingerprint_persisted_and_injected():
    wf = make_wf(["- 句式：短句为主\n- 对话：口语化", "正文。" * 200])
    fp = wf.extract_style_fingerprint(sample_text="样例文字")
    assert "短句" in fp
    assert wf.vs.extra["style_fingerprint"] == fp
    wf.novel_info["outline"] = "第1章 开始 —— 起始"
    ctx = wf._build_chapter_context(1, "开始")
    assert "文风要求" in ctx["context_text"]


# ---------- 4.3 导出 ----------

CHAPTERS = {"2": {"title": "发展", "content": "第二章内容\n第二段"}, "1": {"title": "开始", "content": "第一章内容"}}

def test_build_markdown_sorted():
    md = build_markdown("测试书", CHAPTERS)
    assert md.index("第1章") < md.index("第2章")

def test_build_docx(tmp_path):
    pytest.importorskip("docx")
    path = str(tmp_path / "test.docx")
    build_docx("测试书", CHAPTERS, path)
    assert os.path.getsize(path) > 1000

def test_build_epub(tmp_path):
    path = str(tmp_path / "test.epub")
    build_epub("测试书", CHAPTERS, path)
    import zipfile
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert names[0] == "mimetype"  # EPUB 规范
        assert "OEBPS/content.opf" in names and "OEBPS/toc.ncx" in names
        assert "OEBPS/chap_1.xhtml" in names and "OEBPS/chap_2.xhtml" in names
        toc = zf.read("OEBPS/toc.ncx").decode()
        assert "第1章 开始" in toc
