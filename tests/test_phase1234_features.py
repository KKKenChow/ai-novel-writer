"""TODO 阶段 1-4 新功能测试：
- 1.1 台账 delta 记录/撤销/重建
- 1.2 外部章节导入  1.3 空白章节
- 2.1 角色卡生成/解析/降级  2.3 登场章过滤注入
- 3.1 extend_outline 增量扩展
- 4.1 登场调度检查 + 大纲局部改写  4.2 回溯影响扫描
"""
import json
import pytest
from workflow.novel_workflow import FullNovelWorkflow
from workflow import character_cards as cc


class FakeVS:
    def __init__(self):
        self.added = []
        self.extra = {}
        self.sections = {}
    def delete_section(self, t, title): self.sections.pop((t, title), None)
    def add_section(self, t, title, content):
        self.added.append((t, title, content))
        self.sections[(t, title)] = content
    def update_section(self, t, title, content): self.add_section(t, title, content)
    def get_section(self, t, title): return self.sections.get((t, title))
    def search_related(self, q, n_results=5): return []
    def get_all_by_type(self, t): return []
    def save_extra_data(self, k, v): self.extra[k] = v
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
            # 记忆更新（台账+摘要合并调用）：按章回传不同的伏笔，模拟真实台账 delta
            import re as _re
            m = _re.search(r"第 (\d+) 章正文", prompt)
            n = m.group(1) if m else "0"
            return json.dumps({
                "delta": {"foreshadowing": [
                    {"item": f"伏笔{n}号", "planted_chapter": int(n), "status": "未回收"}]},
                "full_summary": "梗概。",
                "recent_summary": f"截至当前的摘要（含第{len(self.prompts)}次调用）。"},
                ensure_ascii=False)
        return self.outputs.pop(0) if self.outputs else "默认输出内容。" * 100


def make_wf(api_outputs=()):
    return FullNovelWorkflow(FakeAPI(api_outputs), FakeVS())


# ---------- 1.1 台账 delta 记录 + 撤销 + 重建 ----------

def test_ledger_delta_undo_on_regenerate():
    """TODO 1.1(a)：生成 3 章后第 2 章变动，第 2/3 章独有的旧伏笔不再存在于合并台账"""
    wf = make_wf()
    for n in (1, 2, 3):
        wf.novel_info.setdefault("chapters", {})[str(n)] = {"title": f"章{n}", "content": f"第{n}章正文"}
        wf.update_state_ledger(n, f"第{n}章正文内容")
    ledger = wf.vs.extra["state_ledger"]
    assert any(f["item"] == "伏笔2号" for f in ledger["foreshadowing"])

    # 第 2 章被重生成/编辑 → 失效处理撤销第 2、3 章的 delta
    wf.invalidate_memory_from(2)
    ledger = wf.vs.extra["state_ledger"]
    items = [f["item"] for f in ledger["foreshadowing"]]
    assert "伏笔1号" in items and "伏笔2号" not in items and "伏笔3号" not in items
    assert wf.vs.extra["ledger_stale"] is True  # 第 3 章还在但 delta 没了 → stale


def test_rebuild_memory_regen():
    """rebuild_memory(regen=True) 逐章重算 delta 并清除 stale"""
    wf = make_wf()
    for n in (1, 2, 3):
        wf.novel_info.setdefault("chapters", {})[str(n)] = {"title": f"章{n}", "content": f"第{n}章正文"}
        wf.update_state_ledger(n, f"第{n}章正文内容")
    wf.invalidate_memory_from(2)
    result = wf.rebuild_memory(from_chapter=2, regen=True)
    assert result["regenerated"] == 2  # 第 2、3 章重算
    assert wf.vs.extra["ledger_stale"] is False
    items = [f["item"] for f in result["ledger"]["foreshadowing"]]
    assert set(items) == {"伏笔1号", "伏笔2号", "伏笔3号"}


def test_rolling_summary_snapshot_rebuild():
    """摘要快照按章存，失效后回滚到变动章之前的快照"""
    wf = make_wf()
    wf.update_rolling_summary(1, "第一章内容")
    wf.update_rolling_summary(2, "第二章内容")
    wf.invalidate_memory_from(2)
    assert wf.vs.extra["rolling_summary_recent"]  # 回滚到第 1 章快照，而非清空


# ---------- 1.2 / 1.3 导入与空白章节 ----------

def test_import_chapter_overwrites_and_invalidates():
    wf = make_wf()
    wf.import_chapter(5, "外来章节", "导入的正文内容")
    assert wf.vs.get_section("chapter", "chapter_5").startswith("第5章 外来章节")
    assert wf.novel_info["chapters"]["5"]["content"] == "导入的正文内容"
    # 再导入同章号 = 覆盖
    wf.import_chapter(5, "外来章节", "修改后的正文")
    assert wf.novel_info["chapters"]["5"]["content"] == "修改后的正文"


def test_create_blank_chapter():
    wf = make_wf()
    wf.create_blank_chapter(7, "待写章")
    assert wf.novel_info["chapters"]["7"] == {"title": "待写章", "content": ""}
    assert wf.vs.get_section("chapter", "chapter_7") is not None


# ---------- 2.1 角色卡解析 / 生成 / 降级 ----------

TAGGED = """[角色]
姓名：叶凡
类型：主角
身份：落魄少爷
性格：坚毅隐忍
人物关系：与林雪是道侣
登场章节：1
退场章节：
备注：身负太古血脉
[/角色]
[角色]
姓名：玄机子
类型：配角
身份：隐世高人
性格：莫测
人物关系：叶凡的师父
登场章节：160
退场章节：280
备注：
[/角色]"""


def test_parse_character_cards():
    cards, ok = cc.parse_character_cards(TAGGED)
    assert ok and len(cards) == 2
    assert cards[0]["role"] == "main" and cards[0]["appearance_chapter"] == 1
    assert cards[1]["exit_chapter"] == 280


def test_generate_characters_structured_success():
    wf = make_wf([TAGGED])
    result = wf.generate_characters("仙侠")
    assert result["cards"][0]["name"] == "叶凡"
    # JSON 与自由文本渲染双写
    assert json.loads(wf.vs.get_section("character_cards", "cards"))[0]["name"] == "叶凡"
    assert "叶凡" in wf.vs.get_section("character", "all_characters")


def test_generate_characters_fallback_freetext():
    """mock 返回乱格式 → 重试 1 次仍失败 → 降级自由文本（现状行为）"""
    wf = make_wf(["随便一段人物描写，没有标签", "依然没有标签"])
    result = wf.generate_characters("仙侠")
    assert "cards" not in result
    assert result["characters"] == "依然没有标签"
    assert wf.vs.get_section("character_cards", "cards") is None
    assert len(wf.api.prompts) == 2  # 重试了 1 次


def test_migrate_characters_to_cards_preview():
    wf = make_wf([TAGGED])
    wf.novel_info["characters"] = "叶凡是主角……（自由文本）"
    preview = wf.migrate_characters_to_cards()
    assert preview["cards"][1]["name"] == "玄机子"
    # 迁移只返回预览，未入库（等用户确认）
    assert wf.vs.get_section("character_cards", "cards") is None


# ---------- 2.3 章节生成按登场章过滤注入 ----------

def test_appearance_filtered_injection():
    """TODO 2.3：配角 160 章登场，第 100 章只见名字不见详情；第 160 章注入详情"""
    wf = make_wf()
    wf.save_character_cards(cc.parse_character_cards(TAGGED)[0])
    wf.novel_info["outline_total_chapters"] = 200

    ctx100 = wf._build_chapter_context(100, "某章")
    assert "隐世高人" not in ctx100["context_text"]   # 详情不注入
    assert "玄机子" in ctx100["context_text"]          # 名字在兜底名单
    assert "叶凡" in ctx100["context_text"]            # 在场角色正常注入

    ctx160 = wf._build_chapter_context(160, "某章")
    assert "隐世高人" in ctx160["context_text"]        # 登场后详情注入
    # 退场后不再注入详情
    ctx300 = wf._build_chapter_context(300, "某章")
    assert "隐世高人" not in ctx300["context_text"]


# ---------- 角色卡按卷区间过滤（大纲/卷细纲注入） ----------

def test_filter_cards_for_range():
    cards, _ = cc.parse_character_cards(TAGGED)  # 叶凡登场1不退场；玄机子登场160退场280
    def names(seq): return [c["name"] for c in seq]
    assert names(cc.filter_cards_for_range(cards, 1, 40)) == ["叶凡"]
    assert names(cc.filter_cards_for_range(cards, 150, 170)) == ["叶凡", "玄机子"]
    assert names(cc.filter_cards_for_range(cards, 300, 320)) == ["叶凡"]


def test_volume_chapters_injects_volume_characters():
    """卷逐章细纲应注入本卷区间登场角色卡"""
    wf = make_wf([TAGGED])
    wf.save_character_cards(cc.parse_character_cards(TAGGED)[0])
    plan = [{"index": 1, "start": 1, "end": 100, "name": "第一卷"},
            {"index": 2, "start": 101, "end": 200, "name": "第二卷"}]
    wf.novel_info["volume_plan"] = plan
    wf.novel_info["outline"] = "故事主线：xxx"
    out = wf._generate_single_volume_chapters({"start_chapter": 150, "end_chapter": 200,
                                               "name": "第二卷", "plot": "决战"}, "卷级大纲", 4000)
    joined = "".join(wf.api.prompts[-1])
    assert "玄机子" in joined and "隐世高人" in joined   # 本卷登场角色注入


# ---------- 3.1 extend_outline 增量扩展 ----------

def test_extend_outline_appends_verbatim():
    """TODO 3.1：旧大纲逐字前缀保留、总章数更新"""
    old_outline = "故事主线：xxx\n\n第1章：开始 —— 概要\n第2章：发展 —— 概要\n"
    new_entries = "\n".join(f"第 {i} 章：新{i} —— 概要" for i in range(1, 4))  # 模型从1重排→自动偏移
    wf = make_wf([new_entries])
    wf.novel_info["outline"] = old_outline
    wf.novel_info["outline_total_chapters"] = "2"
    wf.extend_outline(3)
    final = wf.novel_info["outline"]
    assert final.startswith(old_outline.rstrip())       # 旧大纲逐字保留
    assert "第 3 章：新1" in final and "第 5 章：新3" in final
    assert wf.vs.extra["outline_total_chapters"] == "5"


# ---------- 4.1 登场调度检查 + 大纲局部改写 ----------

def test_check_appearance_in_outline():
    wf = make_wf()
    wf.novel_info["outline"] = "\n".join(f"第{n}章：标题{n} —— 叶凡行动" for n in range(1, 11))
    assert wf.check_appearance_in_outline("叶凡", 5)["mentioned"] is True
    assert wf.check_appearance_in_outline("玄机子", 5)["mentioned"] is False


def test_rewrite_outline_range_only_rewrites_target():
    wf = make_wf(["第 1 章：改1 —— 新概要\n第 2 章：改2 —— 新概要"])
    wf.novel_info["outline"] = "\n".join(f"第{n}章：旧{n} —— 旧概要" for n in range(1, 6))
    final = wf.rewrite_outline_range(3, 4, "让玄机子提前登场")
    assert "旧1" in final and "旧5" in final      # 范围外逐字保留
    assert "改1" in final and "改2" in final      # 改写后编号自动偏移为 3-4
    assert "旧3" not in final and "旧4" not in final


# ---------- 4.2 回溯影响扫描 ----------

def test_scan_impacted_chapters():
    wf = make_wf()
    wf.novel_info["chapters"] = {
        "1": {"title": "起始", "content": "叶凡走进城里，买了烧饼"},
        "2": {"title": "相遇", "content": "林雪登场"},
        "3": {"title": "", "content": ""},
    }
    results = wf.scan_impacted_chapters(["烧饼", "林雪"])
    assert [r["chapter"] for r in results] == [1, 2]
    assert results[0]["hits"] == {"烧饼": 1}
    assert wf.scan_impacted_chapters([]) == []
