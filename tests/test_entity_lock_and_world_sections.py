"""实体时间锁 / 世界观按档注入 / 节拍校验 / 人物卡精简注入 测试"""
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
        if "状态台账" in prompt:
            return "{}"
        return self.outputs.pop(0) if self.outputs else "默认输出内容。" * 100


def make_wf(api_outputs=()):
    return FullNovelWorkflow(FakeAPI(api_outputs), FakeVS())


WORLD = """# 小说暂定名：《试炼之巅》
## 1. 故事发生的时代背景
剑修世界，灵气复苏的时代。
## 2. 主要的地理/世界构架
九霄大陆，宗门林立。
## 3. 核心的力量体系/规则
练气、筑基、金丹三境。
## 4. 主要的势力格局
玄机阁与幽夜盟两强对峙。
## 5. 写作视角或风格的额外补充
主角从凡人一路成长为仙帝。"""

CARDS = """[角色]
姓名：叶凡
别名/代号：小凡
类型：主角
身份：落魄少爷
性格：坚毅隐忍
人物关系：与林雪是道侣
登场章节：10
退场章节：
[/角色]
[角色]
姓名：玄机子
类型：配角
身份：隐世高人
性格：莫测
人物关系：叶凡的师父
登场章节：1
退场章节：
[/角色]"""


def make_wf_with_world_and_cards():
    wf = make_wf()
    wf.novel_info["world_setting"] = WORLD
    wf.novel_info["outline_total_chapters"] = "100"
    cards, ok = cc.parse_character_cards(CARDS)
    assert ok
    wf.save_character_cards(cards)
    return wf


# ---------- 实体注册表 / 实体时间锁 ----------

def test_entity_registry_contains_alias():
    wf = make_wf_with_world_and_cards()
    reg = wf._build_entity_registry()
    assert reg["叶凡"] == 10
    assert reg["小凡"] == 10    # 别名进注册表
    assert reg["玄机子"] == 1


def test_lock_entities_masks_not_appeared():
    wf = make_wf_with_world_and_cards()
    # 第5章：玄机子已登场，叶凡(10)未登场 → 遮蔽
    out = wf._lock_entities("叶凡和小凡去找玄机子", 5)
    assert "玄机子" in out
    assert "叶凡" not in out and "小凡" not in out
    # 第10章：叶凡登场 → 全部保留
    out2 = wf._lock_entities("叶凡和小凡去找玄机子", 10)
    assert "叶凡" in out2 and "小凡" in out2


# ---------- 世界观按档注入 ----------

def test_split_world_sections_classify():
    wf = make_wf_with_world_and_cards()
    secs = wf._split_world_sections(WORLD)
    kinds = {s["title"]: s["kind"] for s in secs}
    assert kinds["## 1. 故事发生的时代背景"] == "stable"
    assert kinds["## 2. 主要的地理/世界构架"] == "stable"
    assert kinds["## 3. 核心的力量体系/规则"] == "stable"
    assert kinds["## 4. 主要的势力格局"] == "faction"
    assert kinds["## 5. 写作视角或风格的额外补充"] == "style"


def test_stable_world_staged_injection():
    wf = make_wf_with_world_and_cards()
    # stable 档：只有稳定背景，无势力/风格段
    stable = wf._stable_world_text("stable", chapter_num=5)
    assert "灵气复苏" in stable and "九霄大陆" in stable
    assert "玄机阁" not in stable and "仙帝" not in stable
    # faction 档：加上势力段，仍无风格/结局段
    faction = wf._stable_world_text("faction", chapter_num=50)
    assert "玄机阁" in faction and "幽夜盟" in faction
    assert "仙帝" not in faction
    # full 档：全部
    full = wf._stable_world_text("full", chapter_num=90)
    assert "仙帝" in full


def test_world_sections_saved_after_generation():
    wf = make_wf()
    wf.novel_info["world_setting"] = WORLD
    wf._save_world_sections()
    secs = wf.vs.extra["world_sections"]
    assert len(secs) == 5
    assert any(s["kind"] == "faction" for s in secs)


def test_chapter_context_world_and_cards_staged():
    wf = make_wf_with_world_and_cards()
    # 第5章（开篇/strict）：世界观只稳定段 + 人物卡精简（无人物关系弧线）
    ctx5 = wf._build_chapter_context(5, "入门")
    t5 = ctx5["context_text"]
    assert "九霄大陆" in t5 and "灵气复苏" in t5
    assert "玄机阁" not in t5 and "仙帝" not in t5
    assert "人物关系" not in t5      # brief 卡不注入弧线
    assert "身份：隐世高人" in t5    # 已登场角色（玄机子）的 brief 卡
    # 第50章（中期/minimal）：加势力段 + 人物卡全量
    ctx50 = wf._build_chapter_context(50, "对峙")
    t50 = ctx50["context_text"]
    assert "玄机阁" in t50
    assert "仙帝" not in t50
    assert "人物关系" in t50        # 全卡
    # 第90章（后期/none）：全量
    ctx90 = wf._build_chapter_context(90, "决战")
    assert "仙帝" in ctx90["context_text"]


# ---------- 逐章概要：登场章软约束 + 独立编辑 ----------

def test_volume_chapters_prompt_appearance_alignment():
    """逐章概要生成 prompt 注入角色登场章，并要求按标注登场章首次出场"""
    wf = make_wf_with_world_and_cards()
    wf.novel_info["outline"] = WORLD
    prompt = None

    class Cap(FakeAPI):
        def generate(self, p, step="", **kw):
            nonlocal prompt
            prompt = p
            return "第 1 章：入门 —— 概要"
    wf.api = Cap()
    wf._generate_single_volume_chapters(
        {"name": "第一卷", "start_chapter": 1, "end_chapter": 20, "plot": "开局"},
        "故事主线：x", 1000)
    assert "叶凡（第10章登场）" in prompt
    assert "玄机子（第1章登场）" in prompt
    assert "首次出场" in prompt and "登场章" in prompt


def test_volume_chapters_prompt_has_words_baseline():
    """逐章概要生成 prompt 注入每章基准字数（约N字的浮动基准），不再让 AI 凭感觉填"""
    wf = make_wf_with_world_and_cards()
    wf.novel_info["outline_words_per_chapter"] = "10000"
    wf.vs.extra["outline_words_per_chapter"] = "10000"
    prompt = None

    class Cap(FakeAPI):
        def generate(self, p, step="", **kw):
            nonlocal prompt
            prompt = p
            return "第 1 章：入门 —— 概要"
    wf.api = Cap()
    wf._generate_single_volume_chapters(
        {"name": "第一卷", "start_chapter": 1, "end_chapter": 20, "plot": "开局"},
        "故事主线：x", 1000)
    assert "围绕每章基准 10000 字" in prompt
    assert "铺垫章可略短（不少于基准的一半）" in prompt


def test_chapter_summaries_get_update_split():
    """逐章概要独立读写：get 取标记后文本，update 只替换标记后、卷级部分逐字保留"""
    wf = make_wf()
    wf.novel_info["outline"] = "故事主线：x\n\n---\n\n## 逐章概要\n### 第一卷\n第 1 章：旧一 —— 旧概要"
    assert "### 第一卷" in wf.get_chapter_summaries()
    new = wf.update_chapter_summaries("### 第一卷\n第 1 章：新一 —— 新概要")
    assert new.startswith("故事主线：x")          # 卷级部分保留
    assert "旧概要" not in new
    assert "新概要" in new
    assert wf.novel_info["outline"] == new
    # 无标记时追加
    wf.novel_info["outline"] = "只有卷级"
    new2 = wf.update_chapter_summaries("第 1 章：一 —— 概要")
    assert "## 逐章概要" in new2 and "第 1 章：一 —— 概要" in new2


# ---------- 局部改写：角色卡注入 + 格式归一化 ----------

def test_normalize_chapter_lines():
    """格式归一化：剥离加粗/列表/空行，非章节行并作续行"""
    raw = """以下为改写结果：
**第 4 章：首单与邂逅** —— 陈大强完成第一单配送，结识了白领张依依（约35字）

第 5 章：竞争
—— 王猛登场，两人暗中较量（约30字）
"""
    out = FullNovelWorkflow._normalize_chapter_lines(raw)
    lines = out.split("\n")
    assert len(lines) == 2
    assert lines[0] == "第 4 章：首单与邂逅 —— 陈大强完成第一单配送，结识了白领张依依（约35字）"
    assert "**" not in out and "\n\n" not in out
    assert "竞争—— 王猛登场" in lines[1]  # 续行并入


def test_rewrite_outline_prompt_has_character_cards():
    """局部改写 prompt 注入改写范围内的角色卡及登场章，并要求角色一致性"""
    wf = make_wf_with_world_and_cards()
    wf.novel_info["outline"] = "故事主线：x\n\n---\n\n## 逐章概要\n" + "\n".join(
        f"第 {i} 章：章{i} —— 概要{i}" for i in range(1, 12))

    class Cap(FakeAPI):
        def generate(self, p, step="", **kw):
            self.prompts.append(p)
            return "第 8 章：新八 —— 新概要\n第 9 章：新九 —— 新概要"
    wf.api = Cap()
    wf.rewrite_outline_range(8, 9, "让玄机子出场")
    p = wf.api.prompts[0]
    assert "玄机子（第1章登场）" in p          # 范围角色登场章
    assert "角色一致性" in p and "首次出场" in p
    assert "第 8 章：新八 —— 新概要" in wf.novel_info["outline"]  # 替换成功
    assert "第 1 章：章1 —— 概要1" in wf.novel_info["outline"]    # 范围外保留

def test_validate_beats_detects_not_appeared_entity():
    wf = make_wf_with_world_and_cards()
    beats = """## 场景1：拜师
- 地点与出场人物：山门，玄机子、叶凡
- 核心冲突/事件：收徒
- 结尾钩子/进展：下山"""
    r = wf.validate_beats(beats, 5)
    assert r["ok"] is False
    assert any("叶凡" in i for i in r["issues"])
    assert r["scene_feedback"][0]["head"]  # 逐场景定位反馈
    assert any("叶凡" in p for p in r["scene_feedback"][0]["problems"])
    # 第10章（已登场）→ 通过
    r2 = wf.validate_beats(beats, 10)
    assert r2["ok"] is True and r2["issues"] == [] and r2["scene_feedback"] == []


def test_generate_chapter_beats_auto_regenerate_on_validation_fail():
    """节拍越界时自动重生成：第1次输出点名未登场角色，第2次输出干净 → 采用第2次"""
    bad = "## 场景1：偶遇\n- 地点与出场人物：小凡、玄机子\n- 核心冲突/事件：x\n- 结尾钩子/进展：y"
    good = "## 场景1：山门初探\n- 地点与出场人物：玄机子\n- 核心冲突/事件：x\n- 结尾钩子/进展：y"
    wf = make_wf([bad, good])
    wf.novel_info["outline"] = "第 5 章：入门 —— 叶凡拜师玄机子（约30字）"
    wf.save_character_cards(cc.parse_character_cards(CARDS)[0])
    beats = wf.generate_chapter_beats(5, "入门")
    assert "山门初探" in beats            # 采用校验通过的第2次输出
    assert wf.last_beats_warning == ""
    # 第2次重试 prompt 带逐场景定位反馈（场景原文+问题+允许角色名单）
    retry_p = wf.api.prompts[1]
    assert "校验反馈（逐场景定位）" in retry_p
    assert "小凡" in retry_p             # 越界场景原文
    assert "第10章才登场" in retry_p     # 具体问题定位