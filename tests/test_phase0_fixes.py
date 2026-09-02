"""Phase 0 修复的核心逻辑测试：解析 / 过滤 / 阶段分类 / 拒绝入库 / 续写拼接"""
import pytest
from workflow.novel_workflow import FullNovelWorkflow, is_ai_refusal


class FakeVS:
    """存储假对象，记录所有写入"""
    def __init__(self):
        self.added = []
        self.deleted = []
        self.extra = {}
    def delete_section(self, t, title): self.deleted.append((t, title))
    def add_section(self, t, title, content): self.added.append((t, title, content))
    def update_section(self, t, title, content): pass
    def get_section(self, t, title): return None
    def search_related(self, q, n_results=5): return []
    def get_all_by_type(self, t): return []
    def save_extra_data(self, k, v): self.extra[k] = v
    def delete_extra_field(self, k): self.extra.pop(k, None)
    def load_extra_data(self, k=None, default=None):
        return self.extra.get(k, default) if k else self.extra


class FakeAPI:
    """API 假对象，按队列返回预设内容；台账/摘要自动应答"""
    MAX_TOKENS_LIMIT = 32768
    model = "fake"
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
    def generate(self, prompt, step="", **kw):
        self.prompts.append(prompt)
        if "状态台账" in prompt:
            return "{}"
        if "滚动摘要" in prompt:
            return "摘要。"
        return self.outputs.pop(0) if self.outputs else "默认输出内容。" * 100


def make_wf(api_outputs=()):
    return FullNovelWorkflow(FakeAPI(list(api_outputs)), FakeVS())


OUTLINE_MESSY = """故事主线：主角从凡人成长为仙帝。

- 第 1 章：**初入都市** —— 主角来到大城市
* 第2章：遇见师父 —— 偶遇隐世高人
### 第 3 章：获得传承 —— 得到上古功法
【第4章】第一次战斗 —— 小试牛刀
第5章 秘境开启 —— 秘境现世
"""


# ---------- _match_chapter_num ----------

@pytest.mark.parametrize("line,expected", [
    ("第1章 初入都市", 1),
    ("第 1 章：初入都市", 1),
    ("第一章 开始", 1),
    ("第 十二 章 大战", 12),
    ("- 第3章 获得传承", 3),
    ("* 第4章 战斗", 4),
    ("**第5章** 秘境", 5),
    ("【第6章】标题", 6),
    ("### 第7章 标题", 7),
    ("（第8章）标题", 8),
    ("1章 无第变体：标题", 1),
    ("故事主线：主角成长", 0),
    ("第3卷 卷名", 0),
    ("", 0),
])
def test_match_chapter_num(line, expected):
    wf = make_wf()
    assert wf._match_chapter_num(line) == expected


# ---------- get_outline_chapter_titles ----------

def test_titles_messy_formats():
    wf = make_wf()
    wf.novel_info["outline"] = OUTLINE_MESSY
    titles = wf.get_outline_chapter_titles()
    assert set(titles.keys()) == {1, 2, 3, 4, 5}
    assert "初入都市" in titles[1]


# ---------- _extract_relevant_outline ----------

def test_outline_range_opening_isolated():
    """开篇 range=0 只应拿到第1章，不能看到第3章的「获得传承」"""
    wf = make_wf()
    wf.novel_info["outline"] = OUTLINE_MESSY
    result = wf._extract_relevant_outline(OUTLINE_MESSY, 1, capture_range=0, spoiler_level="strict")
    assert "初入都市" in result
    assert "获得传承" not in result
    assert "秘境开启" not in result
    # strict 模式不含总述
    assert "故事主线" not in result


def test_outline_range_plus_minus_one():
    wf = make_wf()
    wf.novel_info["outline"] = OUTLINE_MESSY
    result = wf._extract_relevant_outline(OUTLINE_MESSY, 3, capture_range=1, spoiler_level="none")
    assert "遇见师父" in result
    assert "获得传承" in result
    assert "第一次战斗" in result
    assert "初入都市" not in result
    assert "秘境开启" not in result


# ---------- _strip_spoiler_sentences ----------

SPOILER_TEXT = "主角名叫林凡，外貌俊朗。他最终成为仙界至尊。性格坚韧，出身贫寒。"

def test_strip_spoiler_strict():
    wf = make_wf()
    result = wf._strip_spoiler_sentences(SPOILER_TEXT, level="strict")
    assert "最终成为" not in result
    assert "外貌俊朗" in result


def test_strip_spoiler_never_falls_back_to_original():
    """即使过滤掉超过50%内容，也绝不能放回含剧透的原文"""
    wf = make_wf()
    heavy = "他最终成为至尊。他最终战死了。结局是悲剧。外貌英俊。"
    result = wf._strip_spoiler_sentences(heavy, level="strict")
    assert "最终" not in result
    assert "结局" not in result
    assert len(result) < len(heavy)


def test_strip_spoiler_none_passthrough():
    wf = make_wf()
    assert wf._strip_spoiler_sentences(SPOILER_TEXT, level="none") == SPOILER_TEXT


# ---------- _classify_chapter_phase ----------

def test_phase_classification_ratio():
    wf = make_wf()
    assert wf._classify_chapter_phase(1, 100)["phase"] == "opening"
    assert wf._classify_chapter_phase(20, 100)["phase"] == "early_dev"
    assert wf._classify_chapter_phase(50, 100)["phase"] == "mid_dev"
    assert wf._classify_chapter_phase(70, 100)["phase"] == "late_dev"
    assert wf._classify_chapter_phase(85, 100)["phase"] == "climax"
    assert wf._classify_chapter_phase(99, 100)["phase"] == "resolution"


def test_phase_strategies_progressive():
    wf = make_wf()
    o = wf._classify_chapter_phase(1, 100)
    c = wf._classify_chapter_phase(85, 100)
    assert o["outline_range"] == 0 and o["rag_look_ahead"] == 0 and o["spoiler_level"] == "strict"
    assert c["spoiler_level"] == "none"


# ---------- _estimate_total_chapters ----------

def test_estimate_total_prefers_saved_param():
    wf = make_wf()
    wf.novel_info["outline_total_chapters"] = "120"
    wf.novel_info["outline"] = OUTLINE_MESSY  # 解析只能得到5
    assert wf._estimate_total_chapters() == 120


# ---------- _parse_num ----------

@pytest.mark.parametrize("s,expected", [("3", 3), ("二十三", 23), ("一百零五", 105), ("十", 10), ("abc", 0)])
def test_parse_num(s, expected):
    assert FullNovelWorkflow._parse_num(s) == expected


# ---------- is_ai_refusal ----------

def test_is_ai_refusal():
    assert is_ai_refusal("抱歉，我不能为你创作这类内容")
    assert not is_ai_refusal("主角走进了森林。")
    assert not is_ai_refusal("")


# ---------- 拒绝内容不入库 ----------

def test_chapter_refusal_not_persisted():
    wf = make_wf(["我不能为你创作涉及违规内容的小说"])
    wf.novel_info["outline"] = OUTLINE_MESSY
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=2000)
    assert is_ai_refusal(result)
    # 向量库不应有任何 chapter 写入
    assert not any(t == "chapter" for t, _, _ in wf.vs.added)


def test_world_refusal_not_persisted():
    wf = make_wf(["我无法生成这类内容"])
    wf.generate_world_setting("某些敏感描述")
    assert not any(t == "setting" for t, _, _ in wf.vs.added)


# ---------- 续写拼接分隔符 ----------

def test_continuation_separator():
    # 第一次生成很短（触发续写），续写内容接上
    wf = make_wf(["开头段落无换行结尾", "续写的第二段内容。" * 300])
    wf.novel_info["outline"] = OUTLINE_MESSY
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=2000)
    assert "无换行结尾\n\n续写的第二段" in result


# ---------- max_tokens 自动提升 ----------

def test_max_tokens_auto_bump():
    calls = []
    class RecAPI(FakeAPI):
        def generate(self, prompt, step="", **kw):
            calls.append(kw.get("max_tokens"))
            return super().generate(prompt, step, **kw)
    wf = FullNovelWorkflow(RecAPI(["开头", "续写" + "内容。" * 1000]), FakeVS())
    wf.novel_info["outline"] = OUTLINE_MESSY
    wf.generate_chapter(1, "初入都市", max_tokens=2500, target_words=2000)
    assert calls[0] == 3600  # 2000 * 1.8


# ---------- 范围校验警告 ----------

def test_scope_warning_stored():
    wf = make_wf(["主角进入秘境，获得传承，直接突破境界。"])
    wf.novel_info["outline"] = OUTLINE_MESSY
    wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=2000)
    # 第3章标题关键词「获得传承」出现在第1章正文 → 应产生警告
    assert "获得传承" in wf.last_scope_warning
