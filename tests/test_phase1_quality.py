"""Phase 1 测试：场景卡细纲 / 评审闭环 / 一致性检查覆盖章节 / 黄金开篇择优"""
import pytest
from workflow.novel_workflow import FullNovelWorkflow


class FakeVS:
    def __init__(self):
        self.added = []
        self.extra = {}
    def delete_section(self, t, title): pass
    def add_section(self, t, title, content): self.added.append((t, title, content))
    def update_section(self, t, title, content): pass
    def get_section(self, t, title): return None
    def search_related(self, q, n_results=5): return []
    def get_all_by_type(self, t): return []
    def save_extra_data(self, k, v): self.extra[k] = v
    def load_extra_data(self, k=None, default=None): return self.extra.get(k, default) if k else self.extra


class FakeAPI:
    MAX_TOKENS_LIMIT = 32768
    model = "fake"
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
    def generate(self, prompt, step="", **kw):
        # 台账/摘要等后台自动调用直接应答，不消耗预设输出队列
        if "状态台账" in prompt:
            self.prompts.append(prompt)
            return "{}"
        if "滚动摘要" in prompt:
            self.prompts.append(prompt)
            return "摘要。"
        self.prompts.append(prompt)
        return self.outputs.pop(0) if self.outputs else "默认正文。" * 300


OUTLINE = """故事主线：主角成长。

第1章 初入都市 —— 主角来到大城市
第2章 遇见师父 —— 偶遇高人
"""

BEATS = """## 场景1：火车站
- 地点与出场人物：火车站，主角
- 核心冲突/事件：主角钱包被偷
- 情绪走向：迷茫→警觉
- 结尾钩子/进展：发现神秘纸条

## 场景2：出租屋
- 地点与出场人物：出租屋，主角、房东
- 核心冲突/事件：房租纠纷
- 情绪走向：压抑→爆发
- 结尾钩子/进展：纸条发光
"""


def make_wf(outputs):
    return FullNovelWorkflow(FakeAPI(list(outputs)), FakeVS())


# ---------- 场景卡解析 ----------

def test_parse_beats_markdown():
    beats = FullNovelWorkflow.parse_beats(BEATS)
    assert len(beats) == 2
    assert "火车站" in beats[0] and "出租屋" in beats[1]


def test_parse_beats_plain():
    plain = BEATS.replace("## ", "")
    beats = FullNovelWorkflow.parse_beats(plain)
    assert len(beats) == 2


# ---------- 按场景卡生成 ----------

def main_prompts(wf):
    """过滤掉台账/摘要等后台自动调用，只保留正文生成 prompt"""
    return [p for p in wf.api.prompts if "状态台账" not in p and "滚动摘要" not in p]


def test_generate_by_beats_calls_per_scene():
    wf = make_wf(["火车站场景正文。" * 100, "出租屋场景正文。" * 100])
    wf.novel_info["outline"] = OUTLINE
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=1200, beats=BEATS)
    # 每个场景一次调用
    assert len(main_prompts(wf)) == 2
    # 第二个场景的 prompt 带前文末尾
    assert "前文末尾" in main_prompts(wf)[1]
    # 两个场景都进了正文
    assert "火车站场景正文" in result and "出租屋场景正文" in result


def test_generate_by_beats_fallback_when_unparsable():
    wf = make_wf(["一次性正文。" * 200])
    wf.novel_info["outline"] = OUTLINE
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=1200, beats="没有场景格式的文本")
    assert "一次性正文" in result
    assert len(main_prompts(wf)) == 1  # 回退单次生成


def test_generate_beats_persisted():
    wf = make_wf([BEATS])
    wf.novel_info["outline"] = OUTLINE
    beats = wf.generate_chapter_beats(1, "初入都市")
    assert "场景1" in beats
    assert wf.vs.extra.get("chapter_beats_1") == beats


# ---------- 评审 / 改写 ----------

def test_review_and_revise():
    review_text = "## 总分：6/10\n\n## 问题清单\n1. 开头太慢"
    wf = make_wf([review_text, "改写后的正文。" * 200])
    review = wf.review_chapter(1, "初入都市", "原始正文内容")
    assert "6/10" in review
    revised = wf.revise_chapter(1, "初入都市", "原始正文内容", review)
    assert "改写后的正文" in revised
    # 改写 prompt 应包含评审意见和原文
    assert review_text in wf.api.prompts[1] and "原始正文内容" in wf.api.prompts[1]


# ---------- 黄金开篇择优 ----------

def test_golden_picks_higher_score():
    wf = make_wf([
        "版本A正文。" * 200, "版本B正文。" * 200,
        "## 总分：6/10", "## 总分：9/10",
    ])
    wf.novel_info["outline"] = OUTLINE
    result = wf.generate_golden_chapter(1, "初入都市", max_tokens=16000, target_words=1000)
    assert result["picked"] == 2
    assert "版本B正文" in result["content"]
    assert "版本A正文" in result["alt_content"]
    assert result["scores"] == (6.0, 9.0)


# ---------- 一致性检查覆盖章节 ----------

def test_consistency_includes_chapters():
    wf = make_wf(["设定检查报告", "章节检查报告"])
    wf.novel_info["world_setting"] = "世界观：低魔世界"
    wf.novel_info["characters"] = "主角：林凡"
    wf.novel_info["outline"] = OUTLINE
    wf.novel_info["chapters"] = {"1": {"title": "初入都市", "content": "林凡走进了城市……"}}
    result = wf.check_consistency()
    # 两次调用：设定比对 + 章节批次
    assert len(wf.api.prompts) == 2
    assert "林凡走进了城市" in wf.api.prompts[1]
    assert "设定检查报告" in result and "章节检查报告" in result


def test_consistency_without_chapters_unchanged():
    wf = make_wf(["设定检查报告"])
    wf.novel_info["world_setting"] = "世界观"
    wf.novel_info["characters"] = "人物"
    wf.novel_info["chapters"] = {}
    result = wf.check_consistency()
    assert len(wf.api.prompts) == 1
    assert result == "设定检查报告"
