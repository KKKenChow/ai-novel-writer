"""TODO 阶段 0 与阶段 3.2 修复的回归测试：
- 0.1 人物生成直取世界观
- 0.2 台账/摘要输入覆盖全章（头2000+末4000）
- 0.3 伏笔注入提前到 mid_dev
- 0.4 角色状态台账注入生成 prompt
- 3.2 Bug A/B 两阶段大纲全局章号
"""
import json
import pytest
from workflow.novel_workflow import FullNovelWorkflow


class FakeVS:
    def __init__(self):
        self.added = []
        self.extra = {}
        self.sections = {}
    def delete_section(self, t, title): pass
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
        if "滚动摘要" in prompt:
            return "摘要。"
        return self.outputs.pop(0) if self.outputs else "默认输出内容。" * 100


def make_wf(api_outputs=()):
    return FullNovelWorkflow(FakeAPI(api_outputs), FakeVS())


# ---------- 0.1 人物生成直取世界观 ----------

def test_characters_prompt_contains_world_setting():
    wf = make_wf(["人物设定结果"])
    wf.novel_info["world_setting"] = "九霄大陆以剑修为尊"
    wf.generate_characters("写一部仙侠小说")
    assert "九霄大陆以剑修为尊" in wf.api.prompts[-1]


def test_characters_world_fallback_from_store():
    """workflow 独立实例化时从存储 setting_world_setting 兜底"""
    wf = make_wf(["人物设定结果"])
    wf.vs.sections[("setting", "world_setting")] = "废土末世，辐射尘暴"
    assert "world_setting" not in wf.novel_info or not wf.novel_info["world_setting"]
    wf.generate_characters("写一部末世小说")
    assert "废土末世，辐射尘暴" in wf.api.prompts[-1]
    assert wf.novel_info["world_setting"] == "废土末世，辐射尘暴"


# ---------- 0.2 台账/摘要输入覆盖全章 ----------

def test_ledger_prompt_covers_chapter_tail():
    wf = make_wf()
    content = "开头内容。" + "过程描写。" * 2000 + "章末埋下紫晶钥匙伏笔。"
    wf.update_state_ledger(5, content)
    assert "紫晶钥匙" in wf.api.prompts[-1]
    assert "开头内容" in wf.api.prompts[-1]


def test_rolling_summary_prompt_covers_chapter_tail():
    wf = make_wf()
    content = "开头内容。" + "过程描写。" * 2000 + "结尾主角觉醒太古血脉。"
    wf.update_rolling_summary(5, content)
    assert "太古血脉" in wf.api.prompts[-1]
    assert "开头内容" in wf.api.prompts[-1]


def test_chapter_excerpt_short_content_unchanged():
    assert FullNovelWorkflow._chapter_excerpt("短章节全文") == "短章节全文"


# ---------- 0.3 / 0.4 上下文注入 ----------

def _wf_with_ledger():
    wf = make_wf()
    wf.novel_info["outline_total_chapters"] = 20
    wf.novel_info["outline"] = "\n".join(f"第{n}章：标题{n} —— 概要" for n in range(1, 21))
    wf.vs.extra["state_ledger"] = {
        "characters": [{"name": "林雪", "status": "重伤昏迷"}],
        "timeline": [{"chapter": i, "event": f"事件{i}"} for i in range(1, 6)],
        "foreshadowing": [
            {"item": "神秘罗盘的来历", "planted_chapter": 2, "status": "未回收"},
        ],
    }
    return wf


def test_mid_dev_injects_foreshadowing():
    wf = _wf_with_ledger()
    ctx = wf._build_chapter_context(10, "标题10")  # 10/20 → mid_dev
    assert ctx["phase_config"]["phase"] == "mid_dev"
    assert "神秘罗盘的来历" in ctx["context_text"]


def test_opening_does_not_inject_foreshadowing():
    wf = _wf_with_ledger()
    ctx = wf._build_chapter_context(1, "标题1")
    assert "神秘罗盘的来历" not in ctx["context_text"]


def test_ledger_characters_and_timeline_injected():
    wf = _wf_with_ledger()
    ctx = wf._build_chapter_context(10, "标题10")
    assert "林雪: 重伤昏迷" in ctx["context_text"]
    # 最近 3 条时间线（第3、4、5章）
    assert "事件5" in ctx["context_text"]
    assert "事件3" in ctx["context_text"]
    assert "事件1" not in ctx["context_text"]


def test_rolling_summary_injected_from_chapter_2():
    wf = _wf_with_ledger()
    wf.vs.extra["rolling_summary"] = "主角已离开新手村。"
    ctx2 = wf._build_chapter_context(2, "标题2")
    assert "主角已离开新手村" in ctx2["context_text"]


# ---------- 3.2 Bug A/B 两阶段大纲全局章号 ----------

STAGE1 = """故事主线：主角从凡人成长为仙帝。

[卷]
卷名：风起
章节：第1-34章
剧情：主角踏上修行路
[/卷]

[卷]
卷名：云涌
章节：第35-67章
剧情：宗门大比与背叛
[/卷]

[卷]
卷名：归一
章节：第68-100章
剧情：决战仙域
[/卷]
"""


def _vol_lines(prefix, count):
    return "\n".join(f"第 {i} 章：{prefix}{i} —— 概要" for i in range(1, count + 1))


def test_two_stage_outline_lazy_volume1_only():
    """TODO 3.2.0(a)：生成 100 章大纲，落库仅含卷级规划 + 第一卷细纲"""
    wf = make_wf([STAGE1, _vol_lines("风", 34)])
    outline = wf.generate_outline("仙侠", total_chapters=100)
    wf.novel_info["outline"] = outline
    titles = wf.get_outline_chapter_titles()
    assert sorted(titles.keys()) == list(range(1, 35))  # 只有第一卷细纲
    assert "风1" in titles[1]
    plan = wf.vs.extra.get("volume_plan")
    assert len(plan) == 3
    assert plan[0]["chapters_done"] and not plan[1]["chapters_done"]
    assert (plan[1]["start"], plan[1]["end"]) == (35, 67)


def test_lazy_volume_generation_with_prev_context():
    """TODO 3.2.0(b)(c)：惰性生成第 2 卷时注入前一卷内容；全部卷生成后覆盖 1..100 连续"""
    wf = make_wf([
        STAGE1, _vol_lines("风", 34),
        _vol_lines("云", 33),  # 第2卷：模型从 1 重排 → 自动偏移为 35-67
        _vol_lines("归", 33),  # 第3卷 → 68-100
    ])
    outline = wf.generate_outline("仙侠", total_chapters=100)
    wf.novel_info["outline"] = outline

    # ensure_outline_for_chapter 自动触发第 2 卷细纲生成
    wf.ensure_outline_for_chapter(61)
    vol2_prompt = wf.api.prompts[-1]
    assert "第 35-67 章" in vol2_prompt and "从第 35 章开始" in vol2_prompt
    assert "前一卷逐章大纲" in vol2_prompt and "风34" in vol2_prompt  # 注入前卷内容解决盲写

    # 重复触发不重复生成
    n_prompts = len(wf.api.prompts)
    wf.ensure_outline_for_chapter(40)
    assert len(wf.api.prompts) == n_prompts

    wf.generate_volume_chapters(3)
    titles = wf.get_outline_chapter_titles()
    assert sorted(titles.keys()) == list(range(1, 101))
    assert "云1" in titles[35]
    assert "归1" in titles[68]
    assert "归33" in titles[100]


def test_two_stage_single_volume_failure_not_fatal():
    """第一卷细纲生成失败 → 大纲其余部分正常产出，卷计划保留待补"""
    class FlakyAPI(FakeAPI):
        def generate(self, prompt, step="", **kw):
            self.prompts.append(prompt)
            if "当前需要补全的卷" in prompt:
                raise RuntimeError("API 故障")
            return super().generate(prompt, step=step, **kw)
    wf = FullNovelWorkflow(FlakyAPI([STAGE1]), FakeVS())
    outline = wf.generate_outline("仙侠", total_chapters=100)
    assert "故事主线" in outline  # 卷级规划仍产出
    plan = wf.vs.extra.get("volume_plan")
    assert len(plan) == 3 and not plan[0]["chapters_done"]  # 标记未完成，可后续惰性补


def test_plan_volumes_contiguous():
    plan = FullNovelWorkflow._plan_volumes(100)
    assert plan[0]["start"] == 1
    assert plan[-1]["end"] == 100
    for a, b in zip(plan, plan[1:]):
        assert b["start"] == a["end"] + 1
    assert sum(v["end"] - v["start"] + 1 for v in plan) == 100


def test_adjust_volumes_realigns_ranges():
    """模型声明范围混乱时，程序顺序重排并对齐总章数"""
    wf = make_wf()
    volumes = [
        {"name": "第一卷", "chapters": 10, "plot": "a", "start_chapter": 1, "end_chapter": 10},
        {"name": "第二卷", "chapters": 10, "plot": "b", "start_chapter": 1, "end_chapter": 10},
    ]
    adjusted = wf._adjust_volumes(volumes, 25)
    assert (adjusted[0]["start_chapter"], adjusted[0]["end_chapter"]) == (1, 10)
    assert (adjusted[1]["start_chapter"], adjusted[1]["end_chapter"]) == (11, 25)


def test_two_stage_single_volume_failure_not_fatal():
    """单卷两次都失败 → 记录缺失继续，其余卷正常产出"""
    class FlakyAPI(FakeAPI):
        def generate(self, prompt, step="", **kw):
            self.prompts.append(prompt)
            if "该卷核心剧情：宗门大比" in prompt:
                raise RuntimeError("API 故障")  # 第二卷细纲（含重试）全部失败
            return super().generate(prompt, step=step, **kw)
    wf = FullNovelWorkflow(FlakyAPI([
        STAGE1, _vol_lines("风", 34), _vol_lines("归", 33),
    ]), FakeVS())
    outline = wf.generate_outline("仙侠", total_chapters=100)
    wf.novel_info["outline"] = outline
    # 惰性生成第 2 卷（失败）与第 3 卷（成功）
    assert wf.generate_volume_chapters(2) is None
    assert wf.generate_volume_chapters(3) is not None
    titles = wf.get_outline_chapter_titles()
    assert "风1" in titles[1]
    assert "归1" in titles[68]
    assert 35 not in titles  # 失败卷缺失但不影响其他卷入库