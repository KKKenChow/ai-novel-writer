"""章节断点续写 / 用户决策交互 / max_tokens 覆盖 / 推理模型探测 测试"""
import pytest
from workflow.novel_workflow import FullNovelWorkflow, ChapterPaused
from api.api_client import LLMAPIClient
from api import user_config


# ---------- 公共 Fake ----------

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
    def save_extra_data(self, k, v):
        if v is None:
            self.extra.pop(k, None)
        else:
            self.extra[k] = v
    def delete_extra_field(self, k): self.extra.pop(k, None)
    def load_extra_data(self, k=None, default=None):
        return self.extra.get(k, default) if k else self.extra


class FakeAPI:
    MAX_TOKENS_LIMIT = 32768
    model = "fake"
    def __init__(self, outputs):
        # outputs 元素为 str 或 Exception 实例（抛出）
        self.outputs = list(outputs)
        self.prompts = []
    def generate(self, prompt, step="", **kw):
        if "full_summary" in prompt:
            return '{"delta": {"characters": [], "timeline": [], "foreshadowing": []}, "full_summary": "摘要。", "recent_summary": "摘要。"}'
        self.prompts.append(prompt)
        out = self.outputs.pop(0) if self.outputs else "默认正文。" * 300
        if isinstance(out, Exception):
            raise out
        return out


OUTLINE = """故事主线：主角成长。

第1章 初入都市 —— 主角来到大城市
第2章 遇见师父 —— 偶遇高人
"""

BEATS3 = """## 场景1：火车站
- 核心冲突/事件：钱包被偷

## 场景2：出租屋
- 核心冲突/事件：房租纠纷

## 场景3：天台
- 核心冲突/事件：偶遇神秘人
"""


def make_wf(outputs, confirm=None):
    wf = FullNovelWorkflow(FakeAPI(outputs), FakeVS())
    wf.novel_info["outline"] = OUTLINE
    if confirm:
        wf.on_confirm = confirm
    return wf


# ---------- 断点保护 ----------

def test_partial_saved_and_pause_on_failure():
    """场景3失败 + 选「稍后继续」→ 断点保留前2场景，抛 ChapterPaused"""
    wf = make_wf(["场景一正文。" * 100, "场景二正文。" * 100, RuntimeError("API返回空内容")],
                 confirm=lambda msg, opts: "resume_later")
    with pytest.raises(ChapterPaused, match="已暂停"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    partial = wf.vs.extra.get("chapter_partial_1")
    assert partial and len(partial["parts"]) == 2
    assert partial["beats_text"] == BEATS3


def test_resume_from_partial():
    """检测到断点 + 选「从断点续写」→ 跳过已完成场景，只调用剩余场景"""
    wf = make_wf(["场景二正文。" * 100, "场景三正文。" * 100],
                 confirm=lambda msg, opts: "resume")
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": BEATS3,
        "parts": ["场景一正文。" * 100],
    }
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert "场景一正文" in result and "场景二正文" in result and "场景三正文" in result
    # 只补写了 2 个场景（场景1 未重复调用）
    assert len(wf.api.prompts) == 2
    assert "第 2/3 个场景" in wf.api.prompts[0]
    # 完成后断点已清除
    assert "chapter_partial_1" not in wf.vs.extra


def test_restart_discards_partial():
    """选「从头生成」→ 旧断点作废，全部场景重新生成"""
    wf = make_wf(["场景一正文。" * 100, "场景二正文。" * 100, "场景三正文。" * 100],
                 confirm=lambda msg, opts: "restart")
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": BEATS3,
        "parts": ["旧场景一。" * 100],
    }
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert "旧场景一" not in result
    assert len(wf.api.prompts) == 3


def test_cancel_saves_draft():
    """场景2失败 + 选「取消」→ 已完成场景存为章节草稿，断点清除"""
    wf = make_wf(["场景一正文。" * 100, RuntimeError("boom")],
                 confirm=lambda msg, opts: "cancel")
    with pytest.raises(ChapterPaused, match="草稿"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    chapters = [c for t, _, c in wf.vs.added if t == "chapter"]
    assert chapters and "场景一正文" in chapters[0]
    assert "chapter_partial_1" not in wf.vs.extra


def test_retry_scene_until_success():
    """场景2失败 + 选「重试」→ 同场景重新调用直至成功"""
    calls = []
    def confirm(msg, opts):
        calls.append(msg)
        return "retry"
    wf = make_wf(["场景一正文。" * 100, RuntimeError("boom"), "场景二正文。" * 100, "场景三正文。" * 100],
                 confirm=confirm)
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert "场景二正文" in result
    assert len(calls) == 1  # 只问了一次


def test_no_confirm_callback_keeps_old_behavior():
    """无 on_confirm（CLI）：失败仍抛原始异常，但断点已保存"""
    wf = make_wf(["场景一正文。" * 100, RuntimeError("原始错误")])
    with pytest.raises(RuntimeError, match="原始错误"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert wf.vs.extra.get("chapter_partial_1")


def test_resume_later_keeps_partial_without_generating():
    """初始断点询问 + 选「稍后决定」→ 不生成、不删进度，安全暂停"""
    wf = make_wf([], confirm=lambda msg, opts: "resume_later")
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": BEATS3,
        "parts": ["场景一正文。" * 100],
    }
    with pytest.raises(ChapterPaused, match="已暂停"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert not wf.api.prompts
    assert len(wf.vs.extra["chapter_partial_1"]["parts"]) == 1


def test_disconnect_during_resume_prompt_keeps_partial():
    """前端断开导致确认返回 None 时，也必须安全暂停而不是误选从头生成"""
    wf = make_wf([], confirm=lambda msg, opts: None)
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": BEATS3,
        "parts": ["场景一正文。" * 100],
    }
    with pytest.raises(ChapterPaused, match="已暂停"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert not wf.api.prompts
    assert "chapter_partial_1" in wf.vs.extra


# ---------- max_tokens 覆盖 ----------

@pytest.fixture
def tmp_config(monkeypatch, tmp_path):
    cfg = tmp_path / "user_config.json"
    monkeypatch.setattr(user_config, "CONFIG_PATH", str(cfg))
    return cfg


def test_overrides_roundtrip(tmp_config):
    user_config.set_max_tokens_overrides({"chapter": 20000, "chapter_scene": "8000", "bad": -1, "x": "abc"})
    ov = user_config.get_max_tokens_overrides()
    assert ov == {"chapter": 20000, "chapter_scene": 8000}  # 非法值被剔除


def test_step_max_tokens_reads_override(tmp_config):
    user_config.set_max_tokens_overrides({"chapter_scene": 9999})
    assert FullNovelWorkflow._step_max_tokens("chapter_scene") == 9999
    assert FullNovelWorkflow._step_max_tokens("outline") is None


def test_beat_max_tokens_uses_override(tmp_config):
    """chapter_scene 覆盖值生效：按场景生成时传给 API 的 max_tokens 为覆盖值"""
    user_config.set_max_tokens_overrides({"chapter_scene": 7777})

    class RecAPI(FakeAPI):
        def __init__(self, outputs):
            super().__init__(outputs)
            self.mts = []
        def generate(self, prompt, step="", **kw):
            if "full_summary" not in prompt:
                self.mts.append(kw.get("max_tokens"))
            return super().generate(prompt, step=step, **kw)

    wf = FullNovelWorkflow(RecAPI(["场景一正文。" * 100, "场景二正文。" * 100, "场景三正文。" * 100]), FakeVS())
    wf.novel_info["outline"] = OUTLINE
    wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert wf.api.mts and all(mt == 7777 for mt in wf.api.mts)


# ---------- api_client：实例上限 / effort 注入 / 空内容提示 ----------

class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = str(self._payload)
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")
    def json(self):
        return self._payload


def ok_payload(content="你好"):
    return {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_instance_max_output_limit(monkeypatch):
    sent = {}
    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        sent.update(json)
        return FakeResp(200, ok_payload())
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m", max_output=65536)
    assert c.MAX_TOKENS_LIMIT == 65536
    c.chat([{"role": "user", "content": "hi"}], max_tokens=999999)
    assert sent["max_tokens"] == 65536  # 按实例上限截断而非类默认 32768


def test_reasoning_effort_injected(monkeypatch):
    sent = {}
    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        sent.update(json)
        return FakeResp(200, ok_payload())
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m", reasoning_effort="low")
    c.chat([{"role": "user", "content": "hi"}])
    assert sent.get("reasoning_effort") == "low"
    # 未配置时绝不传该参数
    sent.clear()
    c2 = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    c2.chat([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in sent


def test_empty_content_with_reasoning_hint(monkeypatch):
    payload = {"choices": [{"index": 0, "finish_reason": "length",
                            "message": {"role": "assistant", "content": "",
                                        "reasoning_content": "思考中..."}}]}
    monkeypatch.setattr("requests.post", lambda *a, **kw: FakeResp(200, payload))
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    with pytest.raises(Exception, match="推理模型"):
        c.chat([{"role": "user", "content": "hi"}])


def test_probe_reasoning_detects(monkeypatch):
    reasoning_payload = {
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "2",
                                 "reasoning_content": "1+1=2"}}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 50}}}
    monkeypatch.setattr("requests.post", lambda *a, **kw: FakeResp(200, reasoning_payload))
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    caps = c._probe_reasoning({}, 5)
    assert caps["reasoning"] is True
    assert caps["reasoning_effort_options"] == ["low", "medium", "high"]


def test_probe_reasoning_negative(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **kw: FakeResp(200, ok_payload("2")))
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    caps = c._probe_reasoning({}, 5)
    assert caps == {"reasoning": False, "reasoning_effort_options": None}


def test_probe_reasoning_effort_unsupported(monkeypatch):
    reasoning_payload = {
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "2",
                                 "reasoning_content": "..."}}]}
    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        if json and "reasoning_effort" in json:
            return FakeResp(400, {"error": "unknown param"})
        return FakeResp(200, reasoning_payload)
    monkeypatch.setattr("requests.post", fake_post)
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    caps = c._probe_reasoning({}, 5)
    assert caps["reasoning"] is True
    assert caps["reasoning_effort_options"] is None


# ---------- 新增：SSE 编码 / 思考回调 / 截断检测 / 关闭思考 / 取消 / 断点 bug A/B ----------

import json as _json
from api.api_client import GenerationCancelled


class FakeStreamResp:
    """模拟 SSE 流式响应。lines 为已编码的字符串行列表；encoding 模拟 requests 的
    行为：text/* 无 charset 时默认 ISO-8859-1，client 应强制改为 utf-8。"""
    def __init__(self, lines, status=200):
        self.status_code = status
        self._lines = lines
        self.encoding = "ISO-8859-1"  # 模拟网关未带 charset 时 requests 的默认值
        self.text = ""
        self.closed = False
    def iter_lines(self, decode_unicode=False):
        for ln in self._lines:
            if decode_unicode:
                # 模拟 requests：按 self.encoding 解码 UTF-8 字节
                raw = ln.encode("utf-8")
                yield raw.decode(self.encoding or "utf-8")
            else:
                yield ln
    def close(self):
        self.closed = True


def _sse(lines):
    return ["data: " + _json.dumps(d, ensure_ascii=False) for d in lines] + ["data: [DONE]"]


def test_stream_forces_utf8_encoding(monkeypatch):
    """P0 回归：SSE 响应无 charset 时，中文不得被按 ISO-8859-1 解码成乱码"""
    resp = FakeStreamResp(_sse([{"choices": [{"delta": {"content": "你好世界"}}]}]))
    captured = {}
    def fake_post(url, headers=None, json=None, timeout=None, stream=False, **kw):
        return resp
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    got = []
    out = c.chat([{"role": "user", "content": "hi"}], stream_callback=got.append)
    assert out == "你好世界"  # 若未强制 utf-8，此处为乱码
    assert resp.encoding == "utf-8"


def test_stream_reasoning_callback_and_finish_reason(monkeypatch):
    """流式：reasoning_content 走 reasoning_callback，正文走 stream_callback，finish_reason 被记录"""
    resp = FakeStreamResp(_sse([
        {"choices": [{"delta": {"reasoning_content": "先想想"}}]},
        {"choices": [{"delta": {"content": "正文"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    ]))
    monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    content_parts, reasoning_parts = [], []
    out = c.chat([{"role": "user", "content": "hi"}],
                 stream_callback=content_parts.append,
                 reasoning_callback=reasoning_parts.append)
    assert out == "正文" and "".join(content_parts) == "正文"
    assert "".join(reasoning_parts) == "先想想"
    assert c.last_finish_reason == "length"


def test_thinking_disabled_param_injected(monkeypatch):
    """关闭思考开关：请求体注入 thinking.type=disabled；响应仍含思考时置 ignored 标志"""
    sent = {}
    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        sent.update(json)
        p = ok_payload("好")
        p["choices"][0]["message"]["reasoning_content"] = "仍在思考"
        return FakeResp(200, p)
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m", thinking_disabled=True)
    c.chat([{"role": "user", "content": "hi"}])
    assert sent.get("thinking") == {"type": "disabled"}
    assert c.thinking_disable_ignored is True


def test_thinking_disabled_default_off(monkeypatch):
    """未勾选时绝不注入 thinking 参数"""
    sent = {}
    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        sent.update(json)
        return FakeResp(200, ok_payload())
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    c.chat([{"role": "user", "content": "hi"}])
    assert "thinking" not in sent


def test_cancel_before_and_during_stream(monkeypatch):
    """取消：cancel_check 返回 True 时流式循环立即中断，抛 GenerationCancelled"""
    resp = FakeStreamResp(_sse([{"choices": [{"delta": {"content": "x"}}]}] * 100))
    monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
    monkeypatch.setattr(user_config, "CONFIG_PATH", "/nonexistent/x.json")
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    state = {"n": 0}
    def check():
        state["n"] += 1
        return state["n"] > 3  # 第三个 chunk 后取消
    c.cancel_check = check
    with pytest.raises(GenerationCancelled):
        c.chat([{"role": "user", "content": "hi"}], stream_callback=lambda t: None)


def test_probe_thinking_disable_supported(monkeypatch):
    """探测：思考模型且 thinking 参数被接受 → caps.thinking_disable=True"""
    reasoning_payload = {
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "2",
                                 "reasoning_content": "..."}}]}
    monkeypatch.setattr("requests.post", lambda *a, **kw: FakeResp(200, reasoning_payload))
    c = LLMAPIClient(api_key="k", api_base="http://x", model="m")
    caps = c._probe_reasoning({}, 5)
    assert caps.get("thinking_disable") is True


# ---- 断点 bug A/B 回归 ----

class FakeVSWithDelete(FakeVS):
    def __init__(self):
        super().__init__()
        self.deleted = []
    def delete_section(self, t, title):
        self.deleted.append((t, title))


def test_bug_a_pause_does_not_delete_old_chapter():
    """Bug A：断点弹窗选「稍后决定」时，旧章节正文不得被预删除"""
    wf = make_wf([], confirm=lambda msg, opts: "resume_later")
    wf.vs = FakeVSWithDelete()
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": BEATS3,
        "parts": ["场景一正文。" * 100],
    }
    with pytest.raises(ChapterPaused, match="已暂停"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert ("chapter", "chapter_1") not in wf.vs.deleted  # 旧正文保留


def test_bug_a_confirmed_generation_deletes_old_chapter():
    """Bug A：正常生成（无断点/确认继续）时仍会删除旧章节数据"""
    wf = make_wf(["场景一正文。" * 100, "场景二正文。" * 100, "场景三正文。" * 100])
    wf.vs = FakeVSWithDelete()
    wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert ("chapter", "chapter_1") in wf.vs.deleted


def test_bug_b_beats_mismatch_asks_instead_of_silent_discard():
    """Bug B：beats 不一致且有交互回调 → 弹提示；选「稍后决定」保留断点"""
    asked = []
    def confirm(msg, opts):
        asked.append(msg)
        return "resume_later"
    wf = make_wf([], confirm=confirm)
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": "旧节拍",
        "parts": ["场景一正文。" * 100],
    }
    with pytest.raises(ChapterPaused, match="已暂停"):
        wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert asked and "不一致" in asked[0]
    assert wf.vs.extra.get("chapter_partial_1")  # 断点未被静默丢弃


def test_bug_b_beats_mismatch_confirm_restart_discards():
    """Bug B：beats 不一致，用户确认「放弃旧断点」→ 清除并从头生成"""
    def confirm(msg, opts):
        return "restart"
    wf = make_wf(["场景一正文。" * 100, "场景二正文。" * 100, "场景三正文。" * 100],
                 confirm=confirm)
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": "旧节拍",
        "parts": ["旧场景。" * 100],
    }
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert "旧场景" not in result
    assert len(wf.api.prompts) == 3


def test_bug_b_beats_whitespace_normalized():
    """Bug B：beats 仅首尾空白差异时不应误判为不一致"""
    wf = make_wf(["场景二正文。" * 100, "场景三正文。" * 100],
                 confirm=lambda msg, opts: "resume")
    wf.vs.extra["chapter_partial_1"] = {
        "chapter_num": 1, "title": "初入都市", "beats_text": BEATS3 + "\n\n",
        "parts": ["场景一正文。" * 100],
    }
    result = wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=600, beats=BEATS3)
    assert "场景一正文" in result  # 断点命中并复用，未被当成不一致
