"""评审/黄金开篇数据失效清理测试（A 物理清理 + B 快照指纹过期标记）"""
import zlib
import pytest
from fastapi.testclient import TestClient
from workflow.novel_workflow import FullNovelWorkflow
from storage.json_store import JsonNovelStore


class FakeVS:
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
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.prompts = []
    def generate(self, prompt, step="", **kw):
        if "full_summary" in prompt:
            self.prompts.append(prompt)
            return '{"delta": {"characters": [], "timeline": [], "foreshadowing": []}, "full_summary": "摘要。", "recent_summary": "摘要。"}'
        self.prompts.append(prompt)
        return self.outputs.pop(0) if self.outputs else "默认正文。" * 300


OUTLINE = """故事主线：主角成长。

第1章 初入都市 —— 主角来到大城市
"""


def make_wf(outputs):
    return FullNovelWorkflow(FakeAPI(list(outputs)), FakeVS())


def seed_review(vs, num=1):
    """预置旧评审/黄金开篇数据"""
    vs.extra[f"chapter_review_{num}"] = {"review": "旧评审", "hash": "aaaaaaaa"}
    vs.extra[f"chapter_golden_{num}"] = {
        "content": "旧版", "alt_content": "旧落选", "hash": "aaaaaaaa", "alt_hash": "bbbbbbbb"}


# ---------- A：物理清理（生成/改写/删除/导入时连带失效） ----------

def test_generate_chapter_clears_stale_review():
    wf = make_wf(["新正文。" * 200])
    wf.novel_info["outline"] = OUTLINE
    seed_review(wf.vs)
    wf.generate_chapter(1, "初入都市", max_tokens=16000, target_words=1000)
    assert "chapter_review_1" not in wf.vs.extra
    assert "chapter_golden_1" not in wf.vs.extra


def test_delete_stale_chapter_clears_review_and_golden():
    wf = make_wf([])
    wf.novel_info["outline"] = OUTLINE
    seed_review(wf.vs)
    wf._delete_stale_chapter(1)
    assert "chapter_review_1" not in wf.vs.extra
    assert "chapter_golden_1" not in wf.vs.extra


def test_revise_step_clears_review_after_rewrite():
    import server as srv
    wf = make_wf(["改写后的正文。" * 200])
    seed_review(wf.vs)
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t_revise", "any", "chapter_revise",
                            {"chapter_num": 1, "chapter_title": "初入都市",
                             "content": "旧正文", "review": "评审意见"})
        assert "chapter_review_1" not in wf.vs.extra
        assert "chapter_golden_1" not in wf.vs.extra
    finally:
        mp.undo()


def test_delete_section_clears_review_extra(tmp_path):
    import server as srv
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="review_cleanup")
    store.save_extra_data("chapter_review_1", {"review": "r", "hash": "h"})
    store.save_extra_data("chapter_golden_1", {"content": "c"})
    store.update_section("chapter", "chapter_1", "第1章 测试\n正文")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    try:
        client = TestClient(srv.app)
        resp = client.delete("/api/novels/review_cleanup/section?type=chapter&title=chapter_1")
        assert resp.status_code == 200
        assert "chapter_review_1" not in store._data["extra"]
        assert "chapter_golden_1" not in store._data["extra"]
        # 非章节删除不影响评审数据
        store.save_extra_data("chapter_review_1", {"review": "r", "hash": "h"})
        resp = client.delete("/api/novels/review_cleanup/section?type=setting&title=world_setting")
        assert resp.status_code == 200
        assert "chapter_review_1" in store._data["extra"]
    finally:
        mp.undo()


def test_import_chapter_clears_review_extra(tmp_path):
    import server as srv
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="review_cleanup2")
    store.save_extra_data("chapter_review_2", {"review": "r", "hash": "h"})
    store.save_extra_data("chapter_golden_2", {"content": "c"})
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_wf([]))
    try:
        client = TestClient(srv.app)
        resp = client.post("/api/novels/review_cleanup2/chapters/import",
                           json={"chapter_num": 2, "content": "导入正文", "title": "新章"})
        assert resp.status_code == 200
        assert "chapter_review_2" not in store._data["extra"]
        assert "chapter_golden_2" not in store._data["extra"]
    finally:
        mp.undo()


# ---------- B：快照指纹（hash 确定性 + 与标准 CRC32 一致） ----------

def test_content_hash_deterministic_and_crc32_compatible():
    from server import _content_hash
    assert _content_hash("测试正文") == _content_hash("测试正文")
    assert _content_hash("测试正文") != _content_hash("测试正文2")
    assert len(_content_hash("任意文本")) == 8
    # 与 zlib.crc32（UTF-8）参考向量一致，保证前端 crc32 可比对
    for text in ["hello world", "第1章 初入都市", "中文与 English 混合 content！"]:
        expected = format(zlib.crc32(text.encode("utf-8")), "08x")
        assert _content_hash(text) == expected


def test_review_step_saves_hash_snapshot():
    """评审入库应为 {review, hash} 结构，hash 为评审时正文的指纹"""
    import server as srv
    wf = make_wf(["## 总分：8/10"])
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t_review", "any", "chapter_review",
                            {"chapter_num": 1, "chapter_title": "初入都市",
                             "content": "当前正文内容"})
        saved = wf.vs.extra["chapter_review_1"]
        assert saved["review"] and saved["hash"] == srv._content_hash("当前正文内容")
    finally:
        mp.undo()