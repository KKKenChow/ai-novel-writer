"""AI 评审 Tab 测试：评审/改写历史 CRUD、一键替换/一键还原版本链、章节联动、旧数据兼容"""
import time
import pytest
from fastapi.testclient import TestClient
from workflow.novel_workflow import FullNovelWorkflow
from storage.json_store import JsonNovelStore
import server as srv


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


def make_wf(outputs):
    return FullNovelWorkflow(FakeAPI(list(outputs)), FakeVS())


def make_store_wf(store):
    wf = FullNovelWorkflow(None, store)
    srv._sync_novel_info(wf, store)
    return wf


def seed_rewrite(store, rid="rw_1", ctype="outline", content="改写后大纲", status="draft",
                 chapter_num=None, chapter_title="", original_snapshot=None):
    entry = {"id": rid, "type": ctype, "review_id": "rev_x", "chapter_num": chapter_num,
             "chapter_title": chapter_title, "content": content, "status": status,
             "created_at": time.time()}
    if original_snapshot is not None:
        entry["original_snapshot"] = original_snapshot
        entry["original_hash"] = srv._content_hash(original_snapshot)
    items = [i for i in (store.load_extra_data("ai_rewrite_history", []) or []) if isinstance(i, dict)]
    items.append(entry)
    store.save_extra_data("ai_rewrite_history", items)


# ---------- 评审 / 改写 step ----------

def test_content_review_appends_history_and_cleans_legacy():
    wf = make_wf(["## 总分：8/10\n## 问题清单\n1. 大纲太平。\n## 修改建议\n1. 增加冲突。"])
    wf.novel_info["world_setting"] = "修仙世界"
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t", "any", "content_review",
                            {"type": "outline", "content": "旧大纲内容"})
        hist = wf.vs.extra["ai_review_history"]
        assert len(hist) == 1
        entry = hist[0]
        assert entry["type"] == "outline" and entry["review"].startswith("## 总分：8/10")
        assert entry["snapshot"] == "旧大纲内容"
        assert entry["hash"] == srv._content_hash("旧大纲内容")
        assert entry["chapter_num"] is None
        # 章节评审：清理旧单条评审键
        wf.vs.extra["chapter_review_3"] = {"review": "旧", "hash": "aaaaaaaa"}
        srv._run_generation("t", "any", "content_review",
                            {"type": "chapter", "chapter_num": 3, "chapter_title": "测试章",
                             "content": "章节正文"})
        assert "chapter_review_3" not in wf.vs.extra
        ch_entry = wf.vs.extra["ai_review_history"][1]
        assert ch_entry["type"] == "chapter" and ch_entry["chapter_num"] == 3
    finally:
        mp.undo()


def test_content_review_rejects_bad_type_or_empty():
    wf = make_wf([])
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        # 非法类型 / 空内容：任务以 error 结束，不写入历史
        srv._run_generation("t", "any", "content_review", {"type": "xxx", "content": "a"})
        srv._run_generation("t", "any", "content_review", {"type": "outline", "content": "  "})
        assert "ai_review_history" not in wf.vs.extra
    finally:
        mp.undo()


# ---------- 推理模型预算放大（思考感知，2c） ----------

def _reasoning_api(is_reasoning, thinking_disabled, thinking_disable_supported=True):
    class ReasoningAPI(FakeAPI):
        def generate(self, prompt, step="", **kw):
            self.last_max_tokens = kw.get("max_tokens")
            return super().generate(prompt, step=step, **kw)
    ra = ReasoningAPI(["## 总分：8/10\n1. 大纲太平。"])
    ra.is_reasoning = is_reasoning
    ra.thinking_disabled = thinking_disabled
    ra.thinking_disable_supported = thinking_disable_supported
    return ra


def test_reasoning_model_budget_multiplied():
    """推理模型且未关思考：content_review 默认 3000 → ×3 = 9000（思考感知预算放大）"""
    ra = _reasoning_api(is_reasoning=True, thinking_disabled=False)
    wf = FullNovelWorkflow(ra, FakeVS())
    wf.novel_info["world_setting"] = "修仙世界"
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t", "any", "content_review",
                            {"type": "outline", "content": "旧大纲内容"})
        assert ra.last_max_tokens == 9000
    finally:
        mp.undo()


def test_disabled_thinking_no_budget_multiplier():
    """已关思考（且接口真的生效）：预算维持默认 3000，不放大（省钱）"""
    ra = _reasoning_api(is_reasoning=True, thinking_disabled=True)
    wf = FullNovelWorkflow(ra, FakeVS())
    wf.novel_info["world_setting"] = "修仙世界"
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t", "any", "content_review",
                            {"type": "outline", "content": "旧大纲内容"})
        assert ra.last_max_tokens == 3000
    finally:
        mp.undo()


def test_ignored_thinking_still_multiplies_budget():
    """勾选关闭思考但接口实际忽略（hy3 式）→ 按"实际未关闭"放大预算 ×3"""
    ra = _reasoning_api(is_reasoning=True, thinking_disabled=True, thinking_disable_supported=False)
    wf = FullNovelWorkflow(ra, FakeVS())
    wf.novel_info["world_setting"] = "修仙世界"
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t", "any", "content_review",
                            {"type": "outline", "content": "旧大纲内容"})
        assert ra.last_max_tokens == 9000
    finally:
        mp.undo()


def test_ignored_thinking_writes_back_cap():
    """运行时发现接口忽略关闭思考 → 自愈写回 provider 的 thinking_disable=False"""
    ra = _reasoning_api(is_reasoning=True, thinking_disabled=True, thinking_disable_supported=False)
    ra.thinking_disable_ignored = True
    wf = FullNovelWorkflow(ra, FakeVS())
    wf.novel_info["world_setting"] = "修仙世界"
    updated = {}
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    mp.setattr(srv.user_config, "get_active_provider",
               lambda: {"name": "fake-hy3", "api_key": "k", "api_base": "http://x",
                        "model": "hy3", "reasoning": True})
    mp.setattr(srv.user_config, "update_provider_fields",
               lambda name, fields: updated.update({name: fields}))
    try:
        srv._run_generation("t", "any", "content_review",
                            {"type": "outline", "content": "旧大纲内容"})
        assert updated == {"fake-hy3": {"thinking_disable": False}}
    finally:
        mp.undo()


def test_content_rewrite_appends_draft_history():
    wf = make_wf(["改写后的正文。" * 200])
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_workflow", lambda novel_id, fresh_client=False: wf)
    try:
        srv._run_generation("t", "any", "content_rewrite",
                            {"type": "chapter", "chapter_num": 5, "chapter_title": "风起",
                             "content": "旧正文", "review": "评审意见", "review_id": "rev_1"})
        items = wf.vs.extra["ai_rewrite_history"]
        assert len(items) == 1
        e = items[0]
        assert e["status"] == "draft" and e["type"] == "chapter" and e["chapter_num"] == 5
        assert e["review_id"] == "rev_1" and e["content"].startswith("改写后的正文")
        # 缺评审意见 → 拒绝（任务 error，不写入历史）
        before = len(wf.vs.extra.get("ai_rewrite_history", []))
        srv._run_generation("t", "any", "content_rewrite",
                            {"type": "outline", "content": "a", "review": "  "})
        assert len(wf.vs.extra.get("ai_rewrite_history", [])) == before
    finally:
        mp.undo()


# ---------- 一键替换 / 一键还原（版本链） ----------

def test_apply_and_restore_outline_roundtrip(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="apply_outline")
    store.update_section("outline", "full_outline", "原文大纲")
    store.update_section("setting", "world_setting", "世界")
    seed_rewrite(store, ctype="outline", content="改写后大纲")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        resp = client.post("/api/novels/apply_outline/ai-rewrite/apply", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 200
        assert store.get_section("outline", "full_outline") == "改写后大纲"
        entry = store.load_extra_data("ai_rewrite_history")[0]
        assert entry["status"] == "applied" and entry["original_snapshot"] == "原文大纲"
        # 已应用 → 重复替换拒绝
        resp = client.post("/api/novels/apply_outline/ai-rewrite/apply", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 400
        # 一键还原
        resp = client.post("/api/novels/apply_outline/ai-rewrite/restore", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 200
        assert store.get_section("outline", "full_outline") == "原文大纲"
        entry = store.load_extra_data("ai_rewrite_history")[0]
        assert entry["status"] == "restored"
        # 还原后可再次替换（无限往返）
        resp = client.post("/api/novels/apply_outline/ai-rewrite/apply", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 200
        assert store.get_section("outline", "full_outline") == "改写后大纲"
    finally:
        mp.undo()


def test_apply_chapter_invalidates_memory(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="apply_chapter")
    store.update_section("chapter", "chapter_1", "第1章 起航\n旧正文")
    store.update_section("chapter", "chapter_2", "第2章 风浪\n第二章正文")
    store.save_extra_data("ledger_deltas", {"1": {"x": 1}, "2": {"y": 2}})
    store.save_extra_data("rolling_summaries", {"1": "s1", "2": "s2"})
    seed_rewrite(store, rid="rw_1", ctype="chapter", chapter_num=1, chapter_title="起航",
                 content="新正文内容")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        resp = client.post("/api/novels/apply_chapter/ai-rewrite/apply", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 200
        assert store.get_section("chapter", "chapter_1") == "第1章 起航\n新正文内容"
        # 台账：第1章及其后的 delta 全部失效；第2章仍存在 → stale 标记
        assert store.load_extra_data("ledger_deltas") == {}
        assert store.load_extra_data("rolling_summaries") == {}
        assert store.load_extra_data("ledger_stale") is True
        entry = store.load_extra_data("ai_rewrite_history")[0]
        assert entry["original_snapshot"] == "旧正文"
        # 还原：正文回旧版本，台账按旧章重新可用（无 delta 时重建为空合并态）
        resp = client.post("/api/novels/apply_chapter/ai-rewrite/restore", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 200
        assert store.get_section("chapter", "chapter_1") == "第1章 起航\n旧正文"
    finally:
        mp.undo()


def test_restore_rejected_when_content_modified(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="restore_mismatch")
    store.update_section("outline", "full_outline", "原文")
    seed_rewrite(store, ctype="outline", content="改写后大纲", status="applied",
                 original_snapshot="原文")
    # 正文被其他操作改动（不再等于改写版本）
    store.update_section("outline", "full_outline", "被人手动改过")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        resp = client.post("/api/novels/restore_mismatch/ai-rewrite/restore", json={"rewrite_id": "rw_1"})
        assert resp.status_code == 400
        assert store.get_section("outline", "full_outline") == "被人手动改过"
    finally:
        mp.undo()


def test_apply_world_and_characters(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="apply_wc")
    store.update_section("setting", "world_setting", "旧世界观")
    store.update_section("character", "all_characters", "旧人物")
    seed_rewrite(store, rid="rw_1", ctype="world_setting", content="新世界观")
    seed_rewrite(store, rid="rw_2", ctype="characters", content="新人物")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        assert client.post("/api/novels/apply_wc/ai-rewrite/apply", json={"rewrite_id": "rw_1"}).status_code == 200
        assert store.get_section("setting", "world_setting") == "新世界观"
        assert client.post("/api/novels/apply_wc/ai-rewrite/apply", json={"rewrite_id": "rw_2"}).status_code == 200
        assert store.get_section("character", "all_characters") == "新人物"
    finally:
        mp.undo()


def test_version_chain_supersedes_previous_applied(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="chain")
    store.update_section("outline", "full_outline", "原文V0")
    seed_rewrite(store, rid="rw_1", ctype="outline", content="改写V1")
    seed_rewrite(store, rid="rw_2", ctype="outline", content="改写V2")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        client.post("/api/novels/chain/ai-rewrite/apply", json={"rewrite_id": "rw_1"})
        client.post("/api/novels/chain/ai-rewrite/apply", json={"rewrite_id": "rw_2"})
        items = store.load_extra_data("ai_rewrite_history")
        by_id = {i["id"]: i for i in items}
        assert by_id["rw_2"]["status"] == "applied"
        assert by_id["rw_2"]["original_snapshot"] == "改写V1"
        assert by_id["rw_1"]["status"] == "superseded"
        # 还原 V2 → 回到 V1，V1 自动标记回 applied（版本链不变量）
        resp = client.post("/api/novels/chain/ai-rewrite/restore", json={"rewrite_id": "rw_2"})
        assert resp.status_code == 200
        assert store.get_section("outline", "full_outline") == "改写V1"
        items = store.load_extra_data("ai_rewrite_history")
        by_id = {i["id"]: i for i in items}
        assert by_id["rw_1"]["status"] == "applied"
        assert by_id["rw_2"]["status"] == "restored"
    finally:
        mp.undo()


# ---------- 历史 CRUD 规则 ----------

def test_review_history_delete_and_legacy(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="rev_del")
    store.save_extra_data("ai_review_history", [
        {"id": "rev_1", "type": "outline", "review": "r1", "snapshot": "s1", "hash": "h1", "created_at": 1},
    ])
    store.save_extra_data("chapter_review_2", {"review": "旧评审", "hash": "aaaaaaaa"})
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    try:
        client = TestClient(srv.app)
        resp = client.delete("/api/novels/rev_del/ai-review-history/rev_1")
        assert resp.status_code == 200
        assert store.load_extra_data("ai_review_history") == []
        # legacy 键删除（前端传 legacy_key / 或带 legacy_ 前缀的合并 id 均可）
        resp = client.delete("/api/novels/rev_del/ai-review-history/legacy_chapter_review_2")
        assert resp.status_code == 200
        assert "chapter_review_2" not in store._data["extra"]
        store.save_extra_data("chapter_review_3", {"review": "旧评审3", "hash": "bbbbbbbb"})
        resp = client.delete("/api/novels/rev_del/ai-review-history/chapter_review_3")
        assert resp.status_code == 200
        assert "chapter_review_3" not in store._data["extra"]
        # 不存在的记录 → 404
        resp = client.delete("/api/novels/rev_del/ai-review-history/nope")
        assert resp.status_code == 404
    finally:
        mp.undo()


def test_rewrite_history_edit_and_delete_rules(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="rw_rules")
    seed_rewrite(store, rid="rw_draft", ctype="outline", content="草稿")
    seed_rewrite(store, rid="rw_applied", ctype="outline", content="已应用", status="applied",
                 original_snapshot="原文")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    try:
        client = TestClient(srv.app)
        # 草稿可编辑
        resp = client.put("/api/novels/rw_rules/ai-rewrite-history/rw_draft", json={"content": "改过的草稿"})
        assert resp.status_code == 200
        entry = next(i for i in store.load_extra_data("ai_rewrite_history") if i["id"] == "rw_draft")
        assert entry["content"] == "改过的草稿"
        # 草稿可删除
        resp = client.delete("/api/novels/rw_rules/ai-rewrite-history/rw_draft")
        assert resp.status_code == 200
        assert len(store.load_extra_data("ai_rewrite_history")) == 1
        # 已应用：禁止编辑 / 禁止删除（还原依据）
        resp = client.put("/api/novels/rw_rules/ai-rewrite-history/rw_applied", json={"content": "x"})
        assert resp.status_code == 400
        resp = client.delete("/api/novels/rw_rules/ai-rewrite-history/rw_applied")
        assert resp.status_code == 400
        # 空内容拒绝
        resp = client.put("/api/novels/rw_rules/ai-rewrite-history/rw_applied", json={"content": "  "})
        assert resp.status_code == 400
    finally:
        mp.undo()


# ---------- 旧数据兼容 & 章节删除清理 ----------

def test_get_novel_merges_legacy_reviews(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="legacy_merge")
    store.save_extra_data("chapter_review_1", {"review": "旧评审", "hash": "aaaaaaaa"})
    store.save_extra_data("ai_review_history", [
        {"id": "rev_9", "type": "outline", "review": "新评审", "snapshot": "s", "hash": "h", "created_at": 9},
    ])
    store.update_section("chapter", "chapter_1", "第1章 测试\n正文")
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    try:
        client = TestClient(srv.app)
        resp = client.get("/api/novels/legacy_merge")
        hist = resp.json()["extra"]["ai_review_history"]
        assert len(hist) == 2
        legacy = next(i for i in hist if i.get("legacy_key") == "chapter_review_1")
        assert legacy["review"] == "旧评审" and legacy["hash"] == "aaaaaaaa"
        assert legacy["type"] == "chapter" and legacy["chapter_num"] == 1
        assert any(i["id"] == "rev_9" for i in hist)
    finally:
        mp.undo()


def test_delete_chapter_cleans_history_entries(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="chapter_del")
    store.update_section("chapter", "chapter_2", "第2章 测试\n正文")
    store.save_extra_data("ai_review_history", [
        {"id": "rev_1", "type": "chapter", "chapter_num": 2, "review": "r", "created_at": 1},
        {"id": "rev_2", "type": "outline", "review": "r2", "created_at": 2},
    ])
    store.save_extra_data("ai_rewrite_history", [
        {"id": "rw_1", "type": "chapter", "chapter_num": 2, "content": "c", "status": "draft", "created_at": 1},
    ])
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    try:
        client = TestClient(srv.app)
        resp = client.delete("/api/novels/chapter_del/section?type=chapter&title=chapter_2")
        assert resp.status_code == 200
        reviews = store.load_extra_data("ai_review_history")
        assert [i["id"] for i in reviews] == ["rev_2"]
        assert store.load_extra_data("ai_rewrite_history") == []
    finally:
        mp.undo()