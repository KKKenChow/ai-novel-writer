"""章节编辑 → 账本失效衔接测试：内容未变跳过失效、内容变化返回影响范围、不自动补账"""
import pytest
from fastapi.testclient import TestClient
from workflow.novel_workflow import FullNovelWorkflow
from storage.json_store import JsonNovelStore
import server as srv


def make_store_wf(store):
    wf = FullNovelWorkflow(None, store)
    srv._sync_novel_info(wf, store)
    return wf


def seed_chapters(store, n=3):
    for i in range(1, n + 1):
        store.update_section("chapter", f"chapter_{i}", f"第{i}章 标题{i}\n第{i}章正文")
    store.save_extra_data("ledger_deltas", {str(i): {"timeline": [f"e{i}"]} for i in range(1, n + 1)})
    store.save_extra_data("rolling_summaries", {str(i): f"s{i}" for i in range(1, n + 1)})


def test_put_section_skips_invalidation_when_content_unchanged(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="edit_skip")
    seed_chapters(store)
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        # 相同内容提交（防抖自动保存的重复提交）→ 不做失效，账页保留
        resp = client.put("/api/novels/edit_skip/section",
                          json={"type": "chapter", "title": "chapter_1", "content": "第1章 标题1\n第1章正文"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ledger_invalidated"] is False
        assert len(store.load_extra_data("ledger_deltas")) == 3
        # 内容变化 → 失效，返回受影响章数
        resp = client.put("/api/novels/edit_skip/section",
                          json={"type": "chapter", "title": "chapter_1", "content": "第1章 标题1\n第1章正文改过"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ledger_invalidated"] is True
        assert body["affected_chapters"] == 3
        # 第1章及之后账页全部作废
        assert store.load_extra_data("ledger_deltas") == {}
        assert store.load_extra_data("rolling_summaries") == {}
    finally:
        mp.undo()


def test_put_section_affected_count_from_edited_chapter(tmp_path):
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="edit_mid")
    seed_chapters(store, n=5)
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        resp = client.put("/api/novels/edit_mid/section",
                          json={"type": "chapter", "title": "chapter_3", "content": "第3章 标题3\n第3章正文改"})
        assert resp.status_code == 200
        body = resp.json()
        # 编辑第3章 → 3、4、5 共 3 章受影响
        assert body["ledger_invalidated"] is True
        assert body["affected_chapters"] == 3
        # 第1、2章账页保留
        deltas = store.load_extra_data("ledger_deltas")
        assert sorted(deltas.keys()) == ["1", "2"]
        # 非章节编辑不影响台账也不返回失效标记
        resp = client.put("/api/novels/edit_mid/section",
                          json={"type": "setting", "title": "world_setting", "content": "新世界观"})
        assert resp.status_code == 200
        assert resp.json().get("ledger_invalidated") is None
    finally:
        mp.undo()


def test_edit_warns_but_does_not_auto_rebuild(tmp_path):
    """编辑后只作废不自动补账：账页保持缺失，直到用户手动同步（内存/UI 提示由前端负责）"""
    store = JsonNovelStore(db_path=str(tmp_path), novel_id="edit_noauto")
    seed_chapters(store, n=2)
    mp = pytest.MonkeyPatch()
    mp.setattr(srv, "get_store", lambda novel_id: store)
    mp.setattr(srv, "get_store_workflow", lambda novel_id: make_store_wf(store))
    try:
        client = TestClient(srv.app)
        client.put("/api/novels/edit_noauto/section",
                   json={"type": "chapter", "title": "chapter_1", "content": "第1章 标题1\n改"})
        # 账页已作废且无自动补账（ledger_deltas 无新条目，stale 标记存在提示待补）
        assert store.load_extra_data("ledger_deltas") == {}
        assert store.load_extra_data("ledger_stale") is True
        # 记忆状态接口能探测到缺失章（前端据此提示「同步账本」）
        resp = client.get("/api/novels/edit_noauto/memory/status")
        assert resp.status_code == 200
        assert resp.json()["missing_chapters"] == [1, 2]
    finally:
        mp.undo()