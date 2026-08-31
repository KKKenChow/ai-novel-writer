"""JSON 存储测试：增删改查 / 检索 / extra_data / 多小说管理"""
import os
import json
import pytest
from storage.json_store import JsonNovelStore


@pytest.fixture
def store(tmp_path):
    return JsonNovelStore(db_path=str(tmp_path), novel_id="novel_test1", novel_name="测试小说")


def test_add_get_update_delete(store):
    store.add_section("setting", "world_setting", "世界观内容")
    assert store.get_section("setting", "world_setting") == "世界观内容"
    # 覆盖更新
    store.update_section("setting", "world_setting", "新内容")
    assert store.get_section("setting", "world_setting") == "新内容"
    store.delete_section("setting", "world_setting")
    assert store.get_section("setting", "world_setting") is None


def test_persistence_across_instances(store, tmp_path):
    store.add_section("chapter", "chapter_1", "第1章 开始\n正文内容")
    # 新实例从文件加载
    store2 = JsonNovelStore(db_path=str(tmp_path), novel_id="novel_test1")
    assert store2.get_section("chapter", "chapter_1") == "第1章 开始\n正文内容"
    assert store2._data["novel_name"] == "测试小说"


def test_load_all_to_dict(store):
    store.add_section("setting", "world_setting", "世界观")
    store.add_section("character", "all_characters", "人物")
    store.add_section("outline", "full_outline", "大纲")
    store.add_section("chapter", "chapter_2", "第2章 标题\n正文")
    store.save_extra_data("outline_total_chapters", "50")
    result = store.load_all_to_dict()
    assert result["world_setting"] == "世界观"
    assert result["characters"] == "人物"
    assert result["outline"] == "大纲"
    assert result["chapters"]["2"] == {"title": "标题", "content": "正文"}
    assert result["extra"]["outline_total_chapters"] == "50"


def test_extra_data(store):
    store.save_extra_data("k1", {"a": 1})
    assert store.load_extra_data("k1") == {"a": 1}
    assert store.load_extra_data("missing", default={}) == {}
    store.delete_extra_field("k1")
    assert store.load_extra_data("k1") is None


def test_search_related_keyword(store):
    store.add_section("chapter", "chapter_1", "主角在秘境中获得上古传承，修为大增")
    store.add_section("chapter", "chapter_2", "今天风和日丽，主角出门买菜")
    results = store.search_related("传承觉醒", n_results=2)
    assert results
    assert results[0]["metadata"]["title"] == "chapter_1"
    assert results[0]["metadata"]["type"] == "chapter"


def test_clear_and_delete(store, tmp_path):
    store.add_section("setting", "world_setting", "x")
    store.clear()
    assert store.get_section("setting", "world_setting") is None
    store.add_section("setting", "world_setting", "y")
    store.delete_novel()
    assert not os.path.isfile(store.file_path)


def test_list_and_delete_all(tmp_path):
    JsonNovelStore(db_path=str(tmp_path), novel_id="novel_a", novel_name="甲").add_section("setting", "world_setting", "x")
    JsonNovelStore(db_path=str(tmp_path), novel_id="novel_b", novel_name="乙").add_section("chapter", "chapter_1", "第1章 t\n正文")
    novels = JsonNovelStore.list_all_novels(db_path=str(tmp_path))
    assert len(novels) == 2
    names = {n["name"] for n in novels}
    assert names == {"甲", "乙"}
    b = next(n for n in novels if n["name"] == "乙")
    assert b["type_counts"]["chapter"] == 1
    assert JsonNovelStore.delete_all_novels(db_path=str(tmp_path)) == 2
    assert JsonNovelStore.list_all_novels(db_path=str(tmp_path)) == []


def test_atomic_write_no_tmp_left(store, tmp_path):
    store.add_section("setting", "world_setting", "内容")
    leftovers = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert not leftovers


def test_skill_recommend():
    from skills import skill_manager
    skills = skill_manager.list_skills()
    if not skills:
        pytest.skip("无内置 skill")
    recs = skill_manager.recommend_skills("写一场热血的战斗场面", step="chapter")
    assert isinstance(recs, list)
