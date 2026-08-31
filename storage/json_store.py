"""
本地 JSON 存储 - 每本小说一个 JSON 文件
替代原 ChromaDB 向量库方案：零额外依赖、数据可读可编辑、可直接备份

存储结构（novels_data/<safe_id>.json）：
{
  "novel_id": "...",       # 不可变内部ID
  "novel_name": "...",     # 可修改显示名
  "sections": {            # 所有内容段落，key 为 doc_id（type_title）
    "setting_world_setting": {"type": "setting", "title": "world_setting", "content": "..."},
    "chapter_chapter_1":     {"type": "chapter", "title": "chapter_1", "content": "..."}
  },
  "extra": {...}           # extra_data（原始prompt、检查结果、图谱、参数等）
}

检索说明：search_related 用中文二元字符组（bigram）重合度做关键词检索，
替代原向量语义检索——本项目的核心上下文均为按 ID/章节号的确定性直取，
该检索仅用于补充参考片段与 skill 推荐。
"""
import os
import re
import json
import hashlib
import tempfile
from typing import List, Dict, Optional

DEFAULT_DATA_DIR = "./novels_data"


def sanitize_file_name(name: str) -> str:
    """将 novel_id 转换为安全的文件名（唯一且可识别）"""
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    safe_prefix = re.sub(r'[^a-zA-Z0-9._-]', '_', name)[:50]
    return f"n_{safe_prefix}_{hash_suffix}"


def _bigrams(text: str) -> set:
    """提取中文/英文混合文本的字符二元组集合，用于关键词重合度计算"""
    text = re.sub(r"\s+", "", text)
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else {text}


class JsonNovelStore:
    def __init__(self, db_path: str = DEFAULT_DATA_DIR, novel_id: str = "default", novel_name: str = None):
        self.data_dir = db_path
        os.makedirs(self.data_dir, exist_ok=True)
        self.novel_id = novel_id
        self.file_path = os.path.join(self.data_dir, sanitize_file_name(novel_id) + ".json")
        self._data = self._load_file()
        if novel_name and self._data.get("novel_name") != novel_name:
            self._data["novel_name"] = novel_name
            self._save()

    # ---------- 文件读写（原子写入，防止中断损坏） ----------

    def _load_file(self) -> dict:
        if os.path.isfile(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("sections", {})
                data.setdefault("extra", {})
                return data
            except Exception:
                pass
        return {"novel_id": self.novel_id, "novel_name": self.novel_id, "sections": {}, "extra": {}}

    def _save(self):
        fd, tmp = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.file_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def rename(self, new_name: str):
        """修改小说的显示名称（不改 novel_id 和文件名）"""
        self._data["novel_name"] = new_name
        self._save()

    # ---------- section 增删改查 ----------

    @staticmethod
    def _doc_id(section_type: str, title: str) -> str:
        return f"{section_type}_{title.replace(' ', '_')}"

    def add_section(self, section_type: str, title: str, content: str):
        """添加一个章节/段落到存储（同 ID 覆盖更新）"""
        self._data["sections"][self._doc_id(section_type, title)] = {
            "type": section_type, "title": title, "content": content,
        }
        self._save()

    # upsert 语义与 add_section 相同
    update_section = add_section

    def delete_section(self, section_type: str, title: str):
        self._data["sections"].pop(self._doc_id(section_type, title), None)
        self._save()

    def get_section(self, section_type: str, title: str) -> Optional[str]:
        sec = self._data["sections"].get(self._doc_id(section_type, title))
        return sec["content"] if sec else None

    def iter_sections(self) -> List[Dict]:
        """返回所有 section 的列表 [{type, title, content}]"""
        return list(self._data["sections"].values())

    def get_all_by_type(self, section_type: str) -> List[Dict]:
        return [
            {"content": s["content"], "metadata": {"type": s["type"], "title": s["title"]}}
            for s in self._data["sections"].values() if s["type"] == section_type
        ]

    def load_all_to_dict(self) -> Dict:
        """加载所有内容，返回结构化字典，用于恢复 session_state"""
        result = {"world_setting": "", "characters": "", "character_cards": "", "outline": "", "chapters": {}, "extra": dict(self._data["extra"])}
        for sec in self._data["sections"].values():
            stype, title, content = sec["type"], sec["title"], sec["content"]
            if stype == "setting":
                result["world_setting"] = content
            elif stype == "character":
                result["characters"] = content
            elif stype == "character_cards":
                # 结构化角色卡（JSON 字符串），人物自由文本仍走上面的 character 分支
                result["character_cards"] = content
            elif stype == "outline":
                result["outline"] = content
            elif stype == "chapter":
                m = re.match(r"chapter_(\d+)", title)
                if m:
                    chap_num = m.group(1)
                    lines = content.split("\n", 1)
                    chap_title, chap_content = "", content
                    m2 = re.match(r"第\d+章\s*(.*)", lines[0].strip())
                    if m2:
                        chap_title = m2.group(1).strip()
                        chap_content = lines[1] if len(lines) > 1 else content
                    result["chapters"][chap_num] = {"title": chap_title, "content": chap_content}
        return result

    # ---------- 关键词检索（替代原向量语义检索） ----------

    def search_related(self, query: str, n_results: int = 5) -> List[Dict]:
        """按 bigram 重合度检索相关段落，返回与原向量库相同的结构"""
        q = _bigrams(query)
        if not q:
            return []
        scored = []
        for sec in self._data["sections"].values():
            full = f"{sec['title']}\n{sec['content']}"
            b = _bigrams(full)
            if not b:
                continue
            score = len(q & b) / len(q | b)  # Jaccard
            if score > 0:
                scored.append((score, full, sec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": full, "metadata": {"type": s["type"], "title": s["title"]}, "distance": 1 - score}
            for score, full, s in scored[:n_results]
        ]

    # ---------- 整体操作 ----------

    def clear(self):
        """清空当前小说的所有内容（保留名称）"""
        self._data["sections"] = {}
        self._data["extra"] = {}
        self._save()

    def delete_novel(self):
        """彻底删除当前小说的数据文件"""
        if os.path.isfile(self.file_path):
            os.remove(self.file_path)
        self._data = {"novel_id": self.novel_id, "novel_name": self.novel_id, "sections": {}, "extra": {}}

    # ---------- extra_data 机制 ----------

    def save_extra_data(self, key: str, value):
        if value is None:
            self._data["extra"].pop(key, None)
        else:
            self._data["extra"][key] = value
        self._save()

    def load_extra_data(self, key: str = None, default=None):
        if key is None:
            return dict(self._data["extra"])
        return self._data["extra"].get(key, default)

    def delete_extra_field(self, key: str):
        self.save_extra_data(key, None)

    # ---------- 多小说管理 ----------

    @staticmethod
    def list_all_novels(db_path: str = DEFAULT_DATA_DIR) -> List[Dict]:
        """列出所有小说及其内容摘要"""
        novels = []
        if not os.path.isdir(db_path):
            return novels
        for fname in sorted(os.listdir(db_path)):
            if not (fname.startswith("n_") and fname.endswith(".json")):
                continue
            try:
                with open(os.path.join(db_path, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                type_counts = {"setting": 0, "character": 0, "outline": 0, "chapter": 0}
                for sec in data.get("sections", {}).values():
                    if sec.get("type") in type_counts:
                        type_counts[sec["type"]] += 1
                novels.append({
                    "id": data.get("novel_id", fname[:-5]),
                    "name": data.get("novel_name") or data.get("novel_id", fname[:-5]),
                    "collection_name": fname[:-5],
                    "type_counts": type_counts,
                    "total_docs": len(data.get("sections", {})),
                })
            except Exception:
                continue
        return novels

    @staticmethod
    def delete_all_novels(db_path: str = DEFAULT_DATA_DIR) -> int:
        """删除所有小说数据文件，返回删除数量"""
        count = 0
        if os.path.isdir(db_path):
            for fname in os.listdir(db_path):
                if fname.startswith("n_") and (fname.endswith(".json") or fname.endswith(".tmp")):
                    os.remove(os.path.join(db_path, fname))
                    count += 1
        return count
