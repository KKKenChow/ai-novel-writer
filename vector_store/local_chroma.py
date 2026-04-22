"""
本地向量存储 - 用ChromaDB存储小说上下文
用于检索相关设定和前情，保持一致性
"""
import chromadb
from typing import List, Dict, Optional
import os
import hashlib
import re
import json

def sanitize_collection_name(name: str) -> str:
    """将名称转换为ChromaDB兼容的collection name（仅支持 [a-zA-Z0-9._-]）"""
    # 使用 hash 确保唯一性，保留原始名称的前缀便于识别
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    safe_prefix = re.sub(r'[^a-zA-Z0-9]', '_', name)[:50]
    return f"n_{safe_prefix}_{hash_suffix}"

class LocalNovelVectorStore:
    def __init__(self, db_path="./chroma_db", novel_id="default", novel_name=None):
        self.client = chromadb.PersistentClient(path=db_path)
        self.novel_id = novel_id
        self._collection_name = sanitize_collection_name(novel_id)
        # novel_name 是可修改的显示名称，存入 collection metadata
        display_name = novel_name if novel_name else novel_id
        # 立即创建 collection
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "original_name": self.novel_id, "novel_name": display_name}
        )
        # 如果传入了 novel_name 且与现有不同，更新 metadata
        # 注意：modify 时不能传 hnsw:space，ChromaDB 不允许修改已创建 collection 的 distance function
        if novel_name and self.collection.metadata.get("novel_name") != novel_name:
            safe_metadata = {k: v for k, v in self.collection.metadata.items() if not k.startswith("hnsw:")}
            self.collection.modify(metadata={
                **safe_metadata,
                "novel_name": novel_name
            })
    
    def rename(self, new_name: str):
        """修改小说的显示名称（不改 collection 和 novel_id）"""
        # modify 时不能传 hnsw:space，ChromaDB 不允许修改 distance function
        safe_metadata = {k: v for k, v in self.collection.metadata.items() if not k.startswith("hnsw:")}
        self.collection.modify(metadata={
            **safe_metadata,
            "novel_name": new_name
        })
    
    def add_section(self, section_type: str, title: str, content: str):
        """添加一个章节/段落到向量库（如果ID已存在则覆盖更新）"""
        doc_id = f"{section_type}_{title.replace(' ', '_')}"
        full_text = f"{title}\n{content}"
        
        # 使用 upsert 语义：先尝试删除旧记录再添加
        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass
        
        self.collection.add(
            documents=[full_text],
            metadatas=[{"type": section_type, "title": title}],
            ids=[doc_id]
        )
    
    def update_section(self, section_type: str, title: str, content: str):
        """更新向量库中的某个章节/段落（upsert语义，安全覆盖）"""
        doc_id = f"{section_type}_{title.replace(' ', '_')}"
        full_text = f"{title}\n{content}"
        
        self.collection.upsert(
            documents=[full_text],
            metadatas=[{"type": section_type, "title": title}],
            ids=[doc_id]
        )
    
    def delete_section(self, section_type: str, title: str):
        """删除向量库中的某个章节/段落"""
        doc_id = f"{section_type}_{title.replace(' ', '_')}"
        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass
    
    def get_section(self, section_type: str, title: str) -> Optional[str]:
        """获取向量库中某个特定段落的内容"""
        doc_id = f"{section_type}_{title.replace(' ', '_')}"
        try:
            result = self.collection.get(ids=[doc_id])
            if result and result.get("documents") and result["documents"]:
                # 返回时去掉标题行（add_section时添加的）
                content = result["documents"][0]
                if content.startswith(title + "\n"):
                    content = content[len(title) + 1:]
                return content
        except Exception:
            pass
        return None
    
    def load_all_to_dict(self) -> Dict:
        """加载向量库中所有内容，返回结构化字典，用于恢复session_state"""
        result = {
            "world_setting": "",
            "characters": "",
            "outline": "",
            "chapters": {},
            "extra": {}  # 额外数据：*_original, *_prompt, consistency_result, relation_graph 等
        }
        all_data = self.collection.get()
        if not all_data or not all_data.get("documents"):
            return result
        
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            section_type = meta.get("type", "")
            title = meta.get("title", "")
            content = doc
            # 去掉add_section时添加的标题行
            if content.startswith(title + "\n"):
                content = content[len(title) + 1:]
            
            if section_type == "setting":
                result["world_setting"] = content
            elif section_type == "character":
                result["characters"] = content
            elif section_type == "outline":
                result["outline"] = content
            elif section_type == "chapter":
                # 从 title "chapter_X" 提取章节号
                import re
                match = re.match(r"chapter_(\d+)", title)
                if match:
                    chap_num = match.group(1)
                    # 从内容中提取章节标题
                    lines = content.split("\n", 1)
                    chap_title = ""
                    chap_content = content
                    first_line = lines[0].strip()
                    match2 = re.match(r"第\d+章\s*(.*)", first_line)
                    if match2:
                        chap_title = match2.group(1).strip()
                        chap_content = lines[1] if len(lines) > 1 else content
                    result["chapters"][chap_num] = {
                        "title": chap_title,
                        "content": chap_content
                    }
            elif section_type == "extra_data":
                # 额外数据（JSON 格式）
                try:
                    result["extra"] = json.loads(doc)
                except Exception:
                    pass
        
        return result
    
    def search_related(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜索和当前query相关的上下文"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        contexts = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                contexts.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        return contexts
    
    def get_all_by_type(self, section_type: str) -> List[Dict]:
        """获取某一类型的所有内容"""
        all_data = self.collection.get()
        results = []
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            if meta["type"] == section_type:
                results.append({
                    "content": doc,
                    "metadata": meta
                })
        return results
    
    def clear(self):
        """清空当前小说的向量库（保留collection结构）"""
        try:
            self.client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "original_name": self.novel_id, "novel_name": self.novel_id}
        )

    def delete_novel(self):
        """彻底删除当前小说的collection（不重建）"""
        try:
            self.client.delete_collection(name=self._collection_name)
        except Exception:
            pass
    
    # ---- 额外数据持久化（原始生成文本、prompt、检查结果等非 section 数据） ----
    
    EXTRA_DATA_DOC_ID = "_extra_data"
    
    def _load_extra_data_raw(self) -> dict:
        """读取额外数据的原始字典，不存在则返回空字典"""
        try:
            result = self.collection.get(ids=[self.EXTRA_DATA_DOC_ID])
            if result and result.get("documents") and result["documents"]:
                return json.loads(result["documents"][0])
        except Exception:
            pass
        return {}
    
    def save_extra_data(self, key: str, value):
        """保存单条额外数据（key-value），与已有数据合并后整体写入"""
        data = self._load_extra_data_raw()
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
        self.collection.upsert(
            documents=[json.dumps(data, ensure_ascii=False)],
            metadatas=[{"type": "extra_data"}],
            ids=[self.EXTRA_DATA_DOC_ID]
        )
    
    def load_extra_data(self, key: str = None, default=None):
        """读取额外数据。指定 key 返回单条值，不指定返回全部字典"""
        data = self._load_extra_data_raw()
        if key is None:
            return data
        return data.get(key, default)
    
    def delete_extra_field(self, key: str):
        """删除额外数据中的某个字段"""
        self.save_extra_data(key, None)

    @staticmethod
    def list_all_novels(db_path="./chroma_db") -> List[Dict]:
        """列出向量库中所有小说及其内容摘要"""
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        novels = []
        for col in collections:
            if col.name.startswith("n_") or col.name.startswith("novel_"):
                try:
                    col_data = col.get()
                    type_counts = {"setting": 0, "character": 0, "outline": 0, "chapter": 0}
                    for meta in col_data.get("metadatas", []):
                        if meta and meta.get("type") in type_counts:
                            type_counts[meta["type"]] += 1
                    
                    # novel_name 优先从 metadata 取，兼容旧数据回退到 original_name
                    novel_name = col.metadata.get("novel_name") or col.metadata.get("original_name") or col.name
                    novel_id = col.metadata.get("original_name", col.name)
                    novels.append({
                        "id": novel_id,               # 内部ID，不可变
                        "name": novel_name,            # 显示名称，可修改
                        "collection_name": col.name,
                        "type_counts": type_counts,
                        "total_docs": len(col_data.get("documents", []))
                    })
                except Exception:
                    continue
        return novels
