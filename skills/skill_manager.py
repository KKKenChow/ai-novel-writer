"""
Skill（Prompt 技能包）装载引擎

每个 skill 是 skills/ 下的一个子目录，内含 SKILL.md（YAML frontmatter + 正文）：
---
name: 技能名称
description: 触发条件描述（越详细越容易被正确路由，支持多行）
apply_to: [chapter, continue]   # 生效步骤
enabled: true                   # 全局默认开关（可被每本小说独立覆盖）
source: 来源说明（可选，如蒸馏自哪篇文章）
keywords: [白描, 留白]          # 可选，检索推荐辅助关键词
phases: [climax]                # 可选，建议适用的六阶段（仅作推荐参考）
---
（正文：注入到 Prompt 前的指令内容）

step 取值：world / characters / outline / chapter / continue / polish / consistency / relations

启停状态说明：
- SKILL.md 里的 enabled 是「全局默认」
- 每本小说可在自己的存储 extra_data["skill_states"] 中覆盖（{dir_name: bool}）
- 最终生效 = novel_states.get(dir, 全局默认 enabled)
"""
import os
import re
import json
import shutil
import zipfile
import tempfile
from typing import List, Dict, Optional

import yaml

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

STEP_LABELS = {
    "world": "🌍 世界观",
    "characters": "👤 人物",
    "outline": "📋 大纲",
    "chapter": "📖 章节",
    "continue": "✍️ 续写",
    "polish": "🎨 润色",
    "consistency": "🔍 一致性检查",
    "relations": "🕸️ 角色图谱",
}

PHASE_LABELS = {
    "opening": "开篇",
    "early_dev": "早期发展",
    "mid_dev": "中期发展",
    "late_dev": "后期发展",
    "climax": "高潮",
    "resolution": "收尾",
}

# 单个 skill 注入正文的最大字数默认值，防止 token 膨胀（可在 Skill 管理 Tab 调整，存 user_config.json）
MAX_INJECT_CHARS = 2000


def get_inject_max_chars() -> int:
    """读取用户配置的单技能注入字数上限（默认 MAX_INJECT_CHARS，下限 100）"""
    try:
        from api import user_config
        v = int(user_config.load_config().get("skill_inject_chars", MAX_INJECT_CHARS) or MAX_INJECT_CHARS)
        return max(100, v)
    except Exception:
        return MAX_INJECT_CHARS


# ---------------- frontmatter 解析（pyyaml，兼容多行 description） ----------------

def parse_frontmatter(text: str):
    """解析 YAML frontmatter，返回 (meta, body)。无 frontmatter 时 meta 为空"""
    meta, body = {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if m:
        try:
            loaded = yaml.safe_load(m.group(1))
            if isinstance(loaded, dict):
                meta = loaded
        except yaml.YAMLError:
            pass
        body = m.group(2)
    return meta, body.strip()


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _dump_frontmatter(meta: Dict, body: str) -> str:
    fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{fm}---\n\n{body.strip()}\n"


# ---------------- skill 增删改查 ----------------

def list_skills() -> List[Dict]:
    """扫描 skills/ 目录，返回所有 skill 的元信息（含正文字数）"""
    skills = []
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for entry in sorted(os.listdir(SKILLS_DIR)):
        d = os.path.join(SKILLS_DIR, entry)
        md = os.path.join(d, "SKILL.md")
        if not os.path.isdir(d) or not os.path.isfile(md):
            continue
        try:
            with open(md, "r", encoding="utf-8") as f:
                meta, body = parse_frontmatter(f.read())
        except Exception:
            continue
        skills.append({
            "dir": entry,
            "path": md,
            "name": str(meta.get("name", entry)),
            "description": str(meta.get("description", "") or ""),
            "apply_to": _as_list(meta.get("apply_to")),
            "enabled": bool(meta.get("enabled", True)),
            "source": str(meta.get("source", "") or ""),
            "keywords": _as_list(meta.get("keywords")),
            "phases": _as_list(meta.get("phases")),
            "body": body,
            "chars": len(body),
        })
    return skills


def save_skill(dir_name: str, meta: Dict, body: str):
    d = os.path.join(SKILLS_DIR, dir_name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(_dump_frontmatter(meta, body))


def set_enabled(dir_name: str, enabled: bool):
    """设置全局默认开关（写入 SKILL.md）。按小说覆盖请用 set_novel_skill_enabled"""
    d = os.path.join(SKILLS_DIR, dir_name)
    md = os.path.join(d, "SKILL.md")
    with open(md, "r", encoding="utf-8") as f:
        meta, body = parse_frontmatter(f.read())
    meta["enabled"] = enabled
    save_skill(dir_name, meta, body)


def delete_skill(dir_name: str):
    d = os.path.join(SKILLS_DIR, dir_name)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


# ---------------- 按小说启停（状态存各小说本地存储 extra_data） ----------------

SKILL_STATES_KEY = "skill_states"


def get_skill_states(vs) -> Dict[str, bool]:
    """读取某本小说的 skill 启停覆盖状态 {dir_name: bool}"""
    if vs is None:
        return {}
    try:
        states = vs.load_extra_data(SKILL_STATES_KEY, default={})
        return states if isinstance(states, dict) else {}
    except Exception:
        return {}


def set_novel_skill_enabled(vs, dir_name: str, enabled: Optional[bool]):
    """设置某本小说对某 skill 的覆盖开关；enabled=None 表示清除覆盖（回落到全局默认）"""
    states = get_skill_states(vs)
    if enabled is None:
        states.pop(dir_name, None)
    else:
        states[dir_name] = bool(enabled)
    vs.save_extra_data(SKILL_STATES_KEY, states)


def is_skill_enabled(skill: Dict, novel_states: Dict[str, bool]) -> bool:
    """最终生效开关 = 小说覆盖值 或 全局默认"""
    return novel_states.get(skill["dir"], skill["enabled"])


# ---------------- Prompt 注入 ----------------

def get_active_prompts(step: str, novel_states: Optional[Dict[str, bool]] = None,
                       max_chars: Optional[int] = None) -> str:
    """拼接所有启用且作用于当前 step 的 skill 正文（每个截断至 max_chars，默认读用户配置）"""
    if max_chars is None:
        max_chars = get_inject_max_chars()
    novel_states = novel_states or {}
    parts = []
    for s in list_skills():
        if is_skill_enabled(s, novel_states) and step in s["apply_to"] and s["body"]:
            body = s["body"]
            if len(body) > max_chars:
                body = body[:max_chars] + "\n……（内容过长已截断）"
            parts.append(f"【创作技能：{s['name']}】\n{body}")
    return "\n\n".join(parts)


def inject_skills(prompt: str, step: str, novel_states: Optional[Dict[str, bool]] = None) -> str:
    """供 LLMAPIClient.skill_provider 回调使用：把 skill 文本前置注入 prompt"""
    extra = get_active_prompts(step, novel_states)
    if not extra:
        return prompt
    return f"{extra}\n\n{'=' * 20}\n\n{prompt}"


# ---------------- 本地导入外部 skill ----------------

def import_skill_from_zip(file_obj, original_name: str = "") -> str:
    """从 zip 文件导入 skill（zip 内需包含某个目录下的 SKILL.md）。返回导入的目录名"""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(file_obj.read())
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        # 在解压树中定位 SKILL.md（兼容仓库嵌套结构，如 repo/sun-style-writing/SKILL.md）
        skill_md = None
        for root, _dirs, files in os.walk(extract_dir):
            if "__MACOSX" in root:
                continue
            if "SKILL.md" in files:
                skill_md = os.path.join(root, "SKILL.md")
                break
        if not skill_md:
            raise ValueError("zip 中未找到 SKILL.md，请确认是合法的 skill 包")
        src_dir = os.path.dirname(skill_md)
        dir_name = os.path.basename(src_dir) or "imported_skill"
        return _copy_skill_dir(src_dir, dir_name)


def import_skill_from_md(file_obj, fallback_name: str = "imported_skill") -> str:
    """从单个 SKILL.md 文件导入。返回导入的目录名"""
    with tempfile.TemporaryDirectory() as tmp:
        md_path = os.path.join(tmp, "SKILL.md")
        with open(md_path, "wb") as f:
            f.write(file_obj.read())
        dir_name = re.sub(r'[\\/:*?"<>|]', "_", fallback_name) or "imported_skill"
        return _copy_skill_dir(tmp, dir_name, only_md=True)


def _copy_skill_dir(src_dir: str, dir_name: str, only_md: bool = False) -> str:
    """把 skill 目录复制到 skills/ 下，重名时自动加后缀。返回最终目录名"""
    final_name = dir_name
    n = 1
    while os.path.exists(os.path.join(SKILLS_DIR, final_name)):
        final_name = f"{dir_name}_{n}"
        n += 1
    dst = os.path.join(SKILLS_DIR, final_name)
    os.makedirs(dst, exist_ok=True)
    if only_md:
        shutil.copy2(os.path.join(src_dir, "SKILL.md"), os.path.join(dst, "SKILL.md"))
    else:
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            t = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, t)
            else:
                shutil.copy2(s, t)
    return final_name


def ensure_apply_to(dir_name: str, apply_to: List[str]):
    """导入的外部 skill 若缺 apply_to 字段则补上（其余字段保持原样）"""
    md = os.path.join(SKILLS_DIR, dir_name, "SKILL.md")
    with open(md, "r", encoding="utf-8") as f:
        meta, body = parse_frontmatter(f.read())
    if not _as_list(meta.get("apply_to")):
        meta["apply_to"] = apply_to or ["chapter", "continue", "polish"]
    if "enabled" not in meta:
        meta["enabled"] = True
    if "name" not in meta:
        meta["name"] = dir_name
    save_skill(dir_name, meta, body)


# ---------------- 文章蒸馏 ----------------

DISTILL_PROMPT_TEMPLATE = """你是一位写作方法论专家。请阅读下面的参考文章，从中蒸馏出一个**可被 AI 直接执行的写作技能包（Skill）**。

【参考文章】
{articles}

【输出要求】
严格按以下 Markdown 格式输出（不要输出任何额外解释）：

---
name: （技能名称，简洁有力，10字以内）
description: （详细触发条件：什么样的写作场景/用户需求应该使用此技能，60-150字，列举尽可能多的触发说法）
---

# （技能名称）

（第一段：一句话核心哲学——这套写法最本质的原则，加粗）

（然后分 3-7 条技法展开，每条一个二级标题，每条必须包含以下三部分：）
## 一、（技法名）
**做法：** 具体操作规则，可执行、可检查
**禁止：** 明确列出绝对不能出现的写法/句式
**示例：** 从参考文章中摘取或仿写的正反对照示例（"❌ 普通写法" vs "✅ 本技法写法"）

（最后：）
## 适用边界
**适合：** 什么材料/场景适合用
**不适合：** 什么材料/场景不要用

【质量要求】
1. 每条技法必须是可执行的操作指令，不是空泛的文学评论（❌"要有感染力" ✅"用具体数字代替形容词，如'二十六个人'而非'很多人'"）
2. 示例必须从原文中提炼，让 AI 能模仿具体句式
3. 全文控制在 1500-3000 字
4. 直接输出 Markdown 内容，不要用代码块包裹
"""
