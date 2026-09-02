"""角色卡结构化模块（模块 A）

数据结构（每张卡）：
{
  "name": "...",              # 姓名（必填）
  "role": "main|support",     # 主角/配角
  "identity": "...",          # 身份背景
  "personality": "...",       # 性格
  "relationships": "...",     # 人物关系
  "appearance_chapter": 1,    # 登场章节（用于章节生成时按登场过滤注入）
  "exit_chapter": null,       # 退场章节（null = 不退场）
  "notes": "..."              # 备注
}

设计要点：
- 结构化角色卡存于 sections["character_cards_cards"]（JSON 字符串）；
  旧的角色自由文本 character_all_characters 始终保留，作为备份与降级——
  解析失败或旧小说未迁移时，所有链路自动回退到自由文本模式。
- LLM 输出约定为 [角色]...[/角色] 标签块（TAGGED_FORMAT），解析器对字段缺失容错。
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 角色卡字段定义：(字段key, 中文标签, 是否必填)
CARD_FIELDS = [
    ("name", "姓名", True),
    ("alias", "别名/代号", False),   # 代号/外号/马甲（如「夜莺」），用于实体时间锁遮蔽
    ("role", "类型", False),           # 主角/配角 → main/support
    ("identity", "身份", False),
    ("personality", "性格", False),
    ("relationships", "人物关系", False),
    ("appearance_chapter", "登场章节", False),
    ("exit_chapter", "退场章节", False),
    ("notes", "备注", False),
]

# LLM 输出格式说明（嵌入生成 prompt）
TAGGED_FORMAT = """[角色]
姓名：角色姓名
别名/代号：外号/代号/马甲（如「夜莺」；没有就留空）
类型：主角 或 配角
身份：身份背景一句话
性格：性格特点
人物关系：与其他角色的关系
登场章节：第几章首次登场（填数字，第1章登场填1）
退场章节：第几章退场（填数字，不退场留空）
备注：其他重要设定（外貌/目标/能力等）
[/角色]"""

# 存储用的 section title（doc_id = "character_cards_cards"）
CARDS_SECTION_TITLE = "cards"

# 参与全文查找/替换的文本字段（排除角色类型、登场/退场章节等结构化字段）
CARD_SEARCH_FIELDS = [
    ("name", "姓名"),
    ("alias", "别名/代号"),
    ("identity", "身份"),
    ("personality", "性格"),
    ("relationships", "人物关系"),
    ("notes", "备注"),
]


def new_card(name: str, role: str = "support") -> Dict:
    """创建一张空白角色卡"""
    return {
        "name": name, "alias": "", "role": role, "identity": "", "personality": "",
        "relationships": "", "appearance_chapter": 1, "exit_chapter": None,
        "notes": "",
    }


def _parse_chapter_num(text: str) -> Optional[int]:
    """从"第3章"/"3"/"三章"等文本解析章节号，失败返回 None"""
    if not text:
        return None
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else None


def normalize_card(raw: Dict) -> Dict:
    """清洗/补全一张角色卡：字段缺省、类型归一化、章节号转数字"""
    card = new_card(str(raw.get("name", "")).strip() or "未命名")
    for key, _, _ in CARD_FIELDS:
        if key in raw and raw[key] is not None:
            card[key] = raw[key]
    # role 归一化：主角/main → main，其他 → support
    role_text = str(card.get("role", ""))
    card["role"] = "main" if role_text in ("main", "主角", "主要角色", "核心") else "support"
    # 章节号归一化
    card["appearance_chapter"] = _parse_chapter_num(card.get("appearance_chapter")) or 1
    card["exit_chapter"] = _parse_chapter_num(card.get("exit_chapter"))
    return card


def parse_character_cards(text: str) -> Tuple[List[Dict], bool]:
    """解析 LLM 输出的 [角色]...[/角色] 标签块为角色卡列表。

    返回 (cards, ok)：ok=False 表示一个块都没解析到（调用方应重试或降级自由文本）。
    单个块字段缺失不视为失败（normalize_card 兜底）。
    """
    cards = []
    for block in re.findall(r"\[角色\](.*?)\[/角色\]", text, re.DOTALL):
        raw = {}
        for key, label, _ in CARD_FIELDS:
            m = re.search(rf"{label}[：:]\s*(.*)", block)
            if m:
                raw[key] = m.group(1).strip()
        if raw.get("name"):
            cards.append(normalize_card(raw))
    return cards, bool(cards)


def cards_to_text(cards: List[Dict]) -> str:
    """角色卡渲染为自由文本（用于：兼容旧的整段人物设定注入、结构化→文本切换、备份）"""
    parts = []
    for c in cards:
        role_label = "主角" if c.get("role") == "main" else "配角"
        lines = [f"■ {c['name']}（{role_label}）"]
        if c.get("alias"):
            lines.append(f"别名/代号：{c['alias']}")
        if c.get("identity"):
            lines.append(f"身份：{c['identity']}")
        if c.get("personality"):
            lines.append(f"性格：{c['personality']}")
        if c.get("relationships"):
            lines.append(f"人物关系：{c['relationships']}")
        exit_ch = c.get("exit_chapter")
        span = f"第{c.get('appearance_chapter', 1)}章登场" + (f"，第{exit_ch}章退场" if exit_ch else "")
        lines.append(span)
        if c.get("notes"):
            lines.append(f"备注：{c['notes']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def cards_to_brief(cards: List[Dict]) -> str:
    """角色卡精简渲染（开篇/早期阶段注入用）：只含名字/别名/身份/性格。

    不含人物关系、登场行、备注——这些字段往往带着人物弧线/后续剧情，
    开篇阶段不该让模型看到（信息按叙事时点解锁）。
    """
    parts = []
    for c in cards:
        role_label = "主角" if c.get("role") == "main" else "配角"
        lines = [f"■ {c['name']}（{role_label}）"]
        if c.get("alias"):
            lines.append(f"别名/代号：{c['alias']}")
        if c.get("identity"):
            lines.append(f"身份：{c['identity']}")
        if c.get("personality"):
            lines.append(f"性格：{c['personality']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def cards_to_json(cards: List[Dict]) -> str:
    return json.dumps(cards, ensure_ascii=False, indent=1)


def cards_from_json(text: str) -> List[Dict]:
    """从存储的 JSON 字符串恢复角色卡列表，失败返回 []"""
    try:
        data = json.loads(text or "[]")
        if isinstance(data, list):
            return [normalize_card(c) for c in data if isinstance(c, dict)]
    except Exception as e:
        logger.warning(f"角色卡 JSON 解析失败: {e}")
    return []


def filter_cards_for_chapter(cards: List[Dict], chapter_num: int) -> Tuple[List[Dict], List[str]]:
    """按登场/退场章过滤（TODO 2.3）：
    返回 (active_cards, absent_names)：
    - active_cards: appearance_chapter <= chapter_num <= (exit_chapter or ∞) 的角色完整卡
    - absent_names: 未登场角色的名字列表（仅注入名字作兜底，防止前文提及的角色显得凭空出现）
    """
    active, absent = [], []
    for c in cards:
        appear = c.get("appearance_chapter") or 1
        exit_ch = c.get("exit_chapter")
        if appear <= chapter_num and (exit_ch is None or chapter_num <= exit_ch):
            active.append(c)
        elif chapter_num < appear:
            absent.append(c["name"])
        # 已退场的角色不注入详情也不进兜底名单（退场后一般不再需要）
    return active, absent


def filter_cards_for_range(cards: List[Dict], start_ch: int, end_ch: int) -> List[Dict]:
    """按卷/章节区间过滤角色卡（用于大纲、卷逐章细纲等整段规划场景）：
    返回在该区间任意一章在场合的角色卡（登场章 ≤ 区间末 且 退场章 ≥ 区间首）。
    无登场章按第 1 章，无退场章视为不退场。"""
    if start_ch > end_ch:
        start_ch, end_ch = end_ch, start_ch
    active = []
    for c in cards:
        appear = c.get("appearance_chapter") or 1
        exit_ch = c.get("exit_chapter")
        if appear <= end_ch and (exit_ch is None or exit_ch >= start_ch):
            active.append(c)
    return active
