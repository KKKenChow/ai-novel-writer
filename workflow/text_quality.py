"""文字质量工具：AI 套话黑名单检测 + 规避指令（纯函数，无外部依赖）"""
import re
from typing import List, Tuple

# 高频 AI 腔/网文套话黑名单（可在 UI 中追加自定义词条）
DEFAULT_CLICHE_BLACKLIST = [
    "不禁", "不由得", "嘴角勾起", "勾起一抹", "空气仿佛凝固", "空气瞬间凝固",
    "眼中闪过一丝", "眼底闪过", "眸光一沉", "深吸一口气", "深吸了一口气",
    "心脏猛地一跳", "心跳漏了一拍", "时间仿佛静止", "仿佛过了一个世纪",
    "握紧拳头", "拳头紧握", "指甲深深陷入掌心", "后背发凉", "冷汗直流",
    "不寒而栗", "毛骨悚然", "宛如...", "犹如天人", "不可思议地看着",
    "难以置信", "瞳孔地震", "嘴角微微上扬", "露出意味深长的笑容",
    "空气突然安静", "陷入了沉思", "若有所思", "目光深邃",
]


def detect_cliches(text: str, extra_blacklist: List[str] = None) -> List[Tuple[str, int]]:
    """检测文本中的套话，返回 [(词条, 出现次数)]，按次数降序"""
    blacklist = list(DEFAULT_CLICHE_BLACKLIST)
    for w in (extra_blacklist or []):
        w = w.strip()
        if w and w not in blacklist:
            blacklist.append(w)
    hits = []
    for w in blacklist:
        # "宛如..." 这类模式跳过精确匹配
        if "..." in w:
            continue
        c = text.count(w)
        if c > 0:
            hits.append((w, c))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def cliche_report(text: str, extra_blacklist: List[str] = None) -> str:
    """生成可读的套话检测报告（无命中返回空字符串）"""
    hits = detect_cliches(text, extra_blacklist)
    if not hits:
        return ""
    total = sum(c for _, c in hits)
    top = "、".join(f"「{w}」×{c}" for w, c in hits[:8])
    return f"检测到 {total} 处高频套话/AI腔：{top}{' 等' if len(hits) > 8 else ''}（建议用「🎨 去AI腔」处理）"


def cliche_avoidance_instruction(extra_blacklist: List[str] = None, max_words: int = 15) -> str:
    """注入生成 prompt 的套话规避指令（ proactive 防御）。自定义词条优先保留"""
    words = []
    for w in (extra_blacklist or []):
        w = w.strip()
        if w and w not in words:
            words.append(w)
    for w in DEFAULT_CLICHE_BLACKLIST:
        if "..." not in w and w not in words:
            words.append(w)
    words = words[:max_words]
    return "**避免套话**：禁止使用以下高频 AI 腔/套话：" + "、".join(f"「{w}」" for w in words) + "。用具体的动作、细节、对话代替这些空洞表达。"
