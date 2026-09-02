"""
全链路小说创作工作流
"""
import logging
import re
import json
from typing import Dict, List, Optional
from api.api_client import LLMAPIClient, GenerationCancelled
from storage.json_store import JsonNovelStore
from workflow.text_quality import detect_cliches, cliche_report, cliche_avoidance_instruction
from workflow import character_cards as cc

logger = logging.getLogger(__name__)

# AI 套话黑名单自定义词条的 extra_data key
CUSTOM_BLACKLIST_KEY = "custom_cliche_blacklist"

# 大纲内「逐章概要」小节的分节标记（曾用名「逐章大纲」，需兼容旧数据）
VOLUME_CHAPTER_MARKER = "## 逐章概要"
VOLUME_CHAPTER_MARKER_OLD = "## 逐章大纲"

# AI 内容安全审查拒绝的关键词（拒绝内容不入库，避免污染本地存储）
_REFUSAL_KEYWORDS = [
    "我不能按照你的要求进行创作",
    "不符合健康的价值观和道德准则",
    "我不能为你创作",
    "我无法为你提供",
    "我无法完成这个请求",
    "无法按照您的要求",
    "不符合相关法律法规",
    "涉及违规内容",
    "我不能生成",
    "我无法生成",
]

def is_ai_refusal(text: str) -> bool:
    """检测 AI 是否返回了拒绝生成的内容"""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in _REFUSAL_KEYWORDS)

class ChapterPaused(Exception):
    """章节生成被用户暂停/取消。message 为面向用户的友好说明。"""
    pass


class FullNovelWorkflow:
    def __init__(self, api_client: LLMAPIClient, vector_store: JsonNovelStore):
        self.api = api_client
        self.vs = vector_store
        self.novel_info = {}
        # 最近一次章节生成的范围校验警告（供 UI 展示）
        self.last_scope_warning = ""
        # 最近一次节拍校验警告（供 UI 展示；空串表示校验通过）
        self.last_beats_warning = ""
        # 可选回调（供 Web 界面使用）：阶段进度 fn(str)、正文流式输出 fn(str)
        self.on_progress = None
        self.on_token = None
        # 可选回调：推理模型思考过程流式输出 fn(str)
        self.on_reasoning = None
        # 可选回调：需要用户决策时调用 fn(msg, options: list[dict]) -> action str（阻塞等待）
        # options 形如 [{"action": "retry", "label": "重试"}, ...]；未设置时返回 None
        self.on_confirm = None

    def _report(self, msg: str, **fields):
        """上报阶段进度（未设置回调时为空操作）。
        可附带结构化字段（stage/stage_total/phase 等），供 UI 展示阶段指示。"""
        cb = self.on_progress
        if cb:
            try:
                cb(msg, **fields)
            except TypeError:
                try:
                    cb(msg)
                except Exception:
                    pass
            except Exception:
                pass

    def _confirm(self, msg: str, options: list) -> Optional[str]:
        """请求用户决策（未设置回调时返回 None，调用方走默认行为）"""
        cb = self.on_confirm
        if not cb:
            return None
        try:
            return cb(msg, options)
        except Exception:
            return None
    
    def generate_world_setting(self, user_prompt: str, max_tokens: int = 2000) -> str:
        """第一步：生成世界观设定"""
        prompt = f"""你是一个专业的小说创作者，请根据用户的需求，创作一部小说的世界观设定。
用户需求：{user_prompt}

请包含以下内容：
1. 故事发生的时代背景
2. 主要的地理/世界构架
3. 核心的力量体系/规则（如果是玄幻/科幻）
4. 主要的势力格局
5. 写作视角或风格的额外补充

请用清晰的结构输出："""

        result = self.api.generate(prompt, step="world", temperature=0.8, max_tokens=max_tokens)
        # AI拒绝的内容不入库，避免污染本地存储
        if is_ai_refusal(result):
            logger.warning("世界观生成被AI拒绝（内容安全审查），结果不入库")
            return result
        # 保存到本地存储
        self.vs.add_section("setting", "world_setting", result)
        self.novel_info["world_setting"] = result
        # 按 #/## 小节拆分存档，供正文注入按需取档（稳定/势力/剧情分档，避免整本走向前移）
        self._save_world_sections()
        return result
    
    def generate_characters(self, user_prompt: str, num_main: int = 3, num_support: int = 5, max_tokens: int = 2000) -> Dict:
        """第二步：生成人物设定"""
        # 直取世界观（与 _build_chapter_context 同一做法）。
        # 旧实现用 bigram 检索"世界观设定"，正文里大概率不含这几个字 → 静默退化为无世界观生成。
        world_context = self.novel_info.get("world_setting", "")
        if not world_context:
            # 兜底：workflow 独立实例化（测试/脚本）时从存储直取
            world_context = self.vs.get_section("setting", "world_setting") or ""
            if world_context:
                self.novel_info["world_setting"] = world_context
        if not world_context:
            logger.warning("未找到世界观设定，人物生成将在无世界观上下文下进行")
        
        prompt = f"""请根据以下世界观设定，为这部小说设计主要人物。
世界观：
{world_context}

用户需求：{user_prompt}

请设计：
- {num_main}个主要角色（主角，主要反派）：包含姓名、外貌、性格、背景、目标
- {num_support}个重要配角：简要介绍

【输出格式要求】每个角色一个 [角色]...[/角色] 块，严格按以下格式输出，不要输出其他内容：

{cc.TAGGED_FORMAT}"""

        # 结构化生成：要求按固定标签格式输出，解析失败自动重试 1 次，
        # 再失败降级为自由文本模式（旧行为），保证任何情况下人物设定都能产出。
        result = self.api.generate(prompt, step="characters", temperature=0.7, max_tokens=max_tokens)
        if is_ai_refusal(result):
            logger.warning("人物设定生成被AI拒绝（内容安全审查），结果不入库")
            return {"characters": result}
        cards, ok = cc.parse_character_cards(result)
        if not ok:
            logger.warning("角色卡标签格式解析失败，重试 1 次")
            retry_prompt = prompt + "\n\n【注意】你上次的输出没有使用 [角色]...[/角色] 标签格式，请务必严格按格式重新输出。"
            retry_result = self.api.generate(retry_prompt, step="characters", temperature=0.5, max_tokens=max_tokens)
            if not is_ai_refusal(retry_result):
                cards2, ok2 = cc.parse_character_cards(retry_result)
                if ok2:
                    result, cards, ok = retry_result, cards2, True
                else:
                    result = retry_result  # 降级时也用重试结果（通常更接近要求格式）
        if ok:
            # 结构化模式：角色卡 JSON + 自由文本渲染双写（自由文本保留作备份与降级）
            self.vs.add_section("character_cards", cc.CARDS_SECTION_TITLE, cc.cards_to_json(cards))
            free_text = cc.cards_to_text(cards)
            self.vs.add_section("character", "all_characters", free_text)
            self.novel_info["characters"] = free_text
            self.novel_info["character_cards"] = cards
            logger.info(f"角色卡结构化生成成功: {len(cards)} 张卡")
            return {"characters": free_text, "cards": cards}
        # 降级：自由文本模式（现状行为）
        logger.warning("角色卡解析重试仍失败，降级为自由文本模式")
        self.vs.add_section("character", "all_characters", result)
        self.novel_info["characters"] = result
        return {"characters": result}

    def migrate_characters_to_cards(self, max_tokens: int = 3000) -> Optional[Dict]:
        """旧小说迁移（TODO 2.2）：把既有自由文本人物设定一次性解析为角色卡。

        只返回预览（不直接入库），由 UI 展示给用户确认后再保存，避免 AI 丢信息。
        解析失败返回 None（调用方保持自由文本模式）。
        """
        free_text = self.novel_info.get("characters", "") or \
            self.vs.get_section("character", "all_characters") or ""
        if not free_text.strip():
            return None
        prompt = f"""你是小说设定整理助手。请把下面的人物设定自由文本整理为结构化角色卡，不要新增或丢失人物信息。

【原有人物设定】
{free_text[:6000]}

【输出格式要求】每个角色一个 [角色]...[/角色] 块，严格按以下格式输出：

{cc.TAGGED_FORMAT}

说明：原文没有登场/退场章节信息时，登场章节填 1，退场章节留空。"""
        result = self.api.generate(prompt, step="characters", temperature=0.3, max_tokens=max_tokens)
        if is_ai_refusal(result):
            return None
        cards, ok = cc.parse_character_cards(result)
        if not ok:
            logger.warning("旧人物设定迁移解析失败，保持自由文本模式")
            return None
        return {"cards": cards, "rendered": cc.cards_to_text(cards)}

    def save_character_cards(self, cards: List[Dict]):
        """保存角色卡（UI 编辑器/迁移确认共用）：JSON + 自由文本渲染双写"""
        cards = [cc.normalize_card(c) for c in cards if isinstance(c, dict) and c.get("name")]
        self.vs.add_section("character_cards", cc.CARDS_SECTION_TITLE, cc.cards_to_json(cards))
        free_text = cc.cards_to_text(cards)
        self.vs.add_section("character", "all_characters", free_text)
        self.novel_info["characters"] = free_text
        self.novel_info["character_cards"] = cards
        return cards

    def load_character_cards(self) -> List[Dict]:
        """读取角色卡（优先 novel_info 缓存，其次存储），无则返回 []"""
        cards = self.novel_info.get("character_cards")
        if cards:
            return cards
        raw = self.vs.get_section("character_cards", cc.CARDS_SECTION_TITLE)
        cards = cc.cards_from_json(raw) if raw else []
        if cards:
            self.novel_info["character_cards"] = cards
        return cards

    # ---------- 实体注册表 / 实体时间锁（信息按叙事时点解锁） ----------

    def _build_entity_registry(self) -> Dict[str, int]:
        """构建实体注册表：{名字/别名/代号: 登场章节}

        用于实体时间锁：注入上下文前，把尚未登场角色的名字/代号遮蔽掉，
        防止世界观/检索片段/自由文本里提前点名后续才该登场的角色（防信息前移）。
        角色卡是唯一带登场章信息的来源；自由文本人物无登场信息，不参与遮蔽。
        """
        registry = {}
        for c in self.load_character_cards():
            appear = int(c.get("appearance_chapter") or 1)
            name = str(c.get("name", "")).strip()
            if name and len(name) >= 2:
                registry[name] = appear
                # 兼容旧模板：姓名里直接带代号（如「林晚（代号"夜莺"）」），把代号也注册
                m = re.search(r'[（(]\s*(?:代号|外号|别名)[：:"“”]*([^）)」』]+)', name)
                if m:
                    code = m.group(1).strip().strip('"”\'’')
                    if len(code) >= 2:
                        registry[code] = appear
            alias = str(c.get("alias", "") or "").strip()
            if alias:
                for a in re.split(r"[、,，/／\s]+", alias):
                    if len(a.strip()) >= 2:
                        registry[a.strip()] = appear
        return registry

    def _lock_entities(self, text: str, chapter_num: int) -> str:
        """实体时间锁：把文本中尚未登场角色的名字/代号替换为「某个神秘角色」。

        只替换登场章晚于当前章的实体；已登场/无登场信息的实体原样保留。
        """
        if not text:
            return text
        locked_names = []
        for name, appear in self._build_entity_registry().items():
            if appear > chapter_num and name in text:
                text = text.replace(name, "某个神秘角色")
                locked_names.append(f"{name}(第{appear}章登场)")
        if locked_names:
            logger.info(f"实体时间锁[{chapter_num}章]: 遮蔽未登场实体 {locked_names}")
        return text

    # ---------- 世界观按需注入（稳定/势力/结局 三档，随阶段解锁） ----------

    _WORLD_STABLE_KEYWORDS = ("时代", "背景", "地理", "构架", "规则", "体系", "文明", "历史", "文化", "社会", "经济", "职业", "世界", "设定", "环境")
    _WORLD_FACTION_KEYWORDS = ("势力", "组织", "格局", "阵营", "门派", "宗门", "家族", "帝国", "联盟", "势力格局")
    _WORLD_STYLE_KEYWORDS = ("视角", "风格", "情感", "脉络", "亮点", "总结", "结局", "走向", "主线", "暂定名")

    @staticmethod
    def _classify_world_header(header: str) -> str:
        """按小节标题分类世界观段落：stable（稳定背景）/ faction（势力格局）/ style（剧情走向）"""
        h = header.strip().lstrip("#").strip()
        if any(k in h for k in FullNovelWorkflow._WORLD_FACTION_KEYWORDS):
            return "faction"
        if any(k in h for k in FullNovelWorkflow._WORLD_STYLE_KEYWORDS):
            return "style"
        return "stable"

    def _split_world_sections(self, text: str) -> List[Dict]:
        """把世界观文本按 #/## 小节拆分，返回 [{title, content, kind}]。

        无小节标题的文本整体视为一个 stable 段（保证任何世界观都能注入）。
        """
        if not text:
            return []
        lines = text.split("\n")
        sections = []
        cur_title, cur_lines, cur_kind = None, [], "stable"
        def flush():
            nonlocal cur_title, cur_lines
            if cur_lines:
                sections.append({"title": cur_title or "", "content": "\n".join(cur_lines).strip(),
                                 "kind": cur_kind})
        for line in lines:
            m = re.match(r"^(#{1,4})\s*(.+)$", line.strip())
            if m and len(line.strip()) <= 60:
                flush()
                cur_title = line.strip()
                cur_kind = self._classify_world_header(line)
                cur_lines = []
            else:
                cur_lines.append(line)
        flush()
        if not sections and text.strip():
            sections.append({"title": "", "content": text.strip(), "kind": "stable"})
        return sections

    def _save_world_sections(self):
        """生成/保存世界观后，把小节拆分结果存 extra（供注入时按需取档）"""
        text = self.novel_info.get("world_setting", "") or ""
        sections = self._split_world_sections(text)
        if sections:
            self.vs.save_extra_data("world_sections", sections)

    def _stable_world_text(self, level: str = "stable", chapter_num: int = 0,
                           max_chars: int = 4000) -> str:
        """按档取世界观注入文本：
        - stable：只保留稳定背景段（时代/地理/规则）
        - faction：稳定背景 + 势力格局段
        - full：全部
        返回前应用实体时间锁（防世界观点名未登场角色）。
        """
        text = self.novel_info.get("world_setting", "") or \
            self.vs.get_section("setting", "world_setting") or ""
        if not text:
            return ""
        sections = self.vs.load_extra_data("world_sections", None) or self._split_world_sections(text)
        allowed = {"stable", "faction"} if level == "faction" else {"stable"}
        if level == "full":
            parts = [s["content"] for s in sections]
        else:
            parts = [s["content"] for s in sections if s["kind"] in allowed]
        result = "\n\n".join(parts).strip()
        if not result:
            result = text[:max_chars]
        if chapter_num > 0:
            result = self._lock_entities(result, chapter_num)
        if len(result) > max_chars:
            result = result[:max_chars] + "..."
        return result

    def _character_section(self, limit: int = 4000, start_ch: Optional[int] = None,
                           end_ch: Optional[int] = None) -> str:
        """构建人物设定注入文本：优先从结构化角色卡生成（可按章节/卷区间过滤），
        无卡片时回退到自由文本 characters。返回截断后的文本（空字符串表示无人物设定）。"""
        text = ""
        cards = self.load_character_cards()
        if cards:
            if start_ch is not None and end_ch is not None:
                cards = cc.filter_cards_for_range(cards, start_ch, end_ch)
            text = cc.cards_to_text(cards)
        else:
            text = self.novel_info.get("characters", "") or ""
        text = (text or "").strip()
        if text and limit and len(text) > limit:
            text = text[:limit] + "..."
        return text

    # ---------- 外部章节导入 / 手写章节（TODO 1.2 / 1.3） ----------

    def import_chapter(self, chapter_num: int, title: str, content: str):
        """导入外部章节（或保存手写章节正文）入库。

        - 同章节号覆盖（UI 层负责冲突确认）；
        - 入库后从该章起做记忆失效处理（旧 delta 撤销，后续章 delta 标记 stale），
          调用方可视情况再调 sync_memory(regen=True) 只重算缺失章补齐。
        """
        title = (title or "").strip() or f"第{chapter_num}章"
        content = (content or "").strip()
        self.vs.add_section("chapter", f"chapter_{chapter_num}", f"第{chapter_num}章 {title}\n{content}")
        chapters = self.novel_info.setdefault("chapters", {})
        chapters[str(chapter_num)] = {"title": title, "content": content}
        self.invalidate_memory_from(chapter_num)
        logger.info(f"章节已导入/保存: 第{chapter_num}章「{title}」({len(content)}字)")
        return {"chapter_num": chapter_num, "title": title, "length": len(content)}

    def create_blank_chapter(self, chapter_num: int, title: str = ""):
        """新建空白章节占位（手写新章入口）：content 为空，等用户手写或"填充空白章"。
        批量生成默认跳过空章（尊重手写意图）。"""
        title = (title or "").strip() or f"第{chapter_num}章"
        self.vs.add_section("chapter", f"chapter_{chapter_num}", f"第{chapter_num}章 {title}\n")
        chapters = self.novel_info.setdefault("chapters", {})
        chapters[str(chapter_num)] = {"title": title, "content": ""}
        logger.info(f"已创建空白章节: 第{chapter_num}章「{title}」")
        return {"chapter_num": chapter_num, "title": title}
    
    def generate_outline(self, user_prompt: str, total_chapters: int = 50, words_per_chapter: int = 2000, max_tokens: int = 4000) -> str:
        """第三步：生成总体大纲（大章节量时自动分卷两阶段生成）"""
        # 获取已有上下文：人物设定直接读角色卡（结构化），无卡时回退自由文本
        world_setting = self.novel_info.get("world_setting", "")
        characters = self._character_section(limit=None)  # 大纲需完整角色卡，不设上限
        
        # 章节数较多时（>60），采用两阶段生成：先卷级大纲，再逐卷补全章节
        if total_chapters > 60:
            result = self._generate_outline_two_stage(
                user_prompt, total_chapters, words_per_chapter, max_tokens,
                world_setting, characters
            )
        else:
            result = self._generate_outline_single(
                user_prompt, total_chapters, words_per_chapter, max_tokens,
                world_setting, characters
            )
        
        if is_ai_refusal(result):
            logger.warning("大纲生成被AI拒绝（内容安全审查），结果不入库")
            return result
        self.vs.add_section("outline", "full_outline", result)
        self.novel_info["outline"] = result
        return result
    
    def _generate_outline_single(self, user_prompt, total_chapters, words_per_chapter, max_tokens, world_setting, characters):
        """单次生成大纲（章节数较少时使用）"""
        prompt = f"""请根据以下信息，创作这部小说的分卷/分章大纲。
总共规划 {total_chapters} 章，每章大约 {words_per_chapter} 字。

世界观设定：
{world_setting}

人物设定：
{characters}

用户需求：{user_prompt}

请按照结构输出：
1. 故事主线
2. 分卷规划
3. 每章简要内容，逐章条目格式：第 X 章：章节标题 —— 一句话概括（约N字）
   其中 N 为本章目标字数，围绕 {words_per_chapter} 字按剧情节奏浮动（铺垫章可略短，高潮章可略长）。

大纲要起承转合，有节奏起伏。

【重要】你必须写出全部 {total_chapters} 章的简要内容，不能中途停止或省略。
如果内容较长，请持续输出直到写完所有章节，绝不要用"以此类推"或省略号代替。"""

        # max_tokens 按章数线性放大（每章约 80 tokens），上限取 API 允许值
        max_tokens = min(max(max_tokens, total_chapters * 80 + 500),
                         getattr(self.api, "MAX_TOKENS_LIMIT", max_tokens))
        self._report("正在生成小说大纲…", stage=1, stage_total=1, phase="requesting")
        return self.api.generate(prompt, step="outline", temperature=0.7, max_tokens=max_tokens,
                                 stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
    
    @staticmethod
    def _plan_volumes(total_chapters: int) -> list:
        """程序预分卷：按总章数计算各卷起止章号（全局连续，第一卷从 1 开始，
        最后一卷 end 强制对齐 total_chapters）。
        返回 [{"index": 1, "start": 1, "end": 40}, ...]
        """
        n_volumes = max(2, min(10, (total_chapters + 39) // 40))
        base = total_chapters // n_volumes
        rem = total_chapters % n_volumes
        volumes, start = [], 1
        for i in range(n_volumes):
            count = base + (1 if i < rem else 0)
            volumes.append({"index": i + 1, "start": start, "end": start + count - 1})
            start += count
        volumes[-1]["end"] = total_chapters  # 强制对齐，消除取整漂移
        return volumes

    def _generate_outline_two_stage(self, user_prompt, total_chapters, words_per_chapter, max_tokens, world_setting, characters):
        """两阶段生成大纲（章节数较多时使用）
        第一阶段：生成卷级大纲（故事主线 + 分卷规划 + 每卷概要）
        第二阶段：逐卷补全每章简要内容

        卷划分由程序预计算（_plan_volumes）并直接写进 prompt，模型只填卷名和剧情，
        从根本上避免"各卷从第 1 章重排编号"（Bug A）和"缩放后卷声明与逐章矛盾"（Bug B）。
        """
        plan = self._plan_volumes(total_chapters)
        vol_table = "\n".join(
            f"第{v['index']}卷：第{v['start']}-{v['end']}章" for v in plan
        )

        # ---- 第一阶段：生成卷级大纲（卷数与起止章号已由程序给定） ----
        logger.info(f"大纲两阶段生成 → 第一阶段：卷级大纲（共{total_chapters}章，预分{len(plan)}卷）")
        stage1_prompt = f"""请根据以下信息，创作这部小说的分卷大纲。
总共规划 {total_chapters} 章，每章大约 {words_per_chapter} 字。

世界观设定：
{world_setting}

人物设定：
{characters}

用户需求：{user_prompt}

本书按以下卷划分（卷数和章节范围已确定，不要改动）：
{vol_table}

请严格按照以下格式输出（不要使用其他格式）：

故事主线：200字以内概括整体故事走向

[卷]
卷名：第一卷的名称
章节：第{plan[0]['start']}-{plan[0]['end']}章
剧情：该卷的核心剧情和冲突（100-200字）
[/卷]

（按此格式为上述每一卷各输出一个 [卷]...[/卷] 块，章节范围必须照抄上方给定值）

【重要】
- 只需要写到"卷"的级别，不需要写出每章内容
- 必须用 [卷]...[/卷] 标签包裹每一卷的信息
- 必须输出全部 {len(plan)} 卷
- 大纲要起承转合，有节奏起伏"""

        self._report("大纲第一阶段：正在生成卷级规划…", stage=1, stage_total=2, phase="requesting")
        stage1_result = self.api.generate(stage1_prompt, step="outline", temperature=0.7, max_tokens=max_tokens,
                                          stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
        logger.info(f"大纲第一阶段完成，卷级大纲长度: {len(stage1_result)} 字符")
        logger.debug(f"卷级大纲原文：\n{stage1_result}")

        # ---- 第二阶段：逐卷补全章节大纲 ----
        # 从卷级大纲中解析各卷信息；解析失败/范围异常时回退到程序预分卷
        volumes = self._parse_volumes(stage1_result, total_chapters)
        if not volumes:
            logger.warning("卷级大纲解析失败，使用程序预分卷的卷名兜底")
            cn_nums = "一二三四五六七八九十"
            volumes = [{
                "name": f"第{cn_nums[v['index'] - 1] if v['index'] <= len(cn_nums) else v['index']}卷",
                "chapters": v["end"] - v["start"] + 1,
                "plot": "",
                "start_chapter": v["start"],
                "end_chapter": v["end"],
            } for v in plan]

        logger.info(f"共 {len(volumes)} 卷，惰性生成：本次只补全第一卷逐章概要，后续卷按需生成（generate_volume_chapters）")

        # 持久化卷计划（章节生成遇到无细纲的卷时按需自动生成，UI 也可手动提前触发）
        volume_plan = [{
            "index": i + 1, "name": v["name"],
            "start": v["start_chapter"], "end": v["end_chapter"],
            "plot": v["plot"], "chapters_done": False,
        } for i, v in enumerate(volumes)]

        # 先把卷级大纲作为基础
        all_parts = [stage1_result, f"\n\n---\n\n{VOLUME_CHAPTER_MARKER}\n"]

        # 只生成第一卷的逐章概要；卷间盲写问题由 generate_volume_chapters 解决
        # （后续卷生成时注入前一卷逐章概要 + 滚动摘要）
        vol_result = self._generate_single_volume_chapters(
            volumes[0], stage1_result, max_tokens, words_per_chapter=words_per_chapter)
        if vol_result:
            all_parts.append(f"### {volumes[0]['name']}\n{vol_result}\n\n")
            volume_plan[0]["chapters_done"] = True

        final_result = "".join(all_parts)
        self.vs.save_extra_data("volume_plan", volume_plan)
        self.novel_info["volume_plan"] = volume_plan
        logger.info(f"大纲两阶段生成完成（第一卷细纲已出，共{len(volumes)}卷待惰性补全），总长度: {len(final_result)} 字符")
        return final_result

    def _generate_single_volume_chapters(self, vol: dict, stage1_result: str, max_tokens: int,
                                         prev_volume_detail: str = "",
                                         words_per_chapter: int = 0) -> Optional[str]:
        """为单卷生成逐章概要（两阶段第二阶段 + 惰性补全共用）。

        prev_volume_detail: 前一卷的逐章概要（惰性补全时注入，解决卷间盲写：
        写第 N 卷细纲时能看到第 N-1 卷的章节拆分和章末钩子，节奏/伏笔才能衔接）。
        words_per_chapter: 每章基准目标字数（0 时调用方需自行解析；prompt 中作为「约N字」的浮动基准，
        避免 AI 凭感觉乱填与全局设定严重脱节的字数）。
        失败重试 1 次；返回 None 表示彻底失败（调用方记录缺失，不影响其他卷）。
        生成后自动校验全局章号，模型从 1 重排时自动偏移改写。
        """
        if not words_per_chapter or words_per_chapter <= 0:
            words_per_chapter = self._resolve_global_words_per_chapter()
        vol_name = vol["name"]
        start_ch, end_ch = vol["start_chapter"], vol["end_chapter"]
        vol_chapters = end_ch - start_ch + 1
        # 本卷登场角色：直接从角色卡按卷区间过滤注入（无卡时回退自由文本）。
        # 登场章列表前置（「名字（第X章登场）」，限长截断也不丢登场信息）+ 完整角色卡详情供参考
        characters_brief = ""
        cards = self.load_character_cards()
        if cards:
            vol_cards = cc.filter_cards_for_range(cards, start_ch, end_ch)
            if vol_cards:
                appear_list = "\n".join(
                    f"- {c.get('name', '')}（第{c.get('appearance_chapter', 1)}章登场）" for c in vol_cards)
                detail = cc.cards_to_text(vol_cards)[:2500]
                characters_brief = f"【本卷登场角色及登场章】\n{appear_list}\n\n【角色详情】\n{detail}"
        if not characters_brief:
            characters_brief = self._character_section(limit=2500, start_ch=start_ch, end_ch=end_ch)
        characters_block = f"""
【人物设定（本卷登场角色及登场章）】
{characters_brief}
""" if characters_brief else ""
        # max_tokens 按卷章数线性放大（每章约 80 tokens），上限取 API 允许值
        vol_max_tokens = min(
            max(max_tokens, vol_chapters * 80 + 500),
            getattr(self.api, "MAX_TOKENS_LIMIT", max_tokens)
        )
        prev_block = f"""
【前一卷逐章概要（用于衔接节奏与伏笔，不要重复其桥段）】
{prev_volume_detail[-3000:]}
""" if prev_volume_detail else ""

        stage2_prompt = f"""请为以下这卷小说补全每章的简要内容。

【故事主线与分卷规划】
{stage1_result}

【当前需要补全的卷】
卷名：{vol_name}
本卷覆盖全书第 {start_ch}-{end_ch} 章（共 {vol_chapters} 章）
该卷核心剧情：{vol.get('plot', '')}
{prev_block}
{characters_block}请为这 {vol_chapters} 章逐一写出简要内容（每章一句话概括）。

格式要求：
第 X 章：章节标题 —— 一句话概括（约N字）
（N 为本章目标字数，围绕每章基准 {words_per_chapter} 字按剧情节奏浮动：铺垫章可略短（不少于基准的一半），高潮章可略长）

【重要】
- 章节编号是全书全局编号：必须从第 {start_ch} 章开始，连续编号到第 {end_ch} 章，不得从 1 重新编号
- **角色登场章对齐**：上面「人物设定」里每个角色都标注了「第X章登场」，该角色必须在其标注的登场章**首次出场**（登场章之前的章节概要中不得安排其出场）；如你的剧情安排与标注冲突，请以标注的登场章为准调整概要，不要提前安排出场
- 你必须写完这 {vol_chapters} 章的全部内容，不能中途停止或省略。"""

        vol_result = None
        for attempt in range(2):
            try:
                self._report(f"正在补全卷「{vol_name}」（第{start_ch}-{end_ch}章）的章节概要…",
                             stage=2, stage_total=2, phase="requesting")
                vol_result = self.api.generate(stage2_prompt, step="outline", temperature=0.7, max_tokens=vol_max_tokens,
                                               stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
                break
            except Exception as e:
                logger.warning(f"卷「{vol_name}」第 {attempt + 1} 次生成失败: {e}")
        if vol_result is None:
            logger.error(f"卷「{vol_name}」（第{start_ch}-{end_ch}章）细纲生成失败，该卷暂缺")
            return None
        return self._renumber_volume_outline(vol_result, start_ch, end_ch)

    def _stage1_from_outline(self, outline: str) -> str:
        """从已存大纲文本中截取卷级部分（逐章概要标记之前的原文）"""
        marker = self._find_volume_chapter_marker(outline)
        return outline[:marker].rstrip() if marker >= 0 else outline

    @staticmethod
    def _find_volume_chapter_marker(outline: str) -> int:
        """返回大纲中「逐章概要」小节标记的起始下标；兼容旧数据里的「逐章大纲」标记。

        返回 -1 表示大纲中尚无该小节。
        """
        idx_new = outline.find(VOLUME_CHAPTER_MARKER)
        idx_old = outline.find(VOLUME_CHAPTER_MARKER_OLD)
        candidates = [i for i in (idx_new, idx_old) if i >= 0]
        return min(candidates) if candidates else -1

    def get_chapter_summaries(self) -> str:
        """取大纲中的逐章概要段（「## 逐章概要」标记**之后**的文本）；无标记返回空串。

        供前端「编辑逐章概要」独立编辑——数据仍存于大纲 section 内，卷级部分不受影响。
        """
        outline = self.novel_info.get("outline", "") or \
            self.vs.get_section("outline", "full_outline") or ""
        idx = self._find_volume_chapter_marker(outline)
        if idx < 0:
            return ""
        # 跳过标记行本身（可能是旧标记「## 逐章大纲」）
        line_end = outline.find("\n", idx)
        return outline[line_end + 1:].rstrip() if line_end >= 0 else ""

    def update_chapter_summaries(self, content: str) -> str:
        """整体替换大纲中的逐章概要段（卷级部分逐字保留），返回更新后的大纲全文。

        大纲尚无逐章概要标记时，在末尾追加标记与内容。
        """
        outline = self.novel_info.get("outline", "") or \
            self.vs.get_section("outline", "full_outline") or ""
        idx = self._find_volume_chapter_marker(outline)
        content = content.strip()
        # 防御：内容里误带标记行/多余分隔线时剥掉
        for marker in (VOLUME_CHAPTER_MARKER, VOLUME_CHAPTER_MARKER_OLD):
            if content.startswith(marker):
                content = content[len(marker):].lstrip("\n")
        base = outline[:idx].rstrip() if idx >= 0 else outline.rstrip()
        base = re.sub(r"(?:\n---)+\s*$", "", base)  # 去掉已有分隔线（含历史叠加），统一由本函数补
        if idx >= 0:
            new_outline = base + f"\n\n---\n\n{VOLUME_CHAPTER_MARKER}\n{content}\n"
        else:
            new_outline = base + f"\n\n---\n\n{VOLUME_CHAPTER_MARKER}\n{content}\n"
        self.vs.add_section("outline", "full_outline", new_outline)
        self.novel_info["outline"] = new_outline
        logger.info(f"逐章概要已更新: {len(content)}字，卷级部分保留 {len(base)}字")
        return new_outline

    def generate_volume_chapters(self, volume_index: int, max_tokens: int = 4000, force: bool = False) -> Optional[str]:
        """惰性生成/重新生成指定卷的逐章概要（TODO 3.2.0）。

        prompt 注入：卷级大纲（按卷切分，避免跨卷泄漏）+ 前一卷逐章概要 + 滚动摘要（解决卷间盲写）。
        细纲写入大纲文本对应卷之下；全局章号由 volume_plan 保证。
        已生成过的卷默认跳过（返回 None）；force=True 时强制重新生成并整体替换该卷旧细纲。
        返回写入的大纲文本片段或 None。
        """
        plan = self.novel_info.get("volume_plan") or self.vs.load_extra_data("volume_plan", []) or []
        vol = next((v for v in plan if v.get("index") == volume_index), None)
        if not vol:
            logger.warning(f"volume_plan 中无第 {volume_index} 卷")
            return None
        if vol.get("chapters_done") and not force:
            logger.info(f"第 {volume_index} 卷细纲已存在，跳过")
            return None
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        if not outline:
            logger.warning("无大纲文本，无法补全卷细纲")
            return None

        stage1_result = self._stage1_from_outline(outline)
        # 卷级总述按当前卷切分：只保留当前卷+前一卷概要，避免把后续卷内容喂给本卷细纲
        stage1_result = self._slice_overview_by_volume(stage1_result, vol["start"])
        # 前一卷逐章概要 + 滚动摘要，解决卷间盲写
        prev_detail = ""
        if volume_index > 1:
            prev = next((v for v in plan if v.get("index") == volume_index - 1), None)
            if prev:
                prev_detail = self._extract_relevant_outline(
                    outline, prev["start"], capture_range=max(0, prev["end"] - prev["start"]),
                    spoiler_level="none")
        rolling = self.vs.load_extra_data("rolling_summary", "") or ""
        if rolling:
            stage1_result += f"\n\n【全书剧情摘要（已生成章节）】\n{rolling[:1200]}"

        vol_result = self._generate_single_volume_chapters(
            {"name": vol["name"], "plot": vol.get("plot", ""),
             "start_chapter": vol["start"], "end_chapter": vol["end"]},
            stage1_result, max_tokens, prev_detail,
            words_per_chapter=self._resolve_global_words_per_chapter())
        if vol_result is None:
            return None

        new_outline = self._upsert_volume_detail(outline, vol["name"], vol_result)
        self.vs.add_section("outline", "full_outline", new_outline)
        self.novel_info["outline"] = new_outline
        vol["chapters_done"] = True
        self.vs.save_extra_data("volume_plan", plan)
        self.novel_info["volume_plan"] = plan
        logger.info(f"第 {volume_index} 卷细纲已{'重新生成并替换' if force else '生成并写入'}入库")
        return vol_result

    def _upsert_volume_detail(self, outline: str, vol_name: str, new_text: str) -> str:
        """把卷逐章概要写入大纲：按卷号在「逐章概要」段内定位同卷块（块头带加粗/范围后缀也能命中），
        整体替换并顺带删除同卷其他重复块；找不到则追加到段末。返回更新后的大纲全文。
        """
        vol_num = self._volume_num_from_name(vol_name)
        prefix, parts = self._split_summary_blocks(outline)
        matched = [i for i in range(1, len(parts), 2)
                   if self._header_vol_num(parts[i]) == vol_num]
        if matched:
            first = matched[0]
            keep = set(range(1, len(parts), 2)) - set(matched[1:])
            return prefix + self._rebuild_scope(parts, keep, replace_body_of=first, new_body=new_text.rstrip())
        if outline.strip():
            return outline.rstrip() + f"\n\n### {vol_name}\n{new_text.rstrip()}\n"
        return f"### {vol_name}\n{new_text.rstrip()}\n"

    def ensure_outline_for_chapter(self, chapter_num: int, max_tokens: int = 4000):
        """章节生成前置保障：若该章所属卷尚无逐章概要，自动惰性生成（TODO 3.2.0 触发时机）"""
        plan = self.novel_info.get("volume_plan") or self.vs.load_extra_data("volume_plan", []) or []
        if not plan:
            return
        vol = next((v for v in plan if v.get("start", 0) <= chapter_num <= v.get("end", 0)), None)
        if vol and not vol.get("chapters_done"):
            logger.info(f"第{chapter_num}章所属第{vol['index']}卷无细纲，自动惰性生成")
            self._report(f"第{chapter_num}章所属卷「{vol['name']}」尚无细纲，正在自动生成…")
            self.generate_volume_chapters(vol["index"], max_tokens=max_tokens)

    # ---------- 大纲增量扩展 / 局部改写（TODO 3.1 / 4.1） ----------

    def extend_outline(self, additional_chapters: int, max_tokens: int = 4000) -> str:
        """增量扩展大纲（TODO 3.1）：AI 只输出新增章节条目，**追加**到现有大纲末尾，
        已有部分逐字保留（区别于 generate_outline 的同 ID 覆盖）。

        prompt 注入：已有大纲全文 + 滚动摘要 + 伏笔台账 + 要新增的章数。
        返回追加的文本片段；失败/拒答返回空字符串。
        """
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        if not outline.strip():
            raise ValueError("当前还没有大纲，请先生成大纲")
        cur_total = self._estimate_total_chapters()
        new_start, new_end = cur_total + 1, cur_total + additional_chapters

        rolling = self.vs.load_extra_data("rolling_summary", "") or ""
        pending = self._pending_foreshadowing()
        fs_text = "\n".join(f"- {f['item']}（第{f.get('planted_chapter', '?')}章埋设）"
                            for f in pending[:10]) or "（无）"

        # max_tokens 按新增章数线性放大
        max_tokens = min(max(max_tokens, additional_chapters * 80 + 500),
                         getattr(self.api, "MAX_TOKENS_LIMIT", max_tokens))
        prompt = f"""请为这部小说**续写扩展大纲**：在现有大纲之后新增第 {new_start}-{new_end} 章的条目。

【已有大纲（逐字保留，不得修改或重复）】
{outline[-6000:]}

【全书剧情摘要（已生成章节）】
{rolling[:1200] or "（暂无）"}

【待回收伏笔（扩展大纲应尽量安排回收）】
{fs_text}

【输出要求】
- 只输出新增的第 {new_start}-{new_end} 章条目，每章一行
- 格式：第 X 章：章节标题 —— 一句话概括（约N字），N 为本章目标字数
- 章节编号必须从第 {new_start} 章开始连续到第 {new_end} 章
- 如需要可在条目前加一行"### 第X卷 卷名"分卷标题
- 剧情要承接已有大纲的结尾走向，起承转合，不要重复已有桥段"""

        self._report(f"正在扩展大纲：新增第 {new_start}-{new_end} 章…")
        result = self.api.generate(prompt, step="outline", temperature=0.7, max_tokens=max_tokens)
        if is_ai_refusal(result):
            logger.warning("大纲扩展被AI拒绝，不入库")
            return ""
        # 校验扩展段编号，模型从 1 重排时自动偏移
        result = self._renumber_volume_outline(result, new_start, new_end)
        new_outline = outline.rstrip() + "\n\n" + result.strip() + "\n"
        self.vs.add_section("outline", "full_outline", new_outline)
        self.novel_info["outline"] = new_outline
        self.vs.save_extra_data("outline_total_chapters", str(new_end))
        self.novel_info["outline_total_chapters"] = str(new_end)
        logger.info(f"大纲已扩展到第 {new_end} 章（新增 {additional_chapters} 章条目）")
        return result

    def check_appearance_in_outline(self, name: str, chapter_num: int, span: int = 2) -> Dict:
        """登场调度检查（TODO 4.1）：大纲第 N±span 章文本是否提及该角色名。
        返回 {"mentioned": bool, "excerpt": 相关大纲文本}"""
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        excerpt = self._extract_relevant_outline(outline, chapter_num, capture_range=span,
                                                 spoiler_level="none") if outline else ""
        return {"mentioned": bool(name and name in excerpt), "excerpt": excerpt}

    def rewrite_outline_range(self, start: int, end: int, instruction: str, max_tokens: int = 3000) -> str:
        """局部改写大纲（TODO 4.1）：只重写第 start-end 章的大纲条目，其余逐字保留。

        替换策略：正则定位大纲中该范围的章节行并移除，把 AI 新条目插到原位置；
        若原文中找不到任何该范围的行，则追加到大纲末尾（兜底不丢内容）。
        返回更新后的完整大纲文本。
        """
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        if not outline.strip():
            raise ValueError("当前还没有大纲")
        old_excerpt = self._extract_relevant_outline(outline, (start + end) // 2,
                                                     capture_range=(end - start) // 2 + 1,
                                                     spoiler_level="none")
        # 改写范围内的登场角色：注入「名字（第X章登场）」+ 详情，保证角色设定与角色卡一致
        cards = self.load_character_cards()
        char_block = ""
        if cards:
            range_cards = cc.filter_cards_for_range(cards, start, end)
            if range_cards:
                appear_list = "\n".join(
                    f"- {c.get('name', '')}（第{c.get('appearance_chapter', 1)}章登场）" for c in range_cards)
                detail = cc.cards_to_text(range_cards)[:2500]
                char_block = f"\n【该范围登场角色（必须与其设定一致）】\n{appear_list}\n\n【角色详情】\n{detail}\n"
        prompt = f"""请局部改写小说大纲中第 {start}-{end} 章的条目。

【改写要求】
{instruction}

【这部分章节的现有大纲】
{old_excerpt or "（大纲中未找到该范围条目，请直接创作）"}
{char_block}
【输出要求】
- 只输出改写后的第 {start}-{end} 章条目，每章一行，行与行之间不要空行
- 格式严格为：第 X 章：章节标题 —— 一句话概括（约N字），N 为本章目标字数
- **不要使用加粗（**）、列表符号（- ）、标题符号（###）等任何装饰，直接输出纯文本条目**
- 编号从第 {start} 章连续到第 {end} 章
- 只改这些章，前后章节的安排视为已定，需与之衔接
- **角色一致性**：上面【该范围登场角色】标注了登场章与设定，条目中涉及的角色必须与其身份/性格/登场章一致，不得编造与角色卡冲突的身份；改写范围覆盖某角色登场章时，该角色须在对应章节首次出场"""

        result = self.api.generate(prompt, step="outline", temperature=0.7, max_tokens=max_tokens)
        if is_ai_refusal(result):
            return outline
        result = self._normalize_chapter_lines(result)
        result = self._renumber_volume_outline(result, start, end)

        # 移除大纲中该范围的旧章节行，记录首个移除位置作为插入点
        lines = outline.split("\n")
        kept, insert_at = [], None
        chap_re = re.compile(r"^\s*(?:#{1,6}\s*|[-\*]\s+|\*{1,3})?\[?第\s*(\d+)\s*章")
        for line in lines:
            m = chap_re.match(line)
            if m and start <= int(m.group(1)) <= end:
                if insert_at is None:
                    insert_at = len(kept)
                continue
            kept.append(line)
        new_block = result.strip().split("\n")
        if insert_at is None:
            new_lines = kept + ["", f"### 大纲修订（第{start}-{end}章）"] + new_block
        else:
            new_lines = kept[:insert_at] + new_block + kept[insert_at:]
        new_outline = "\n".join(new_lines)
        self.vs.add_section("outline", "full_outline", new_outline)
        self.novel_info["outline"] = new_outline
        logger.info(f"大纲第{start}-{end}章已局部改写")
        return new_outline

    # ---------- 回溯影响分析（TODO 4.2） ----------

    def scan_impacted_chapters(self, keywords: List[str]) -> List[Dict]:
        """扫描已生成章节，找出提及任一关键词的章节（新增主角/核心设定变更的回溯分析）。
        返回 [{"chapter": n, "title": ..., "hits": {关键词: 次数}}]，按章号排序。
        只扫描，不改写——改写由用户逐章确认后走 generate_chapter(extra_instruction=...)。
        """
        keywords = [k.strip() for k in keywords if k and k.strip()]
        if not keywords:
            return []
        results = []
        chapters = self.novel_info.get("chapters", {})
        for k in sorted(chapters, key=lambda x: int(x) if str(x).isdigit() else 0):
            ch = chapters[k]
            content = (ch.get("content") or "")
            if not content.strip():
                continue
            hits = {kw: content.count(kw) for kw in keywords if content.count(kw)}
            if hits:
                results.append({"chapter": int(k), "title": ch.get("title", ""), "hits": hits})
        return results

    def preview_chapter_rewrite(self, chapter_num: int, instruction: str, max_tokens: int = 800) -> str:
        """为指定章节生成"AI 改写建议预览"（不直接改正文，供用户确认后走重写流程）"""
        ch = self.novel_info.get("chapters", {}).get(str(chapter_num), {})
        content = (ch.get("content") or "")
        if not content.strip():
            return ""
        prompt = f"""你是小说编辑。下面这一章需要融入新的设定变更，请给出具体的改写建议（不要直接改写全文）。

【新设定/变更要求】
{instruction}

【第 {chapter_num} 章「{ch.get('title', '')}」正文（节选）】
{self._chapter_excerpt(content)}

请输出：
1. 本章哪些段落/情节与新设定冲突或需要呼应
2. 具体的改写方案（逐条列出）
3. 若本章其实无需改动，直接说明"无需改动"及理由"""
        result = self.api.generate(prompt, step="consistency", temperature=0.5, max_tokens=max_tokens)
        return "" if is_ai_refusal(result) else result

    @staticmethod
    def _normalize_chapter_lines(text: str) -> str:
        """归一化逐章条目格式：剥离加粗/列表/标题等装饰，合并空行与换行续行，
        保证每章一条「第 X 章：标题 —— 概要（约N字）」纯文本行。

        非章节行（AI 多输出的引言/卷标题）若前面已有章节行则并作续行，否则丢弃。
        """
        if not text:
            return text
        out = []
        for l in text.split("\n"):
            l = l.strip()
            if not l:
                continue
            # 剥离装饰：### / - / * / ** 等
            cleaned = re.sub(r"^#{1,6}\s*", "", l)
            cleaned = re.sub(r"^[-\*]\s+", "", cleaned)
            cleaned = re.sub(r"^\*{1,3}", "", cleaned)
            cleaned = cleaned.replace("**", "").strip()
            if not cleaned:
                continue
            if re.match(r"第\s*\d+\s*章", cleaned):
                out.append(cleaned)
            else:
                # 非章节行：并到上一行末尾（续行），无上一行则丢弃（引言/卷标题）
                if out:
                    out[-1] += cleaned
        return "\n".join(out)

    def _renumber_volume_outline(self, vol_result: str, start_ch: int, end_ch: int) -> str:
        """校验并修正单卷逐章概要的章节号：
        - 若章节行数与卷章数一致但编号从 1 开始（模型无视全局编号），按顺序改写为 start_ch..end_ch；
        - 若编号已落在 [start_ch, end_ch] 区间内，原样返回；
        - 其他异常情况记告警并原样返回（不破坏文本）。
        """
        expected = end_ch - start_ch + 1
        pattern = re.compile(r"(第\s*)(\d+)(\s*章)")
        matches = list(pattern.finditer(vol_result))
        if not matches:
            logger.warning(f"卷大纲（第{start_ch}-{end_ch}章）未检测到任何章节行")
            return vol_result
        nums = [int(m.group(2)) for m in matches]
        if nums[0] == start_ch and nums[-1] == end_ch and len(matches) == expected:
            return vol_result  # 编号正确
        if len(matches) == expected:
            # 条数正确但编号错位（典型：从 1 重排）→ 顺序改写为全局编号
            logger.warning(f"卷大纲章节号错位（{nums[0]}-{nums[-1]}），自动重编号为 {start_ch}-{end_ch}")
            pieces, last = [], 0
            for i, m in enumerate(matches):
                pieces.append(vol_result[last:m.start()])
                pieces.append(f"{m.group(1)}{start_ch + i}{m.group(3)}")
                last = m.end()
            pieces.append(vol_result[last:])
            return "".join(pieces)
        logger.warning(f"卷大纲章节条数异常: 期望{expected}条，实得{len(matches)}条（第{start_ch}-{end_ch}章）")
        return vol_result

    @staticmethod
    def _cn_to_int(s: str) -> Optional[int]:
        """中文数字/阿拉伯数字字符串转 int，支持 一~十、二十 等；失败返回 None"""
        s = str(s).strip()
        if s.isdigit():
            return int(s)
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}
        if s in digits:
            return digits[s]
        if "十" in s:
            parts = s.split("十", 1)
            tens = digits.get(parts[0], 1) if parts[0] else 1
            ones = digits.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            if (parts[0] and parts[0] not in digits) or (len(parts) > 1 and parts[1] and parts[1] not in digits):
                return None
            return tens * 10 + ones
        return None

    @classmethod
    def _normalize_volume_name(cls, name: str) -> str:
        """剥离卷名开头自带的「第N卷：」前缀（可能多重），返回纯卷名。
        例：'第一卷：第一卷：三色骑士' -> '三色骑士'
        """
        name = (name or "").strip()
        prev = None
        while prev != name:
            prev = name
            name = re.sub(r'^第\s*[一二三四五六七八九十两\d]+\s*卷\s*[：:、\s]*', '', name).strip()
        return name

    @staticmethod
    def _volume_num_from_name(name: str) -> Optional[int]:
        """从带「第N卷」前缀的完整卷名中提取卷号（用于去重/排序）"""
        m = re.match(r'^第\s*([一二三四五六七八九十两\d]+)\s*卷', (name or "").strip())
        if not m:
            return None
        return FullNovelWorkflow._cn_to_int(m.group(1))

    @staticmethod
    def _header_vol_num(header: str) -> Optional[int]:
        """从「### 卷名」块头提取卷号（忽略 markdown 加粗与（第X-Y章）范围后缀）"""
        return FullNovelWorkflow._volume_num_from_name(
            re.sub(r'^#{1,6}\s*', '', header).replace("*", "").strip())

    @staticmethod
    def _header_plain_name(header: str) -> str:
        """块头去掉 markdown（###/加粗）与（第X-Y章）范围后缀后的纯卷名"""
        s = re.sub(r'^#{1,6}\s*', '', header).replace("*", "").strip()
        s = re.sub(r'[（(]\s*第\s*[\d一二三四五六七八九十两]+\s*[-~—至]\s*[\d一二三四五六七八九十两]+\s*章?\s*[)）].*$', '', s)
        return s.strip()

    @classmethod
    def _split_summary_blocks(cls, outline: str) -> tuple:
        """把大纲切为逐章概要段（marker 之后）内的「### 卷名」块。

        返回 (prefix, parts)：prefix 为 marker 之前原文；parts 为 re.split 结果，
        偶数下标为块间文本（含 marker 前导/分隔线），奇数下标为「### 卷名」块头行。
        无 marker 时视整篇为段（兼容旧数据）。
        """
        marker_idx = cls._find_volume_chapter_marker(outline)
        scope_start = marker_idx if marker_idx >= 0 else 0
        return outline[:scope_start], re.split(r'(?m)^(###[^\n]*)$', outline[scope_start:])

    @staticmethod
    def _rebuild_scope(parts: list, keep_headers: set, replace_body_of=None, new_body="") -> str:
        """按保留块头集合重建逐章概要段：被删块的正文一并丢弃；
        replace_body_of 指定的块正文用 new_body 替换。"""
        out = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                if i in keep_headers:
                    out.append(part)
            elif i == 0 or (i - 1) in keep_headers:
                out.append(new_body if (i - 1) == replace_body_of else part)
        return "".join(out).rstrip() + "\n"

    @classmethod
    def deduplicate_summary_blocks(cls, outline: str, plan: Optional[list] = None) -> tuple:
        """清理逐章概要段内同一卷的重复「### 卷名」块：每组同卷块只保留一个，其余删除。

        保留规则：优先保留块头与 volume_plan 卷名完全一致的块；否则保留该卷最后一块。
        返回 (新大纲, 删除块数)；无重复时原样返回。
        """
        if not outline:
            return outline, 0
        prefix, parts = cls._split_summary_blocks(outline)
        groups = {}
        for i in range(1, len(parts), 2):
            num = cls._header_vol_num(parts[i])
            if num is not None:
                groups.setdefault(num, []).append(i)
        plan_names = {}
        for v in (plan or []):
            n = cls._volume_num_from_name(v.get("name", ""))
            if n is not None:
                plan_names.setdefault(n, (v.get("name") or "").strip())
        keep = set(range(1, len(parts), 2))
        removed = 0
        for num, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            chosen = None
            plan_name = plan_names.get(num)
            if plan_name:
                chosen = next((i for i in idxs if cls._header_plain_name(parts[i]) == plan_name), None)
            if chosen is None:
                chosen = idxs[-1]
            for i in idxs:
                if i != chosen:
                    keep.discard(i)
                    removed += 1
        if not removed:
            return outline, 0
        return prefix + cls._rebuild_scope(parts, keep), removed

    
    def _parse_volumes(self, volume_outline: str, total_chapters: int) -> list:
        """从卷级大纲中解析出各卷信息，返回
        [{"name", "chapters", "plot", "start_chapter", "end_chapter"}]

        优先解析 [卷]...[/卷] 结构化格式（prompt 要求的固定格式），
        如果解析失败则尝试从自由文本中提取。
        起止章号以程序预分卷为准（_adjust_volumes 顺序重排），模型文本中的
        范围声明仅作参考，避免文本与内存计数自相矛盾。
        """

        volumes = []

        # ---- 优先解析结构化格式 [卷]...[/卷] ----
        vol_blocks = re.findall(r'\[卷\](.*?)\[/卷\]', volume_outline, re.DOTALL)
        if vol_blocks:
            seen_keys = set()  # 按卷号/章节范围去重，LLM 重复输出同一卷时只保留首个
            for block in vol_blocks:
                vol_name = ""
                chapters = 0
                start_chapter, end_chapter = None, None
                plot = ""

                # 提取卷名（剥离模型自带的「第N卷：」前缀，避免拼接时重复）
                name_m = re.search(r'卷名[：:]\s*(.+)', block)
                declared_num = None
                if name_m:
                    raw_name = name_m.group(1).strip()
                    declared_num = self._volume_num_from_name(raw_name)
                    vol_name = self._normalize_volume_name(raw_name)

                # 提取章节范围（同时记录起始章号）
                ch_m = re.search(r'章节[：:]\s*(?:第)?\s*(\d+)\s*[-–—]\s*(?:第)?\s*(\d+)\s*章?', block)
                if ch_m:
                    start_chapter = int(ch_m.group(1))
                    end_chapter = int(ch_m.group(2))
                    chapters = end_chapter - start_chapter + 1

                # 去重 key：优先声明的卷号，否则用章节范围
                dedup_key = ("num", declared_num) if declared_num else (
                    ("range", start_chapter, end_chapter) if start_chapter is not None else None)
                if dedup_key is not None:
                    if dedup_key in seen_keys:
                        logger.warning(f"检测到重复卷块（{dedup_key}），已跳过")
                        continue
                    seen_keys.add(dedup_key)

                # 提取剧情
                plot_m = re.search(r'剧情[：:]\s*(.+)', block, re.DOTALL)
                if plot_m:
                    plot = plot_m.group(1).strip()

                if vol_name or chapters > 0:
                    cn_nums = "一二三四五六七八九十"
                    idx = len(volumes)
                    cn_num = cn_nums[idx] if idx < len(cn_nums) else str(idx + 1)
                    volumes.append({
                        "name": f"第{cn_num}卷：{vol_name}" if vol_name else f"第{cn_num}卷",
                        "chapters": chapters,
                        "plot": plot or vol_name,
                        "start_chapter": start_chapter,
                        "end_chapter": end_chapter,
                    })

            if volumes:
                logger.info(f"结构化格式解析成功，共 {len(volumes)} 卷")
                # 校验和调整章节范围
                return self._adjust_volumes(volumes, total_chapters)

        # ---- 兜底：从自由文本中解析 ----
        logger.info("未检测到 [卷]...[/卷] 结构化格式，尝试自由文本解析")
        seen_nums = set()  # 按规范化卷号去重（"第五卷"与"第5卷"视为同卷）
        for line in volume_outline.split("\n"):
            line_stripped = line.strip()
            vol_m = re.search(r'第([一二三四五六七八九十两\d]+)卷', line_stripped)
            if not vol_m:
                continue

            # 提取章节范围（整行搜索，同时记录起始章号）
            range_m = re.search(r'(\d+)\s*[-–—]\s*(\d+)\s*章', line_stripped)
            # 只接受"规划行"：含章节范围，或行内有卷名/剧情标记；
            # 正文/故事主线里提及"第五卷"的句子不生成卷条目
            if not range_m and not re.search(r'卷名|剧情', line_stripped):
                continue

            vol_num_raw = vol_m.group(1)
            vol_num = self._cn_to_int(vol_num_raw)
            if vol_num is not None:
                if vol_num in seen_nums:
                    continue
                seen_nums.add(vol_num)

            rest = line_stripped[vol_m.end():].strip()
            chapters = 0
            start_chapter, end_chapter = None, None
            if range_m:
                start_chapter = int(range_m.group(1))
                end_chapter = int(range_m.group(2))
                chapters = end_chapter - start_chapter + 1

            # 提取卷名：去掉章节范围和标点（rest 已不含「第N卷」，再做一次兜底剥离）
            vol_name = rest
            vol_name = re.sub(r'[（(]\s*\d+\s*[-–—]\s*\d+\s*章\s*[）)]', '', vol_name)
            vol_name = re.sub(r'第?\s*\d+\s*[-–—]\s*\d+\s*章', '', vol_name)
            vol_name = re.sub(r'^[：:，,、\s]+', '', vol_name).strip()
            vol_name = re.sub(r'[：:，,、\s]+$', '', vol_name).strip()
            vol_name = self._normalize_volume_name(vol_name)

            volumes.append({
                "name": f"第{vol_num_raw}卷：{vol_name}" if vol_name else f"第{vol_num_raw}卷",
                "chapters": chapters,
                "plot": vol_name if vol_name else f"第{vol_num_raw}卷",
                "start_chapter": start_chapter,
                "end_chapter": end_chapter,
                "_num": vol_num,
            })

        if not volumes:
            logger.warning(f"自由文本解析也失败，AI原文前500字：{volume_outline[:500]}")
            return []

        # 按卷号排序（无卷号的保持相对顺序排最后），保证卷序正确
        volumes.sort(key=lambda v: (v["_num"] is None, v["_num"] or 0))
        for v in volumes:
            v.pop("_num", None)

        return self._adjust_volumes(volumes, total_chapters)

    def _adjust_volumes(self, volumes: list, total_chapters: int) -> list:
        """校验卷划分并重排起止章号：无论模型声明的范围如何，
        一律按顺序重排为全局连续编号（第一卷从 1 开始，每卷 start = 前一卷 end + 1，
        最后一卷 end 强制对齐 total_chapters），保证程序计数唯一权威。
        """
        total_parsed = sum(v["chapters"] for v in volumes)

        # 所有卷都没有章节数，平均分配
        if total_parsed == 0:
            avg = max(1, total_chapters // len(volumes))
            for v in volumes:
                v["chapters"] = avg
            total_parsed = sum(v["chapters"] for v in volumes)

        # 总章节数与预期差距大，按比例调整
        if abs(total_parsed - total_chapters) > total_chapters * 0.3:
            scale = total_chapters / total_parsed
            for v in volumes:
                v["chapters"] = max(1, round(v["chapters"] * scale))

        # 顺序重排起止章号，最后一卷强制对齐 total_chapters
        start = 1
        for v in volumes:
            v["start_chapter"] = start
            v["end_chapter"] = start + v["chapters"] - 1
            start = v["end_chapter"] + 1
        drift = total_chapters - volumes[-1]["end_chapter"]
        if drift:
            volumes[-1]["end_chapter"] = total_chapters
            volumes[-1]["chapters"] = volumes[-1]["end_chapter"] - volumes[-1]["start_chapter"] + 1

        return volumes

    @classmethod
    def sanitize_volume_plan(cls, plan: list) -> list:
        """清理存量 volume_plan（旧数据可能有重复卷名前缀、重复卷、编号错乱）：
        - 卷名剥离多重「第N卷：」前缀后按顺序重新拼接规范前缀；
        - 按卷号（或章节范围）去重，保留首个；按卷号/起始章排序；
        - 重排 index 与 start/end 保证全局连续，最后一卷 end 对齐原最大章号。
        """
        if not plan:
            return plan
        cn_nums = "一二三四五六七八九十"
        cleaned, seen = [], set()
        for v in plan:
            if not isinstance(v, dict):
                continue
            name = str(v.get("name", "") or "")
            num = cls._volume_num_from_name(name)
            pure = cls._normalize_volume_name(name)
            key = ("num", num) if num is not None else ("range", v.get("start"), v.get("end"))
            if key in seen:
                continue
            seen.add(key)
            item = dict(v)
            item["_num"] = num
            item["_pure"] = pure
            cleaned.append(item)
        if not cleaned:
            return plan
        cleaned.sort(key=lambda v: (v["_num"] is None, v["_num"] or 0, v.get("start") or 0))
        total_end = max((v.get("end") or 0) for v in cleaned)
        out, start = [], 1
        for i, v in enumerate(cleaned):
            cn = cn_nums[i] if i < len(cn_nums) else str(i + 1)
            pure = v.pop("_pure")
            v.pop("_num", None)
            count = (v.get("end") or 0) - (v.get("start") or 0) + 1
            if count <= 0:
                count = 1
            v["index"] = i + 1
            v["name"] = f"第{cn}卷：{pure}" if pure else f"第{cn}卷"
            v["start"] = start
            v["end"] = start + count - 1
            start = v["end"] + 1
            out.append(v)
        if total_end and out[-1]["end"] != total_end:
            out[-1]["end"] = total_end
        return out

    def _estimate_total_chapters(self) -> int:
        """估算小说总章节数，用于阶段分类
        
        优先级：
        1. novel_info 中的 outline_total_chapters（大纲生成时持久化，由 app 层同步）
        2. 从 outline 文本中解析出的最大章节号
        """
        # 优先读持久化的总章数参数
        try:
            saved = int(self.novel_info.get("outline_total_chapters", 0) or 0)
            if saved > 0:
                return saved
        except (ValueError, TypeError):
            pass
        # 从大纲文本解析
        outline = self.novel_info.get("outline", "")
        if outline:
            titles = self.get_outline_chapter_titles()
            if titles:
                max_outline = max(titles.keys())
                if max_outline > 0:
                    return max_outline
        return 0
    
    def _classify_chapter_phase(self, chapter_num: int, total_chapters: int = 0) -> dict:
        """根据当前章节位置判断故事阶段，返回策略配置
        
        六阶段渐进模型（核心改进：策略随进度渐进变化，而非跳跃式切换）：
        - 开篇 (opening)：前 12% 或第 1-3 章 → range=0, rag=0, 严格隔离
        - 早期发展 (early_dev)：12%-35% → range=1, rag=0, 小范围展开
        - 中期发展 (mid_dev)：35%-60% → range=2, rag=1, 正常展开
        - 后期发展 (late_dev)：60%-78% → range=2, rag=1, 开始加速
        - 高潮 (climax)：78%-92% → range=2, rag=2, 允许跨章联动
        - 收尾 (resolution)：92%+ 或最后 2-3 章 → range=1, rag=1, 聚焦收束
        
        关键原则：
        - outline_range 渐进增长：0→1→2（不是一步到位）
        - rag_look_ahead 渐进增长：0→0→1→1→2→1
        - 叙事指令随阶段细化，每个阶段都有明确的节奏边界
        
        返回:
        {
            "phase": str,           # 阶段名称
            "outline_range": int,   # 大纲提取范围 (前后±N章)
            "rag_look_ahead": int,  # RAG 允许看后面几章 (0=不看, 1=只看下一章, >=2=正常)
            "pacing_instruction": str,  # 阶段专属叙事指令
            "spoiler_level": str    # 剧透过滤级别: strict/moderate/minimal/none
        }
        """
        if total_chapters <= 0:
            # 无法获取总章数时退回到基于绝对值的判断
            if chapter_num <= 3:
                phase = "opening"
                outline_range = 0
                rag_look_ahead = 0
                spoiler_level = "strict"
            elif chapter_num <= 8:
                phase = "early_dev"
                outline_range = 1
                rag_look_ahead = 0
                spoiler_level = "moderate"
            elif chapter_num <= 15:
                phase = "mid_dev"
                outline_range = 2
                rag_look_ahead = 1
                spoiler_level = "minimal"
            elif chapter_num <= 22:
                phase = "late_dev"
                outline_range = 2
                rag_look_ahead = 1
                spoiler_level = "minimal"
            else:
                phase = "climax"
                outline_range = 2
                rag_look_ahead = 2
                spoiler_level = "none"
        else:
            ratio = chapter_num / total_chapters
            chapters_from_end = total_chapters - chapter_num
            
            if chapter_num <= 3 or ratio <= 0.12:
                phase = "opening"
                outline_range = 0
                rag_look_ahead = 0
                spoiler_level = "strict"
            elif chapters_from_end <= 2 or ratio >= 0.92:
                phase = "resolution"
                outline_range = 1
                rag_look_ahead = 1
                spoiler_level = "none"
            elif ratio <= 0.35:
                phase = "early_dev"
                outline_range = 1
                rag_look_ahead = 0
                spoiler_level = "moderate"
            elif ratio <= 0.60:
                phase = "mid_dev"
                outline_range = 2
                rag_look_ahead = 1
                spoiler_level = "minimal"
            elif ratio <= 0.78 or chapters_from_end <= 5:
                phase = "late_dev"
                outline_range = 2
                rag_look_ahead = 1
                spoiler_level = "minimal"
            else:
                phase = "climax"
                outline_range = 2
                rag_look_ahead = 2
                spoiler_level = "none"
        
        # 各阶段的叙事指导 prompt（每个阶段都有明确的节奏边界和防抢跑措辞）
        pacing_prompts = {
            "opening": """
【本章处于故事开篇阶段，请特别注意叙事节奏】
- **循序渐进**：不要急于交代全部设定，只展示当前场景下读者需要知道的信息
- **悬念留白**：人物背景、身世秘密、世界深层规则等应随着剧情逐步揭示，不要在开篇就倾倒
- **聚焦当前**：只写好当前章节的核心事件（穿越/觉醒/相遇等），不要提前写后续章节才该发生的剧情
- **避免剧透**：大纲中后期出现的角色、势力、转折点，在本章中不应直接出现或详细描述
- **开篇质量**：用具体场景、细节描写、对话来吸引读者，而不是用大段说明性文字介绍世界观
- **严禁抢跑**：绝对不要把后续章节的剧情提前到本章写。本章只需要完成「引入」这一件事——让读者认识主角、了解初始状态、对世界产生初步好奇就够了。后续章节该有的冲突、转折、配角登场，各自有各自的章节，不需要你在这里抢先写完""",
            
            "early_dev": """
【本章处于故事早期发展阶段，请控制节奏稳步推进】
- **小步前进**：每章只推进1-2个情节节点，不要试图在一章内解决太多问题
- **打好基础**：这一阶段的目标是让读者深入了解世界和角色，建立情感联结，而不是急于推动主线冲突
- **适度展开**：可以开始引入新角色和新设定，但每章最多引入1-2个新元素，并给足展开空间
- **保持悬念**：大纲中标记为后期出现的角色、能力和事件，不要在本章提前揭示或暗示
- **严禁抢跑**：只写当前章节大纲指定的事件。如果大纲说本章"主角初入宗门"，那写完入门的第一印象和初步冲突就够了，不要直接写到"通过考核"或"获得传承"等后续章节的内容
- **节奏标尺**：本章应该像是电视剧的第4-6集——世界观在扩展，但主线还远未到高潮""",
            
            "mid_dev": """
【本章处于故事中期发展阶段，可以适当加速但仍需稳住节奏】
- **稳定推进**：每章推进2-3个情节节点，可以开始构建更大的冲突格局
- **伏笔回收**：早期埋下的伏笔可以开始部分回收，同时铺设新的悬念
- **多方势力**：可以引入更多势力和角色的交错互动，但要确保每个角色出场都有足够的铺垫
- **节奏把控**：这一阶段最容易出现"赶进度"的问题——觉得大纲内容多，想在一章内把好几件事都推进完。请克制这种冲动，每章只做好大纲中本章该做的事
- **严禁抢跑**：只写当前章节大纲指定的事件范围。不要因为看到后续大纲中某个精彩场面就提前把它写进来，每个高潮都有它该出现的章节
- **节奏标尺**：本章应该像是电视剧的中间几集——矛盾在升级，但还没到终极对决的时候""",
            
            "late_dev": """
【本章处于故事后期发展阶段，可以适度提速为高潮做铺垫】
- **加速布局**：冲突的密度可以增加，各方势力的行动可以更频繁
- **蓄势待发**：这一阶段的核心任务是为高潮做铺垫——把弓弦拉满但不松手
- **多线收拢**：之前分散的故事线可以开始逐渐汇聚，但不要急于给出结论
- **情感积累**：在冲突爆发前，给角色和读者足够的情感准备
- **严禁跳章**：即使接近高潮，也不要提前写出高潮章节才该出现的决战/突破/真相揭露。高潮之所以是高潮，是因为前期积累了足够的张力
- **节奏标尺**：本章应该像是暴风雨前的闷热——所有征兆都在，但雷还没劈下来""",
            
            "climax": """
【本章处于高潮/关键转折阶段，可以让情绪和冲突全面爆发】
- **张力拉满**：这是故事的高能区域，可以加快节奏、提高冲突密度
- **前期铺垫的回收**：之前埋下的伏笔、积累的情感可以在此时集中爆发/回收
- **多线交汇**：多条故事线在此处汇合是正常的，但要确保每条线的收束都有足够的铺垫
- **切忌仓促**：即使节奏快，关键转折也需要足够的细节支撑，不要用几句话草草带过
- **节奏标尺**：这是全书的最高潮——但高潮不是一口气写完所有决战，每个重大转折仍然需要足够的篇幅来呈现""",
            
            "resolution": """
【本章处于故事收尾阶段，请专注收束】
- **聚焦收尾**：不要再开启新的重大支线或引入新角色（除非是彩蛋性质）
- **回响前文**：适当回顾故事早期的场景、台词或细节，形成首尾呼应
- **角色归宿**：主要角色的结局需要符合其性格弧线，给读者完整的交代
- **克制留白**：好的结尾不需要解释一切，适度留白比过度说明更有余韵
- **不要拖沓**：收尾阶段不要为了凑字数而反复回味已解决的冲突，干净利落地收束"""
        }
        
        result = {
            "phase": phase,
            "outline_range": outline_range,
            "rag_look_ahead": rag_look_ahead,
            "pacing_instruction": pacing_prompts.get(phase, ""),
            "spoiler_level": spoiler_level
        }
        
        logger.info(
            f"章节阶段分类: chapter={chapter_num}/{total_chapters}章 → "
            f"phase={phase}, outline_range=±{outline_range}, "
            f"rag_look_ahead={rag_look_ahead}, spoiler_level={spoiler_level}"
        )
        return result
    
    # 防抢跑通用指令：根据阶段调整严格程度
    ANTI_RUSH_INSTRUCTIONS = {
        "strict": "**严禁抢跑**：你只负责写本章大纲范围内的事件。绝不要为了丰富内容或推进剧情而把后续章节才该出现的角色、场景、冲突、转折提前写进来。后续章节有后续章节的篇幅，不需要你在这里抢先完成。",
        "moderate": '**不要抢跑**：只写本章大纲范围内的事件。如果大纲说本章只到「主角进入秘境」，就不要写到「获得传承」——获得传承是后续章节的事。可以铺垫和暗示，但不要提前写出结果。',
        "minimal": "**控制节奏**：本章只负责推进大纲中标记给本章的情节，不要一口气把后续几章的走向都写完。关键转折要有铺垫过程，不要跳过中间环节直接给结果。",
        "none": '**保持节奏**：即使到了故事高潮/收尾阶段，每个重大事件仍需要足够的篇幅来展开，不要因为「快结束了」就草草带过。'
    }

    def _build_chapter_context(self, chapter_num: int, chapter_title: str, previous_summary: str = "") -> dict:
        """构建章节生成的分类上下文（阶段感知 + 剧透过滤 + 关键词检索补充）
        
        返回 dict:
            context_text: 拼接好的上下文字符串
            phase_config: 阶段策略配置
            outline_for_chapter: 当前章节相关大纲（范围校验/续写复用）
            setting_text / character_text / outline_text: 原始设定（续写精简上下文复用）
            anti_rush: 当前阶段的防抢跑指令
        """
        # ---- 阶段分类：根据章节位置决定上下文策略 ----
        total_chapters = self._estimate_total_chapters()
        phase_config = self._classify_chapter_phase(chapter_num, total_chapters)
        
        context_parts = []
        
        # 1. 核心设定：直接从 novel_info 获取（已生成的内容，比检索更完整更精准）
        setting_text = self.novel_info.get("world_setting", "")
        character_text = self.novel_info.get("characters", "")
        outline_text = self.novel_info.get("outline", "")
        
        if setting_text:
            # 世界观按档注入（信息按叙事时点解锁）：开篇/早期只给稳定背景，
            # 中期加势力格局，后期全量；注入前套实体时间锁遮蔽未登场角色名
            SETTING_MAX = 4000
            world_level = {"strict": "stable", "moderate": "stable",
                           "minimal": "faction"}.get(phase_config.get("spoiler_level", "minimal"), "full")
            world_text = self._stable_world_text(level=world_level, chapter_num=chapter_num,
                                                 max_chars=SETTING_MAX)
            if world_text:
                context_parts.append(f"【世界观设定】\n{world_text}")
                logger.info(f"世界观设定[{world_level}]: 原始{len(setting_text)}字 → 注入{len(world_text)}字 (上限{SETTING_MAX})")
        else:
            logger.info("世界观设定: 无")
            
        spoiler_level = phase_config.get("spoiler_level", "minimal")
        cards = self.load_character_cards()
        if cards:
            # 结构化模式（TODO 2.3）：按登场/退场章过滤注入角色卡详情；
            # 未登场角色只注入名字列表作兜底，防止前文提及的角色显得凭空出现
            active, absent = cc.filter_cards_for_chapter(cards, chapter_num)
            if active:
                # 开篇/早期（strict/moderate）：只注入精简卡（名字/身份/性格），
                # 不带人物关系/备注弧线，防止把后续剧情写进当前章
                if spoiler_level in ("strict", "moderate"):
                    cards_text = cc.cards_to_brief(active)
                else:
                    cards_text = cc.cards_to_text(active)
                CHAR_MAX = 6000
                trunc = cards_text[:CHAR_MAX] + ("..." if len(cards_text) > CHAR_MAX else "")
                if spoiler_level != "none":
                    trunc = self._strip_spoiler_sentences(trunc, level=spoiler_level)
                context_parts.append(f"【人物设定】\n{trunc}")
                logger.info(f"角色卡注入: {len(active)}/{len(cards)} 张在场角色卡 (模式={'brief' if spoiler_level in ('strict', 'moderate') else 'full'})")
            if absent:
                context_parts.append(f"【尚未登场角色（仅名字，本章不得安排其出场或揭示其信息）】\n{'、'.join(absent)}")
            character_text = cc.cards_to_text(active)
        elif character_text:
            # 人物设定截断，最多 6000 字；开篇/早期只取精简字段行的免费文本受限——
            # 自由文本无结构化字段，退而求其次：strict/moderate 阶段截断前 800 字并套实体锁
            CHAR_MAX = 6000
            if spoiler_level in ("strict", "moderate"):
                trunc = character_text[:800]
            else:
                trunc = character_text[:CHAR_MAX]
            # 渐进式前瞻信息过滤：根据阶段决定过滤力度
            if spoiler_level != "none":
                trunc = self._strip_spoiler_sentences(trunc, level=spoiler_level)
                logger.info(f"人物设定[{spoiler_level}]: 已过滤前瞻信息")
            trunc = self._lock_entities(trunc, chapter_num)
            if trunc:
                context_parts.append(f"【人物设定】\n{trunc}")
            logger.info(f"人物设定: 原始{len(character_text)}字 → 传入{len(trunc)}字 (上限{CHAR_MAX})")
        else:
            logger.info("人物设定: 无")
        
        outline_for_chapter = ""
        if outline_text:
            effective_range = phase_config["outline_range"]
            plan = self.novel_info.get("volume_plan") or self.vs.load_extra_data("volume_plan", []) or []
            cur_vol = self._current_volume(plan, chapter_num)
            outline_for_chapter = self._extract_relevant_outline(
                outline_text, chapter_num, 
                capture_range=effective_range,
                spoiler_level=spoiler_level,
                current_volume=cur_vol
            )
            context_parts.append(f"【小说大纲（当前章节相关）】\n{outline_for_chapter}")
            logger.info(f"小说大纲: 原始{len(outline_text)}字 → 提取相关{len(outline_for_chapter)}字 (range=±{effective_range}, spoiler={spoiler_level})")
        else:
            logger.info("小说大纲: 无")
        
        # 2. 前几章摘要：只取最近2章的末尾段落作为前情回顾，不传全文
        prev_chapters_summary = self._get_previous_chapters_summary(chapter_num, max_chars=1500)
        if prev_chapters_summary:
            context_parts.append(f"【前情回顾】\n{prev_chapters_summary}")
            logger.info(f"前情回顾: {len(prev_chapters_summary)}字")
        else:
            logger.info("前情回顾: 无")
        
        # 3. 关键词检索补充：用当前章节标题在本地存储检索相关片段(n_results=4)
        #    跳过已直接包含的设定/人物/大纲(避免重复)
        #    跳过当前章节自身(避免旧版本标题/内容污染上下文)
        #    根据阶段策略决定是否允许看后续章节
        query = f"第{chapter_num}章 {chapter_title}"
        related = self.vs.search_related(query, n_results=4)
        extra_context = []
        for ctx in related:
            content = ctx["content"]
            meta = ctx.get("metadata", {})
            # 跳过已经直接包含的核心设定（避免重复）
            if meta.get("type") in ("setting", "character", "outline"):
                continue
            # 跳过当前章节自身（避免旧版本标题/内容污染上下文）
            if meta.get("type") == "chapter" and meta.get("title", "") == f"chapter_{chapter_num}":
                logger.info(f"关键词检索跳过当前章节自身: chapter_{chapter_num}")
                continue
            # 根据阶段策略过滤后续章节：rag_look_ahead=0 表示完全不看后续
            if meta.get("type") == "chapter":
                chap_match = re.search(r"chapter_(\d+)", meta.get("title", ""))
                if chap_match:
                    ref_chap_num = int(chap_match.group(1))
                    allowed_max = chapter_num + phase_config["rag_look_ahead"]
                    if ref_chap_num > allowed_max:
                        logger.info(
                            f"[{phase_config['phase']}] 关键词检索跳过第{ref_chap_num}章 "
                            f"(允许范围≤第{allowed_max}章)，避免信息泄漏"
                        )
                        continue
            # 前章内容只取短片段，且去除标题行（避免旧标题污染当前生成）
            if meta.get("type") == "chapter":
                # 去掉"第X章 标题"行，只保留正文
                lines = content.split("\n", 1)
                if lines and re.match(r"第\d+章", lines[0].strip()):
                    content = lines[1] if len(lines) > 1 else content
                content = content[:800] + ("..." if len(content) > 800 else "")
            # 实体时间锁：检索片段里若点名尚未登场的角色，一律遮蔽
            content = self._lock_entities(content, chapter_num)
            extra_context.append(content[:1000])
        
        if extra_context:
            combined_extra = "\n---\n".join(extra_context)
            context_parts.append(f"【相关参考片段】\n{combined_extra}")
            logger.info(f"关键词检索补充: {len(extra_context)}条, 共{len(combined_extra)}字")
        else:
            logger.info("关键词检索补充: 无")
        
        # 4. 滚动全书摘要：长篇连载的长线记忆（Phase 3）
        # 4. 双摘要长线记忆（B6 分层）：全书梗概（早期信息长期保留）+ 近 N 章摘要（细节）
        if chapter_num > 1:
            summary_full = self.vs.load_extra_data("rolling_summary_full", "") or \
                self.vs.load_extra_data("rolling_summary", "") or ""
            summary_recent = self.vs.load_extra_data("rolling_summary_recent", "") or ""
            if summary_recent:
                context_parts.append(f"【近期剧情摘要（近N章）】\n{summary_recent[:500]}")
                logger.info(f"近期摘要: 传入{min(len(summary_recent),500)}字")
            if summary_full:
                context_parts.append(f"【全书剧情梗概（截至上一章）】\n{summary_full[:300]}")
                logger.info(f"全书梗概: 传入{min(len(summary_full),300)}字")

        # 4b. 角色状态台账 + 近期事件 + 待回收伏笔（A1 分块配额 + A3 时间线分层 + B1 伏笔排序）
        ledger_brief = self._ledger_brief(chapter_num, phase_config)
        if ledger_brief:
            context_parts.append(f"【角色状态与剧情台账】\n{ledger_brief}")
            logger.info(f"台账注入: {len(ledger_brief)}字（分块配额）")
        
        # 6. 文风指纹：生成时锁定文风（Phase 2）
        fingerprint = self.vs.load_extra_data("style_fingerprint", "") or ""
        if fingerprint:
            context_parts.append(f"【文风要求（必须严格遵守）】\n{fingerprint[:1500]}")
            logger.info(f"文风指纹: 传入{min(len(fingerprint),1500)}字")
        
        context_text = "\n\n".join(context_parts)
        logger.info(f"上下文总计: {len(context_text)}字")
        
        if previous_summary:
            context_text += f"\n\n上一章内容回顾：{previous_summary}\n"
            logger.info(f"上一章回顾: {len(previous_summary)}字")
        
        return {
            "context_text": context_text,
            "phase_config": phase_config,
            "outline_for_chapter": outline_for_chapter,
            "setting_text": setting_text,
            "character_text": character_text,
            "outline_text": outline_text,
            "anti_rush": self.ANTI_RUSH_INSTRUCTIONS.get(spoiler_level, self.ANTI_RUSH_INSTRUCTIONS["minimal"]),
        }

    def generate_chapter(self, chapter_num: int, chapter_title: str, previous_summary: str = "", max_tokens: int = 2500, target_words: int = 2000, beats: str = "", temperature: float = 0.8, extra_instruction: str = "") -> str:
        """生成单章正文 — 泛化节奏控制：根据章节所处阶段自动调整上下文策略
        
        beats: 可选的章节细纲（场景卡）。提供时按场景逐段生成，质量更稳定。
        extra_instruction: 可选的额外改写要求（如"需融入新设定X"，回溯改写确认后传入）。
        """
        
        logger.info(f"===== generate_chapter 开始 =====")
        logger.info(f"参数: chapter_num={chapter_num}, chapter_title={chapter_title}, max_tokens={max_tokens}, target_words={target_words}, beats={'有' if beats else '无'}")

        # 惰性细纲保障：当前章所属卷无逐章概要时自动生成（TODO 3.2.0）
        self.ensure_outline_for_chapter(chapter_num)
        if not chapter_title.strip():
            chapter_title = self.get_outline_chapter_titles().get(chapter_num, "")
        if not chapter_title.strip():
            # 大纲也没有标题 → 由 AI 根据上下文拟定
            chapter_title = self.generate_chapter_title(chapter_num) or f"第{chapter_num}章"
            logger.info(f"第{chapter_num}章标题为空，AI 拟定为「{chapter_title}」")
        self.last_chapter_title = chapter_title
        if not target_words or target_words <= 0:
            # 字数未显式指定 → 从大纲带过来（逐章字数 → 全局每章字数 → 默认）
            target_words = self.resolve_target_words(chapter_num)
            logger.info(f"第{chapter_num}章字数未指定，从大纲解析为 {target_words} 字")

        # 记忆自动闭环：前面的章被改动导致后续章台账/摘要 stale 时，自动补齐
        # （否则本章生成时前情回顾/台账注入的是残缺记忆，会破坏一致性）
        # 只重算缺失记录的章（不浪费 API 全量重算）
        if self.vs.load_extra_data("ledger_stale", False):
            logger.info("检测到台账 stale，生成前自动重建记忆")
            self._report("正在自动重建账本与摘要（章节变动导致记忆过期）…")
            try:
                self.sync_memory(regen=True)
            except Exception as e:
                logger.warning(f"自动记忆重建失败（不影响本章生成）: {e}")
        
        # max_tokens 必须覆盖目标字数（中文1字≈1.5-2 token），否则首轮必截断
        needed_mt = min(int(target_words * 1.8), self.api.MAX_TOKENS_LIMIT)
        if max_tokens < needed_mt:
            logger.info(f"max_tokens={max_tokens} 低于目标字数估算({needed_mt})，自动提升")
            max_tokens = needed_mt
        
        # 先删除存储中当前章节的旧数据，避免旧标题/内容污染搜索结果
        # 注意：删除发生在用户确认生成方式之后（见下方 _delete_stale_chapter 调用点），
        # 否则用户在断点弹窗选择"稍后决定"时旧正文会被误删
        ctx = self._build_chapter_context(chapter_num, chapter_title, previous_summary)
        context_text = ctx["context_text"]
        if extra_instruction.strip():
            # 回溯改写等场景的额外要求，注入上下文末尾（权重最高）
            context_text += f"\n\n【本次改写的额外要求（必须体现）】\n{extra_instruction.strip()}\n"
        phase_config = ctx["phase_config"]
        outline_for_chapter = ctx["outline_for_chapter"]
        setting_text = ctx["setting_text"]
        character_text = ctx["character_text"]
        outline_text = ctx["outline_text"]
        anti_rush = ctx["anti_rush"]
        phase = phase_config["phase"]
        spoiler_level = phase_config.get("spoiler_level", "minimal")
        
        # 构建阶段感知的叙事节奏指令
        pacing_instruction = phase_config["pacing_instruction"]
        if pacing_instruction:
            logger.info(f"[{phase_config['phase']}] 已注入阶段专属叙事指导")
        
        # ---- 有细纲时：按场景卡逐段生成 ----
        if beats and beats.strip():
            # 断点决策（续写/暂停/重来）在 _generate_chapter_by_beats 内部完成，
            # 旧章节内容的删除发生在决策之后，避免用户选择"稍后决定"时误删旧正文
            result = self._generate_chapter_by_beats(
                chapter_num, chapter_title, beats, context_text, pacing_instruction,
                anti_rush, target_words, max_tokens, temperature
            )
        else:
            # 旧章节内容的删除在 _generate_chapter_single 内部（生成前）执行
            result = self._generate_chapter_single(
                chapter_num, chapter_title, context_text, pacing_instruction,
                anti_rush, phase, target_words, max_tokens, temperature
            )
        
        # 自动续写：如果输出太短，续写到接近目标字数
        min_chars = int(target_words * 0.7)  # 至少达到目标的70%
        max_continuations = 3  # 最多续写3轮，防止无限循环
        continuation_round = 0
        while len(result) < min_chars and continuation_round < max_continuations:
            remaining = target_words - len(result)
            continuation_round += 1

            # 续写时也需要核心设定上下文，避免人设跑偏
            # 使用精简版上下文（比首次生成更短，节省token）
            continue_context_parts = []
            if setting_text:
                continue_context_parts.append(f"【世界观设定（摘要）】\n{setting_text[:1500]}")
            if character_text:
                continue_context_parts.append(f"【人物设定（摘要）】\n{character_text[:2000]}")
            if outline_text:
                continue_context_parts.append(f"【当前章节大纲】\n{outline_for_chapter[:500]}")
            continue_context = "\n\n".join(continue_context_parts) if continue_context_parts else ""

            continue_prompt = f"""请继续往下写第 {chapter_num} 章 "{chapter_title}" 的剩余内容。

当前已写 {len(result)} 字，还需要再写至少 {remaining} 字才能完成本章。
{continue_context}

【前文末尾】
{result[-1500:]}

【硬性要求】
- 再写至少 {remaining} 字，但如果当前场景已自然收束到合适位置，允许略少于目标
- 保持人物设定一致（参照上方人物设定）
- {anti_rush}
- 不要写总结性结尾，不要写"本章完"
- 如果情节还有发展空间，在段落中间自然断开
- 直接输出续写内容，不要解释

续写内容："""

            # 续写时的 token 预算：按剩余字数的2倍估算（中文字符≈1.5-2 token）
            continue_max = min(max_tokens, max(2000, int(remaining * 2)))
            self._report(f"第{chapter_num}章已写{len(result)}字，进行第{continuation_round}轮续写…")
            continuation = self.api.generate(continue_prompt, step="chapter", temperature=temperature, max_tokens=continue_max, stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
            # 续写拼接时补分隔符，避免段落粘连
            if result and not result.endswith("\n"):
                result += "\n\n"
            result = result + continuation

            logger.info(f"章节第{chapter_num}章续写第{continuation_round}轮：原始长度={len(result) - len(continuation)}字 + 续写{len(continuation)}字 = 总计{len(result)}字")

        if len(result) < min_chars:
            logger.warning(f"章节第{chapter_num}章经{max_continuations}轮续写后仍仅{len(result)}字，目标{target_words}字")

        # 生成后范围校验：检查是否可能覆盖了超出当前大纲范围的事件
        scope_warning = self._check_chapter_scope(result, chapter_num, outline_for_chapter, spoiler_level)
        self.last_scope_warning = scope_warning or ""
        
        # 生成后套话检测（Phase 2）
        cliche_warn = self.detect_chapter_cliches(result)
        if cliche_warn:
            self.last_scope_warning = (self.last_scope_warning + "；" if self.last_scope_warning else "") + cliche_warn
            logger.info(f"第{chapter_num}章套话检测: {cliche_warn}")
        
        # 伏笔回收率告警（高潮/收尾阶段，Phase 3）
        if phase in ("climax", "resolution"):
            fs_warn = self.foreshadowing_recovery_warning()
            if fs_warn:
                self.last_scope_warning = (self.last_scope_warning + "；" if self.last_scope_warning else "") + fs_warn
        
        if self.last_scope_warning:
            logger.warning(f"章节{chapter_num}章生成后检查: {self.last_scope_warning}")
        
        # AI拒绝的内容不入库，避免污染本地存储
        if is_ai_refusal(result):
            logger.warning(f"第{chapter_num}章生成被AI拒绝（内容安全审查），结果不入库")
            return result
        
        # 保存到本地存储，方便后续章节检索
        self.vs.add_section("chapter", f"chapter_{chapter_num}", f"第{chapter_num}章 {chapter_title}\n{result}")
        
        # Phase 3：章节生成后自动更新记忆（台账 delta + 全书梗概 + 近期摘要，一次调用）
        # 重生成场景：先失效本章及之后的旧 delta（旧章引入的伏笔/状态随之撤销），再重新产出
        self.invalidate_memory_from(chapter_num)
        self._report("正文完成，正在更新伏笔台账与摘要…")
        self.update_memory(chapter_num, result)
        
        return result
    
    def _delete_stale_chapter(self, chapter_num: int):
        """删除存储中当前章节的旧数据，避免旧标题/内容污染检索结果。
        必须在用户确认生成方式之后调用（断点弹窗选择"稍后决定"时不得删除旧正文）。
        同步清理该章的评审/黄金开篇数据（旧评审针对旧正文，正文被替换后必然失效）。"""
        self.vs.delete_section("chapter", f"chapter_{chapter_num}")
        self.vs.delete_extra_field(f"chapter_review_{chapter_num}")
        self.vs.delete_extra_field(f"chapter_golden_{chapter_num}")
        logger.info(f"已预删除存储中 chapter_{chapter_num} 的旧数据（含评审/黄金开篇）")

    def _chapter_hard_requirements(self, phase: str, target_words: int, anti_rush: str) -> str:
        """按阶段生成章节硬性要求文本"""
        if phase == "opening":
            return f"""【硬性要求】
- 本章目标字数约 {target_words} 字，但如果章节自然结尾在 {int(target_words * 0.7)} 字以上也是完全可以接受的——开篇章节的质量远比字数重要
- 如果一次写不完，请在段落中间自然断开，不要写总结性结尾，不要写"本章完"之类的结束语
- {anti_rush}
- 保持人物设定一致性
- 文笔流畅，用具体场景、细节描写、对话、心理活动来展开（不要用大段说明性文字凑字数）
- 直接输出正文，不要解释

正文："""
        elif phase == "resolution":
            return f"""【硬性要求】
- 本章目标字数约 {target_words} 字，可以略少（收尾重在质量而非篇幅）
- 如果一次写不完，请在段落中间自然断开，不要写总结性结尾，不要写"本章完"之类的结束语
- {anti_rush}
- 保持人物设定一致性
- 文笔流畅，有画面感、细节描写丰富
- 直接输出正文，不要解释

正文："""
        else:
            return f"""【硬性要求】
- 本章目标字数约 {target_words} 字（允许±20%浮动，但不要低于80%）
- 如果一次写不完，请在段落中间自然断开，不要写总结性结尾，不要写"本章完"之类的结束语
- {anti_rush}
- 情节符合大纲走向，保持人物设定一致性
- 文笔流畅，有画面感、细节描写丰富（大量使用对话、动作、心理、环境描写来展开）
- 直接输出正文，不要解释

正文："""
    
    def _generate_chapter_single(self, chapter_num, chapter_title, context_text, pacing_instruction,
                                 anti_rush, phase, target_words, max_tokens, temperature) -> str:
        """无细纲的传统一次性生成路径"""
        prompt = f"""请你根据以下信息，写出小说第 {chapter_num} 章 "{chapter_title}" 的完整正文。

{context_text}
{pacing_instruction}"""
        prompt += self._chapter_hard_requirements(phase, target_words, anti_rush)
        # 套话规避指令（proactive 防御）
        prompt = prompt.replace("正文：", self._cliche_instruction() + "\n\n正文：")
        
        # 兜底删除旧章节数据（beats 解析失败回退到本路径时，主流程的删除点不会被经过）
        self._delete_stale_chapter(chapter_num)
        
        logger.info(f"完整prompt长度: {len(prompt)}字 (约{len(prompt)*2}token)")
        logger.info(f"API调用参数: model={self.api.model}, max_tokens={max_tokens}, temperature={temperature}")
        
        return self.api.generate(prompt, step="chapter", temperature=temperature, max_tokens=max_tokens, stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
    
    @staticmethod
    def parse_beats(beats_text: str) -> list:
        """把细纲文本解析为场景卡列表。兼容 '## 场景1：xxx' 和 '场景1：xxx' 两种格式"""
        blocks = re.split(r"(?m)^#{1,4}\s*(?=场景\s*\d+)", beats_text)
        if len(blocks) <= 1:
            blocks = re.split(r"(?m)^(?=场景\s*\d+\s*[：:])", beats_text)
        beats = [b.strip() for b in blocks if b.strip() and re.search(r"场景\s*\d+", b)]
        return beats
    
    def _generate_chapter_by_beats(self, chapter_num, chapter_title, beats_text, context_text,
                                   pacing_instruction, anti_rush, target_words, max_tokens, temperature) -> str:
        """按场景卡逐段生成：每个场景一次 API 调用，带前文末尾保持连贯，最后拼接。
        断点保护：每完成一个场景立即落盘（extra/chapter_partial_N）；场景失败时
        通过 on_confirm 询问用户：重试 / 稍后继续（保留断点） / 取消（已生成场景存为草稿）。"""
        beats = self.parse_beats(beats_text)
        if not beats:
            logger.warning("细纲解析失败，回退到一次性生成")
            return self._generate_chapter_single(
                chapter_num, chapter_title, context_text, pacing_instruction,
                anti_rush, self._classify_chapter_phase(chapter_num, self._estimate_total_chapters())["phase"],
                target_words, max_tokens, temperature
            )

        words_per_beat = max(400, target_words // len(beats))
        # 每场景 max_tokens：用户可在「高级设置」用 chapter_scene 覆盖（推理模型思考也占额度，默认可能不够）
        beat_max_tokens = self._step_max_tokens("chapter_scene") or \
            min(max(1500, int(words_per_beat * 2)), self.api.MAX_TOKENS_LIMIT)
        logger.info(f"按细纲生成：{len(beats)} 个场景，每场景约 {words_per_beat} 字，beat_max_tokens={beat_max_tokens}")

        # ---- 断点检测：同章同细纲的未完成进度可续写 ----
        partial_key = f"chapter_partial_{chapter_num}"
        parts = []
        partial = self.vs.load_extra_data(partial_key)
        # 比对前归一化空白，避免 trim/换行差异导致断点被误判为不匹配
        saved_beats = (partial.get("beats_text") or "").strip() if partial else ""
        if partial and saved_beats == beats_text.strip() and partial.get("parts"):
            saved = partial["parts"]
            choice = self._confirm(
                f"检测到第{chapter_num}章有未完成进度（已生成 {len(saved)}/{len(beats)} 个场景），是否从断点续写？",
                [{"action": "resume", "label": "从断点续写"},
                 {"action": "resume_later", "label": "稍后决定（保留进度）"},
                 {"action": "restart", "label": "放弃进度，从头生成"}])
            if choice == "resume":
                parts = list(saved)
                logger.info(f"第{chapter_num}章从断点续写：已完成 {len(parts)}/{len(beats)} 场景")
                self._report(f"第{chapter_num}章：从断点续写（已完成 {len(parts)}/{len(beats)} 场景）")
            elif choice in (None, "resume_later"):
                # 前端断开或用户点关闭：安全暂停，绝不能把未响应误判为“从头生成”
                logger.info(f"第{chapter_num}章断点决策暂停：已完成 {len(saved)}/{len(beats)} 场景")
                self._report(f"第{chapter_num}章：已暂停，断点进度已保留")
                raise ChapterPaused(
                    f"已暂停：第{chapter_num}章断点进度已保留（{len(saved)}/{len(beats)} 场景），稍后生成该章时可继续选择")
            else:
                self.vs.save_extra_data(partial_key, None)
                logger.info(f"第{chapter_num}章放弃断点进度，从头生成")
        elif partial and partial.get("parts") and self.on_confirm is not None:
            # 细纲已变更但旧断点仍有进度：不再静默丢弃，交由用户决定（Web 交互路径）
            saved = partial["parts"]
            choice = self._confirm(
                f"检测到第{chapter_num}章有旧断点进度（已完成 {len(saved)} 个场景），但与当前场景节拍不一致。\n"
                f"继续使用旧断点会按当前节拍重新生成（旧进度作废）。",
                [{"action": "restart", "label": "放弃旧断点，从头生成"},
                 {"action": "resume_later", "label": "稍后决定（保留进度）"}])
            if choice in (None, "resume_later"):
                logger.info(f"第{chapter_num}章旧断点保留（节拍不一致，用户暂停）")
                raise ChapterPaused(
                    f"已暂停：第{chapter_num}章旧断点进度已保留（{len(saved)} 个场景），"
                    f"恢复原来的场景节拍后可从断点续写")
            self.vs.save_extra_data(partial_key, None)
            logger.info(f"第{chapter_num}章节拍已变更，用户确认放弃旧断点")
        elif partial:
            # 非交互路径（CLI/测试/批量）：细纲已变更，旧断点作废
            self.vs.save_extra_data(partial_key, None)

        # 断点决策已完成（继续/重来），此处才删除旧章节内容
        self._delete_stale_chapter(chapter_num)

        def _save_partial():
            self.vs.save_extra_data(partial_key, {
                "chapter_num": chapter_num, "title": chapter_title,
                "beats_text": beats_text, "parts": parts,
            })

        for i, beat in enumerate(beats):
            if i < len(parts):
                continue  # 断点续写：跳过已完成场景
            # 用户取消检查点：场景间及时止损（断点已逐场景落盘，取消不丢进度）
            if getattr(self.api, "cancel_check", None) and self.api.cancel_check():
                _save_partial()
                raise GenerationCancelled(f"已取消：第{chapter_num}章进度已保存（{len(parts)}/{len(beats)} 场景）")
            prev_tail = ""
            if parts:
                prev_tail = f"\n\n【前文末尾】\n{(''.join(parts))[-1200:]}\n\n请紧接前文继续，不要重复已写内容。"

            prompt = f"""你正在写小说第 {chapter_num} 章 "{chapter_title}"。本章已按场景卡规划好，你只负责写**第 {i+1}/{len(beats)} 个场景**的正文。

{context_text}
{pacing_instruction}

【本章场景卡】
{beats_text}

【本次要写的场景】
{beat}
{prev_tail}

【硬性要求】
- 本场景目标约 {words_per_beat} 字，用具体场景、细节描写、对话、心理活动展开，不要概括缩写
- 只写这个场景的内容，不要写其他场景的事件
- {anti_rush}
- 保持人物设定一致性
- {self._cliche_instruction()}
- 直接输出正文，不要输出场景标题，不要解释

正文："""
            # 单场景生成 + 失败时的用户决策循环
            while True:
                self._report(f"第{chapter_num}章：正在生成场景 {i+1}/{len(beats)}…")
                try:
                    part = self.api.generate(prompt, step="chapter", temperature=temperature,
                                             max_tokens=beat_max_tokens, stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
                    parts.append(part.strip())
                    _save_partial()  # 每场景落盘，崩溃/取消不丢进度
                    logger.info(f"场景 {i+1}/{len(beats)} 完成，{len(part)} 字")
                    break
                except Exception as e:
                    logger.error(f"第{chapter_num}章场景 {i+1}/{len(beats)} 生成失败: {e}")
                    _save_partial()
                    if self.on_confirm is None:
                        raise  # 无交互回调（CLI/测试）：保持原有行为，直接抛错（断点已保存）
                    action = self._confirm(
                        f"第{chapter_num}章场景 {i+1}/{len(beats)} 生成失败：\n{str(e)[:300]}\n\n"
                        f"已完成的 {len(parts)} 个场景进度已保存。",
                        [{"action": "retry", "label": "重试本场景"},
                         {"action": "resume_later", "label": "稍后继续（保留进度）"},
                         {"action": "cancel", "label": "取消（已生成场景存为草稿）"}])
                    if action == "retry":
                        continue
                    if action == "cancel" and parts:
                        # 已完成场景存为该章草稿正文，避免浪费已烧的 token
                        draft = "\n\n".join(parts)
                        self.vs.add_section("chapter", f"chapter_{chapter_num}",
                                            f"第{chapter_num}章 {chapter_title}\n{draft}")
                        self.vs.save_extra_data(partial_key, None)
                        raise ChapterPaused(
                            f"已取消：前 {len(parts)} 个场景已保存为第{chapter_num}章草稿，可手动编辑或重新生成")
                    raise ChapterPaused(
                        f"已暂停：第{chapter_num}章进度已保存（{len(parts)}/{len(beats)} 场景），"
                        f"下次生成该章时可从断点续写。失败原因：{str(e)[:150]}")

        # 整章完成，清除断点
        self.vs.save_extra_data(partial_key, None)
        return "\n\n".join(parts)

    @staticmethod
    def _step_max_tokens(step: str) -> Optional[int]:
        """读取用户对某步骤的 max_tokens 覆盖值（未设置返回 None）。失败静默。"""
        try:
            from api import user_config
            return user_config.get_max_tokens_overrides().get(step)
        except Exception:
            return None
    
    # ---------- Phase 2: 文风与合规 ----------

    def _cliche_instruction(self) -> str:
        """套话规避指令（含用户自定义词条）"""
        custom = self.vs.load_extra_data(CUSTOM_BLACKLIST_KEY, []) or []
        if isinstance(custom, str):
            custom = [w.strip() for w in re.split(r"[,，、\n]", custom) if w.strip()]
        return cliche_avoidance_instruction(custom)

    def detect_chapter_cliches(self, content: str) -> str:
        """生成后套话检测，返回可读报告（无命中返回空）"""
        custom = self.vs.load_extra_data(CUSTOM_BLACKLIST_KEY, []) or []
        if isinstance(custom, str):
            custom = [w.strip() for w in re.split(r"[,，、\n]", custom) if w.strip()]
        return cliche_report(content, custom)

    def extract_style_fingerprint(self, sample_text: str = "", description: str = "", max_tokens: int = 1200) -> str:
        """从样例文本或描述提取文风指纹，注入后续所有生成（生成时锁定文风）"""
        if not sample_text and not description:
            return ""

        source = ""
        if sample_text:
            source += f"【文风样例文本】\n{sample_text[:3000]}\n\n"
        if description:
            source += f"【用户文风描述】\n{description}\n\n"

        prompt = f"""你是一位文风分析专家。请从以下材料中提炼一份**可被 AI 直接执行的文风指纹**。

{source}
【输出要求】
按以下维度输出，每条必须具体可执行（❌"文笔优美" ✅"对话占比约40%，短句为主"）：

- 句式：句长偏好、断句习惯、是否多用短句/长句
- 对话：对话占比、对话风格（口语化/书面化/含蓄）
- 描写：环境/动作/心理描写的比例和偏好
- 用词：词汇风格（朴素/华丽/冷峻）、标志性用词习惯
- 节奏：叙事推进速度、场景切换频率
- 禁忌：这种文风下绝对不能出现的写法

全文 500-800 字，直接输出，不要解释。"""

        fingerprint = self.api.generate(prompt, step="polish", temperature=0.3, max_tokens=max_tokens)
        if not is_ai_refusal(fingerprint):
            self.vs.save_extra_data("style_fingerprint", fingerprint)
        return fingerprint

    def humanize_text(self, text: str, max_tokens: int = 2000) -> str:
        """去 AI 腔改写：句长随机化、清除套话、增加口语化质感，降低平台 AI 检测率"""
        needed = min(int(len(text) * 1.8) + 500, self.api.MAX_TOKENS_LIMIT)
        max_tokens = max(max_tokens, needed)
        cliches = detect_cliches(text)
        cliche_hint = "、".join(f"「{w}」" for w, _ in cliches[:10]) if cliches else ""

        prompt = f"""你是一位资深人类作家，请改写以下文本，去除"AI 腔"，让它读起来像真人作家写的。

【原文】
{text}

【改写要求】
1. **句长参差**：刻意混合 3-5 字的短句和 20 字以上的长句，避免均匀的中等句式（这是 AI 文最典型的特征）
2. **清除套话**：{"重点清除：" + cliche_hint + "；" if cliche_hint else ""}删除所有空洞的套话表达，用具体动作/细节替代
3. **口语质感**：对话更口语化，允许语气词、打断、不完整的句子
4. **删减解释**：删掉"他意识到""她明白"这类直接解释心理的标签句，改用行为暗示
5. 保持情节和信息完全不变，字数与原文相当
6. 直接输出改写后的文本，不要解释

改写后："""

        return self.api.generate(prompt, step="polish", temperature=0.85, max_tokens=max_tokens)

    # ---------- Phase 3: 长篇记忆（状态台账 / 伏笔 / 滚动摘要） ----------

    def _pending_foreshadowing(self) -> list:
        """从状态台账读取未回收伏笔列表"""
        ledger = self.vs.load_extra_data("state_ledger", {}) or {}
        fs = ledger.get("foreshadowing", [])
        return [f for f in fs if isinstance(f, dict) and f.get("status") != "已回收"]

    def _ledger_brief(self, chapter_num: int, phase_config: dict = None, quotas: tuple = (400, 200, 200)) -> str:
        """构建台账注入文本（正文生成与章节评审共用）。

        A1 分块配额：角色状态/近期事件/待回收伏笔各自独立裁剪，互不挤占；
        A2 活跃度排序：角色按最后活跃章降序，沉睡角色（占满配额前）自动省略；
        A3 时间线分层：最近 3 条 + 伏笔埋设章事件（里程碑）；
        B1 伏笔排序：逾期 > 临近（5 章内）> 远期，按 target_chapter 优先提醒。
        """
        ledger = self.vs.load_extra_data("state_ledger", {}) or {}
        blocks = []
        char_q, tl_q, fs_q = quotas

        # 角色状态：按最后活跃章降序（A2）
        chars = [c for c in ledger.get("characters", []) if isinstance(c, dict) and c.get("name")]
        chars.sort(key=lambda c: int(c.get("updated_chapter") or 0), reverse=True)
        char_lines, used = [], 0
        for c in chars:
            line = f"- {c['name']}: {c.get('status', '')}"
            if used + len(line) > char_q:
                break
            char_lines.append(line)
            used += len(line)
        if char_lines:
            blocks.append("【角色状态】\n" + "\n".join(char_lines))

        # 近期事件 + 伏笔埋设章里程碑（A3）
        timeline = [t for t in ledger.get("timeline", []) if isinstance(t, dict)]
        recent = sorted(timeline, key=lambda t: t.get("chapter", 0))[-3:]
        milestone_chs = {f.get("planted_chapter") for f in ledger.get("foreshadowing", [])
                         if isinstance(f, dict) and f.get("planted_chapter")}
        picked = list(recent)
        for t in timeline:
            if t.get("chapter") in milestone_chs and t not in picked:
                picked.append(t)
        picked.sort(key=lambda t: t.get("chapter", 0))
        tl_lines, used = [], 0
        for t in picked:
            line = f"- 第{t.get('chapter', '?')}章: {t.get('event', '')}"
            if used + len(line) > tl_q:
                break
            tl_lines.append(line)
            used += len(line)
        if tl_lines:
            blocks.append("【近期剧情事件】\n" + "\n".join(tl_lines))

        # 待回收伏笔：mid_dev 起注入，按目标章排序（B1）
        phase = (phase_config or {}).get("phase", "")
        if phase in ("mid_dev", "late_dev", "climax", "resolution"):
            pending = self._pending_foreshadowing()
            if pending:
                now = chapter_num
                def fs_key(f):
                    t = f.get("target_chapter") or 0
                    if t and t < now:
                        return (0, -t)      # 逾期最优先
                    if t and t - now <= 5:
                        return (1, -t)      # 临近
                    return (2, -t)          # 远期
                pending.sort(key=fs_key)
                fs_lines, used = [], 0
                for f in pending:
                    tail = f"，目标第{f.get('target_chapter')}章" if f.get("target_chapter") else ""
                    line = f"- {f.get('item', '')}（第{f.get('planted_chapter', '?')}章埋设{tail}）"
                    if used + len(line) > fs_q:
                        break
                    fs_lines.append(line)
                    used += len(line)
                if fs_lines:
                    blocks.append("【待回收伏笔（本章应尽量回收或推进）】\n" + "\n".join(fs_lines))
        return "\n\n".join(blocks)

    def foreshadowing_recovery_warning(self) -> str:
        """伏笔回收率告警：回收率低于 70% 时返回警告文本"""
        ledger = self.vs.load_extra_data("state_ledger", {}) or {}
        fs = ledger.get("foreshadowing", [])
        if not fs:
            return ""
        total = len(fs)
        done = sum(1 for f in fs if isinstance(f, dict) and f.get("status") == "已回收")
        rate = done / total
        if rate < 0.7:
            return f"伏笔回收率仅 {done}/{total}（{rate:.0%}），后期章节建议优先回收伏笔"
        return ""

    @staticmethod
    def merge_ledger(old: dict, delta: dict, chapter_num: int = None) -> dict:
        """合并状态台账增量（纯函数）：
        - characters: 按 name 覆盖更新（附加 updated_chapter 记录最后活跃章，供活跃度排序）
        - timeline: 追加（按 chapter 去重）
        - foreshadowing: 按 item 覆盖更新（AI 回报"已回收"时更新状态）
        """
        merged = {"characters": [], "timeline": [], "foreshadowing": []}
        merged.update({k: v for k, v in old.items() if k in merged})

        if isinstance(delta.get("characters"), list):
            by_name = {c.get("name"): c for c in merged["characters"] if isinstance(c, dict)}
            for c in delta["characters"]:
                if isinstance(c, dict) and c.get("name"):
                    upd = {**by_name.get(c["name"], {}), **c}
                    if chapter_num is not None:
                        upd["updated_chapter"] = chapter_num
                    by_name[c["name"]] = upd
            merged["characters"] = list(by_name.values())

        if isinstance(delta.get("timeline"), list):
            seen = {t.get("chapter") for t in merged["timeline"] if isinstance(t, dict)}
            for t in delta["timeline"]:
                if isinstance(t, dict) and t.get("chapter") not in seen:
                    merged["timeline"].append(t)
                    seen.add(t.get("chapter"))

        if isinstance(delta.get("foreshadowing"), list):
            by_item = {f.get("item"): f for f in merged["foreshadowing"] if isinstance(f, dict)}
            for f in delta["foreshadowing"]:
                if isinstance(f, dict) and f.get("item"):
                    by_item[f["item"]] = {**by_item.get(f["item"], {}), **f}
            merged["foreshadowing"] = list(by_item.values())

        return merged

    def rebuild_memory_from_deltas(self, upto_chapter: int = None):
        """由按章 delta 顺序重建合并态 state_ledger / rolling_summary。

        设计：ledger_deltas / rolling_summaries 按章号单独存每章生成时的 LLM 产出，
        合并态只是派生物——任何一章变动都可以精确撤销其影响（删掉对应 delta 后重建），
        避免旧的"覆盖式 merge 无法撤销"导致的台账漂移。

        upto_chapter: 只重建到该章为止（含），None 表示全部。
        返回 (merged_ledger, summary_recent)。
        """
        deltas = self.vs.load_extra_data("ledger_deltas", {}) or {}
        merged = {"characters": [], "timeline": [], "foreshadowing": []}
        for k in sorted(deltas.keys(), key=lambda x: int(x)):
            if upto_chapter is not None and int(k) > upto_chapter:
                continue
            if isinstance(deltas[k], dict):
                merged = self.merge_ledger(merged, deltas[k], chapter_num=int(k))
        # 应用人工修正层（rebuild 后调用，防 AI 重建覆盖人工修改）
        merged = self._apply_ledger_fixes(merged)
        self.vs.save_extra_data("state_ledger", merged)

        # 双摘要通道分别重建：rolling_summaries(近期) / rolling_summaries_full(全书梗概)
        def _latest_snapshot(key: str, save_key: str) -> str:
            snaps = self.vs.load_extra_data(key, {}) or {}
            out, best = "", -1
            for k in snaps:
                kn = int(k)
                if upto_chapter is not None and kn > upto_chapter:
                    continue
                if kn > best:
                    best, out = kn, snaps[k]
            if out:
                self.vs.save_extra_data(save_key, out)
            elif best < 0:
                self.vs.save_extra_data(save_key, "")
            return out

        recent = _latest_snapshot("rolling_summaries", "rolling_summary_recent")
        full = _latest_snapshot("rolling_summaries_full", "rolling_summary_full")
        # 兼容旧数据：只有旧 rolling_summary（全书压缩）时，作为近期摘要兜底
        if not recent:
            legacy = self.vs.load_extra_data("rolling_summary", "") or ""
            if legacy:
                recent = legacy
        logger.info(f"记忆重建完成: 台账角色{len(merged['characters'])}条, 梗概{len(full)}字, 近期{len(recent)}字")
        return merged, recent

    def _apply_ledger_fixes(self, merged: dict) -> dict:
        """应用人工修正层（ledger_manual_fixes）：
        - characters: {名字: {字段: 覆盖值}} 覆盖角色状态
        - foreshadowing: {伏笔: {status: "已回收"}} 标记回收
        每次 rebuild 后应用——人工修正永远胜过 AI 重建结果。
        """
        fixes = self.vs.load_extra_data("ledger_manual_fixes", {}) or {}
        chars = fixes.get("characters") or {}
        if isinstance(chars, dict) and chars:
            by_name = {c.get("name"): c for c in merged["characters"] if isinstance(c, dict)}
            for name, patch in chars.items():
                if isinstance(patch, dict) and name in by_name:
                    by_name[name] = {**by_name[name], **patch}
            merged["characters"] = list(by_name.values())
        fs = fixes.get("foreshadowing") or {}
        if isinstance(fs, dict) and fs:
            by_item = {f.get("item"): f for f in merged["foreshadowing"] if isinstance(f, dict)}
            for item, patch in fs.items():
                if isinstance(patch, dict) and item in by_item:
                    by_item[item] = {**by_item[item], **patch}
            merged["foreshadowing"] = list(by_item.values())
        return merged

    def apply_ledger_fix(self, fix_type: str, key: str, patch: dict) -> dict:
        """写入人工修正层并立即应用（防 AI 重建覆盖人工修改）。

        fix_type: "character"（key=角色名，patch 覆盖 status 等）| "foreshadowing"（key=伏笔，patch 如 {"status": "已回收"}）
        返回应用后的合并态。
        """
        fixes = self.vs.load_extra_data("ledger_manual_fixes", {}) or {}
        section = "characters" if fix_type == "character" else "foreshadowing"
        fixes.setdefault(section, {})
        fixes[section][key] = {**(fixes[section].get(key) or {}), **patch}
        self.vs.save_extra_data("ledger_manual_fixes", fixes)
        merged, _ = self.rebuild_memory_from_deltas()
        logger.info(f"人工修正已应用: {fix_type}={key} {patch}")
        return merged

    def clear_memory(self, deep: bool = False) -> dict:
        """清空记忆：
        deep=False：只清展示层（合并态/当前摘要），保留按章快照 → 下次生成章节时自动重建；
        deep=True：彻底清空（含按章快照 ledger_deltas/摘要快照/人工修正层），真正的"失忆"。
        """
        for k in ("state_ledger", "rolling_summary", "rolling_summary_full", "rolling_summary_recent"):
            self.vs.delete_extra_field(k)
        if deep:
            for k in ("ledger_deltas", "rolling_summaries", "rolling_summaries_full",
                      "ledger_manual_fixes", "ledger_stale", "ledger_stale_from"):
                self.vs.delete_extra_field(k)
        else:
            self.vs.save_extra_data("ledger_stale", False)
        logger.info(f"记忆已清空: deep={deep}")
        return {"cleared": True, "deep": deep}

    def invalidate_memory_from(self, chapter_num: int):
        """第 chapter_num 章内容发生变动（重生成/手动编辑/导入）后的记忆失效处理：
        1. 删除该章及之后所有章的 delta 与摘要快照（它们基于旧内容产出，已不可信）；
        2. 用剩余 delta 重建合并态（旧章引入的伏笔/状态被精确撤销）；
        3. 若之后还存在已生成章节，标记 ledger_stale——这些章的 delta 缺失，
           可通过 sync_memory(regen=True) 逐章重新调 LLM 补齐（只重算缺失章，有 token 成本）。
        """
        deltas = self.vs.load_extra_data("ledger_deltas", {}) or {}
        summaries = self.vs.load_extra_data("rolling_summaries", {}) or {}
        summaries_full = self.vs.load_extra_data("rolling_summaries_full", {}) or {}
        deltas = {k: v for k, v in deltas.items() if int(k) < chapter_num}
        summaries = {k: v for k, v in summaries.items() if int(k) < chapter_num}
        summaries_full = {k: v for k, v in summaries_full.items() if int(k) < chapter_num}
        self.vs.save_extra_data("ledger_deltas", deltas)
        self.vs.save_extra_data("rolling_summaries", summaries)
        self.vs.save_extra_data("rolling_summaries_full", summaries_full)
        self.rebuild_memory_from_deltas(upto_chapter=chapter_num - 1)

        chapters = self.novel_info.get("chapters", {})
        later = [int(k) for k in chapters if str(k).isdigit() and int(k) > chapter_num
                 and (chapters[k].get("content") or "").strip()]
        if later:
            self.vs.save_extra_data("ledger_stale", True)
            self.vs.save_extra_data("ledger_stale_from", chapter_num + 1)
            logger.info(f"第{chapter_num}章变动，第{min(later)}章起的台账 delta 标记为 stale")
        else:
            self.vs.save_extra_data("ledger_stale", False)

    def missing_memory_chapters(self) -> list:
        """有正文但缺少按章记忆记录的章节（账本过期/缺失章）。免费计算，不调用 API。

        按章 delta 是每章 AI 记的"原始账页"；章节被改写/编辑/导入后旧账页会被作废
        （见 invalidate_memory_from），此处即找出"有正文但没账页"的章——这些章
        必须让 AI 重新读取正文重写账页才能补齐。
        """
        chapters = self.novel_info.get("chapters", {})
        deltas = self.vs.load_extra_data("ledger_deltas", {}) or {}
        return sorted(int(k) for k in chapters
                      if str(k).isdigit()
                      and (chapters[k].get("content") or "").strip()
                      and str(k) not in deltas)

    def sync_memory(self, regen: bool = False, all_chapters: bool = False, max_tokens: int = 1500) -> dict:
        """同步账本（UI「同步账本」按钮的底层逻辑）。

        regen=False（默认，零成本）：只用已有按章记录重算合并态/摘要展示层，
        不调用 AI；缺失记录的章保持缺失状态（返回 missing 供前端提示）。
        regen=True（烧 token，UI 需明示成本）：只重算"有正文但缺按章记录"的章
        （见 missing_memory_chapters）；all_chapters=True 时从第 1 章全量重算。
        全部补齐后自动清除过期标记。
        返回 {"regenerated": 重算章数, "ledger": 合并台账, "summary": 近期摘要,
              "missing": 仍缺失的章}
        """
        chapters = self.novel_info.get("chapters", {})
        if not regen:
            merged, summary = self.rebuild_memory_from_deltas()
            return {"regenerated": 0, "ledger": merged, "summary": summary,
                    "missing": self.missing_memory_chapters()}

        if all_chapters:
            targets = sorted(int(k) for k in chapters
                             if str(k).isdigit() and (chapters[k].get("content") or "").strip())
        else:
            targets = self.missing_memory_chapters()
        if targets:
            # 只清目标章的旧账页（其他章账页原样保留），逐章重新调 LLM 产出
            deltas = {k: v for k, v in (self.vs.load_extra_data("ledger_deltas", {}) or {}).items()
                      if int(k) not in targets}
            self.vs.save_extra_data("ledger_deltas", deltas)
            for snap_key in ("rolling_summaries", "rolling_summaries_full"):
                snaps = {k: v for k, v in (self.vs.load_extra_data(snap_key, {}) or {}).items()
                         if int(k) not in targets}
                self.vs.save_extra_data(snap_key, snaps)
            self.rebuild_memory_from_deltas()
            for n in targets:
                self._report(f"正在重建第{n}章的账本与摘要（{targets.index(n) + 1}/{len(targets)}）…")
                content = chapters[str(n)]["content"]
                self.update_memory(n, content, max_tokens=max_tokens)
        if not self.missing_memory_chapters():
            self.vs.save_extra_data("ledger_stale", False)
            self.vs.save_extra_data("ledger_stale_from", None)
        merged, summary = self.rebuild_memory_from_deltas()
        return {"regenerated": len(targets), "ledger": merged, "summary": summary,
                "missing": self.missing_memory_chapters()}

    @staticmethod
    def _chapter_excerpt(content: str, head: int = 2000, tail: int = 4000) -> str:
        """取章节"前 head 字 + 末 tail 字"拼接（总长约 6000 字，成本可控）。

        取舍说明：长章（可调字数 + 自动续写）常超 4000 字，只取开头会系统性漏记
        章末的伏笔/钩子；全文分段摘要合并成本高一倍。头尾拼接用一次调用覆盖
        章节开局与结尾，中间段的细节允许遗漏（多为过程描写）。
        """
        if len(content) <= head + tail:
            return content
        return content[:head] + "\n……（中间省略）……\n" + content[-tail:]

    def update_memory(self, chapter_num: int, content: str, max_tokens: int = 2500) -> dict:
        """章节生成后**一次调用**更新全部记忆（原台账+摘要两次调用合并，每章省一次）。

        单 prompt 产出 delta + 全书梗概 + 近期摘要（B2/B6）：
        - B3 结构化喂料：全部角色名 + 近 15 章活跃角色状态（消灭旧 2000 字截断盲区）
        - B4 失败自动重试 1 次（带格式纠正），仍失败置 ledger_stale 并告警（不再静默）
        返回 {"delta": ..., "full": ..., "recent": ..., "ok": bool}
        """
        ledger = self.vs.load_extra_data("state_ledger", {}) or {}
        old_full = self.vs.load_extra_data("rolling_summary_full", "") or \
            self.vs.load_extra_data("rolling_summary", "") or ""
        old_recent = self.vs.load_extra_data("rolling_summary_recent", "") or ""

        char_names = "、".join(
            c.get("name", "") for c in ledger.get("characters", [])
            if isinstance(c, dict) and c.get("name")) or "（暂无角色）"
        active_lines = []
        for c in ledger.get("characters", []):
            if isinstance(c, dict) and c.get("name"):
                upd = int(c.get("updated_chapter") or 0)
                if upd >= chapter_num - 15:
                    active_lines.append(f"- {c['name']}: {c.get('status', '')}")
        old_brief = "\n".join(active_lines) or "（空）"
        ledger_full_brief = json.dumps(ledger, ensure_ascii=False)[:1500]

        prompt = f"""你是小说设定管理助手。请阅读最新一章正文，一次性产出三个结果（只输出合法 JSON，不要输出其他内容）：

{{
  "delta": {{
    "characters": [{{"name": "角色名", "status": "本章后的状态变化（位置/伤势/关系/实力等，无变化则不列出）"}}],
    "timeline": [{{"chapter": {chapter_num}, "event": "本章核心事件一句话"}}],
    "foreshadowing": [
      {{"item": "伏笔内容", "planted_chapter": {chapter_num}, "target_chapter": null, "status": "未回收"}}
    ]
  }},
  "full_summary": "全书剧情梗概（≤300字）：从第1章到现在的整体主线，早期关键设定/主角目标必须长期保留，不随章节增多而丢失",
  "recent_summary": "近期剧情摘要（≤500字）：最近约10章（第 {max(1, chapter_num - 10)} 章至今）的详细剧情，保留事件细节与状态变化"
}}

【全部登场角色名单（角色状态如有变化必须列出；未提及的视为无变化）】
{char_names}

【角色当前状态（近15章内活跃角色）】
{old_brief}

【既有伏笔与时间线（参考）】
{ledger_full_brief[:800]}

【已有全书梗概】
{old_full or "（空）"}

【已有近期摘要】
{old_recent or "（空）"}

【第 {chapter_num} 章正文（头尾节选）】
{self._chapter_excerpt(content)}

注意：
- delta 只报告本章**新增或发生变化**的条目：角色无变化不要列出；伏笔本章新埋标记"未回收"，本章回收旧伏笔也要列出并标记"已回收"
- full_summary 是全书级梗概：早期内容必须保留要点，不要因为后续章节多就把开头删掉
- recent_summary 聚焦最近章节细节，与 full_summary 分工互补
- 必须只输出 JSON"""

        for attempt in range(2):
            try:
                result = self.api.generate(prompt, step="consistency", temperature=0.2, max_tokens=max_tokens)
                if is_ai_refusal(result):
                    break
                m = re.search(r"\{.*\}", result, re.DOTALL)
                if not m:
                    raise ValueError("输出中未找到 JSON 对象")
                data = json.loads(m.group(0))
                delta = data.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ValueError("delta 不是对象")
                # 按章存 delta（可追溯/可撤销），合并态由全部 delta 顺序重建
                deltas = self.vs.load_extra_data("ledger_deltas", {}) or {}
                deltas[str(chapter_num)] = delta
                self.vs.save_extra_data("ledger_deltas", deltas)
                merged, _ = self.rebuild_memory_from_deltas()
                # 双摘要分通道存储（含按章快照，供失效重建）
                full = (data.get("full_summary") or "").strip()
                recent = (data.get("recent_summary") or "").strip()
                if full:
                    snaps = self.vs.load_extra_data("rolling_summaries_full", {}) or {}
                    snaps[str(chapter_num)] = full
                    self.vs.save_extra_data("rolling_summaries_full", snaps)
                    self.vs.save_extra_data("rolling_summary_full", full)
                if recent:
                    snaps = self.vs.load_extra_data("rolling_summaries", {}) or {}
                    snaps[str(chapter_num)] = recent
                    self.vs.save_extra_data("rolling_summaries", snaps)
                    self.vs.save_extra_data("rolling_summary_recent", recent)
                self.vs.save_extra_data("ledger_stale", False)
                logger.info(f"记忆已更新: 角色{len(merged['characters'])} 伏笔{len(merged['foreshadowing'])} 梗概{len(full)}字 近期{len(recent)}字")
                return {"delta": delta, "full": full, "recent": recent, "ok": True}
            except Exception as e:
                logger.warning(f"第{chapter_num}章记忆更新第 {attempt + 1} 次失败: {e}")
                if attempt == 0:
                    prompt += "\n\n【注意】你上次的输出解析失败（不是合法 JSON 或结构不对）。请严格按给定 JSON 结构输出，不要多解释。"
        # 重试仍失败：标记 stale，交 UI 提示（不再静默吞掉）
        self.vs.save_extra_data("ledger_stale", True)
        if not self.vs.load_extra_data("ledger_stale_from", None):
            self.vs.save_extra_data("ledger_stale_from", chapter_num)
        logger.warning(f"第{chapter_num}章记忆更新失败，已标记待重建")
        return {"delta": {}, "full": "", "recent": "", "ok": False}

    def update_state_ledger(self, chapter_num: int, content: str, max_tokens: int = 1500) -> dict:
        """兼容薄封装：仅更新台账部分（内部走 update_memory，保持旧返回类型）"""
        r = self.update_memory(chapter_num, content, max_tokens=max_tokens)
        return self.vs.load_extra_data("state_ledger", {}) or {}

    def update_rolling_summary(self, chapter_num: int, content: str, max_tokens: int = 1000) -> str:
        """兼容薄封装：仅更新摘要部分（内部走 update_memory，保持旧返回类型）"""
        r = self.update_memory(chapter_num, content, max_tokens=max_tokens)
        return r.get("recent") or ""

    def generate_chapter_beats(self, chapter_num: int, chapter_title: str, target_words: int = 2000, max_tokens: int = 2000) -> str:
        """生成章节细纲（场景卡）：3-6 个场景节拍，供用户编辑确认后按场景生成正文

        节拍只锚定本章：**不注入**全书走向/世界观/人物弧线/台账（它们描述的是整个故事设计，
        写第一章不该带进后续章节内容），只给本章那一行逐章概要 + 本章登场角色名 + 阶段叙事指令。
        """
        # 前置保障：确保本章所属卷已有逐章概要，否则提取不到章节标记会兜底返回整本书卷级概览
        self.ensure_outline_for_chapter(chapter_num)
        if not target_words or target_words <= 0:
            target_words = self.resolve_target_words(chapter_num)
        # 标题为空时从大纲补齐，确保场景节拍对齐到具体章节
        if not chapter_title.strip():
            chapter_title = self.get_outline_chapter_titles().get(chapter_num, "")
        n_beats = min(6, max(3, target_words // 800))

        # 本章那一行逐章概要（唯一大纲依据）：spoiler_level=strict 剔除顶部总述（含全部后续卷剧情），
        # capture_range=0 只保留本章条目
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        chapter_anchor = self._extract_relevant_outline(
            outline, chapter_num, capture_range=0, spoiler_level="strict") if outline else ""

        # 本章登场角色：只注入名字（含主/配角标注），不注入身份/人物关系/备注弧线，防止把后续剧情带进来
        char_names = ""
        cards = self.load_character_cards()
        if cards:
            active, _absent = cc.filter_cards_for_chapter(cards, chapter_num)
            if active:
                parts = []
                for c in active:
                    role_label = "主角" if c.get("role") == "main" else "配角"
                    parts.append(f"{c.get('name', '')}（{role_label}）")
                char_names = "、".join(parts)

        # 当前阶段叙事指令（开篇严禁抢跑等）
        total_chapters = self._estimate_total_chapters()
        phase_config = self._classify_chapter_phase(chapter_num, total_chapters)

        prompt = f"""你是一位专业的小说结构编辑。请为第 {chapter_num} 章 "{chapter_title}" 设计**场景细纲**（不是正文）。

【本章登场角色】
{char_names or "（暂无）"}

【本章大纲（唯一依据，场景节拍只围绕本章，不得覆盖后续章节）】
{chapter_anchor or "（暂无本章大纲条目）"}

{phase_config["pacing_instruction"]}

【要求】
- 把本章拆成 {n_beats} 个场景节拍，每个场景有可识别的地点/人物/冲突推进
- 场景节拍内容**严格限定在第 {chapter_num} 章「{chapter_title}」概要描述的事件范围内**，只展开本章该发生的事
- **严禁抢跑**：不要把后续章节才该出现的冲突、转折、角色提前放进本章节拍。本章节拍只覆盖第 {chapter_num} 章本身，绝不能把第 {chapter_num + 1} 章及以后的内容写进来
- 每个场景按以下格式输出（严格遵守，方便程序解析）：

## 场景1：场景名
- 地点与出场人物：...
- 核心冲突/事件：...
- 情绪走向：...（如 压抑→爆发）
- 结尾钩子/进展：...（这个场景结束时留下的悬念或状态变化）

- 场景之间要有递进，最后一个场景的钩子要让人想点开下一章
- 直接输出场景卡，不要解释

场景细纲："""
        
        beats = self.api.generate(prompt, step="outline", temperature=0.7, max_tokens=max_tokens,
                              stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
        # 节拍自动校验：生成后程序化检查各场景是否点名尚未登场的角色（实体时间锁）
        # 越界则带「逐场景定位反馈」自动重生成，最多 2 次；仍越界则保留最近结果并记录告警
        for attempt in range(3):
            if is_ai_refusal(beats):
                break
            check = self.validate_beats(beats, chapter_num)
            if check["ok"]:
                self.last_beats_warning = ""
                self.vs.save_extra_data(f"chapter_beats_{chapter_num}", beats)
                return beats
            self.last_beats_warning = f"节拍校验未通过：{'；'.join(check['issues'][:3])}"
            logger.warning(f"第{chapter_num}章节拍校验未通过（第{attempt + 1}次）: {self.last_beats_warning}")
            if attempt >= 2:
                break
            # 逐场景定位反馈：列出越界场景原文 + 具体问题 + 本章允许的角色名单
            fb_parts = []
            for sc in check.get("scene_feedback", []):
                probs = "；".join(sc.get("problems", []))
                fb_parts.append(
                    f"场景「{sc.get('head', '')}」原文：\n{sc.get('text', '')[:300]}\n"
                    f"问题：{probs}")
            allowed = f"本章允许出现的角色：{char_names or '（本章大纲中的人物）'}。"
            beats = self.api.generate(
                prompt + f"\n\n【校验反馈（逐场景定位）】上次输出的以下场景越界了：\n"
                         + "\n\n".join(fb_parts[:5])
                         + f"\n\n{allowed}请把越界场景改写为本章范围内的事件（不得点名未登场角色），重新输出完整场景卡。",
                step="outline", temperature=0.7, max_tokens=max_tokens,
                stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
        if not is_ai_refusal(beats):
            self.vs.save_extra_data(f"chapter_beats_{chapter_num}", beats)
        return beats

    def validate_beats(self, beats_text: str, chapter_num: int) -> dict:
        """程序化校验节拍是否越界：解析各场景，检查是否点名尚未登场的角色/代号。

        返回 {"ok": bool, "issues": [..], "scene_feedback": [{head, text, problems}]}
        issues 为空即 ok。这是「实体时间锁」在节拍侧的落地：
        纯程序字符串比对，**不需要任何 LLM 介入**——模型输出再自律，
        也比不上写完后逐场景对一遍角色注册表。
        """
        issues = []
        scene_feedback = []
        registry = self._build_entity_registry()
        for beat in self.parse_beats(beats_text or ""):
            head = beat.strip().split("\n", 1)[0].strip()[:24]
            problems = []
            for name, appear in registry.items():
                if appear > chapter_num and name in beat:
                    problems.append(f"出现未登场角色「{name}」（第{appear}章才登场）")
            if problems:
                issues.append(f"场景「{head}」出现未登场角色" + "；".join(p for p in problems))
                scene_feedback.append({"head": head, "text": beat.strip()[:300], "problems": problems})
        return {"ok": not issues, "issues": issues, "scene_feedback": scene_feedback}
    
    def review_chapter(self, chapter_num: int, chapter_title: str, content: str, max_tokens: int = 2500) -> str:
        """AI 评审章节：多维度评分 + 问题清单 + 修改建议（商业化质量标准）"""
        total = self._estimate_total_chapters()
        pc = self._classify_chapter_phase(chapter_num, total)
        plan = self.novel_info.get("volume_plan") or self.vs.load_extra_data("volume_plan", []) or []
        cur_vol = self._current_volume(plan, chapter_num)

        outline_for_chapter = ""
        outline_text = self.novel_info.get("outline", "")
        if outline_text:
            outline_for_chapter = self._extract_relevant_outline(
                outline_text, chapter_num, capture_range=pc["outline_range"],
                spoiler_level="none", current_volume=cur_vol)

        world_brief = self.novel_info.get("world_setting", "")[:2000]

        # B5：台账注入评审——角色状态/近期事件/待回收伏笔，让"设定一致性"维度能抓到
        # 正文与角色当前状态的矛盾（如正文写主角还在县城、台账说他已在魔都）
        ledger_brief = self._ledger_brief(chapter_num, pc)

        # 人物：优先只注入当前章在场角色卡（与生成一致），无卡片时回退全文摘要
        characters_brief = ""
        cards = self.load_character_cards()
        if cards:
            active, _absent = cc.filter_cards_for_chapter(cards, chapter_num)
            if active:
                characters_brief = cc.cards_to_text(active)[:2000]
        if not characters_brief:
            characters_brief = self.novel_info.get("characters", "")[:2000]

        prompt = f"""你是一位极其严格的网文/出版编辑。请评审第 {chapter_num} 章 "{chapter_title}" 的正文质量。

【世界观设定（摘要）】
{world_brief}

【人物设定（摘要）】
{characters_brief}

【角色状态与剧情台账（一致性对照依据）】
{ledger_brief or "（暂无台账）"}

【本章大纲】
{outline_for_chapter[:1500]}

【章节正文】
{content}

请按以下格式输出评审报告（严格遵守格式）：

## 总分：X/10

## 维度评分
- 钩子与悬念：X/10 —— 一句话说明
- 节奏与信息量：X/10 —— 一句话说明
- 对话质量：X/10 —— 一句话说明
- 设定一致性：X/10 —— 一句话说明（对照人物设定和大纲，指出矛盾）
- 文字表现力：X/10 —— 一句话说明（是否有 AI 腔/套话/流水账）

## 问题清单
1. （最严重的问题，引用原文句子）
2. ...

## 修改建议
1. （每条建议要具体可执行，说明改哪里、怎么改）
2. ...

评审标准参考：商业化连载小说要求开头 200 字内抓住读者、每章结尾有钩子、对话推动剧情而非灌水、设定零矛盾、避免"不禁/嘴角勾起/空气仿佛凝固"类 AI 套话。"""
        
        review = self.api.generate(prompt, step="consistency", temperature=0.3, max_tokens=max_tokens)
        return review
    
    def revise_chapter(self, chapter_num: int, chapter_title: str, content: str, review: str, max_tokens: int = 2000) -> str:
        """根据评审意见改写章节正文"""
        # 输出长度需覆盖原文
        needed = min(int(len(content) * 1.8) + 500, self.api.MAX_TOKENS_LIMIT)
        max_tokens = max(max_tokens, needed)
        
        prompt = f"""你是一位资深小说编辑。请根据评审意见改写第 {chapter_num} 章 "{chapter_title}" 的正文。

【评审意见】
{review}

【原正文】
{content}

【改写要求】
- 逐条落实评审意见中的修改建议
- 保持核心情节和人物不变，只改进表达、节奏、对话质量
- 改写后总字数与原文相当（不少于原文的 85%）
- 直接输出改写后的正文，不要解释

改写后正文："""
        
        return self.api.generate(prompt, step="chapter", temperature=0.7, max_tokens=max_tokens, stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
    
    def generate_golden_chapter(self, chapter_num: int, chapter_title: str, max_tokens: int = 2500,
                                target_words: int = 2000, beats: str = "") -> dict:
        """黄金开篇专项：生成两个版本 → 分别评审 → 选总分更高者。
        返回 {"content": 最佳版本, "review": 最佳评审, "alt_content": 另一版, "alt_review": 另一版评审}
        """
        if not chapter_title.strip():
            # 空标题先统一拟一次，保证两个版本共用同一标题
            chapter_title = (self.get_outline_chapter_titles().get(chapter_num, "")
                             or self.generate_chapter_title(chapter_num) or f"第{chapter_num}章")
            logger.info(f"黄金开篇：第{chapter_num}章标题为空，AI 拟定为「{chapter_title}」")
        v1 = self.generate_chapter(chapter_num, chapter_title, max_tokens=max_tokens,
                                   target_words=target_words, beats=beats, temperature=0.8)
        self._report("黄金开篇：版本A完成，正在生成版本B…")
        v2 = self.generate_chapter(chapter_num, chapter_title, max_tokens=max_tokens,
                                   target_words=target_words, beats=beats, temperature=0.95)
        self._report("黄金开篇：两个版本已完成，正在评审…")
        r1 = self.review_chapter(chapter_num, chapter_title, v1)
        r2 = self.review_chapter(chapter_num, chapter_title, v2)
        
        def _score(review: str) -> float:
            m = re.search(r"总分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10", review)
            return float(m.group(1)) if m else 5.0
        
        if _score(r2) > _score(r1):
            return {"content": v2, "review": r2, "alt_content": v1, "alt_review": r1,
                    "picked": 2, "scores": (_score(r1), _score(r2))}
        return {"content": v1, "review": r1, "alt_content": v2, "alt_review": r2,
                "picked": 1, "scores": (_score(r1), _score(r2))}
    
    def _match_chapter_num(self, line: str) -> int:
        """识别一行是否为章节标题行，返回章节号；不是则返回 0。
        
        兼容格式：第1章 / 第 1 章 / 第一章 / 第 一 章 / 1章 / - 第1章 / **第1章** /
        【第1章】/ ### 第1章 / (第1章) 等常见变体。
        """
        s = line.strip()
        if not s:
            return 0
        s = re.sub(r"^#{1,6}\s*", "", s)            # markdown 标题
        s = re.sub(r"^[-\*]\s+", "", s)             # 列表前缀
        s = re.sub(r"^\*{1,3}", "", s)              # 加粗开头
        s = re.sub(r"^[【\[（(]+\s*", "", s)          # 括号开头
        m = re.match(r"第\s*(\d+|[一二三四五六七八九十百零〇]+)\s*章", s)
        if m:
            return self._parse_num(m.group(1))
        m = re.match(r"(\d+)\s*章[\s：:：\-—–]", s)  # "1章 xxx"（行首"第"缺失的变体）
        if m:
            return int(m.group(1))
        return 0

    # 大纲章节条目尾缀的目标字数标记，如「（约2000字）」（闭括号允许缺失，兼容上游 strip）
    _WORDS_SUFFIX_RE = re.compile(r"[（(]\s*约?\s*(\d{3,6})\s*字\s*[)）]?\s*$")

    def get_outline_chapter_words(self) -> Dict[int, int]:
        """从大纲章节条目解析每章目标字数（条目尾缀「（约N字）」），返回 {章节号: 字数}"""
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        words = {}
        for line in outline.split("\n"):
            if not line.strip():
                continue
            num = self._match_chapter_num(line)
            if num <= 0:
                continue
            m = self._WORDS_SUFFIX_RE.search(line.strip())
            if m:
                words[num] = int(m.group(1))
        logger.info(f"get_outline_chapter_words: 从大纲解析出 {len(words)} 个章节字数")
        return words

    def _resolve_global_words_per_chapter(self, fallback: int = 2000) -> int:
        """解析全局每章基准字数（outline_words_per_chapter），失败回退 fallback"""
        try:
            saved = (self.novel_info.get("outline_words_per_chapter")
                     or self.vs.load_extra_data("outline_words_per_chapter", "") or "")
            if str(saved).strip():
                return int(saved)
        except (ValueError, TypeError):
            pass
        return fallback

    def resolve_target_words(self, chapter_num: int, fallback: int = 2000) -> int:
        """章节目标字数解析链：大纲逐章字数 → 大纲全局每章字数 → fallback"""
        w = self.get_outline_chapter_words().get(chapter_num)
        if w:
            return w
        return self._resolve_global_words_per_chapter(fallback)

    def generate_chapter_title(self, chapter_num: int, max_tokens: int = 100) -> str:
        """章节标题缺失时由 AI 根据大纲与剧情摘要拟定标题（小 token 调用）"""
        outline = self.novel_info.get("outline", "") or self.vs.get_section("outline", "full_outline") or ""
        excerpt = self._extract_relevant_outline(outline, chapter_num, capture_range=2,
                                                 spoiler_level="none") if outline else ""
        rolling = self.vs.load_extra_data("rolling_summary", "") or ""
        prompt = f"""请为小说第 {chapter_num} 章拟一个章节标题。

【相邻章节大纲】
{excerpt or "（暂无）"}

【已生成剧情摘要】
{rolling[:800] or "（暂无）"}

【要求】
- 2-12 个字，概括本章核心事件或意象
- 不要带"第X章"前缀，不要带书名号、引号或结尾标点
- 只输出标题本身，不要解释

标题："""
        try:
            raw = self.api.generate(prompt, step="outline", temperature=0.7, max_tokens=max_tokens)
        except Exception as e:
            logger.warning(f"第{chapter_num}章 AI 拟题失败: {e}")
            return ""
        title = (raw or "").strip().split("\n")[0].strip()
        title = re.sub(r"^第\s*\d+\s*章\s*[：:]?\s*", "", title).strip("\"'「」《》<>。，, ")
        if not title or len(title) > 30 or is_ai_refusal(title):
            logger.warning(f"第{chapter_num}章 AI 拟题结果异常（{raw!r}），放弃")
            return ""
        return title

    def get_outline_chapter_titles(self) -> Dict[int, str]:
        """从大纲中解析出所有章节标题，返回 {章节号: 标题} 字典
        
        支持多种格式：第1章 xxx、第一章 xxx、1. xxx、【第1章】xxx 等
        兼容 markdown 列表（- / *）、加粗（**）等前缀
        """
        outline = self.novel_info.get("outline", "")
        if not outline:
            logger.debug("get_outline_chapter_titles: novel_info 中无大纲数据")
            return {}
        
        # 调试日志：输出大纲前15行，方便排查格式问题
        preview_lines = [l for l in outline.split("\n")[:15] if l.strip()]
        # logger.info(f"get_outline_chapter_titles: 大纲总长={len(outline)}字，前15行预览:")
        # for i, l in enumerate(preview_lines):
        #     logger.info(f"  [{i}] {l[:120]}")
        
        titles = {}
        lines = outline.split("\n")
        
        # 先检测大纲中是否包含"第X章"/"第X节"等标记
        has_marker = any(
            re.search(r"[第\d章节]", l) or re.match(r"\s*[\d]+[\.\、．\)]", l)
            for l in lines
        )
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            raw = line_stripped
            
            # 去除常见的行首标记：markdown 标题/列表、加粗等
            cleaned = re.sub(r"^#{1,6}\s*", "", line_stripped)      # "### " 标题
            cleaned = re.sub(r"^[-\*]\s+", "", cleaned)      # "- " 或 "* " 开头
            cleaned = re.sub(r"^\*{1,3}\s*", "", cleaned)           # "**" 加粗开头
            cleaned = re.sub(r"^\[?\[?第?", "", cleaned)             # 【【 第 开头
            
            # 格式1: 第1章 标题 / 第 1 章：标题 / 第一章 标题 / 第 1 章:标题
            #   数字前后可能有空格，冒号可能是全角或半角，也可能没有冒号
            match = re.match(
                r"第\s*(\d+|[一二三四五六七八九十百零〇]+)\s*章\s*[：:\-—–]*\s*(.+)", 
                cleaned
            )
            if match:
                num_str = match.group(1)
                title = match.group(2).strip()
                num = self._parse_num(num_str)
                if num > 0 and title:
                    titles[num] = title
                continue
            
            # 格式1b: 1 章 标题 / 1 章:标题 — 行首"第"被清理步骤去掉后的格式
            #         对应原始格式如 "第 1 章：穿越华国——..."
            match = re.match(r"\s*(\d+)\s*章\s*[：:\-—–]*\s*(.+)", cleaned)
            if match:
                num = int(match.group(1))
                title = match.group(2).strip()
                if num > 0 and title and len(title) >= 2:
                    titles[num] = title
                    continue
            
            # 格式2: 1. 标题 / 1、标题 / 1）标题 / (1) 标题
            match = re.match(r"(?:^|\()(\d+)\s*[\.、．\)\]]\s*(.+)", cleaned)
            if match:
                num = int(match.group(1))
                title = match.group(2).strip()
                if num > 0 and title and len(title) >= 2:
                    titles[num] = title
                    continue
            
            # 格式3: 章一 标题 / 章一：标题 (较少见但有些AI会用)
            match = re.match(r"章\s*([一二三四五六七八九十百零〇]+)\s*[：:：]?\s*(.+)", cleaned)
            if match:
                num = self._cn_num_to_int(match.group(1))
                title = match.group(2).strip()
                if num > 0 and title:
                    titles[num] = title
                    continue
            
            # 格式4: 【第 1 章】标题 / （第1章）标题 — 括号包裹的完整章节标记
            match = re.search(r"[【\(（](第?\s*\d+\s*章)[\)】】]\s*[：:\-—–]*\s*(.+)", raw)
            if match:
                num_match = re.search(r"(\d+)", match.group(1))
                if num_match:
                    num = int(num_match.group(1))
                    title = match.group(2).strip().rstrip("】）)")
                    if num > 0 and title:
                        titles[num] = title
                        continue
        
        # 剥离大纲条目尾缀的目标字数标记（如「（约2000字）」），避免混进标题
        titles = {k: (self._WORDS_SUFFIX_RE.sub("", v).strip().rstrip("，,—–- ") or v)
                  for k, v in titles.items()}
        logger.info(f"get_outline_chapter_titles: 从大纲解析出 {len(titles)} 个章节标题")
        if titles:
            sample_keys = sorted(titles.keys())[:8]
            sample = {k: f"{titles[k][:30]}..." if len(titles[k])>30 else titles[k] for k in sample_keys}
            # logger.info(f"章节标题示例: {sample}")
        elif has_marker:
            logger.warning("get_outline_chapter_titles: 大纲中疑似有章节标记但未能解析任何标题！"
                         "请检查大纲实际格式是否与预期不同")
        
        return titles
    
    @staticmethod
    def _parse_num(s: str) -> int:
        """尝试将字符串解析为整数，支持阿拉伯数字和中文数字"""
        try:
            return int(s)
        except ValueError:
            pass
        cn_digits = {"零":0,"〇":0,"一":1,"二":2,"三":3,"四":4,"五":5,
                     "六":6,"七":7,"八":8,"九":9,"十":10,"百":100}
        result = 0
        current = 0
        for ch in s:
            val = cn_digits.get(ch, None)
            if val is None:
                return 0
            if val == 10 or val == 100:
                if current == 0:
                    current = 1
                result += current * val
                current = 0
            else:
                current = val
        result += current
        return result

    def _cn_num_to_int(self, cn_str: str) -> int:
        """中文数字转整数，支持到999"""
        cn_digits = {"零":0,"〇":0,"一":1,"二":2,"三":3,"四":4,"五":5,
                     "六":6,"七":7,"八":8,"九":9,"十":10,"百":100}
        if not cn_str:
            return 0
        # 简单情况：单字
        if len(cn_str) == 1:
            return cn_digits.get(cn_str, 0)
        # 处理如：十二(12)、二十(20)、二十三(23)、一百(100)、一百零五(105)
        result = 0
        current = 0
        for ch in cn_str:
            val = cn_digits.get(ch, 0)
            if val == 10 or val == 100:
                if current == 0:
                    current = 1
                result += current * val
                current = 0
            else:
                current = val
        result += current
        return result

    def _check_chapter_scope(self, chapter_text: str, chapter_num: int, outline_for_chapter: str, spoiler_level: str) -> str:
        """章节生成后的轻量级范围校验：检查生成内容是否可能超出了当前大纲范围
        
        检测方式：
        1. 如果生成内容中出现了"后续章节标题"中的关键词，可能是抢跑
        2. 如果 spoiler_level 较严格但内容中出现了"最终/结局"等词汇，可能是泄露
        
        返回：警告信息字符串（为空表示无问题）
        """
        warnings = []
        
        # 检查1：生成内容是否引用了后续章节的标题关键词
        outline = self.novel_info.get("outline", "")
        if outline:
            all_titles = self.get_outline_chapter_titles()
            for num, title in all_titles.items():
                if num <= chapter_num:
                    continue  # 只检查后续章节
                # 提取标题中的关键词（2字以上的实词）
                title_keywords = [w for w in re.findall(r'[\u4e00-\u9fff]{2,}', title) if len(w) >= 2]
                for kw in title_keywords:
                    if kw in chapter_text and kw not in outline_for_chapter:
                        # 关键词出现在正文中但不在我方提供的大纲范围内
                        # 排除常见通用词（避免误报）
                        generic_words = {"之后", "然后", "但是", "因为", "所以", "虽然", "如果", "已经", "可以", "不是", "就是", "还是", "之后", "发现", "出现", "开始", "准备", "终于", "突然", "决定", "选择", "离开", "回来", "回去", "之间", "成为", "关于", "对于"}
                        if kw not in generic_words:
                            warnings.append(f"疑似引用后续章节(第{num}章)关键词「{kw}」")
                        break  # 每个后续章节只报一次
        
        # 检查2：对于 spoiler_level 较严格的阶段，检查是否出现了前瞻性内容
        if spoiler_level in ("strict", "moderate"):
            spoiler_indicators = ["最终战胜", "最终击败", "结局是", "最终成为", "故事结局"]
            for indicator in spoiler_indicators:
                if indicator in chapter_text:
                    warnings.append(f"内容包含前瞻性表述「{indicator}」，与当前阶段spoiler_level={spoiler_level}不符")
        
        # 限制警告数量，避免日志刷屏
        if len(warnings) > 3:
            warnings = warnings[:3] + [f"...还有{len(warnings)-3}条警告"]
        
        return "；".join(warnings) if warnings else ""

    def _current_volume(self, plan: list, chapter_num: int) -> Optional[dict]:
        """从 volume_plan 中定位当前章节所属卷。返回卷 dict 或 None。
        卷计划缺失/未匹配时返回 None（调用方回退到不按卷切分的现状）。
        """
        if not plan:
            return None
        for v in plan:
            start = v.get("start") if v.get("start") is not None else v.get("start_chapter")
            end = v.get("end") if v.get("end") is not None else v.get("end_chapter")
            if start is not None and end is not None and start <= chapter_num <= end:
                return v
        return None

    def _slice_overview_by_volume(self, overview_text: str, chapter_num: int) -> str:
        """把大纲总述中的 [卷]...[/卷] 块按当前章节卷切分：只保留「故事主线」+
        当前卷 + 前一卷块，丢弃后续卷块，避免跨卷信息泄漏到当前章节的生成/审阅。
        无 [卷] 块或无法定位当前卷时原样返回（安全兜底）。
        """
        if not overview_text or not overview_text.strip():
            return overview_text
        blocks = list(re.finditer(r'\[卷\](.*?)\[/卷\]', overview_text, re.DOTALL))
        if not blocks:
            return overview_text
        spans = []
        for m in blocks:
            s_ch, e_ch = None, None
            ch = re.search(r'章节[：:]\s*(?:第)?\s*(\d+)\s*[-–—]\s*(?:第)?\s*(\d+)\s*章?', m.group(1))
            if ch:
                s_ch, e_ch = int(ch.group(1)), int(ch.group(2))
            spans.append((m, s_ch, e_ch))
        cur_idx = None
        for i, (m, s, e) in enumerate(spans):
            if s is not None and e is not None and s <= chapter_num <= e:
                cur_idx = i
                break
        if cur_idx is None:
            return overview_text
        keep_indices = {cur_idx, cur_idx - 1}
        parts, pos = [], 0
        for i, (m, s, e) in enumerate(spans):
            parts.append(overview_text[pos:m.start()])
            if i in keep_indices:
                parts.append(m.group(0))
            pos = m.end()
        parts.append(overview_text[pos:])
        return "".join(parts)

    def _extract_relevant_outline(self, outline: str, chapter_num: int, capture_range: int = 2, spoiler_level: str = "minimal", current_volume: Optional[dict] = None) -> str:
        """从大纲中提取当前章节附近的内容
        
        参数：
        - capture_range: 前后各取几章的范围，由 _classify_chapter_phase() 根据阶段动态决定
          opening → 0（只取本章）
          early_dev → 1（±1章）
          mid_dev/late_dev/climax → 2（±2章）
        - spoiler_level: 剧透过滤级别
          strict → 完全不包含总述部分
          moderate → 总述部分过滤重大剧透
          minimal → 总述部分只过滤结局性剧透
          none → 总述部分不过滤
        - current_volume: 当前章节所属卷（可选）。提供时总述中的 [卷] 块按卷切分，
          只保留当前卷+前一卷，丢弃后续卷，防止跨卷信息泄漏。为 None 时保留全部总述。
        """
        lines = outline.split("\n")
        has_chapter_markers = any(self._match_chapter_num(l) > 0 for l in lines)
        
        # 如果大纲没有章节标记格式，直接截断返回
        if not has_chapter_markers:
            return outline[:3000] + ("..." if len(outline) > 3000 else "")
        
        # 分两步：先提取总述和章节内容，再根据 spoiler_level 处理总述
        overview_lines = []  # 大纲开头总述部分（第1章标记之前的内容）
        chapter_lines = []   # 章节标记后的内容
        
        found_first_chapter = False
        for line in lines:
            if not found_first_chapter:
                if self._match_chapter_num(line) > 0:
                    found_first_chapter = True
                    chapter_lines.append(line)
                else:
                    overview_lines.append(line)
            else:
                chapter_lines.append(line)
        
        # 处理总述部分：根据 spoiler_level 决定保留多少
        filtered_overview = []
        if spoiler_level == "strict":
            # 严格模式：完全不包含总述（总述往往概括全书走向）
            filtered_overview = []
        elif spoiler_level in ("moderate", "minimal"):
            # 中等/最小模式：对总述部分做剧透句子过滤（若指定当前卷，先按卷切分）
            overview_text = "\n".join(overview_lines)
            if current_volume:
                overview_text = self._slice_overview_by_volume(overview_text, chapter_num)
            if overview_text.strip():
                filtered_overview_text = self._strip_spoiler_sentences(overview_text, level=spoiler_level)
                if filtered_overview_text.strip():
                    filtered_overview = [filtered_overview_text]
        else:
            # none: 保留完整总述（若指定当前卷，则按卷切分防跨卷泄漏）
            if current_volume:
                overview_text = self._slice_overview_by_volume("\n".join(overview_lines), chapter_num)
                filtered_overview = [overview_text] if overview_text.strip() else []
            else:
                filtered_overview = overview_lines
        
        # 提取章节范围内的大纲行
        relevant_lines = []
        capturing = len(filtered_overview) > 0  # 如果有过滤后的总述，默认捕获
        
        for line in chapter_lines:
            num = self._match_chapter_num(line)
            if num > 0:
                capturing = abs(num - chapter_num) <= capture_range
            
            if capturing:
                relevant_lines.append(line)
        
        # 组合：过滤后的总述 + 范围内的章节大纲
        final_lines = filtered_overview + relevant_lines
        result = "\n".join(final_lines).strip()
        # 如果提取结果为空或太短，退回到截断策略（但仍然过滤剧透）
        if not result or len(result) < 20:
            fallback = outline[:3000] + ("..." if len(outline) > 3000 else "")
            if spoiler_level != "none":
                fallback = self._strip_spoiler_sentences(fallback, level=spoiler_level)
            result = fallback
        elif len(result) > 3000:
            result = result[:3000] + "..."
        return result
    
    def _strip_spoiler_sentences(self, text: str, level: str = "strict") -> str:
        """过滤文本中的前瞻/剧透信息（用于人物设定等）
        
        level 级别控制过滤力度：
        - strict: 过滤所有前瞻信息（最终命运/结局/目标/后来/死亡等）
        - moderate: 过滤重大剧透（结局/死亡/最终命运），但保留中期发展方向
        - minimal: 只过滤最终结局/最终命运（允许保留"后来""逐渐"等渐进信息）
        - none: 不过滤
        
        保留：外貌、性格、初始背景、能力等不涉及后续走向的信息
        """
        # 严格模式：过滤所有前瞻信息
        strict_patterns = [
            r'最终[成为到达].{2,20}',
            r'结局.{0,5}[是为].{2,20}',
            r'命运.{0,5}[是为将].{2,20}',
            r'归宿.{0,5}[是为].{2,20}',
            r'后来.{1,3}(成为|达到|发现|获得|修炼|领悟)',
            r'最终(战死|牺牲|死亡|陨落|身亡)',
            r'(最终|后来|末尾|结尾|故事末).{0,10}(成为|达到|获得|领悟|修炼|突破)',
            r'(最终|后来).{0,8}(战胜|击败|消灭|制服|收服)',
        ]
        
        # 中等模式：只过滤重大剧透（结局/死亡/最终命运），保留中期发展
        moderate_patterns = [
            r'最终[成为到达].{2,20}',
            r'结局.{0,5}[是为].{2,20}',
            r'命运.{0,5}[是为将].{2,20}',
            r'最终(战死|牺牲|死亡|陨落|身亡)',
            r'(最终|末尾|结尾|故事末).{0,10}(成为|达到|获得|领悟|修炼|突破)',
            r'(最终|故事末).{0,8}(战胜|击败|消灭|制服|收服)',
        ]
        
        # 最小模式：只过滤最终结局/最终命运
        minimal_patterns = [
            r'最终[成为到达].{2,20}',
            r'结局.{0,5}[是为].{2,20}',
            r'(最终|故事末).{0,8}(战胜|击败|消灭|制服|收服)',
        ]
        
        if level == "none":
            return text
        elif level == "minimal":
            spoiler_patterns = minimal_patterns
        elif level == "moderate":
            spoiler_patterns = moderate_patterns
        else:
            spoiler_patterns = strict_patterns
        
        combined = '|'.join(f'({p})' for p in spoiler_patterns)
        
        # 按中文标点分句
        sentences = re.split(r'([。！？；\n])', text)
        filtered = []
        skip_next = False
        for i, part in enumerate(sentences):
            if skip_next:
                skip_next = False
                continue
            if re.search(combined, part):
                # 标记：如果下一个片段是分隔符（。！？等），也一并跳过
                if i + 1 < len(sentences) and re.match(r'[。！？；\n]', sentences[i + 1]):
                    skip_next = True
                continue
            filtered.append(part)
        
        result = ''.join(filtered)
        # 过滤比例高时只记录告警，绝不回退到原文（回退会把完整剧透放回去，违背过滤目的）
        if len(result) < len(text) * 0.5:
            logger.warning(f"前瞻信息过滤移除了超过50%的内容({len(text)}→{len(result)})，level={level}，请人工检查是否误伤")
        return result
    
    def _get_previous_chapters_summary(self, current_chapter_num: int, max_chars: int = 800) -> str:
        """获取前几章的摘要，只取最近2章的末尾段落，不传全文"""
        summaries = []
        # 只取最近2章
        for n in range(max(1, current_chapter_num - 2), current_chapter_num):
            content = self.vs.get_section("chapter", f"chapter_{n}")
            if content:
                # 去掉"第X章 标题"行，只保留正文（避免旧标题污染）
                lines = content.split("\n", 1)
                if lines and re.match(r"第\d+章", lines[0].strip()):
                    content = lines[1] if len(lines) > 1 else content
                # 取最后 max_chars 字符作为"前情"
                if len(content) > max_chars:
                    summary = f"第{n}章（末尾段落）：...{content[-max_chars:]}"
                else:
                    summary = f"第{n}章：{content}"
                summaries.append(summary)
        return "\n".join(summaries)
    
    def generate_chapter_with_rag(self, chapter_num: int, chapter_title: str, max_tokens: int = 2500, target_words: int = 2000, beats: str = "", temperature: float = 0.8, extra_instruction: str = "") -> str:
        """自动检索上下文生成章节（支持可选细纲与额外改写要求）"""
        return self.generate_chapter(chapter_num, chapter_title, max_tokens=max_tokens, target_words=target_words, beats=beats, temperature=temperature, extra_instruction=extra_instruction)
    
    def continue_writing(self, current_chapter: str, prompt: str = "继续往下写", target_length: int = 1500, max_tokens: int = 2000) -> str:
        """续写当前章节"""
        # 检索相关上下文，但过滤后续章节内容防止抢跑
        related = self.vs.search_related(current_chapter[-1000:], n_results=5)
        context_text = "前文内容和相关设定：\n"
        for ctx in related:
            content = ctx["content"]
            meta = ctx.get("metadata", {})
            # 跳过后续章节内容（无法确定当前在写第几章，保守跳过所有chapter类型）
            if meta.get("type") == "chapter":
                continue
            context_text += f"{content[:500]}\n---\n"
        
        full_prompt = f"""{context_text}

当前写到这里：
{current_chapter[-6000:] if len(current_chapter) > 6000 else current_chapter}

请继续往下写：{prompt}

要求：
- 续写大约{target_length}字
- 保持与前文风格一致
- **严禁抢跑**：只继续展开当前正在发生的场景和事件，绝不要把后续章节才该出现的剧情、角色、转折提前写进来
- 如果当前场景已自然收束，可以直接结束，不需要为了凑字数而加速推进剧情
- 直接输出续写内容，不要解释

续写内容："""
        
        result = self.api.generate(full_prompt, step="continue", temperature=0.8, max_tokens=max_tokens, stream_callback=self.on_token, reasoning_callback=self.on_reasoning)
        return result
    
    def check_consistency(self, max_tokens: int = 4000, include_chapters: bool = True) -> str:
        """AI一致性检查：找出各设定之间的矛盾和不一致，逐对交叉比对
        
        include_chapters=True 时，分批检查所有章节正文与设定/大纲的矛盾
        （章节正文是矛盾最高发区），每批最多 4 章、每章截取前 1500 字。
        """
        gc = self.novel_info
        sections = []
        section_names = []
        
        if gc.get("world_setting"):
            sections.append(f"【世界观设定】\n{gc['world_setting']}")
            section_names.append("世界观设定")
        if gc.get("characters"):
            sections.append(f"【人物设定】\n{gc['characters']}")
            section_names.append("人物设定")
        if gc.get("outline"):
            sections.append(f"【小说大纲】\n{gc['outline']}")
            section_names.append("小说大纲")
        
        chapters = gc.get("chapters", {})
        if not chapters:
            # 兜底：从本地存储读取章节
            for sec in self.vs.get_all_by_type("chapter"):
                m = re.match(r"chapter_(\d+)", sec["metadata"].get("title", ""))
                if m:
                    chapters[m.group(1)] = {"content": sec["content"]}
        
        if len(sections) < 2 and not chapters:
            return "⚠️ 至少需要完成两个步骤才能进行一致性检查。"
        
        all_text = "\n\n---\n\n".join(sections)
        
        # 构建逐对检查指引
        pair_checks = []
        for i in range(len(section_names)):
            for j in range(i + 1, len(section_names)):
                pair_checks.append(f"- 「{section_names[i]}」 vs 「{section_names[j]}」")
        pair_check_str = "\n".join(pair_checks) if pair_checks else "- （仅有章节正文，跳过设定间比对）"
        
        settings_block = f"""{all_text}

**你必须逐一比对以下每一对内容，不得遗漏：**
{pair_check_str}""" if sections else ""
        
        base_instruction = """**检查步骤（必须严格执行）：**

**第一步：逐对交叉比对**
对上述每一对，分别检查：
1. **人名不一致**：同一个人在不同部分是否名字不同？名字是否有错别字或简写差异？（这是最常见的错误）
2. **设定冲突**：世界观中的力量体系/规则/社会结构是否与人物能力/地位矛盾？
3. **情节矛盾**：大纲/章节中的情节是否与人物背景或世界观设定冲突？
4. **角色特征矛盾**：同一角色在不同部分的性格/外貌/背景/能力描述是否矛盾？
5. **地理/时间矛盾**：地点、时代、时间线在不同部分是否一致？
6. **因果关系矛盾**：事件的前因后果在不同部分是否一致？

**第二步：汇总输出**
请用以下格式输出完整结果：

🔴 **严重矛盾**（必须修改，否则后续生成会混乱）：
- [列出所有严重矛盾，每条都要明确指出：在哪个部分的哪句话，与哪个部分的哪句话矛盾]

🟡 **潜在问题**（建议修改，可能影响连贯性）：
- [列出所有潜在问题]

✅ **一致无问题**：
- [列出已检查且无问题的方面]

**重要：**
- 请务必检查每一对内容，不要因为内容多就跳过
- 宁可多报疑似问题，也不要遗漏真实矛盾
- 发现矛盾时必须引用原文具体语句"""
        
        reports = []
        
        # 第一部分：设定间交叉比对（至少两个设定块时才做）
        if len(sections) >= 2:
            self._report("一致性检查：正在比对设定间矛盾…")
            prompt = f"""你是一个极其严谨的专业小说编辑，请对以下小说的各部分设定进行**全面彻底**的一致性检查。

{settings_block}

{base_instruction}"""
            reports.append(self.api.generate(prompt, step="consistency", temperature=0.2, max_tokens=max_tokens))
        
        # 第二部分：章节正文分批检查（每批 4 章，每章截取前 1500 字）
        if include_chapters and chapters:
            sorted_chaps = sorted(chapters.items(), key=lambda kv: int(kv[0]))
            settings_brief = ""
            if gc.get("characters"):
                settings_brief += f"【人物设定（摘要）】\n{gc['characters'][:2500]}\n\n"
            if gc.get("world_setting"):
                settings_brief += f"【世界观设定（摘要）】\n{gc['world_setting'][:2000]}\n\n"
            if gc.get("outline"):
                settings_brief += f"【小说大纲（摘要）】\n{gc['outline'][:2000]}"
            
            BATCH = 4
            for bi in range(0, len(sorted_chaps), BATCH):
                batch = sorted_chaps[bi:bi + BATCH]
                chap_blocks = []
                for num, data in batch:
                    c = data.get("content", "")[:1500]
                    chap_blocks.append(f"【第{num}章「{data.get('title','')}」正文（节选）】\n{c}")
                batch_label = f"第{batch[0][0]}-{batch[-1][0]}章"
                
                prompt = f"""你是一个极其严谨的专业小说编辑。请检查以下**章节正文**与设定之间的一致性（{batch_label}）。

{settings_brief}

{"".join(chap_blocks)}

**检查重点：**
1. 人名/称呼是否与人物设定一致（最常见错误）
2. 角色的能力、性格、背景是否与设定矛盾
3. 情节走向是否与大纲冲突或抢跑
4. 世界观规则（力量体系/地理/时代）是否被违反

{base_instruction}"""
                logger.info(f"一致性检查：送检章节批次 {batch_label}")
                self._report(f"一致性检查：正在检查 {batch_label}…")
                reports.append(f"### 📖 章节检查（{batch_label}）\n" + self.api.generate(prompt, step="consistency", temperature=0.2, max_tokens=max_tokens))
        
        return "\n\n---\n\n".join(reports)
    
    def polish_with_style(self, text: str, style_reference: str, style_type: str = "作品", max_tokens: int = 2000) -> str:
        """模仿指定作品/作家的风格润色文本"""
        # 输出长度至少覆盖原文（中文1字≈1.5-2 token），防止润色结果被截断
        needed = min(int(len(text) * 1.8) + 500, self.api.MAX_TOKENS_LIMIT)
        if max_tokens < needed:
            logger.info(f"润色 max_tokens={max_tokens} 不足以覆盖原文{len(text)}字，自动提升至 {needed}")
            max_tokens = needed
        # 检索相关上下文
        related = self.vs.search_related(text[:500], n_results=3)
        context_text = ""
        if related:
            context_text = "相关设定参考：\n"
            for ctx in related:
                context_text += f"{ctx['content'][:500]}\n---\n"

        style_guide = ""
        if style_type == "作品":
            style_guide = f"请仔细模仿《{style_reference}》的写作风格来润色以下文本。"
        elif style_type == "描述":
            # 合规模式：直接按文风特征描述润色，不点名模仿在世作家
            style_guide = f"请按以下文风特征描述来润色以下文本：\n{style_reference}"
        else:
            style_guide = f"请仔细模仿{style_reference}的写作风格来润色以下文本。"

        prompt = f"""你是一位精通文学风格的编辑大师。{style_guide}

**风格模仿要点**：
- 句式节奏：模仿目标风格的句子长短、断句习惯
- 修辞手法：模仿其常用的比喻、拟人、排比等修辞
- 叙事视角：模仿其叙事方式和人称运用
- 情感基调：模仿其情感表达的浓淡与节奏
- 用词偏好：模仿其遣词造句的特色与倾向
- 描写侧重：模仿其对环境、人物、动作的描写偏好

{context_text}

**原文**：
{text}

**要求**：
- 保持原文的核心情节和信息不变
- 只改写表达方式，不增删情节
- 让文本读起来就像是用目标风格写出来的一样
- 直接输出润色后的文本，不要解释

**润色后**："""

        result = self.api.generate(prompt, step="polish", temperature=0.7, max_tokens=max_tokens)
        return result

    def global_find_replace(self, find_text: str, replace_text: str, gc: dict) -> dict:
        """全局查找替换：在所有内容中查找并替换文本，返回变更报告"""
        changes = []
        
        # 世界观设定
        if gc.get("world_setting") and find_text in gc["world_setting"]:
            count = gc["world_setting"].count(find_text)
            gc["world_setting"] = gc["world_setting"].replace(find_text, replace_text)
            changes.append(f"🌍 世界观设定：替换了 {count} 处")
            self.vs.update_section("setting", "world_setting", gc["world_setting"])
            self.novel_info["world_setting"] = gc["world_setting"]
        
        # 人物设定
        if gc.get("characters") and find_text in gc["characters"]:
            count = gc["characters"].count(find_text)
            gc["characters"] = gc["characters"].replace(find_text, replace_text)
            changes.append(f"👤 人物设定：替换了 {count} 处")
            self.vs.update_section("character", "all_characters", gc["characters"])
            self.novel_info["characters"] = gc["characters"]
        
        # 大纲
        if gc.get("outline") and find_text in gc["outline"]:
            count = gc["outline"].count(find_text)
            gc["outline"] = gc["outline"].replace(find_text, replace_text)
            changes.append(f"📋 小说大纲：替换了 {count} 处")
            self.vs.update_section("outline", "full_outline", gc["outline"])
            self.novel_info["outline"] = gc["outline"]
        
        # 章节
        if gc.get("chapters"):
            for chap_num, chap_data in gc["chapters"].items():
                content = chap_data.get("content", "")
                title = chap_data.get("title", "")
                content_changed = False
                title_changed = False
                
                if find_text in content:
                    count = content.count(find_text)
                    content = content.replace(find_text, replace_text)
                    chap_data["content"] = content
                    changes.append(f"📖 第{chap_num}章（正文）：替换了 {count} 处")
                    content_changed = True
                
                if find_text in title:
                    title = title.replace(find_text, replace_text)
                    chap_data["title"] = title
                    changes.append(f"📖 第{chap_num}章（标题）：替换了 1 处")
                    title_changed = True
                
                if content_changed or title_changed:
                    full_text = f"第{chap_num}章 {title}\n{content}"
                    self.vs.update_section("chapter", f"chapter_{chap_num}", full_text)
        
        return {"changes": changes, "updated_gc": gc}
    
    def global_find(self, find_text: str, gc: dict) -> list:
        """全局查找：在所有内容中查找文本，返回匹配位置列表"""
        results = []
        
        if gc.get("world_setting") and find_text in gc["world_setting"]:
            count = gc["world_setting"].count(find_text)
            results.append(f"🌍 世界观设定：找到 {count} 处")
        
        if gc.get("characters") and find_text in gc["characters"]:
            count = gc["characters"].count(find_text)
            results.append(f"👤 人物设定：找到 {count} 处")
        
        if gc.get("outline") and find_text in gc["outline"]:
            count = gc["outline"].count(find_text)
            results.append(f"📋 小说大纲：找到 {count} 处")
        
        if gc.get("chapters"):
            for chap_num, chap_data in gc["chapters"].items():
                content = chap_data.get("content", "")
                title = chap_data.get("title", "")
                count = content.count(find_text) + title.count(find_text)
                if count > 0:
                    results.append(f"📖 第{chap_num}章「{title}」：找到 {count} 处")
        
        return results
    
    def extract_character_relations(self, max_tokens: int = 2000) -> str:
        """AI提取角色关系，返回JSON格式的关系数据"""
        gc = self.novel_info
        sections = []
        
        if gc.get("world_setting"):
            sections.append(f"【世界观设定】\n{gc['world_setting']}")
        if gc.get("characters"):
            sections.append(f"【人物设定】\n{gc['characters']}")
        if gc.get("outline"):
            sections.append(f"【小说大纲】\n{gc['outline']}")
        
        if not gc.get("characters"):
            return ""
        
        all_text = "\n\n---\n\n".join(sections)
        
        prompt = f"""请根据以下小说设定，提取角色之间的关系。

{all_text}

请严格按照以下JSON格式输出（不要输出其他任何内容，只输出JSON）：
{{
  "characters": [
    {{"name": "角色名", "role": "主角/反派/配角", "desc": "一句话描述"}},
    ...
  ],
  "relations": [
    {{"from": "角色A", "to": "角色B", "type": "师徒/恋人/敌人/朋友/主仆/同门/亲属", "desc": "关系描述"}},
    ...
  ]
}}

注意：
- characters数组列出所有有名字的角色
- relations数组列出角色之间的关系
- type只从以下选择：师徒、恋人、敌人、朋友、主仆、同门、亲属、对手、盟友
- 最多列出15个最重要的角色和20条最重要的关系
- 必须只输出合法的JSON，不要有其他文字"""

        result = self.api.generate(prompt, step="relations", temperature=0.3, max_tokens=max_tokens)
        return result
