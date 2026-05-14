"""
全链路小说创作工作流
"""
import logging
import re
from typing import Dict, List, Optional
from api.api_client import LLMAPIClient
from vector_store.local_chroma import LocalNovelVectorStore

logger = logging.getLogger(__name__)

class FullNovelWorkflow:
    def __init__(self, api_client: LLMAPIClient, vector_store: LocalNovelVectorStore):
        self.api = api_client
        self.vs = vector_store
        self.novel_info = {}
    
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

        result = self.api.generate(prompt, temperature=0.8, max_tokens=max_tokens)
        # 保存到向量库
        self.vs.add_section("setting", "world_setting", result)
        self.novel_info["world_setting"] = result
        return result
    
    def generate_characters(self, user_prompt: str, num_main: int = 3, num_support: int = 5, max_tokens: int = 2000) -> Dict:
        """第二步：生成人物设定"""
        # 从向量库获取世界观作为上下文
        contexts = self.vs.search_related("世界观设定", n_results=2)
        world_context = contexts[0]["content"] if contexts else ""
        
        prompt = f"""请根据以下世界观设定，为这部小说设计主要人物。
世界观：
{world_context}

用户需求：{user_prompt}

请设计：
- {num_main}个主要角色（主角，主要反派）：包含姓名、外貌、性格、背景、目标
- {num_support}个重要配角：简要介绍

输出格式要清晰分明："""

        result = self.api.generate(prompt, temperature=0.7, max_tokens=max_tokens)
        self.vs.add_section("character", "all_characters", result)
        self.novel_info["characters"] = result
        return {"characters": result}
    
    def generate_outline(self, user_prompt: str, total_chapters: int = 50, words_per_chapter: int = 2000, max_tokens: int = 4000) -> str:
        """第三步：生成总体大纲（大章节量时自动分卷两阶段生成）"""
        # 获取已有上下文
        world_setting = self.novel_info.get("world_setting", "")
        characters = self.novel_info.get("characters", "")
        
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
3. 每章简要内容（一句话概括）

大纲要起承转合，有节奏起伏。

【重要】你必须写出全部 {total_chapters} 章的简要内容，不能中途停止或省略。
如果内容较长，请持续输出直到写完所有章节，绝不要用"以此类推"或省略号代替。"""

        return self.api.generate(prompt, temperature=0.7, max_tokens=max_tokens)
    
    def _generate_outline_two_stage(self, user_prompt, total_chapters, words_per_chapter, max_tokens, world_setting, characters):
        """两阶段生成大纲（章节数较多时使用）
        第一阶段：生成卷级大纲（故事主线 + 分卷规划 + 每卷概要）
        第二阶段：逐卷补全每章简要内容
        """
        # ---- 第一阶段：生成卷级大纲 ----
        logger.info(f"大纲两阶段生成 → 第一阶段：卷级大纲（共{total_chapters}章）")
        stage1_prompt = f"""请根据以下信息，创作这部小说的分卷大纲。
总共规划 {total_chapters} 章，每章大约 {words_per_chapter} 字。

世界观设定：
{world_setting}

人物设定：
{characters}

用户需求：{user_prompt}

请严格按照以下格式输出（不要使用其他格式）：

故事主线：200字以内概括整体故事走向

[卷]
卷名：第一卷的名称
章节：1-{total_chapters // 5}章
剧情：该卷的核心剧情和冲突（100-200字）
[/卷]

[卷]
卷名：第二卷的名称
章节：{total_chapters // 5 + 1}-{2 * total_chapters // 5}章
剧情：该卷的核心剧情和冲突（100-200字）
[/卷]

（按此格式继续列出所有卷）

【重要】
- 只需要写到"卷"的级别，不需要写出每章内容
- 必须用 [卷]...[/卷] 标签包裹每一卷的信息
- 章节范围必须是"起始章-结束章"的格式，如"1-30章"
- 必须覆盖全部 {total_chapters} 章
- 大纲要起承转合，有节奏起伏"""

        stage1_result = self.api.generate(stage1_prompt, temperature=0.7, max_tokens=max_tokens)
        logger.info(f"大纲第一阶段完成，卷级大纲长度: {len(stage1_result)} 字符")
        logger.debug(f"卷级大纲原文：\n{stage1_result}")
        
        # ---- 第二阶段：逐卷补全章节大纲 ----
        # 从卷级大纲中解析出各卷信息
        volumes = self._parse_volumes(stage1_result, total_chapters)
        
        if not volumes:
            # 解析失败，回退到单次生成（加上强制写完的指令）
            logger.warning("卷级大纲解析失败，回退到单次生成")
            return self._generate_outline_single(
                user_prompt, total_chapters, words_per_chapter, max_tokens,
                world_setting, characters
            )
        
        logger.info(f"解析出 {len(volumes)} 卷，开始逐卷补全章节大纲")
        
        # 先把卷级大纲作为基础
        all_parts = [stage1_result, "\n\n---\n\n## 逐章大纲\n"]
        
        for vol in volumes:
            vol_name = vol["name"]
            vol_chapters = vol["chapters"]
            vol_plot = vol["plot"]
            
            if vol_chapters <= 0:
                continue
            
            stage2_prompt = f"""请为以下这卷小说补全每章的简要内容。

【故事主线与分卷规划】
{stage1_result}

【当前需要补全的卷】
卷名：{vol_name}
该卷章节数：{vol_chapters} 章
该卷核心剧情：{vol_plot}

请为这 {vol_chapters} 章逐一写出简要内容（每章一句话概括）。

格式要求：
第 X 章：章节标题 —— 一句话概括

【重要】你必须写完这 {vol_chapters} 章的全部内容，不能中途停止或省略。"""

            vol_result = self.api.generate(stage2_prompt, temperature=0.7, max_tokens=max_tokens)
            all_parts.append(f"### {vol_name}\n{vol_result}\n\n")
            logger.info(f"卷「{vol_name}」补全完成，长度: {len(vol_result)} 字符")
        
        final_result = "".join(all_parts)
        logger.info(f"大纲两阶段生成完成，总长度: {len(final_result)} 字符")
        return final_result
    
    def _parse_volumes(self, volume_outline: str, total_chapters: int) -> list:
        """从卷级大纲中解析出各卷信息，返回 [{"name": "卷名", "chapters": 章节数, "plot": "剧情概要"}]
        
        优先解析 [卷]...[/卷] 结构化格式（prompt 要求的固定格式），
        如果解析失败则尝试从自由文本中提取。
        """
        
        volumes = []
        
        # ---- 优先解析结构化格式 [卷]...[/卷] ----
        vol_blocks = re.findall(r'\[卷\](.*?)\[/卷\]', volume_outline, re.DOTALL)
        if vol_blocks:
            for block in vol_blocks:
                vol_name = ""
                chapters = 0
                plot = ""
                
                # 提取卷名
                name_m = re.search(r'卷名[：:]\s*(.+)', block)
                if name_m:
                    vol_name = name_m.group(1).strip()
                
                # 提取章节范围
                ch_m = re.search(r'章节[：:]\s*(\d+)\s*[-–—]\s*(\d+)\s*章?', block)
                if ch_m:
                    chapters = int(ch_m.group(2)) - int(ch_m.group(1)) + 1
                
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
                        "plot": plot or vol_name
                    })
            
            if volumes:
                logger.info(f"结构化格式解析成功，共 {len(volumes)} 卷")
                # 校验和调整章节数
                return self._adjust_volumes(volumes, total_chapters)
        
        # ---- 兜底：从自由文本中解析 ----
        logger.info("未检测到 [卷]...[/卷] 结构化格式，尝试自由文本解析")
        for line in volume_outline.split("\n"):
            line_stripped = line.strip()
            vol_m = re.search(r'第([一二三四五六七八九十\d]+)卷', line_stripped)
            if not vol_m:
                continue
            
            vol_num = vol_m.group(1)
            rest = line_stripped[vol_m.end():].strip()
            
            # 提取章节范围（整行搜索）
            range_m = re.search(r'(\d+)\s*[-–—]\s*(\d+)\s*章', line_stripped)
            chapters = 0
            if range_m:
                chapters = int(range_m.group(2)) - int(range_m.group(1)) + 1
            
            # 提取卷名：去掉章节范围和标点
            vol_name = rest
            vol_name = re.sub(r'[（(]\s*\d+\s*[-–—]\s*\d+\s*章\s*[）)]', '', vol_name)
            vol_name = re.sub(r'第?\s*\d+\s*[-–—]\s*\d+\s*章', '', vol_name)
            vol_name = re.sub(r'^[：:，,、\s]+', '', vol_name).strip()
            vol_name = re.sub(r'[：:，,、\s]+$', '', vol_name).strip()
            
            volumes.append({
                "name": f"第{vol_num}卷：{vol_name}" if vol_name else f"第{vol_num}卷",
                "chapters": chapters,
                "plot": vol_name if vol_name else f"第{vol_num}卷"
            })
        
        if not volumes:
            logger.warning(f"自由文本解析也失败，AI原文前500字：{volume_outline[:500]}")
            return []
        
        return self._adjust_volumes(volumes, total_chapters)
    
    def _adjust_volumes(self, volumes: list, total_chapters: int) -> list:
        """校验和调整卷的章节数，确保总和接近 total_chapters"""
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
        
        return volumes

    def _estimate_total_chapters(self) -> int:
        """估算小说总章节数，用于阶段分类
        
        优先级：
        1. session_state 中记录的 outline_total_chapters（大纲生成时保存）
        2. 从 outline 文本中解析出的章节数
        3. 从已生成章节的最大编号推断
        """
        # 尝试从 novel_info 的 outline 解析
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
    
    def generate_chapter(self, chapter_num: int, chapter_title: str, previous_summary: str = "", max_tokens: int = 2500, target_words: int = 2000) -> str:
        """生成单章正文 — 泛化节奏控制：根据章节所处阶段自动调整上下文策略"""
        
        logger.info(f"===== generate_chapter 开始 =====")
        logger.info(f"参数: chapter_num={chapter_num}, chapter_title={chapter_title}, max_tokens={max_tokens}, target_words={target_words}")
        
        # 先删除向量库中当前章节的旧数据，避免旧标题/内容污染搜索结果
        self.vs.delete_section("chapter", f"chapter_{chapter_num}")
        logger.info(f"已预删除向量库中 chapter_{chapter_num} 的旧数据")
        
        # ---- 阶段分类：根据章节位置决定上下文策略 ----
        total_chapters = self._estimate_total_chapters()
        phase_config = self._classify_chapter_phase(chapter_num, total_chapters)
        
        # ---- 分类构建上下文，避免一次性塞入过多内容 ----
        context_parts = []
        
        # 1. 核心设定：直接从 novel_info 获取（已生成的内容，比向量检索更完整更精准）
        setting_text = self.novel_info.get("world_setting", "")
        character_text = self.novel_info.get("characters", "")
        outline_text = self.novel_info.get("outline", "")
        phase = phase_config["phase"]
        
        if setting_text:
            # 世界观设定截断，最多 4000 字
            SETTING_MAX = 4000
            trunc = setting_text[:SETTING_MAX] + ("..." if len(setting_text) > SETTING_MAX else "")
            context_parts.append(f"【世界观设定】\n{trunc}")
            logger.info(f"世界观设定: 原始{len(setting_text)}字 → 传入{len(trunc)}字 (上限{SETTING_MAX})")
        else:
            logger.info("世界观设定: 无")
            
        if character_text:
            # 人物设定截断，最多 6000 字
            CHAR_MAX = 6000
            trunc = character_text[:CHAR_MAX] + ("..." if len(character_text) > CHAR_MAX else "")
            # 渐进式前瞻信息过滤：根据阶段决定过滤力度
            spoiler_level = phase_config.get("spoiler_level", "minimal")
            if spoiler_level == "strict":
                # 开篇阶段：过滤所有前瞻信息（最终命运/结局/目标等）
                trunc = self._strip_spoiler_sentences(trunc, level="strict")
                logger.info(f"人物设定[strict]: 已过滤所有前瞻信息")
            elif spoiler_level == "moderate":
                # 早期发展：过滤结局/死亡等重大剧透，但保留中期发展方向
                trunc = self._strip_spoiler_sentences(trunc, level="moderate")
                logger.info(f"人物设定[moderate]: 已过滤重大剧透")
            elif spoiler_level == "minimal":
                # 中后期发展：只过滤最终结局/最终命运
                trunc = self._strip_spoiler_sentences(trunc, level="minimal")
                logger.info(f"人物设定[minimal]: 已过滤最终结局剧透")
            # none: 不过滤
            context_parts.append(f"【人物设定】\n{trunc}")
            logger.info(f"人物设定: 原始{len(character_text)}字 → 传入{len(trunc)}字 (上限{CHAR_MAX})")
        else:
            logger.info("人物设定: 无")
        
        outline_for_chapter = ""
        if outline_text:
            # 根据剧透级别决定大纲范围和是否过滤总述
            # - strict (opening): range=0, 过滤总述
            # - moderate (early_dev): range=1, 过滤总述中的剧透句
            # - minimal (mid_dev/late_dev): range=2, 过滤总述中的"结局"剧透
            # - none (climax/resolution): range=2, 不过滤
            spoiler_level = phase_config.get("spoiler_level", "minimal")
            effective_range = phase_config["outline_range"]
            outline_for_chapter = self._extract_relevant_outline(
                outline_text, chapter_num, 
                capture_range=effective_range,
                spoiler_level=spoiler_level
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
        
        # 3. 语义搜索补充：用当前章节标题在向量库做语义搜索(n_results=4)
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
                logger.info(f"语义搜索跳过当前章节自身: chapter_{chapter_num}")
                continue
            # 根据阶段策略过滤后续章节：rag_look_ahead=0 表示完全不看后续
            if meta.get("type") == "chapter":
                chap_match = re.search(r"chapter_(\d+)", meta.get("title", ""))
                if chap_match:
                    ref_chap_num = int(chap_match.group(1))
                    allowed_max = chapter_num + phase_config["rag_look_ahead"]
                    if ref_chap_num > allowed_max:
                        logger.info(
                            f"[{phase_config['phase']}] 语义搜索跳过第{ref_chap_num}章 "
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
            extra_context.append(content[:1000])
        
        if extra_context:
            combined_extra = "\n---\n".join(extra_context)
            context_parts.append(f"【相关参考片段】\n{combined_extra}")
            logger.info(f"语义搜索补充: {len(extra_context)}条, 共{len(combined_extra)}字")
        else:
            logger.info("语义搜索补充: 无")
        
        # 构建阶段感知的叙事节奏指令
        pacing_instruction = phase_config["pacing_instruction"]
        if pacing_instruction:
            logger.info(f"[{phase_config['phase']}] 已注入阶段专属叙事指导")
        
        context_text = "\n\n".join(context_parts)
        logger.info(f"上下文总计: {len(context_text)}字")
        
        if previous_summary:
            context_text += f"\n\n上一章内容回顾：{previous_summary}\n"
            logger.info(f"上一章回顾: {len(previous_summary)}字")

        prompt = f"""请你根据以下信息，写出小说第 {chapter_num} 章 "{chapter_title}" 的完整正文。

{context_text}
{pacing_instruction}"""
        
        # 所有阶段的硬性要求都包含防抢跑指令，但根据阶段调整力度
        spoiler_level = phase_config.get("spoiler_level", "minimal")
        
        # 防抢跑通用指令：根据阶段调整严格程度
        anti_rush_instructions = {
            "strict": "**严禁抢跑**：你只负责写本章大纲范围内的事件。绝不要为了丰富内容或推进剧情而把后续章节才该出现的角色、场景、冲突、转折提前写进来。后续章节有后续章节的篇幅，不需要你在这里抢先完成。",
            "moderate": '**不要抢跑**：只写本章大纲范围内的事件。如果大纲说本章只到「主角进入秘境」，就不要写到「获得传承」——获得传承是后续章节的事。可以铺垫和暗示，但不要提前写出结果。',
            "minimal": "**控制节奏**：本章只负责推进大纲中标记给本章的情节，不要一口气把后续几章的走向都写完。关键转折要有铺垫过程，不要跳过中间环节直接给结果。",
            "none": '**保持节奏**：即使到了故事高潮/收尾阶段，每个重大事件仍需要足够的篇幅来展开，不要因为「快结束了」就草草带过。'
        }
        anti_rush = anti_rush_instructions.get(spoiler_level, anti_rush_instructions["minimal"])
        
        if phase == "opening":
            prompt += f"""【硬性要求】
- 本章目标字数约 {target_words} 字，但如果章节自然结尾在 {int(target_words * 0.7)} 字以上也是完全可以接受的——开篇章节的质量远比字数重要
- 如果一次写不完，请在段落中间自然断开，不要写总结性结尾，不要写"本章完"之类的结束语
- {anti_rush}
- 保持人物设定一致性
- 文笔流畅，用具体场景、细节描写、对话、心理活动来展开（不要用大段说明性文字凑字数）
- 直接输出正文，不要解释

正文："""
        elif phase == "resolution":
            prompt += f"""【硬性要求】
- 本章目标字数约 {target_words} 字，可以略少（收尾重在质量而非篇幅）
- 如果一次写不完，请在段落中间自然断开，不要写总结性结尾，不要写"本章完"之类的结束语
- {anti_rush}
- 保持人物设定一致性
- 文笔流畅，有画面感、细节描写丰富
- 直接输出正文，不要解释

正文："""
        else:
            prompt += f"""【硬性要求】
- 本章目标字数约 {target_words} 字（允许±20%浮动，但不要低于80%）
- 如果一次写不完，请在段落中间自然断开，不要写总结性结尾，不要写"本章完"之类的结束语
- {anti_rush}
- 情节符合大纲走向，保持人物设定一致性
- 文笔流畅，有画面感、细节描写丰富（大量使用对话、动作、心理、环境描写来展开）
- 直接输出正文，不要解释

正文："""

        logger.info(f"完整prompt长度: {len(prompt)}字 (约{len(prompt)*2}token)")
        logger.info(f"API调用参数: model={self.api.model}, max_tokens={max_tokens}, temperature=0.8")

        result = self.api.generate(prompt, temperature=0.8, max_tokens=max_tokens)

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
- {anti_rush_instructions.get(spoiler_level, anti_rush_instructions['minimal'])}
- 不要写总结性结尾，不要写"本章完"
- 如果情节还有发展空间，在段落中间自然断开
- 直接输出续写内容，不要解释

续写内容："""

            # 续写时的 token 预算：按剩余字数的2倍估算（中文字符≈1.5-2 token）
            continue_max = min(max_tokens, max(2000, int(remaining * 2)))
            continuation = self.api.generate(continue_prompt, temperature=0.8, max_tokens=continue_max)
            result = result + continuation

            logger.info(f"章节第{chapter_num}章续写第{continuation_round}轮：原始长度={len(result) - len(continuation)}字 + 续写{len(continuation)}字 = 总计{len(result)}字")

        if len(result) < min_chars:
            logger.warning(f"章节第{chapter_num}章经{max_continuations}轮续写后仍仅{len(result)}字，目标{target_words}字")

        # 生成后范围校验：检查是否可能覆盖了超出当前大纲范围的事件
        scope_warning = self._check_chapter_scope(result, chapter_num, outline_for_chapter, spoiler_level)
        if scope_warning:
            logger.warning(f"章节{chapter_num}章范围校验: {scope_warning}")
        
        # 保存到向量库，方便后续章节检索
        self.vs.add_section("chapter", f"chapter_{chapter_num}", f"第{chapter_num}章 {chapter_title}\n{result}")
        return result
    
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
            
            # 去除常见的行首标记：markdown 列表、加粗等
            cleaned = re.sub(r"^[-\*]\s+", "", line_stripped)      # "- " 或 "* " 开头
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

    def _extract_relevant_outline(self, outline: str, chapter_num: int, capture_range: int = 2, spoiler_level: str = "minimal") -> str:
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
        """
        lines = outline.split("\n")
        has_chapter_markers = any(re.match(r"第(\d+|[一二三四五六七八九十百]+)章", l) for l in lines)
        
        # 如果大纲没有章节标记格式，直接截断返回
        if not has_chapter_markers:
            return outline[:3000] + ("..." if len(outline) > 3000 else "")
        
        # 分两步：先提取总述和章节内容，再根据 spoiler_level 处理总述
        overview_lines = []  # 大纲开头总述部分（第1章标记之前的内容）
        chapter_lines = []   # 章节标记后的内容
        
        found_first_chapter = False
        for line in lines:
            if not found_first_chapter:
                match = re.match(r"第(\d+|[一二三四五六七八九十百]+)章", line)
                if match:
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
            # 中等/最小模式：对总述部分做剧透句子过滤
            overview_text = "\n".join(overview_lines)
            if overview_text.strip():
                filtered_overview_text = self._strip_spoiler_sentences(overview_text, level=spoiler_level)
                if filtered_overview_text.strip():
                    filtered_overview = [filtered_overview_text]
        else:
            # none: 保留完整总述
            filtered_overview = overview_lines
        
        # 提取章节范围内的大纲行
        relevant_lines = []
        capturing = len(filtered_overview) > 0  # 如果有过滤后的总述，默认捕获
        cn_map = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        
        for line in chapter_lines:
            match = re.match(r"第(\d+|[一二三四五六七八九十百]+)章", line)
            if match:
                num_str = match.group(1)
                try:
                    num = int(num_str)
                except ValueError:
                    num = cn_map.get(num_str, 0)
                
                if abs(num - chapter_num) <= capture_range:
                    capturing = True
                else:
                    capturing = False
            
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
        # 如果过滤后内容大幅减少（超过50%），说明可能误伤太多，回退到原文
        if len(result) < len(text) * 0.5:
            logger.warning(f"前瞻信息过滤移除了超过50%的内容({len(text)}→{len(result)})，level={level}，可能误伤，回退到原文")
            return text
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
    
    def generate_chapter_with_rag(self, chapter_num: int, chapter_title: str, max_tokens: int = 2500, target_words: int = 2000) -> str:
        """用RAG自动检索上下文生成章节"""
        # RAG就是这里：自动从本地向量库找相关上下文给模型
        return self.generate_chapter(chapter_num, chapter_title, max_tokens=max_tokens, target_words=target_words)
    
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
{current_chapter}

请继续往下写：{prompt}

要求：
- 续写大约{target_length}字
- 保持与前文风格一致
- **严禁抢跑**：只继续展开当前正在发生的场景和事件，绝不要把后续章节才该出现的剧情、角色、转折提前写进来
- 如果当前场景已自然收束，可以直接结束，不需要为了凑字数而加速推进剧情
- 直接输出续写内容，不要解释

续写内容："""
        
        result = self.api.generate(full_prompt, temperature=0.8, max_tokens=max_tokens)
        return result
    
    def check_consistency(self, max_tokens: int = 4000) -> str:
        """AI一致性检查：找出各设定之间的矛盾和不一致，逐对交叉比对"""
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
        
        if len(sections) < 2:
            return "⚠️ 至少需要完成两个步骤才能进行一致性检查。"
        
        all_text = "\n\n---\n\n".join(sections)
        
        # 构建逐对检查指引
        pair_checks = []
        for i in range(len(section_names)):
            for j in range(i + 1, len(section_names)):
                pair_checks.append(f"- 「{section_names[i]}」 vs 「{section_names[j]}」")
        pair_check_str = "\n".join(pair_checks)
        
        prompt = f"""你是一个极其严谨的专业小说编辑，请对以下小说的各部分设定进行**全面彻底**的一致性检查。

{all_text}

**你必须逐一比对以下每一对内容，不得遗漏：**
{pair_check_str}

**检查步骤（必须严格执行）：**

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

        result = self.api.generate(prompt, temperature=0.2, max_tokens=max_tokens)
        return result
    
    def polish_with_style(self, text: str, style_reference: str, style_type: str = "作品", max_tokens: int = 2000) -> str:
        """模仿指定作品/作家的风格润色文本"""
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

        result = self.api.generate(prompt, temperature=0.7, max_tokens=max_tokens)
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

        result = self.api.generate(prompt, temperature=0.3, max_tokens=max_tokens)
        return result
