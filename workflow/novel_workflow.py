"""
全链路小说创作工作流
"""
from typing import Dict, List, Optional
from api.api_client import LLMAPIClient
from vector_store.local_chroma import LocalNovelVectorStore

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
        """第三步：生成总体大纲"""
        # 获取已有上下文
        world_setting = self.novel_info.get("world_setting", "")
        characters = self.novel_info.get("characters", "")
        
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

大纲要起承转合，有节奏起伏："""

        result = self.api.generate(prompt, temperature=0.7, max_tokens=max_tokens)
        self.vs.add_section("outline", "full_outline", result)
        self.novel_info["outline"] = result
        return result
    
    def generate_chapter(self, chapter_num: int, chapter_title: str, previous_summary: str = "", max_tokens: int = 2500, target_words: int = 2000) -> str:
        """生成单章正文 — 优化上下文：分类检索设定+前章，避免臃肿"""
        
        # ---- 分类构建上下文，避免一次性塞入过多内容 ----
        context_parts = []
        
        # 1. 核心设定：直接从 novel_info 获取（已生成的内容，比向量检索更完整更精准）
        setting_text = self.novel_info.get("world_setting", "")
        character_text = self.novel_info.get("characters", "")
        outline_text = self.novel_info.get("outline", "")
        
        if setting_text:
            # 世界观设定截断，最多 1500 字
            trunc = setting_text[:1500] + ("..." if len(setting_text) > 1500 else "")
            context_parts.append(f"【世界观设定】\n{trunc}")
        if character_text:
            # 人物设定截断，最多 2000 字
            trunc = character_text[:2000] + ("..." if len(character_text) > 2000 else "")
            context_parts.append(f"【人物设定】\n{trunc}")
        if outline_text:
            # 大纲中提取当前章节附近的内容（优先取相关部分）
            outline_for_chapter = self._extract_relevant_outline(outline_text, chapter_num)
            context_parts.append(f"【小说大纲（当前章节相关）】\n{outline_for_chapter}")
        
        # 2. 前几章摘要：只取最近2章的末尾段落作为前情回顾，不传全文
        prev_chapters_summary = self._get_previous_chapters_summary(chapter_num, max_chars=800)
        if prev_chapters_summary:
            context_parts.append(f"【前情回顾】\n{prev_chapters_summary}")
        
        # 3. 语义搜索补充：用当前章节标题在向量库做语义搜索(n_results=3)
        #    跳过已直接包含的设定/人物/大纲(避免重复)，其他章节内容每条截取前500字
        query = f"第{chapter_num}章 {chapter_title}"
        related = self.vs.search_related(query, n_results=3)
        extra_context = []
        for ctx in related:
            content = ctx["content"]
            # 跳过已经直接包含的核心设定（避免重复）
            meta = ctx.get("metadata", {})
            if meta.get("type") in ("setting", "character", "outline"):
                continue  # 这些已经从 novel_info 直接获取了
            # 前章内容只取短片段
            if meta.get("type") == "chapter":
                content = content[:500] + ("..." if len(content) > 500 else "")
            extra_context.append(content[:600])
        
        if extra_context:
            context_parts.append(f"【相关参考片段】\n" + "\n---\n".join(extra_context))
        
        context_text = "\n\n".join(context_parts)
        
        if previous_summary:
            context_text += f"\n\n上一章内容回顾：{previous_summary}\n"
        
        prompt = f"""请你根据以下信息，写出小说第 {chapter_num} 章 "{chapter_title}" 的完整正文。

{context_text}

要求：
- 字数大约在{target_words}字左右
- 情节符合大纲走向
- 保持人物设定一致性
- 文笔流畅，有画面感
- 直接输出正文，不要解释

正文："""

        result = self.api.generate(prompt, temperature=0.8, max_tokens=max_tokens)
        # 保存到向量库，方便后续章节检索
        self.vs.add_section("chapter", f"chapter_{chapter_num}", f"第{chapter_num}章 {chapter_title}\n{result}")
        return result
    
    def _extract_relevant_outline(self, outline: str, chapter_num: int) -> str:
        """从大纲中提取当前章节附近的内容（前后各2章的范围）"""
        import re
        lines = outline.split("\n")
        has_chapter_markers = any(re.match(r"第(\d+|[一二三四五六七八九十百]+)章", l) for l in lines)
        
        # 如果大纲没有章节标记格式，直接截断返回
        if not has_chapter_markers:
            return outline[:2000] + ("..." if len(outline) > 2000 else "")
        
        relevant_lines = []
        capturing = True  # 默认捕获（大纲开头总述部分也要）
        capture_range = 2  # 前后各取2章的描述
        cn_map = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        
        for line in lines:
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
        
        result = "\n".join(relevant_lines).strip()
        # 如果提取结果为空或太短，退回到截断策略
        if not result or len(result) < 20:
            result = outline[:2000] + ("..." if len(outline) > 2000 else "")
        elif len(result) > 2000:
            result = result[:2000] + "..."
        return result
    
    def _get_previous_chapters_summary(self, current_chapter_num: int, max_chars: int = 800) -> str:
        """获取前几章的摘要，只取最近2章的末尾段落，不传全文"""
        summaries = []
        # 只取最近2章
        for n in range(max(1, current_chapter_num - 2), current_chapter_num):
            content = self.vs.get_section("chapter", f"chapter_{n}")
            if content:
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
        # 检索相关上下文
        related = self.vs.search_related(current_chapter[-1000:], n_results=5)
        context_text = "前文内容和相关设定：\n"
        for ctx in related:
            context_text += f"{ctx['content']}\n---\n"
        
        full_prompt = f"""{context_text}

当前写到这里：
{current_chapter}

请继续往下写：{prompt}

要求：
- 续写大约{target_length}字
- 保持与前文风格一致
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
