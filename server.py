#!/usr/bin/env python3
"""
AI小说创作工具 - FastAPI + 单页H5 版本
运行: python server.py  （或 uvicorn server:app --port 8501）
打开: http://localhost:8501
"""
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from api.api_client import LLMAPIClient, GenerationCancelled
from api import user_config
from skills import skill_manager
from storage.json_store import JsonNovelStore
from storage import exporters
from workflow.novel_workflow import FullNovelWorkflow, is_ai_refusal, ChapterPaused

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "novels_data")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

user_config.ensure_env_migrated()

app = FastAPI(title="AI小说创作")

# 每步骤 max_tokens 默认值（与原 Streamlit 侧边栏一致）
DEFAULT_MAX_TOKENS = {
    "world_setting": 6000, "characters": 6000, "outline": 8000,
    "chapter": 16000, "golden_chapter": 16000, "batch_chapters": 16000,
    "continue": 8000, "polish": 8000, "consistency": 6000, "relation_graph": 6000,
    "extend_outline": 8000, "volume_chapters": 8000, "rewrite_outline": 6000,
    "migrate_cards": 6000, "memory_rebuild": 4000, "rewrite_preview": 2000,
}

# ---------- API 客户端缓存（按激活配置） ----------

_client_lock = threading.Lock()
_client_cache = {"key": None, "client": None}


def _build_client(provider: dict) -> LLMAPIClient:
    return LLMAPIClient(
        api_key=provider["api_key"],
        api_base=provider.get("api_base") or None,
        model=provider.get("model") or "doubao-pro-32k",
        max_output=provider.get("max_output") or None,
        reasoning_effort=provider.get("reasoning_effort") or None,
        thinking_disabled=bool(provider.get("thinking_disabled")),
    )


def get_client(fresh: bool = False) -> LLMAPIClient:
    """获取 API 客户端。fresh=True 时创建独立实例（生成任务用，
    避免并发任务共享单例导致取消标志/截断状态互相覆盖）。"""
    provider = user_config.get_active_provider()
    if not provider or not provider.get("api_key"):
        raise HTTPException(400, "尚未配置模型 API Key，请先在侧边栏「模型配置」中添加")
    if fresh:
        return _build_client(provider)
    key = (provider.get("api_key"), provider.get("api_base"), provider.get("model"),
           provider.get("max_output"), provider.get("reasoning_effort"),
           provider.get("thinking_disabled"))
    with _client_lock:
        if _client_cache["key"] != key:
            _client_cache["client"] = _build_client(provider)
            _client_cache["key"] = key
        return _client_cache["client"]


def get_store(novel_id: str) -> JsonNovelStore:
    return JsonNovelStore(db_path=DATA_DIR, novel_id=novel_id)


def get_workflow(novel_id: str, fresh_client: bool = False) -> FullNovelWorkflow:
    """构建 workflow 并从存储同步 novel_info（等价于原 init_app + load_from_vectorstore）"""
    client = get_client(fresh=fresh_client)
    store = get_store(novel_id)
    workflow = FullNovelWorkflow(client, store)
    client.skill_provider = lambda prompt, step: skill_manager.inject_skills(
        prompt, step, skill_manager.get_skill_states(store)
    )
    _sync_novel_info(workflow, store)
    return workflow


def _sync_novel_info(workflow: FullNovelWorkflow, store):
    """从存储同步 novel_info；顺带做旧数据迁移标记（无 delta 的旧小说台账标记 stale）"""
    data = store.load_all_to_dict()
    extra = data.get("extra", {})
    workflow.novel_info.update({
        "world_setting": data.get("world_setting", ""),
        "characters": data.get("characters", ""),
        "outline": data.get("outline", ""),
        "outline_total_chapters": extra.get("outline_total_chapters", ""),
        "chapters": data.get("chapters", {}),
    })
    if data.get("character_cards"):
        from workflow import character_cards as _cc
        workflow.novel_info["character_cards"] = _cc.cards_from_json(data["character_cards"])
    # 旧小说迁移标记：已有合并态台账但无按章 delta → 标记 stale，首次触发重建时全量重算
    if extra.get("state_ledger") and not extra.get("ledger_deltas") and "ledger_stale" not in extra:
        store.save_extra_data("ledger_stale", True)
    # 存量 volume_plan 清理：重复卷名前缀/重复卷/编号错乱，加载时自动修复并回写
    vp = extra.get("volume_plan")
    if vp:
        sanitized = FullNovelWorkflow.sanitize_volume_plan(vp)
        if sanitized != vp:
            store.save_extra_data("volume_plan", sanitized)
            workflow.novel_info["volume_plan"] = sanitized


def get_store_workflow(novel_id: str) -> FullNovelWorkflow:
    """构建**不依赖模型 API** 的 workflow（仅用于存储层操作：导入章节/空白章/失效处理等）"""
    store = get_store(novel_id)
    workflow = FullNovelWorkflow(None, store)
    _sync_novel_info(workflow, store)
    return workflow


# ---------- 任务队列 + SSE ----------

TASKS = {}
TASKS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=2)
TASK_TTL = 3600  # 已完成任务保留1小时


def _task_emit(task_id: str, event: str, data):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task:
            task["queue"].put((event, data))


def _task_confirm(task_id: str, msg: str, options: list):
    """emit need_confirm 并阻塞等待用户响应（POST /api/tasks/{id}/respond）。
    无时间上限；但前端 SSE 断开（关页面/断网）或任务被清理时返回 None，
    由调用方按「安全暂停」处理，避免线程无限阻塞 / 无人值守反复重试烧 token。"""
    ev = threading.Event()
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return None
        task["confirm_event"] = ev
        task["confirm_result"] = None
    try:
        _task_emit(task_id, "need_confirm", {"msg": msg, "options": options})
        while True:
            if ev.wait(timeout=10):
                with TASKS_LOCK:
                    t = TASKS.get(task_id) or {}
                    return t.get("confirm_result")
            # 每 10s 做一次存活检查：前端断开则放弃等待（进度已落盘，可下次继续）
            with TASKS_LOCK:
                t = TASKS.get(task_id)
                if not t or t.get("client_disconnected"):
                    return None
    finally:
        with TASKS_LOCK:
            t = TASKS.get(task_id)
            if t:
                t.pop("confirm_event", None)
                t.pop("confirm_result", None)


def _run_generation(task_id: str, novel_id: str, step: str, params: dict):
    """在线程池中执行生成任务，通过队列推送 progress/token 事件，最后推送 done/error"""
    emit = lambda e, d: _task_emit(task_id, e, d)
    try:
        workflow = get_workflow(novel_id, fresh_client=True)
        client = workflow.api
        # 记录任务使用的配置快照，便于排查"以为在用 A 模型其实在用 B"
        provider = user_config.get_active_provider() or {}
        logger.info(f"任务开始 → provider={provider.get('name', '?')}, model={client.model}, step={step}")
        with TASKS_LOCK:
            t = TASKS.get(task_id)
            if t is not None:
                t["client"] = client
                t["model"] = client.model
        # 取消检查：任务被标记 cancelled 时，API 层在 chunk 级/请求前中断
        client.cancel_check = lambda: TASKS.get(task_id, {}).get("cancelled", False)
        # progress 事件 payload 统一为对象 {msg, stage?, stage_total?, phase?}，兼容纯文本回调
        workflow.on_progress = lambda msg, **fields: emit("progress", {"msg": msg, **fields})
        # 黄金开篇会生成两个版本，逐字流会造成两个版本文本混在一起，只用进度提示
        if step != "golden_chapter":
            workflow.on_token = lambda tok: emit("token", tok)
            workflow.on_reasoning = lambda tok: emit("reasoning", tok)
        # 用户决策回调：emit need_confirm 后阻塞等待 /respond，前端断开则安全暂停
        workflow.on_confirm = lambda msg, options: _task_confirm(task_id, msg, options)
        vs = workflow.vs
        # max_tokens 优先级：本次请求参数 > 用户全局覆盖表 > 内置默认值
        mt = int(params.get("max_tokens") or
                 user_config.get_max_tokens_overrides().get(step) or
                 DEFAULT_MAX_TOKENS.get(step, 4000))
        result_payload = {}
        warning = ""

        if step == "world_setting":
            user_prompt = params.get("prompt", "").strip()
            if not user_prompt:
                raise ValueError("请输入世界观描述")
            result = workflow.generate_world_setting(user_prompt, max_tokens=mt)
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次生成请求，请修改描述内容后重试（可能触发了内容安全审查）")
            vs.save_extra_data("world_setting_original", result)
            vs.save_extra_data("world_setting_prompt", user_prompt)

        elif step == "characters":
            user_prompt = params.get("prompt", "").strip()
            if not user_prompt:
                raise ValueError("请输入人物设定要求")
            num_main = int(params.get("num_main", 2))
            num_support = int(params.get("num_support", 5))
            result = workflow.generate_characters(user_prompt, num_main, num_support, max_tokens=mt)
            char_text = result["characters"]
            if is_ai_refusal(char_text):
                raise ValueError("AI拒绝了本次生成请求，请修改描述内容后重试（可能触发了内容安全审查）")
            vs.save_extra_data("characters_original", char_text)
            vs.save_extra_data("characters_prompt", user_prompt)
            vs.save_extra_data("characters_num_main", str(num_main))
            vs.save_extra_data("characters_num_support", str(num_support))

        elif step == "outline":
            user_prompt = params.get("prompt", "").strip()
            if not user_prompt:
                raise ValueError("请输入大纲要求")
            total_chapters = int(params.get("total_chapters", 50))
            words_per_chapter = int(params.get("words_per_chapter", 2000))
            result = workflow.generate_outline(user_prompt, total_chapters, words_per_chapter, max_tokens=mt)
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次生成请求，请修改描述内容后重试（可能触发了内容安全审查）")
            vs.save_extra_data("outline_original", result)
            vs.save_extra_data("outline_prompt", user_prompt)
            vs.save_extra_data("outline_total_chapters", str(total_chapters))
            vs.save_extra_data("outline_words_per_chapter", str(words_per_chapter))

        elif step == "chapter":
            chapter_num = int(params.get("chapter_num", 1))
            chapter_title = params.get("chapter_title", "").strip()
            # 标题留空 → workflow 内先从大纲取，再由 AI 拟定
            # 字数留空（0）→ workflow 内从大纲带过来（逐章字数 → 全局每章字数 → 默认）
            target_words = int(params.get("target_words") or 0)
            result = workflow.generate_chapter_with_rag(
                chapter_num, chapter_title, max_tokens=mt,
                target_words=target_words, beats=params.get("beats", ""),
                extra_instruction=params.get("extra_instruction", ""))
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次生成请求，请修改章节内容或标题后重试（可能触发了内容安全审查）")
            warning = getattr(workflow, "last_scope_warning", "")
            generated_title = getattr(workflow, "last_chapter_title", "")
            if not chapter_title and generated_title:
                result_payload["generated_title"] = generated_title

        elif step == "chapter_beats":
            chapter_num = int(params.get("chapter_num", 1))
            chapter_title = params.get("chapter_title", "").strip()
            target_words = int(params.get("target_words") or 0)
            beats = workflow.generate_chapter_beats(chapter_num, chapter_title, target_words=target_words, max_tokens=mt)
            if is_ai_refusal(beats):
                raise ValueError("AI拒绝了本次细纲生成请求，请修改内容后重试")
            result_payload["beats"] = beats
            warning = getattr(workflow, "last_beats_warning", "")
            if warning:
                result_payload["warning"] = warning

        elif step == "validate_beats":
            # 节拍程序化校验（实体时间锁）：检查各场景是否点名尚未登场的角色
            chapter_num = int(params.get("chapter_num", 1))
            beats_text = params.get("beats", "")
            check = workflow.validate_beats(beats_text, chapter_num)
            result_payload["ok"] = check["ok"]
            result_payload["issues"] = check["issues"]

        elif step == "golden_chapter":
            chapter_num = int(params.get("chapter_num", 1))
            chapter_title = params.get("chapter_title", "").strip()
            # 标题留空 → workflow 内先从大纲取，再由 AI 拟定（两个版本共用同一标题）
            target_words = int(params.get("target_words") or 0)
            golden = workflow.generate_golden_chapter(
                chapter_num, chapter_title, max_tokens=mt,
                target_words=target_words, beats=params.get("beats", ""))
            if is_ai_refusal(golden["content"]):
                raise ValueError("AI拒绝了本次生成请求，请修改内容后重试（可能触发了内容安全审查）")
            vs.save_extra_data(f"chapter_review_{chapter_num}", golden["review"])
            vs.save_extra_data(f"chapter_golden_{chapter_num}", golden)
            s1, s2 = golden["scores"]
            warning = (f"🏆 黄金开篇择优：版本A {s1}分 / 版本B {s2}分，"
                       f"已选版本{'AB'[golden['picked'] - 1]}。另一版可在章节下方对比查看。")
            result_payload["golden"] = golden
            generated_title = getattr(workflow, "last_chapter_title", "")
            if not chapter_title and generated_title:
                result_payload["generated_title"] = generated_title

        elif step == "continue":
            current_text = params.get("continue_text", "")
            if not current_text.strip():
                raise ValueError("请输入需要续写的内容")
            target_length = int(params.get("continue_length", 1500))
            result = workflow.continue_writing(
                current_text, params.get("continue_prompt", "继续往下写"),
                target_length, max_tokens=mt)
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次续写请求，请修改内容后重试（可能触发了内容安全审查）")
            result_payload["result"] = result
            result_payload["merged"] = current_text + "\n\n" + result

        elif step == "polish":
            polish_text = params.get("polish_text", "")
            style_reference = params.get("style_reference", "").strip()
            if not polish_text.strip() or not style_reference:
                raise ValueError("请输入要润色的文本和风格参考")
            style_type = params.get("style_type", "作品")
            result = workflow.polish_with_style(polish_text, style_reference, style_type, max_tokens=mt)
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次润色请求，请修改内容后重试（可能触发了内容安全审查）")
            result_payload["result"] = result
            result_payload["original"] = polish_text
            result_payload["style_label"] = f"《{style_reference}》" if style_type == "作品" else style_reference

        elif step == "chapter_review":
            chapter_num = int(params.get("chapter_num", 1))
            content = params.get("content", "")
            if not content.strip():
                raise ValueError("没有可评审的章节内容")
            review = workflow.review_chapter(chapter_num, params.get("chapter_title", ""), content, max_tokens=mt)
            vs.save_extra_data(f"chapter_review_{chapter_num}", review)
            result_payload["review"] = review

        elif step == "extend_outline":
            # TODO 3.1：增量扩展大纲（旧大纲逐字保留）
            additional = int(params.get("additional_chapters", 20))
            if additional <= 0:
                raise ValueError("扩展章数必须大于 0")
            appended = workflow.extend_outline(additional, max_tokens=mt)
            if not appended:
                raise ValueError("AI拒绝了本次扩展请求，或输出为空，请重试")
            result_payload["appended"] = appended

        elif step == "volume_chapters":
            # TODO 3.2.0：手动提前触发/重新生成指定卷的逐章概要
            volume_index = int(params.get("volume_index", 0))
            force = bool(params.get("force", False))
            vol_result = workflow.generate_volume_chapters(volume_index, max_tokens=mt, force=force)
            if vol_result is None:
                raise ValueError("该卷细纲已存在（如需重写请用「重新生成概要」）或卷号无效")
            result_payload["result"] = vol_result

        elif step == "rewrite_outline":
            # TODO 4.1：局部改写指定章节范围的大纲条目，其余逐字保留
            start, end = int(params.get("start", 1)), int(params.get("end", 1))
            instruction = params.get("instruction", "").strip()
            if not instruction or end < start:
                raise ValueError("请填写改写要求，且章节范围有效")
            workflow.rewrite_outline_range(start, end, instruction, max_tokens=mt)

        elif step == "migrate_cards":
            # TODO 2.2：旧自由文本人物设定 → 角色卡（只返回预览，确认后由 PUT 接口入库）
            preview = workflow.migrate_characters_to_cards(max_tokens=mt)
            if not preview:
                raise ValueError("迁移解析失败（AI 未返回有效的角色卡格式），已保持自由文本模式")
            result_payload["preview"] = preview

        elif step == "memory_rebuild":
            # TODO 1.1：手动重建台账与摘要；regen=true 逐章重调 LLM（有 token 成本，UI 已明示）
            from_chapter = int(params.get("from_chapter", 1))
            regen = bool(params.get("regen", False))
            result_payload["rebuild"] = workflow.rebuild_memory(from_chapter, regen=regen, max_tokens=mt)

        elif step == "rewrite_preview":
            # TODO 4.2：单章"AI 改写建议预览"（不改正文，用户确认后再走 chapter 重写）
            chapter_num = int(params.get("chapter_num", 1))
            instruction = params.get("instruction", "").strip()
            if not instruction:
                raise ValueError("请填写新设定/变更说明")
            preview = workflow.preview_chapter_rewrite(chapter_num, instruction, max_tokens=mt)
            result_payload["preview"] = preview

        elif step == "chapter_revise":
            chapter_num = int(params.get("chapter_num", 1))
            chapter_title = params.get("chapter_title", "")
            content = params.get("content", "")
            review = params.get("review", "")
            if not content.strip() or not review.strip():
                raise ValueError("需要先评审再改写")
            revised = workflow.revise_chapter(chapter_num, chapter_title, content, review, max_tokens=mt)
            if is_ai_refusal(revised):
                raise ValueError("AI拒绝了本次改写请求，请修改内容后重试")
            vs.update_section("chapter", f"chapter_{chapter_num}",
                              f"第{chapter_num}章 {chapter_title}\n{revised}")
            # 正文被改写 → 该章及之后的台账 delta 失效重建（TODO 1.1）
            workflow.novel_info.setdefault("chapters", {})[str(chapter_num)] = {
                "title": chapter_title, "content": revised}
            workflow.invalidate_memory_from(chapter_num)

        elif step == "style_fingerprint":
            fp = workflow.extract_style_fingerprint(
                sample_text=params.get("sample", ""), description=params.get("description", ""), max_tokens=mt)
            if is_ai_refusal(fp):
                raise ValueError("AI拒绝了本次提取请求，请修改内容后重试")
            if not fp:
                raise ValueError("请提供文风样例或描述")

        elif step == "humanize":
            text = params.get("humanize_text", "")
            if not text.strip():
                raise ValueError("没有需要去AI腔的文本")
            result = workflow.humanize_text(text, max_tokens=mt)
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次改写请求，请修改内容后重试")
            result_payload["result"] = result

        elif step == "batch_chapters":
            start = int(params.get("start", 1))
            count = int(params.get("count", 3))
            target_words = int(params.get("target_words") or 0)  # 0 → 每章按大纲字数（workflow 内逐章解析）
            use_auto_beats = bool(params.get("auto_beats", False))
            fill_blank = bool(params.get("fill_blank", False))  # 是否填充空白章（默认跳过，尊重手写意图）
            titles = workflow.get_outline_chapter_titles()
            chapters = workflow.novel_info.get("chapters", {})
            done, skipped, failed = [], [], []
            for n in range(start, start + count):
                existing = chapters.get(str(n))
                # 已有正文的章跳过；空白章（手写占位）默认也跳过，勾选"填充空白章"才生成
                if existing is not None and (existing.get("content") or "").strip():
                    skipped.append(n)
                    continue
                if existing is not None and not fill_blank:
                    skipped.append(n)
                    continue
                title = (existing or {}).get("title") or titles.get(n, "")
                if not title:
                    # 大纲也无标题 → 由 AI 根据上下文拟定
                    try:
                        title = workflow.generate_chapter_title(n)
                    except Exception as e:
                        logger.warning(f"批量生成第{n}章 AI 拟题失败: {e}")
                        title = ""
                    if not title:
                        failed.append(f"第{n}章（无法确定标题）")
                        continue
                try:
                    emit("progress", f"批量生成：开始第{n}章「{title}」（{len(done) + 1}/{count}）")
                    beats = ""
                    if use_auto_beats:
                        emit("progress", f"第{n}章：正在生成场景节拍…")
                        beats = workflow.generate_chapter_beats(n, title, target_words=target_words)
                        if is_ai_refusal(beats):
                            beats = ""
                    result = workflow.generate_chapter_with_rag(
                        n, title, max_tokens=mt, target_words=target_words, beats=beats)
                    if is_ai_refusal(result):
                        failed.append(f"第{n}章（被AI拒绝）")
                        continue
                    chapters[str(n)] = {"title": title, "content": result}
                    done.append(n)
                    emit("chapter_done", {"num": n, "title": title})
                except Exception as e:
                    logger.error(f"批量生成第{n}章失败: {e}")
                    failed.append(f"第{n}章（{str(e)[:50]}）")
            summary = f"✅ 成功 {len(done)} 章"
            if skipped:
                summary += f"，⏭️ 跳过已有 {len(skipped)} 章"
            if failed:
                summary += f"，❌ 失败：{'、'.join(failed)}"
            result_payload["summary"] = summary

        elif step == "consistency":
            result = workflow.check_consistency(max_tokens=mt)
            vs.save_extra_data("consistency_result", result)

        elif step == "relation_graph":
            raw_result = workflow.extract_character_relations(max_tokens=mt)
            json_start = raw_result.find("{")
            json_end = raw_result.rfind("}") + 1
            if json_start < 0 or json_end <= json_start:
                raise ValueError("AI返回格式异常，请重试")
            try:
                graph_data = json.loads(raw_result[json_start:json_end])
            except json.JSONDecodeError as e:
                result_payload["raw"] = raw_result
                raise ValueError(f"解析失败: {str(e)[:100]}（可在 AI 返回原文中查看）")
            vs.save_extra_data("relation_graph", graph_data)
            result_payload["graph"] = graph_data

        elif step == "distill":
            articles = params.get("articles", "").strip()
            if not articles:
                raise ValueError("请粘贴参考文章内容")
            prompt = skill_manager.DISTILL_PROMPT_TEMPLATE.format(articles=articles)
            emit("progress", "正在蒸馏写作技能…")
            result = workflow.api.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.5, max_tokens=mt or 6000)
            if is_ai_refusal(result):
                raise ValueError("AI拒绝了本次蒸馏请求，请修改内容后重试")
            result_payload["result"] = result

        else:
            raise ValueError(f"未知的生成步骤: {step}")

        if warning:
            result_payload["warning"] = warning
        # 截断/思考参数未生效的告警（最后一次调用的状态；多次调用时以最后为准，聊胜于无）
        if getattr(client, "last_finish_reason", None) == "length":
            trunc_warn = "⚠️ 输出达到 token 上限被截断（finish_reason=length），建议调大该步骤 max_tokens"
            result_payload["warning"] = (result_payload.get("warning", "") + "；" + trunc_warn).strip("；")
        if getattr(client, "thinking_disable_ignored", False):
            td_warn = "⚠️ 已请求关闭思考，但模型仍在输出思考内容——该服务商可能不支持 thinking 参数"
            result_payload["warning"] = (result_payload.get("warning", "") + "；" + td_warn).strip("；")
        emit("done", result_payload)
    except ChapterPaused as e:
        # 用户暂停/取消：进度已保存，不加"已有内容保持不变"后缀
        logger.info(f"任务 {task_id} 已暂停: {e}")
        emit("error", str(e))
    except GenerationCancelled as e:
        logger.info(f"任务 {task_id} 已被用户取消: {e}")
        emit("error", f"⏹ 已取消：{e}")
    except Exception as e:
        logger.exception(f"任务 {task_id} 失败")
        emit("error", f"{str(e)}（已有内容保持不变）")
    finally:
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["finished_at"] = time.time()
        _task_emit(task_id, "__end__", None)


def _gc_tasks():
    now = time.time()
    with TASKS_LOCK:
        for tid in [t for t, v in TASKS.items()
                    if v.get("finished_at") and now - v["finished_at"] > TASK_TTL]:
            del TASKS[tid]


class GenerateRequest(BaseModel):
    novel_id: str
    step: str
    params: dict = {}


@app.post("/api/generate")
def start_generation(req: GenerateRequest):
    _gc_tasks()
    task_id = uuid.uuid4().hex[:12]
    with TASKS_LOCK:
        TASKS[task_id] = {"queue": queue.Queue(), "created_at": time.time(),
                          "step": req.step, "novel_id": req.novel_id}
    EXECUTOR.submit(_run_generation, task_id, req.novel_id, req.step, req.params or {})
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}/stream")
def task_stream(task_id: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在或已过期")
    q = task["queue"]
    with TASKS_LOCK:
        # 新连接建立（含浏览器自动重连），清除断开标记
        if task_id in TASKS:
            TASKS[task_id]["client_disconnected"] = False

    async def event_source():
        # 异步生成器：客户端断开时 uvicorn 会取消协程，finally 一定执行
        # （同步生成器跑在线程池里，断开时无法可靠感知）
        try:
            while True:
                try:
                    event, data = await asyncio.to_thread(q.get, True, 30)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if event == "__end__":
                    break
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            # 客户端断开（关页面/断网/浏览器重连前）→ 标记，_task_confirm 据此放弃等待
            with TASKS_LOCK:
                t = TASKS.get(task_id)
                if t and not t.get("finished_at"):
                    t["client_disconnected"] = True
                    # 刷新/关闭页面≠取消：任务与 token 消耗仍在后台继续
                    logger.warning(f"任务 {task_id}（{t.get('step', '?')}）前端已断开，"
                                   f"任务仍在后台执行，token 继续消耗；"
                                   f"如需停止请重新打开页面后使用取消功能")

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """用户手动取消任务：置 cancelled 标志并强断进行中的 HTTP 连接。
    已落盘的断点进度保留；注意刷新/关闭页面不会触发本接口（任务在后台继续）。"""
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or task.get("finished_at"):
            raise HTTPException(404, "任务不存在或已结束")
        task["cancelled"] = True
        client = task.get("client")
        # 若有正在等待用户决策的确认框，用 resume_later 解除阻塞，让任务线程走到取消检查点
        ev = task.get("confirm_event")
        if ev is not None:
            task["confirm_result"] = "resume_later"
    if client is not None:
        client.cancel()  # 强断阻塞中的请求，流式/非流式都会立即抛错
    if ev is not None:
        ev.set()
    logger.info(f"任务 {task_id} 收到取消请求")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/respond")
def respond_task(task_id: str, body: dict):
    """用户对 need_confirm 事件做出决策：{"action": "retry"|"resume_later"|"cancel"|...}"""
    action = (body.get("action") or "").strip()
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task or "confirm_event" not in task:
            raise HTTPException(404, "任务不存在或没有待响应的确认")
        task["confirm_result"] = action
        task["confirm_event"].set()
    return {"ok": True}


# ---------- 模型配置 ----------

class ProviderIn(BaseModel):
    name: str
    api_key: str = ""
    api_base: str = ""
    model: str = ""
    max_output: Optional[int] = None           # 单次输出 token 上限（查服务商文档填写）
    reasoning_effort: Optional[str] = None     # 思考强度：low/medium/high，空 = 不传参
    thinking_disabled: Optional[bool] = None   # 关闭思考（注入 thinking.type=disabled）


@app.get("/api/providers")
def list_providers():
    cfg = user_config.load_config()
    return {"providers": cfg.get("providers", []),
            "active": user_config.get_active_provider() or {}}


@app.post("/api/providers")
def upsert_provider(p: ProviderIn):
    if not p.name.strip():
        raise HTTPException(400, "配置名称不能为空")
    data = {k: v for k, v in p.dict().items() if v not in (None, "")}
    data["name"] = p.name  # name 必保留
    # 保留探测得到的能力信息（reasoning / reasoning_effort_options / thinking_disable），表单不提交这些字段
    old = user_config.get_provider(p.name) or {}
    for k in ("reasoning", "reasoning_effort_options", "thinking_disable"):
        if k in old and k not in data:
            data[k] = old[k]
    user_config.upsert_provider(data)
    user_config.set_active_provider(p.name)
    with _client_lock:
        _client_cache["key"] = None
    return {"ok": True}


@app.post("/api/providers/active")
def set_active_provider(body: dict):
    user_config.set_active_provider(body.get("name", ""))
    with _client_lock:
        _client_cache["key"] = None
    # 切换不影响正在运行的任务（它们持有旧 client 直到完成）；返回数量供前端提示
    with TASKS_LOCK:
        running = [t for t in TASKS.values() if not t.get("finished_at")]
    return {"ok": True, "running_tasks": len(running),
            "running_models": sorted({t.get("model", "?") for t in running})}


@app.delete("/api/providers/{name}")
def delete_provider(name: str):
    user_config.delete_provider(name)
    with _client_lock:
        _client_cache["key"] = None
    return {"ok": True}


@app.post("/api/providers/test")
def test_provider(p: ProviderIn):
    client = LLMAPIClient(api_key=p.api_key, api_base=p.api_base or None,
                          model=p.model or "doubao-pro-32k",
                          max_output=p.max_output or None)
    ok, msg, latency, caps = client.test_connection()
    # 探测成功 → 把能力信息写回已有同名配置（reasoning / 支持的思考强度档位）
    if ok and p.name and user_config.get_provider(p.name):
        fields = {"reasoning": caps.get("reasoning", False),
                  "thinking_disable": bool(caps.get("thinking_disable"))}
        if caps.get("reasoning_effort_options"):
            fields["reasoning_effort_options"] = caps["reasoning_effort_options"]
            # 当前所选强度不在支持列表中 → 清空，避免发送无效参数
            cur = (user_config.get_provider(p.name) or {}).get("reasoning_effort")
            if cur and cur not in caps["reasoning_effort_options"]:
                fields["reasoning_effort"] = None
        else:
            fields["reasoning_effort_options"] = None
            fields["reasoning_effort"] = None
        user_config.update_provider_fields(p.name, fields)
        with _client_lock:
            _client_cache["key"] = None
    return {"ok": ok, "msg": msg, "latency": round(latency), "caps": caps}


@app.get("/api/usage")
def get_usage():
    try:
        session = dict(get_client().session_usage)
    except HTTPException:
        session = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "total_chars_in": 0, "total_chars_out": 0}
    return {"session": session, "cumulative": user_config.get_cumulative_usage(),
            "by_model": user_config.get_usage_by_model(),
             "skill_inject_chars": skill_manager.get_inject_max_chars()}


# ---------- 每步 max_tokens 覆盖设置 ----------

# 覆盖表中额外暴露的非 DEFAULT_MAX_TOKENS 步骤（内部计算、不在请求参数里的）
EXTRA_TOKEN_STEPS = {
    "chapter_scene": "章节·每场景（按场景卡分段生成时单次调用，推理模型需调大）",
}


@app.get("/api/settings/max-tokens")
def get_max_tokens_settings():
    return {"defaults": DEFAULT_MAX_TOKENS,
            "extra_steps": EXTRA_TOKEN_STEPS,
            "overrides": user_config.get_max_tokens_overrides()}


@app.post("/api/settings/max-tokens")
def set_max_tokens_settings(body: dict):
    allowed = set(DEFAULT_MAX_TOKENS) | set(EXTRA_TOKEN_STEPS)
    overrides = {k: v for k, v in (body.get("overrides") or {}).items() if k in allowed}
    user_config.set_max_tokens_overrides(overrides)
    return {"ok": True, "overrides": user_config.get_max_tokens_overrides()}


@app.post("/api/usage/clear")
def clear_usage(body: dict):
    """清除用量统计：body.model 为空 → 全部清除；否则只清除该模型"""
    model = (body.get("model") or "").strip()
    user_config.clear_usage(model)
    if not model:
        # 全部清除时同时重置当前会话用量
        with _client_lock:
            c = _client_cache.get("client")
            if c:
                for k in c.session_usage:
                    c.session_usage[k] = 0
    return {"ok": True}


# ---------- 小说管理 ----------

@app.get("/api/novels")
def list_novels():
    return {"novels": JsonNovelStore.list_all_novels(db_path=DATA_DIR)}


@app.post("/api/novels")
def create_novel(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "请输入小说名称")
    novel_id = f"novel_{int(time.time() * 1000)}"
    JsonNovelStore(db_path=DATA_DIR, novel_id=novel_id, novel_name=name)
    return {"id": novel_id, "name": name}


@app.get("/api/novels/{novel_id}/chapter-partials")
def list_chapter_partials(novel_id: str):
    """列出该小说所有未完成的章节断点（按场景生成的中途进度）"""
    store = get_store(novel_id)
    extra = store.load_extra_data() or {}
    partials = []
    for k, v in extra.items():
        if k.startswith("chapter_partial_") and isinstance(v, dict):
            try:
                num = int(k.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                continue
            total = len(FullNovelWorkflow.parse_beats(v.get("beats_text", "")) or [])
            partials.append({"chapter_num": num, "title": v.get("title", ""),
                             "done_scenes": len(v.get("parts") or []), "total_scenes": total})
    return {"partials": sorted(partials, key=lambda x: x["chapter_num"])}


@app.get("/api/novels/{novel_id}")
def get_novel(novel_id: str):
    store = get_store(novel_id)
    data = store.load_all_to_dict()
    data["id"] = novel_id
    data["name"] = store._data.get("novel_name") or novel_id
    # 存量 volume_plan 清理：重复卷名前缀/重复卷/编号错乱，读取时自动修复并回写
    vp = data.get("extra", {}).get("volume_plan")
    if vp:
        sanitized = FullNovelWorkflow.sanitize_volume_plan(vp)
        if sanitized != vp:
            store.save_extra_data("volume_plan", sanitized)
            data["extra"]["volume_plan"] = sanitized
    # 大纲正文中相邻重复的「第N卷：第N卷：」前缀一次性修复并回写
    outline = data.get("outline") or ""
    if outline:
        fixed = re.sub(r'(第\s*[一二三四五六七八九十两\d]+\s*卷\s*[：:])\s*第\s*[一二三四五六七八九十两\d]+\s*卷\s*[：:]', r'\1', outline)
        if fixed != outline:
            store.add_section("outline", "full_outline", fixed)
            data["outline"] = fixed
    return data


@app.post("/api/novels/{novel_id}/rename")
def rename_novel(novel_id: str, body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "名称不能为空")
    get_store(novel_id).rename(name)
    return {"ok": True}


@app.delete("/api/novels/{novel_id}")
def delete_novel(novel_id: str):
    get_store(novel_id).delete_novel()
    return {"ok": True}


@app.delete("/api/novels")
def delete_all_novels():
    count = JsonNovelStore.delete_all_novels(db_path=DATA_DIR)
    return {"ok": True, "count": count}


class SectionIn(BaseModel):
    type: str
    title: str
    content: str = ""


@app.put("/api/novels/{novel_id}/section")
def put_section(novel_id: str, sec: SectionIn):
    get_store(novel_id).update_section(sec.type, sec.title, sec.content)
    # 手动编辑章节正文 → 触发记忆失效处理（TODO 1.1：编辑保存处挂钩）
    if sec.type == "chapter":
        m = re.match(r"chapter_(\d+)", sec.title)
        if m:
            wf = get_store_workflow(novel_id)
            body = sec.content.split("\n", 1)
            ch = wf.novel_info.setdefault("chapters", {})
            ch[m.group(1)] = {"title": "", "content": body[1] if len(body) > 1 else sec.content}
            wf.invalidate_memory_from(int(m.group(1)))
    return {"ok": True}


@app.delete("/api/novels/{novel_id}/section")
def delete_section(novel_id: str, type: str, title: str):
    get_store(novel_id).delete_section(type, title)
    return {"ok": True}


class ChapterSummariesIn(BaseModel):
    content: str = ""


@app.get("/api/novels/{novel_id}/chapter_summaries")
def get_chapter_summaries(novel_id: str):
    """取大纲中的逐章概要段（「## 逐章概要」标记之后），供独立编辑"""
    wf = get_store_workflow(novel_id)
    return {"summaries": wf.get_chapter_summaries()}


@app.put("/api/novels/{novel_id}/chapter_summaries")
def put_chapter_summaries(novel_id: str, body: ChapterSummariesIn):
    """整体替换逐章概要段（卷级大纲逐字保留），数据仍存于大纲 section 内"""
    wf = get_store_workflow(novel_id)
    if not (body.content or "").strip():
        raise HTTPException(400, "逐章概要内容不能为空")
    new_outline = wf.update_chapter_summaries(body.content)
    return {"ok": True, "outline": new_outline}


@app.put("/api/novels/{novel_id}/extra/{key}")
def put_extra(novel_id: str, key: str, body: dict):
    get_store(novel_id).save_extra_data(key, body.get("value"))
    return {"ok": True}


@app.delete("/api/novels/{novel_id}/extra/{key}")
def delete_extra(novel_id: str, key: str):
    get_store(novel_id).delete_extra_field(key)
    return {"ok": True}


# ---------- 章节导入 / 空白章节 / 记忆状态（TODO 1.2 / 1.3 / 1.1） ----------

@app.post("/api/novels/{novel_id}/chapters/import")
def import_chapter(novel_id: str, body: dict):
    """导入外部章节（.txt/.md 内容或粘贴文本）入库；章节号冲突由前端确认后覆盖。
    入库后自动从该章起做记忆失效处理。"""
    chapter_num = int(body.get("chapter_num", 0))
    content = (body.get("content") or "").strip()
    if chapter_num <= 0 or not content:
        raise HTTPException(400, "请提供有效的章节号与正文内容")
    title = (body.get("title") or "").strip()
    wf = get_store_workflow(novel_id)
    result = wf.import_chapter(chapter_num, title, content)
    store = get_store(novel_id)
    result["ledger_stale"] = bool(store.load_extra_data("ledger_stale", False))
    return result


@app.post("/api/novels/{novel_id}/chapters/blank")
def create_blank_chapter(novel_id: str, body: dict):
    """新建空白章节占位（手写新章入口）；批量生成默认跳过空章"""
    chapter_num = int(body.get("chapter_num", 0))
    if chapter_num <= 0:
        raise HTTPException(400, "请提供有效的章节号")
    wf = get_store_workflow(novel_id)
    existing = wf.novel_info.get("chapters", {}).get(str(chapter_num))
    if existing is not None and (existing.get("content") or "").strip():
        raise HTTPException(400, f"第{chapter_num}章已有正文，不能创建空白占位")
    return wf.create_blank_chapter(chapter_num, body.get("title", ""))


@app.get("/api/novels/{novel_id}/memory/status")
def memory_status(novel_id: str):
    """台账/摘要状态：stale 标记（有章节变动但 delta 未重算）与 delta 覆盖范围"""
    store = get_store(novel_id)
    deltas = store.load_extra_data("ledger_deltas", {}) or {}
    ledger = store.load_extra_data("state_ledger", {}) or {}
    # 台账是否有实质内容（rebuild 会产生空壳 {"characters": [], ...}，不算有台账）
    has_ledger = any(ledger.get(k) for k in ("characters", "timeline", "foreshadowing"))
    return {
        "ledger_stale": bool(store.load_extra_data("ledger_stale", False)),
        "ledger_stale_from": store.load_extra_data("ledger_stale_from", None),
        "delta_chapters": sorted(int(k) for k in deltas),
        "has_ledger": has_ledger,
        "has_summary": bool(store.load_extra_data("rolling_summary", "")),
    }


# ---------- 角色卡（TODO 2.x）与登场调度 / 回溯分析（TODO 4.x） ----------

@app.get("/api/novels/{novel_id}/character_cards")
def get_character_cards(novel_id: str):
    """读取角色卡；无角色卡但有自由文本人物设定时 mode=freetext（可触发迁移）"""
    wf = get_store_workflow(novel_id)
    cards = wf.load_character_cards()
    return {"cards": cards,
            "mode": "structured" if cards else "freetext",
            "has_freetext": bool(wf.novel_info.get("characters", "").strip())}


@app.put("/api/novels/{novel_id}/character_cards")
def put_character_cards(novel_id: str, body: dict):
    """保存角色卡（编辑器保存 / 迁移预览确认共用）"""
    cards = body.get("cards") or []
    if not isinstance(cards, list):
        raise HTTPException(400, "cards 必须是列表")
    wf = get_store_workflow(novel_id)
    saved = wf.save_character_cards(cards)
    return {"ok": True, "cards": saved}


@app.delete("/api/novels/{novel_id}/character_cards")
def delete_character_cards(novel_id: str):
    """切回自由文本模式：删除角色卡 section（自由文本人物设定保留）"""
    get_store(novel_id).delete_section("character_cards", "cards")
    return {"ok": True}


@app.post("/api/novels/{novel_id}/characters/check_appearance")
def check_appearance(novel_id: str, body: dict):
    """登场调度检查（TODO 4.1）：大纲第 N±2 章是否提及该角色"""
    name = (body.get("name") or "").strip()
    chapter = int(body.get("chapter", 1))
    if not name:
        raise HTTPException(400, "缺少角色名")
    wf = get_store_workflow(novel_id)
    return wf.check_appearance_in_outline(name, chapter)


@app.post("/api/novels/{novel_id}/impact_scan")
def impact_scan(novel_id: str, body: dict):
    """回溯影响分析（TODO 4.2 第 1 步）：扫描已生成章节中提及关键词的章节清单"""
    keywords = body.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in re.split(r"[,，、\n]", keywords) if k.strip()]
    if not keywords:
        raise HTTPException(400, "请提供至少一个关键词（如新增主角的名字）")
    wf = get_store_workflow(novel_id)
    return {"impacted": wf.scan_impacted_chapters(keywords)}


# ---------- 导出 ----------

@app.get("/api/novels/{novel_id}/export/{fmt}")
def export_novel(novel_id: str, fmt: str):
    store = get_store(novel_id)
    data = store.load_all_to_dict()
    name = store._data.get("novel_name") or novel_id
    chapters = data.get("chapters", {})
    if not chapters:
        raise HTTPException(400, "还没有任何章节内容，无法导出")
    if fmt == "md":
        # 完整版：含世界观/人物/大纲/正文
        parts = [f"# {name}\n"]
        if data.get("world_setting"):
            parts.append(f"\n## 世界观设定\n\n{data['world_setting']}\n")
        if data.get("characters"):
            parts.append(f"\n## 人物设定\n\n{data['characters']}\n")
        if data.get("outline"):
            parts.append(f"\n## 小说大纲\n\n{data['outline']}\n")
        parts.append("\n---\n\n# 正文\n")
        path = OUTPUT_DIR / f"{name}.md"
        path.write_text("".join(parts) + "\n" +
                        exporters.build_markdown(name, chapters).split("\n", 1)[1],
                        encoding="utf-8")
        return FileResponse(path, filename=f"{name}.md", media_type="text/markdown")
    if fmt == "docx":
        path = OUTPUT_DIR / f"{name}.docx"
        exporters.build_docx(name, chapters, str(path))
        return FileResponse(path, filename=f"{name}.docx",
                            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    if fmt == "epub":
        path = OUTPUT_DIR / f"{name}.epub"
        exporters.build_epub(name, chapters, str(path))
        return FileResponse(path, filename=f"{name}.epub", media_type="application/epub+zip")
    raise HTTPException(400, f"不支持的格式: {fmt}")


# ---------- 大纲标题 / 查找替换 ----------

@app.get("/api/novels/{novel_id}/outline_titles")
def outline_titles(novel_id: str):
    workflow = get_workflow(novel_id)
    return {"titles": {str(k): v for k, v in workflow.get_outline_chapter_titles().items()}}


@app.post("/api/novels/{novel_id}/find")
def global_find(novel_id: str, body: dict):
    find_text = body.get("find_text", "")
    if not find_text:
        return {"results": []}
    workflow = get_workflow(novel_id)
    return {"results": workflow.global_find(find_text, workflow.novel_info.copy())}


@app.post("/api/novels/{novel_id}/replace")
def global_replace(novel_id: str, body: dict):
    find_text = body.get("find_text", "")
    replace_text = body.get("replace_text", "")
    if not find_text:
        raise HTTPException(400, "请输入查找内容")
    workflow = get_workflow(novel_id)
    result = workflow.global_find_replace(find_text, replace_text, workflow.novel_info.copy())
    return {"changes": result["changes"]}


# ---------- Skill 管理 ----------

@app.get("/api/skills")
def list_skills(novel_id: str = ""):
    skills = skill_manager.list_skills()
    states = skill_manager.get_skill_states(get_store(novel_id)) if novel_id else {}
    for s in skills:
        s["effective_enabled"] = skill_manager.is_skill_enabled(s, states)
        s["novel_override"] = states.get(s["dir"])
    return {"skills": skills, "step_labels": skill_manager.STEP_LABELS,
            "phase_labels": skill_manager.PHASE_LABELS,
            "inject_chars": skill_manager.get_inject_max_chars()}


@app.get("/api/skills/inject_limit")
def get_inject_limit():
    return {"skill_inject_chars": skill_manager.get_inject_max_chars()}


@app.put("/api/skills/inject_limit")
def set_inject_limit(body: dict):
    try:
        v = int(body.get("value", 0))
    except (ValueError, TypeError):
        raise HTTPException(400, "无效数值")
    if not 100 <= v <= 20000:
        raise HTTPException(400, "注入上限范围为 100-20000 字")
    user_config.set_skill_inject_chars(v)
    return {"ok": True, "skill_inject_chars": user_config.get_skill_inject_chars()}


@app.post("/api/skills/{dir_name}/toggle")
def toggle_skill(dir_name: str, body: dict):
    novel_id = body.get("novel_id", "")
    enabled = body.get("enabled")  # true/false/null
    if not novel_id:
        raise HTTPException(400, "缺少 novel_id")
    skill_manager.set_novel_skill_enabled(get_store(novel_id), dir_name, enabled)
    return {"ok": True}


class SkillIn(BaseModel):
    meta: dict
    body: str = ""


@app.put("/api/skills/{dir_name}")
def save_skill(dir_name: str, s: SkillIn):
    skill_manager.save_skill(dir_name, s.meta, s.body)
    return {"ok": True}


@app.post("/api/skills")
def create_skill(body: dict):
    dir_name = (body.get("dir_name") or "").strip()
    if not dir_name:
        raise HTTPException(400, "目录名不能为空")
    meta = body.get("meta") or {}
    skill_manager.save_skill(dir_name, meta, body.get("body", ""))
    return {"ok": True, "dir": dir_name}


@app.delete("/api/skills/{dir_name}")
def delete_skill(dir_name: str):
    skill_manager.delete_skill(dir_name)
    return {"ok": True}


@app.post("/api/skills/recommend")
def recommend_skills(body: dict):
    query = body.get("query", "")
    novel_id = body.get("novel_id", "")
    exclude = []
    if novel_id:
        states = skill_manager.get_skill_states(get_store(novel_id))
        all_skills = skill_manager.list_skills()
        exclude = [s["dir"] for s in all_skills if skill_manager.is_skill_enabled(s, states)]
    return {"skills": skill_manager.recommend_skills(query, exclude_dirs=exclude)}


@app.post("/api/skills/import")
async def import_skill(file: UploadFile):
    name = file.filename or "imported"
    try:
        if name.lower().endswith(".zip"):
            dir_name = skill_manager.import_skill_from_zip(file.file, name)
        else:
            fallback = os.path.splitext(os.path.basename(name))[0]
            dir_name = skill_manager.import_skill_from_md(file.file, fallback)
        skill_manager.ensure_apply_to(dir_name, [])
        return {"ok": True, "dir": dir_name}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- 静态页面 ----------

@app.middleware("http")
async def no_cache_static(request, call_next):
    """静态页面（html/js/css）不缓存，避免浏览器拿旧文件与新文件混用导致白屏"""
    resp = await call_next(request)
    path = request.url.path
    if not path.startswith("/api/") and path.rsplit(".", 1)[-1] in ("html", "js", "css", "/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=BASE_DIR / "web", html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    import webbrowser

    url = "http://127.0.0.1:8501"
    # 延迟打开浏览器，等服务启动完成
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"打开: {url}")
    uvicorn.run(app, host="127.0.0.1", port=8501)
