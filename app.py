#!/usr/bin/env python3
"""
AI小说创作工具 - 通用API + 本地向量库版本
运行: streamlit run app.py
"""
import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import chromadb
import graphviz

# 配置日志 - 在终端输出API调用信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from api.api_client import LLMAPIClient
from vector_store.local_chroma import LocalNovelVectorStore
from workflow.novel_workflow import FullNovelWorkflow

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="AI小说创作", page_icon="📖", layout="wide")

# 隐藏 Streamlit 自带右上角按钮和footer + 生成时全屏遮罩
st.markdown("""
<style>
    /* 隐藏右上角菜单和底部footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 隐藏header中的工具栏，但保留侧边栏切换按钮 */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] [data-testid="stHeaderToolbar"],
    header[data-testid="stHeader"] [data-testid="stHeaderActionElements"],
    header[data-testid="stHeader"] a {
        display: none !important;
    }
    /* 侧边栏切换按钮始终可见 */
    header[data-testid="stHeader"] button[aria-label="Toggle sidebar"],
    header[data-testid="stHeader"] button[title="Toggle sidebar"],
    header[data-testid="stHeader"] button[kind="header"] {
        visibility: visible !important;
    }
    /* 自定义侧边栏展开按钮 */
    #custom-sidebar-toggle {
        position: fixed;
        top: 6px;
        left: 6px;
        z-index: 99990;
        width: 38px;
        height: 38px;
        background: rgba(30,30,30,0.9);
        border: 1px solid #555;
        border-radius: 8px;
        color: #ccc;
        font-size: 20px;
        cursor: pointer;
        display: none;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    #custom-sidebar-toggle:hover {
        background: rgba(60,60,60,0.95);
        border-color: #4fc3f7;
    }
    
    /* 生成中全屏遮罩 - 覆盖主区域和侧边栏 */
    .generating-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.65);
        z-index: 99999;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: all;
    }
    /* 生成时禁用侧边栏交互 */
    .generating-sidebar-overlay {
        position: fixed;
        top: 0; left: 0; bottom: 0;
        width: 100%;
        max-width: 600px;
        background: transparent;
        z-index: 99998;
        pointer-events: all;
        cursor: not-allowed;
    }
    /* 生成中侧边栏禁用样式 */
    .generating-active section[data-testid="stSidebar"] {
        pointer-events: none !important;
        opacity: 0.5 !important;
    }
    .generating-overlay * { pointer-events: all; }
    .generating-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #4fc3f7;
        border-radius: 16px;
        padding: 40px 50px;
        text-align: center;
        box-shadow: 0 0 40px rgba(79, 195, 247, 0.3);
        min-width: 360px;
    }
    .generating-card .spinner {
        display: inline-block;
        width: 48px; height: 48px;
        border: 4px solid rgba(79, 195, 247, 0.2);
        border-top-color: #4fc3f7;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-bottom: 16px;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .generating-card .title {
        color: #4fc3f7;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .generating-card .subtitle {
        color: #aaa;
        font-size: 14px;
    }
</style>
<!-- 自定义侧边栏展开按钮：当 Streamlit 自带的折叠按钮被隐藏时，此按钮可见 -->
<div id="custom-sidebar-toggle" onclick="
    const btn = document.querySelector('button[aria-label=\\'Toggle sidebar\\']') 
                || document.querySelector('button[title=\\'Toggle sidebar\\']')
                || document.querySelector('header button[kind=\\'header\\']');
    if(btn) btn.click();
    document.getElementById('custom-sidebar-toggle').style.display = 'none';
" title="展开侧边栏">☰</div>
<script>
// 检测侧边栏状态，折叠时显示自定义按钮
function checkSidebarState() {
    const sidebar = document.querySelector('section[data-testid=\"stSidebar\"]');
    const customBtn = document.getElementById('custom-sidebar-toggle');
    if (!sidebar || !customBtn) return;
    // 侧边栏折叠时 width 为 0 或很小，或 display 为 none
    const rect = sidebar.getBoundingClientRect();
    const isCollapsed = rect.width < 10 || getComputedStyle(sidebar).display === 'none';
    // 生成中不显示自定义按钮（避免与遮罩冲突）
    const isGenerating = document.body.classList.contains('generating-active');
    if (isCollapsed && !isGenerating) {
        customBtn.style.display = 'flex';
    } else {
        customBtn.style.display = 'none';
    }
}
// 初始检测 + 定时检测（Streamlit rerun 会重建 DOM）
setTimeout(checkSidebarState, 500);
setInterval(checkSidebarState, 1000);
</script>
""", unsafe_allow_html=True)

# 创建输出目录
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 初始化会话状态
def init_session():
    if "novel_id" not in st.session_state:
        st.session_state["novel_id"] = ""  # 内部ID，不可变
    if "novel_name" not in st.session_state:
        st.session_state["novel_name"] = ""  # 显示名称，可修改
    if "current_novel" not in st.session_state:
        st.session_state["current_novel"] = None
    if "generated_content" not in st.session_state:
        st.session_state["generated_content"] = {}
    if "generating" not in st.session_state:
        st.session_state["generating"] = None  # 当前正在生成的步骤标识
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = 0  # 记住当前激活的tab索引

# 初始化
def init_app():
    init_session()
    # 获取API配置（优先使用侧边栏输入的值，其次使用.env中的值）
    api_key = st.session_state.get("api_key", "") or os.getenv("VOLC_API_KEY", "")
    api_base = st.session_state.get("api_base", "") or os.getenv("VOLC_API_BASE", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    model = st.session_state.get("model", "") or os.getenv("VOLC_MODEL", "doubao-pro-32k")
    
    if not api_key:
        st.sidebar.warning("请输入API Key")
        return None
    
    # 初始化组件
    api_client: LLMAPIClient = LLMAPIClient(api_key=api_key, api_base=api_base, model=model)
    # 获取当前小说ID（内部标识），如果为空则不创建向量库
    novel_id = st.session_state.get("novel_id", "")
    if not novel_id:
        # 没有选择小说，返回一个不包含workflow的标记
        st.session_state["no_novel_selected"] = True
        return None
    st.session_state.pop("no_novel_selected", None)
    novel_name = st.session_state.get("novel_name", "") or novel_id
    vector_store = LocalNovelVectorStore(db_path="./chroma_db", novel_id=novel_id, novel_name=novel_name)
    # 从向量库 metadata 同步 novel_name（兼容旧数据升级）
    if not st.session_state.get("novel_name"):
        stored_name = vector_store.collection.metadata.get("novel_name", "")
        if stored_name:
            st.session_state["novel_name"] = stored_name
    workflow = FullNovelWorkflow(api_client, vector_store)
    
    return workflow

# 从向量库恢复内容到session_state
def load_from_vectorstore(workflow):
    """切换小说时，从向量库加载数据恢复到 generated_content 和 session_state"""
    if workflow is None:
        return
    loaded = workflow.vs.load_all_to_dict()
    gc = st.session_state.get("generated_content", {})
    
    # 只恢复session_state中没有的字段（不覆盖正在编辑的）
    if "world_setting" not in gc and loaded.get("world_setting"):
        gc["world_setting"] = loaded["world_setting"]
    if "characters" not in gc and loaded.get("characters"):
        gc["characters"] = loaded["characters"]
    if "outline" not in gc and loaded.get("outline"):
        gc["outline"] = loaded["outline"]
    if "chapters" not in gc:
        gc["chapters"] = {}
    # 合并向量库中的章节
    for chap_num, chap_data in loaded.get("chapters", {}).items():
        if chap_num not in gc["chapters"]:
            gc["chapters"][chap_num] = chap_data
    
    # 恢复额外数据（*_original, *_prompt, consistency_result, relation_graph 等）
    extra = loaded.get("extra", {})
    for key, value in extra.items():
        if key.endswith("_original") or key.endswith("_prompt"):
            # 原始文本和 prompt 恢复到 generated_content
            if key not in gc:
                gc[key] = value
        elif key == "consistency_result":
            if not st.session_state.get("consistency_result"):
                st.session_state["consistency_result"] = value
        elif key == "relation_graph":
            if not st.session_state.get("relation_graph"):
                st.session_state["relation_graph"] = value
    
    st.session_state["generated_content"] = gc
    # 同步到 workflow.novel_info
    if gc.get("world_setting"):
        workflow.novel_info["world_setting"] = gc["world_setting"]
    if gc.get("characters"):
        workflow.novel_info["characters"] = gc["characters"]
    if gc.get("outline"):
        workflow.novel_info["outline"] = gc["outline"]

# 检查前置步骤是否完成
def check_prerequisite(step: str) -> bool:
    """检查前置步骤是否已完成"""
    gc = st.session_state.get("generated_content", {})
    if step == "characters":
        return bool(gc.get("world_setting"))
    elif step == "outline":
        return bool(gc.get("world_setting")) and bool(gc.get("characters"))
    elif step == "chapter":
        return bool(gc.get("world_setting")) and bool(gc.get("characters")) and bool(gc.get("outline"))
    return True

def get_missing_hints(step: str) -> list:
    """获取缺失的前置步骤提示"""
    gc = st.session_state.get("generated_content", {})
    hints = []
    if step in ("characters", "outline", "chapter"):
        if not gc.get("world_setting"):
            hints.append("🌍 世界观设定")
    if step in ("outline", "chapter"):
        if not gc.get("characters"):
            hints.append("👤 人物设定")
    if step == "chapter":
        if not gc.get("outline"):
            hints.append("📋 小说大纲")
    return hints

def get_downstream_steps(step: str) -> list:
    """获取修改某步骤后可能受影响的下游步骤"""
    gc = st.session_state.get("generated_content", {})
    downstream = []
    if step == "world_setting":
        if gc.get("characters"):
            downstream.append("👤 人物设定")
        if gc.get("outline"):
            downstream.append("📋 小说大纲")
        if gc.get("chapters"):
            downstream.append("📖 正文章节")
    elif step == "characters":
        if gc.get("outline"):
            downstream.append("📋 小说大纲")
        if gc.get("chapters"):
            downstream.append("📖 正文章节")
    elif step == "outline":
        if gc.get("chapters"):
            downstream.append("📖 正文章节")
    return downstream

# 保存小说到文件
def save_novel_to_file(novel_id, content):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{novel_id}_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

# 检测 AI 是否拒绝生成（内容安全审查拒绝）
def is_ai_refusal(text: str) -> bool:
    """检测 AI 是否返回了拒绝生成的内容"""
    if not text:
        return False
    refusal_keywords = [
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
    text_lower = text.lower()
    return any(kw in text_lower for kw in refusal_keywords)

# 接口异常时清空对应步骤的内容，保证最终内容一定是当次接口当次生成的
def _clear_step_content(step: str, gc: dict, workflow=None):
    """当API调用失败时，清空对应步骤的内容，避免残留旧数据"""
    vs = workflow.vs if workflow else None
    if step == "world_setting":
        gc.pop("world_setting", None)
        gc.pop("world_setting_original", None)
        gc.pop("world_setting_prompt", None)
        if vs:
            vs.delete_extra_field("world_setting_original")
            vs.delete_extra_field("world_setting_prompt")
    elif step == "characters":
        gc.pop("characters", None)
        gc.pop("characters_original", None)
        gc.pop("characters_prompt", None)
        if vs:
            vs.delete_extra_field("characters_original")
            vs.delete_extra_field("characters_prompt")
    elif step == "outline":
        gc.pop("outline", None)
        gc.pop("outline_original", None)
        gc.pop("outline_prompt", None)
        if vs:
            vs.delete_extra_field("outline_original")
            vs.delete_extra_field("outline_prompt")
    elif step == "chapter":
        params = st.session_state.get("gen_params", {})
        chapter_num = f"{params.get('chapter_num', 1)}"
        gc.get("chapters", {}).pop(chapter_num, None)
    elif step == "continue":
        st.session_state.pop("continue_result", None)
        st.session_state.pop("continue_merged", None)
    elif step == "consistency":
        st.session_state.pop("consistency_result", None)
        if vs:
            vs.delete_extra_field("consistency_result")
    elif step == "relation_graph":
        st.session_state.pop("relation_graph", None)
        if vs:
            vs.delete_extra_field("relation_graph")
    elif step == "polish":
        st.session_state.pop("polish_result", None)

# 统一执行生成逻辑（遮罩显示后调用）
def execute_generation(workflow):
    """根据 session_state 中的 generating 状态执行对应的 API 调用"""
    step = st.session_state.get("generating")
    gc = st.session_state.get("generated_content", {})
    error_msg = None
    
    # 统一同步 novel_info：每次生成前从 session_state 恢复到 workflow
    # （Streamlit 每次 rerun 会创建新 workflow 实例，novel_info 默认为空）
    workflow.novel_info["world_setting"] = gc.get("world_setting", "")
    workflow.novel_info["characters"] = gc.get("characters", "")
    workflow.novel_info["outline"] = gc.get("outline", "")
    
    try:
        if step == "world_setting":
            params = st.session_state.get("gen_params", {})
            user_prompt = params.get("world_prompt", "")
            max_tokens = params.get("max_tokens", 2000)
            if not user_prompt:
                error_msg = "请输入世界观描述"
            else:
                result = workflow.generate_world_setting(user_prompt, max_tokens=max_tokens)
                st.session_state.generated_content["world_setting"] = result
                st.session_state.generated_content["world_setting_original"] = result
                st.session_state.generated_content["world_setting_prompt"] = user_prompt
                workflow.vs.save_extra_data("world_setting_original", result)
                workflow.vs.save_extra_data("world_setting_prompt", user_prompt)
                if is_ai_refusal(result):
                    error_msg = "AI拒绝了本次生成请求，请修改描述内容后重试（可能触发了内容安全审查）"
        
        elif step == "characters":
            params = st.session_state.get("gen_params", {})
            user_prompt = params.get("char_prompt", "")
            num_main = params.get("num_main", 2)
            num_support = params.get("num_support", 5)
            max_tokens = params.get("max_tokens", 2000)
            if not user_prompt:
                error_msg = "请输入人物设定要求"
            else:
                result = workflow.generate_characters(user_prompt, num_main, num_support, max_tokens=max_tokens)
                char_text = result["characters"]
                st.session_state.generated_content["characters"] = char_text
                st.session_state.generated_content["characters_original"] = char_text
                st.session_state.generated_content["characters_prompt"] = user_prompt
                workflow.vs.save_extra_data("characters_original", char_text)
                workflow.vs.save_extra_data("characters_prompt", user_prompt)
                if is_ai_refusal(char_text):
                    error_msg = "AI拒绝了本次生成请求，请修改描述内容后重试（可能触发了内容安全审查）"
        
        elif step == "outline":
            params = st.session_state.get("gen_params", {})
            user_prompt = params.get("outline_prompt", "")
            total_chapters = params.get("total_chapters", 50)
            words_per_chapter = params.get("words_per_chapter", 2000)
            max_tokens = params.get("max_tokens", 4000)
            if not user_prompt:
                error_msg = "请输入大纲要求"
            else:
                result = workflow.generate_outline(user_prompt, int(total_chapters), int(words_per_chapter), max_tokens=max_tokens)
                st.session_state.generated_content["outline"] = result
                st.session_state.generated_content["outline_original"] = result
                st.session_state.generated_content["outline_prompt"] = user_prompt
                workflow.vs.save_extra_data("outline_original", result)
                workflow.vs.save_extra_data("outline_prompt", user_prompt)
                if is_ai_refusal(result):
                    error_msg = "AI拒绝了本次生成请求，请修改描述内容后重试（可能触发了内容安全审查）"
        
        elif step == "chapter":
            params = st.session_state.get("gen_params", {})
            chapter_num = params.get("chapter_num", 1)
            chapter_title = params.get("chapter_title", "")
            max_tokens = params.get("max_tokens", 2500)
            target_words = params.get("target_words", 2000)
            if not chapter_title:
                error_msg = "请输入章节标题"
            else:
                result = workflow.generate_chapter_with_rag(int(chapter_num), chapter_title, max_tokens=max_tokens, target_words=target_words)
                st.session_state.generated_content["chapters"][f"{chapter_num}"] = {
                    "title": chapter_title,
                    "content": result
                }
                if is_ai_refusal(result):
                    error_msg = "AI拒绝了本次生成请求，请修改章节内容或标题后重试（可能触发了内容安全审查）"
        
        elif step == "continue":
            params = st.session_state.get("gen_params", {})
            current_text = params.get("continue_text", "")
            user_prompt = params.get("continue_prompt", "继续往下写")
            target_length = params.get("continue_length", 1500)
            max_tokens = params.get("max_tokens", 2000)
            if not current_text:
                error_msg = "请输入需要续写的内容"
            else:
                result = workflow.continue_writing(current_text, user_prompt, target_length, max_tokens=max_tokens)
                st.session_state["continue_result"] = result
                st.session_state["continue_merged"] = current_text + "\n\n" + result
                if is_ai_refusal(result):
                    error_msg = "AI拒绝了本次续写请求，请修改内容后重试（可能触发了内容安全审查）"
        
        elif step == "polish":
            params = st.session_state.get("gen_params", {})
            polish_text = params.get("polish_text", "")
            style_reference = params.get("style_reference", "")
            style_type = params.get("style_type", "作品")
            max_tokens = params.get("max_tokens", 2000)
            if not polish_text or not style_reference:
                error_msg = "请输入要润色的文本和风格参考"
            else:
                result = workflow.polish_with_style(polish_text, style_reference, style_type, max_tokens=max_tokens)
                st.session_state["polish_result"] = result
                st.session_state["polish_original"] = polish_text
                style_label = f"《{style_reference}》" if style_type == "作品" else style_reference
                st.session_state["polish_style_label"] = style_label
                # 保存来源信息供后续使用
                st.session_state["polish_source"] = params.get("polish_source", "")
                st.session_state["polish_key"] = params.get("polish_key", "")
                if is_ai_refusal(result):
                    error_msg = "AI拒绝了本次润色请求，请修改内容后重试（可能触发了内容安全审查）"
        
        elif step == "consistency":
            params = st.session_state.get("gen_params", {})
            max_tokens = params.get("max_tokens", 4000)
            
            result = workflow.check_consistency(max_tokens=max_tokens)
            st.session_state["consistency_result"] = result
            workflow.vs.save_extra_data("consistency_result", result)
        
        elif step == "relation_graph":
            params = st.session_state.get("gen_params", {})
            max_tokens = params.get("max_tokens", 2000)
            
            raw_result = workflow.extract_character_relations(max_tokens=max_tokens)
            json_start = raw_result.find("{")
            json_end = raw_result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_result[json_start:json_end]
                graph_data = json.loads(json_str)
                st.session_state["relation_graph"] = graph_data
                workflow.vs.save_extra_data("relation_graph", graph_data)
            else:
                error_msg = "AI返回格式异常，请重试"
    
    except json.JSONDecodeError as e:
        if step == "relation_graph":
            st.session_state["relation_graph_raw_error"] = str(e)[:100]
            st.session_state["relation_graph_raw"] = raw_result if 'raw_result' in locals() else ""
        error_msg = f"解析失败: {str(e)[:100]}"
        # 接口异常时清空对应步骤的内容，保证最终内容一定是当次生成的
        _clear_step_content(step, gc, workflow)
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        # 接口异常时清空对应步骤的内容，保证最终内容一定是当次生成的
        _clear_step_content(step, gc, workflow)
    
    # 清理生成状态
    step_to_tab = {
        "world_setting": 0, "characters": 1, "outline": 2, "chapter": 3,
        "continue": 4, "polish": 5, "consistency": 6, "relation_graph": 8,
    }
    if step in step_to_tab:
        st.session_state["active_tab"] = step_to_tab[step]
    st.session_state["generating"] = None
    st.session_state["gen_params"] = None
    st.session_state["gen_error"] = error_msg
    st.rerun()

# 主界面
def main():
    st.title("📖 AI 全链路小说创作工具")
    st.caption(" powered by 大语言模型API + 本地Chroma向量库 (RAG上下文记忆) ")
    
    init_session()
    
    # 是否正在生成
    is_generating = st.session_state.get("generating") is not None
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # 从 .env 获取默认值，未获取到则使用空字符串
        default_api_key = os.getenv("VOLC_API_KEY", "")
        default_api_base = os.getenv("VOLC_API_BASE", "")
        default_model = os.getenv("VOLC_MODEL", "")
        
        # API配置（优先从.env读取，未获取到则手动输入）
        saved_api_key = st.session_state.get("api_key", default_api_key)
        api_key = st.text_input(
            "API Key", 
            type="password", 
            value=saved_api_key if saved_api_key else default_api_key,
            help="优先从.env读取，未配置请手动输入"
        )
        st.session_state["api_key"] = api_key
        
        saved_api_base = st.session_state.get("api_base", default_api_base)
        api_base = st.text_input(
            "API Base URL", 
            value=saved_api_base if saved_api_base else default_api_base,
            help="优先从.env读取，未配置请手动输入"
        )
        st.session_state["api_base"] = api_base
        os.environ["VOLC_API_BASE"] = api_base
        
        saved_model = st.session_state.get("model", default_model)
        model = st.text_input(
            "模型名称", 
            value=saved_model if saved_model else default_model,
            help="优先从.env读取，未配置请手动输入"
        )
        st.session_state["model"] = model
        
        # 各步骤 max_tokens 配置
        # 模型最大输出token硬限制（大多数32k模型为32768）
        MAX_MT = 32768
        with st.expander("🔧 高级参数 (max_tokens)", expanded=False):
            st.caption(f"控制每步API调用的最大输出token数（上限{MAX_MT}），值越大输出越长但费用越高")
            mt_cols1 = st.columns(2)
            with mt_cols1[0]:
                mt_world = st.number_input("🌍 世界观设定", min_value=500, max_value=MAX_MT, 
                    value=st.session_state.get("mt_world", 6000), step=500, key="mt_world_input")
                st.session_state["mt_world"] = mt_world
            with mt_cols1[1]:
                mt_chars = st.number_input("👤 人物设定", min_value=500, max_value=MAX_MT,
                    value=st.session_state.get("mt_chars", 6000), step=500, key="mt_chars_input")
                st.session_state["mt_chars"] = mt_chars
            mt_cols2 = st.columns(2)
            with mt_cols2[0]:
                mt_outline = st.number_input("📋 小说大纲", min_value=1000, max_value=MAX_MT,
                    value=st.session_state.get("mt_outline", 8000), step=500, key="mt_outline_input")
                st.session_state["mt_outline"] = mt_outline
            with mt_cols2[1]:
                mt_chapter = st.number_input("📖 生成章节", min_value=500, max_value=MAX_MT,
                    value=st.session_state.get("mt_chapter", 16000), step=500, key="mt_chapter_input")
                st.session_state["mt_chapter"] = mt_chapter
            mt_cols3 = st.columns(2)
            with mt_cols3[0]:
                mt_continue = st.number_input("✍️ 续写", min_value=500, max_value=MAX_MT,
                    value=st.session_state.get("mt_continue", 8000), step=500, key="mt_continue_input")
                st.session_state["mt_continue"] = mt_continue
            with mt_cols3[1]:
                mt_polish = st.number_input("🎨 风格润色", min_value=500, max_value=MAX_MT,
                    value=st.session_state.get("mt_polish", 8000), step=500, key="mt_polish_input")
                st.session_state["mt_polish"] = mt_polish
            mt_cols4 = st.columns(2)
            with mt_cols4[0]:
                mt_consistency = st.number_input("🔍 一致性检查", min_value=1000, max_value=MAX_MT,
                    value=st.session_state.get("mt_consistency", 6000), step=500, key="mt_consistency_input")
                st.session_state["mt_consistency"] = mt_consistency
            with mt_cols4[1]:
                mt_relation = st.number_input("🕸️ 角色图谱", min_value=1000, max_value=MAX_MT,
                    value=st.session_state.get("mt_relation", 6000), step=500, key="mt_relation_input")
                st.session_state["mt_relation"] = mt_relation
        
        st.divider()
        
        # 小说管理
        st.header("📚 小说管理")
        
        # 初始化 session state
        if "confirm_delete_all" not in st.session_state:
            st.session_state["confirm_delete_all"] = False
        if "delete_target_id" not in st.session_state:
            st.session_state["delete_target_id"] = None
        
        # --- 新建小说 ---
        new_novel_name = st.text_input(
            "新建小说", 
            value="",
            placeholder="输入小说名称，点击创建",
            key="new_novel_name_input"
        )
        if st.button("➕ 创建新小说", type="primary", use_container_width=True, disabled=is_generating):
            if new_novel_name.strip():
                # 用纯时间戳生成唯一内部ID，不包含书名（避免中文路径/编码问题）
                new_id = f"novel_{int(time.time())}"
                # 检查是否已存在同名小说
                existing = LocalNovelVectorStore.list_all_novels(db_path="./chroma_db")
                existing_names = [n["name"] for n in existing]
                display_name = new_novel_name.strip()
                if display_name in existing_names:
                    st.warning(f"已存在同名小说《{display_name}》，将使用相同名称但内部ID不同")
                st.session_state["novel_id"] = new_id
                st.session_state["novel_name"] = display_name
                st.session_state["generated_content"] = {}
                st.rerun()
            else:
                st.error("请输入小说名称")
        
        st.divider()
        
        # --- 当前小说信息 + 改名 ---
        current_novel_id = st.session_state.get("novel_id", "")
        current_novel_name = st.session_state.get("novel_name", "")
        if current_novel_id:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#1a3a5c,#2d5f8a);color:#fff;'
                f'padding:10px 14px;border-radius:8px;border-left:4px solid #4fc3f7;margin:4px 0;">'
                f'<span style="font-size:16px;">📖</span> '
                f'<b style="font-size:16px;">{current_novel_name}</b>'
                f'<br><span style="font-size:11px;opacity:0.6;">ID: {current_novel_id}</span>'
                f'</div>', unsafe_allow_html=True
            )
            # 改名功能
            rename_cols = st.columns([3, 1])
            with rename_cols[0]:
                new_name = st.text_input(
                    "改名", value=current_novel_name,
                    key="rename_novel_input", label_visibility="collapsed",
                    placeholder="输入新名称"
                )
            with rename_cols[1]:
                if st.button("✏️", key="rename_btn", use_container_width=True, disabled=is_generating):
                    if new_name and new_name.strip() and new_name.strip() != current_novel_name:
                        try:
                            vs = LocalNovelVectorStore(db_path="./chroma_db", novel_id=current_novel_id, novel_name=new_name.strip())
                            vs.rename(new_name.strip())
                            st.session_state["novel_name"] = new_name.strip()
                            st.success(f"已改名为《{new_name.strip()}》")
                            st.rerun()
                        except Exception as e:
                            st.error(f"改名失败: {str(e)}")
        else:
            st.info("👆 请创建新小说或从下方选择已有小说")
        
        # --- 已有小说列表 ---
        with st.expander("📂 已有小说", expanded=True):
            try:
                all_novels = LocalNovelVectorStore.list_all_novels(db_path="./chroma_db")
                
                if all_novels:
                    for novel in all_novels:
                        is_current = novel["id"] == current_novel_id
                        counts = novel["type_counts"]
                        display_name = novel["name"]
                        
                        # 统计信息
                        total_items = sum(counts.values())
                        summary_parts = []
                        if counts["setting"] > 0:
                            summary_parts.append(f"世界观{counts['setting']}")
                        if counts["character"] > 0:
                            summary_parts.append(f"人物{counts['character']}")
                        if counts["outline"] > 0:
                            summary_parts.append(f"大纲{counts['outline']}")
                        if counts["chapter"] > 0:
                            summary_parts.append(f"章节{counts['chapter']}")
                        summary_str = " | ".join(summary_parts)
                        
                        # 卡片式布局 - 当前小说用醒目样式
                        if is_current:
                            st.markdown(
                                f'<div style="background:linear-gradient(135deg,#1a3a5c,#2d5f8a);color:#fff;'
                                f'padding:10px 14px;border-radius:8px;border-left:4px solid #4fc3f7;margin:4px 0;">'
                                f'<span style="font-size:18px;">📖</span> '
                                f'<b style="font-size:15px;">{display_name}</b> '
                                f'<span style="background:#4fc3f7;color:#0d2137;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold;">当前选中</span>'
                                f'<br><span style="font-size:12px;opacity:0.85;">共{total_items}项 | {summary_str}</span>'
                                f'</div>', unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div style="background:#1e1e1e;color:#ccc;'
                                f'padding:8px 14px;border-radius:6px;border-left:3px solid #555;margin:4px 0;">'
                                f'<span style="font-size:15px;">📕</span> '
                                f'<b>{display_name}</b> '
                                f'<span style="font-size:12px;opacity:0.7;">共{total_items}项 | {summary_str}</span>'
                                f'</div>', unsafe_allow_html=True
                            )
                        
                        # 操作按钮行
                        btn_cols = st.columns([1, 1])
                        with btn_cols[0]:
                            if st.button("✏️ 打开" if not is_current else "📌 当前", key=f"sel_{novel['id']}", use_container_width=True, type="primary" if is_current else "secondary"):
                                st.session_state["novel_id"] = novel["id"]
                                st.session_state["novel_name"] = novel["name"]
                                st.session_state["generated_content"] = {}
                                st.rerun()
                        with btn_cols[1]:
                            if st.button("🗑️ 删除", key=f"del_{novel['id']}", use_container_width=True):
                                st.session_state["delete_target_id"] = novel["id"]
                                st.session_state["delete_target_name"] = novel["name"]
                                st.rerun()
                        st.divider()
                else:
                    st.caption("暂无小说，请在上方创建！")
            except Exception as e:
                st.caption(f"读取失败: {str(e)[:50]}")
        
        # 删除单本小说的确认弹窗
        if st.session_state.get("delete_target_id"):
            target_id = st.session_state["delete_target_id"]
            target_name = st.session_state.get("delete_target_name", target_id)
            st.warning(f"⚠️ 确认删除《{target_name}》？此操作不可恢复！")
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("✅ 确认删除", use_container_width=True, type="primary"):
                    try:
                        vector_store = LocalNovelVectorStore(db_path="./chroma_db", novel_id=target_id)
                        vector_store.delete_novel()
                        # 如果删除的是当前小说，清空 session
                        if st.session_state.get("novel_id") == target_id:
                            st.session_state["generated_content"] = {}
                            st.session_state["novel_id"] = ""
                            st.session_state["novel_name"] = ""
                        st.session_state["delete_target_id"] = None
                        st.session_state.pop("delete_target_name", None)
                        st.success(f"已删除《{target_name}》")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {str(e)}")
                        st.session_state["delete_target_id"] = None
                        st.session_state.pop("delete_target_name", None)
            with cols[1]:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state["delete_target_id"] = None
                    st.session_state.pop("delete_target_name", None)
                    st.rerun()
        
        # 删除所有小说
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.get("confirm_delete_all", False):
                if st.button("💣 删除所有小说", use_container_width=True):
                    st.session_state["confirm_delete_all"] = True
                    st.rerun()
            else:
                st.warning("⚠️ 确认删除所有小说？此操作不可恢复！")
                cols = st.columns([1, 1])
                with cols[0]:
                    if st.button("✅ 确认删除", use_container_width=True, type="primary"):
                        try:
                            client = chromadb.PersistentClient(path="./chroma_db")
                            collections = client.list_collections()
                            for col in collections:
                                if col.name.startswith("n_") or col.name.startswith("novel_"):
                                    client.delete_collection(name=col.name)
                            st.session_state["generated_content"] = {}
                            st.session_state["novel_id"] = ""
                            st.session_state["novel_name"] = ""
                            st.session_state["confirm_delete_all"] = False
                            st.success("已删除所有小说向量库")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {str(e)}")
                            st.session_state["confirm_delete_all"] = False
                with cols[1]:
                    if st.button("❌ 取消", use_container_width=True):
                        st.session_state["confirm_delete_all"] = False
                        st.rerun()
        
        # 显示当前小说的详细内容（可折叠）
        if current_novel_id:
            try:
                current_vs = LocalNovelVectorStore(db_path="./chroma_db", novel_id=current_novel_id)
                all_data = current_vs.collection.get()
                
                if all_data and all_data.get("documents"):
                    with st.expander("📋 当前小说详细内容", expanded=False):
                        # 按类型分组显示
                        sections = {
                            "setting": ("🌍 世界观设定", []),
                            "character": ("👤 人物设定", []),
                            "outline": ("📋 小说大纲", []),
                            "chapter": ("📖 正文章节", [])
                        }
                        
                        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                            section_type = meta.get("type", "other")
                            if section_type in sections:
                                sections[section_type][1].append({
                                    "title": meta.get("title", "未命名"),
                                    "content": doc
                                })
                        
                        for sec_type, (sec_name, items) in sections.items():
                            if items:
                                with st.expander(f"{sec_name} ({len(items)})", expanded=False):
                                    for item in items:
                                        st.markdown(f"**{item['title']}**")
                                        # 显示内容摘要（太长则截断）
                                        content_preview = item['content'][:300] + "..." if len(item['content']) > 300 else item['content']
                                        st.text(content_preview)
                                        st.divider()
                else:
                    st.caption("当前小说暂无内容，请开始创作")
            except Exception as e:
                st.caption(f"读取内容失败")
        
        st.divider()
        st.markdown("""
        ### 💡 特点
        - ✅ 向量库**完全本地**免费
        - ✅ RAG自动上下文记忆，不遗忘人设
        - ✅ 兼容任何OpenAI格式API
        - ✅ 导出Markdown保存
        - ✅ 按量付费，具体看你的API使用价格
        """)
    
    workflow = init_app()
    if not workflow:
        if st.session_state.get("no_novel_selected"):
            st.info("👈 请先在左侧创建或选择小说")
        else:
            st.info("👈 请先在左侧输入API Key")
        return
    
    # 从向量库恢复数据到session_state（首次加载或切换小说时）
    gc = st.session_state.get("generated_content", {})
    core_keys = ["world_setting", "characters", "outline"]
    needs_load = not gc or all(not gc.get(k) for k in core_keys)
    if needs_load:
        load_from_vectorstore(workflow)
    
    # 生成中全屏遮罩 - 阻止所有页面交互
    if is_generating:
        # 注入侧边栏禁用class
        st.markdown('<script>document.body.classList.add("generating-active");</script>', unsafe_allow_html=True)
        step_labels_map = {
            "world_setting": "🌍 世界观设定",
            "characters": "👤 人物设定",
            "outline": "📋 小说大纲",
            "chapter": "📖 正文章节",
            "continue": "✍️ 续写",
            "polish": "🎨 风格润色",
            "consistency": "🔍 一致性检查",
            "relation_graph": "🕸️ 角色图谱",
        }
        current_step = step_labels_map.get(st.session_state.get("generating", ""), "生成内容")
        # 侧边栏遮罩
        st.markdown('<div class="generating-sidebar-overlay"></div>', unsafe_allow_html=True)
        # 主区域遮罩
        st.markdown(
            f'<div class="generating-overlay">'
            f'<div class="generating-card">'
            f'<div class="spinner"></div>'
            f'<div class="title">正在生成中...</div>'
            f'<div class="subtitle">{current_step} · 请稍候，API调用中</div>'
            f'<div class="subtitle" style="margin-top:12px;color:#666;font-size:12px;">'
            f'⏳ 期间请勿操作页面，生成完成后自动恢复</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )
        # 渲染遮罩后，立即执行实际生成逻辑
        execute_generation(workflow)
    
    # 显示上一次生成的错误信息（如果有的话）
    if st.session_state.get("gen_error"):
        st.error(f"❌ {st.session_state['gen_error']}")
        st.session_state["gen_error"] = None
    
    # 流程进度提示
    gc = st.session_state.get("generated_content", {})
    steps_done = {
        "世界观": bool(gc.get("world_setting")),
        "人物": bool(gc.get("characters")),
        "大纲": bool(gc.get("outline")),
        "章节": bool(gc.get("chapters"))
    }
    done_count = sum(1 for v in steps_done.values() if v)
    progress_pct = int(done_count / 4 * 100)
    
    with st.container():
        prog_cols = st.columns([1, 6])
        with prog_cols[0]:
            st.caption("创作进度")
        with prog_cols[1]:
            st.progress(progress_pct)
            step_labels = []
            for name, done in steps_done.items():
                icon = "✅" if done else "⬜"
                step_labels.append(f"{icon} {name}")
            st.caption(" → ".join(step_labels))
        
        # 字数统计概览
        word_parts = []
        if gc.get("world_setting"):
            word_parts.append(f"🌍 世界观 {len(gc['world_setting'])}字")
        if gc.get("characters"):
            word_parts.append(f"👤 人物 {len(gc['characters'])}字")
        if gc.get("outline"):
            word_parts.append(f"📋 大纲 {len(gc['outline'])}字")
        if gc.get("chapters"):
            total_chap_words = sum(len(c.get("content", "")) for c in gc["chapters"].values())
            chap_count = len(gc["chapters"])
            word_parts.append(f"📖 章节 {chap_count}章/{total_chap_words}字")
        if word_parts:
            st.caption("📊 字数统计：" + " ｜ ".join(word_parts))
        
        # 流程依赖说明
        st.info(
            "📌 **创作流程说明（必须按顺序）**：\n"
            "1️⃣ 世界观设定 → 2️⃣ 人物设定 → 3️⃣ 小说大纲 → 4️⃣ 生成章节\n\n"
            "每一步都依赖前一步的内容作为上下文。修改已完成的步骤可能导致下游内容不一致，"
            "建议修改后重新生成受影响的下游步骤。"
        )
    
    # 分标签页界面
    active_tab_idx = st.session_state.get("active_tab", 0)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🌍 世界观设定", 
        "👤 人物设定", 
        "📋 小说大纲", 
        "📖 生成章节", 
        "✍️ 续写",
        "🎨 风格润色",
        "🔍 一致性检查",
        "🔎 查找替换",
        "🕸️ 角色图谱",
        "📤 导出小说"
    ])
    # 用 JS 强制切换到目标 tab（Streamlit 原生不支持指定默认 tab）
    if active_tab_idx > 0:
        tab_switch_js = f"""
        <script>
        function switchTab() {{
            var parentDoc = window.parent.document;
            var tabs = parentDoc.querySelectorAll('[data-testid="stTabs"] button[role="tab"]');
            if (tabs.length > {active_tab_idx}) {{
                tabs[{active_tab_idx}].click();
                return true;
            }}
            return false;
        }}
        var attempts = 0;
        var interval = setInterval(function() {{
            attempts++;
            if (switchTab() || attempts > 30) {{
                clearInterval(interval);
            }}
        }}, 100);
        </script>
        """
        components.html(tab_switch_js, height=0, width=0)
        # 只在生成完成后的首次渲染中切换，之后恢复默认
        st.session_state["active_tab"] = 0
    
    # 1. 世界观设定
    with tab1:
        st.subheader("第一步：创作世界观设定")
        st.caption("🌍 世界观是整个小说的基础，后续的人物、大纲、章节都将基于此生成。")
        
        has_world = bool(gc.get("world_setting"))
        if has_world:
            downstream = get_downstream_steps("world_setting")
            if downstream:
                st.warning(f"⚠️ 世界观已存在。修改世界观可能导致以下内容不一致：{'、'.join(downstream)}。建议修改后重新生成受影响的步骤。")
        
        user_prompt = st.text_area("描述你想要的小说题材和背景", 
            placeholder="例如：一部现代都市背景的修真小说，主角在大城市打工偶然获得修炼传承...",
            height=120,
            key="world_prompt")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            btn_label = "🔄 重新生成世界观" if has_world else "生成世界观"
            if st.button(btn_label, type="primary", use_container_width=True, disabled=is_generating):
                if user_prompt:
                    st.session_state["generating"] = "world_setting"
                    st.session_state["gen_params"] = {"world_prompt": user_prompt, "max_tokens": st.session_state.get("mt_world", 2000)}
                    st.rerun()
        with col2:
            if has_world:
                if st.button("🗑️ 清除世界观", use_container_width=True, disabled=is_generating, key="clear_world"):
                    st.session_state.generated_content.pop("world_setting", None)
                    st.session_state.generated_content.pop("world_setting_original", None)
                    st.session_state.generated_content.pop("world_setting_prompt", None)
                    workflow.vs.delete_section("setting", "world_setting")
                    workflow.vs.delete_extra_field("world_setting_original")
                    workflow.vs.delete_extra_field("world_setting_prompt")
                    workflow.novel_info.pop("world_setting", None)
                    st.rerun()
        
        if "world_setting" in st.session_state.generated_content:
            # 显示原始生成 prompt
            original_prompt = st.session_state.generated_content.get("world_setting_prompt", "")
            if original_prompt:
                with st.expander("📝 原始生成需求", expanded=False):
                    st.info(original_prompt)
            # 显示原始生成文本（编辑后可对比）
            original_text = st.session_state.generated_content.get("world_setting_original", "")
            if original_text and original_text != st.session_state.generated_content["world_setting"]:
                with st.expander("📄 原始生成文本（未被修改前的AI输出）", expanded=False):
                    st.text_area("AI原始输出", value=original_text, height=300, key="world_original_view", disabled=True)
            result = st.session_state.generated_content["world_setting"]
            st.caption(f"📊 当前字数：**{len(result)}** 字")
            edited = st.text_area("✏️ 编辑世界观设定（修改后自动保存）", value=result, height=300)
            if edited != result:
                st.session_state.generated_content["world_setting"] = edited
                workflow.vs.update_section("setting", "world_setting", edited)
                workflow.novel_info["world_setting"] = edited
                downstream = get_downstream_steps("world_setting")
                if downstream:
                    st.warning(f"⚠️ 世界观已更新！以下内容可能需要重新生成：{'、'.join(downstream)}")
                else:
                    st.success("✅ 已更新并保存到向量库")
            else:
                st.success("✅ 已保存到本地向量库，后续生成会自动引用")
    
    # 2. 人物设定
    with tab2:
        st.subheader("第二步：设计人物设定")
        st.caption("👤 人物设定基于世界观，确保人物的力量体系、背景等与世界观一致。")
        
        # 前置步骤检查 - 严格约束
        prereq_ok = check_prerequisite("characters")
        if not prereq_ok:
            missing = get_missing_hints("characters")
            st.error(f"🚫 无法生成人物设定！请先完成：{' → '.join(missing)}\n\n人物的力量体系、社会地位等必须基于世界观来设计，否则会产生逻辑矛盾。")
        
        has_chars = bool(gc.get("characters"))
        if has_chars and prereq_ok:
            downstream = get_downstream_steps("characters")
            if downstream:
                st.warning(f"⚠️ 人物设定已存在。修改人物可能导致以下内容不一致：{'、'.join(downstream)}。建议修改后重新生成受影响的步骤。")
        
        user_prompt = st.text_area("补充对人物的要求", 
            placeholder="例如：主角是社畜，反派是修仙界大佬转生...",
            height=80,
            key="char_prompt")
        
        col1, col2 = st.columns(2)
        num_main = col1.number_input("主角人数", min_value=1, max_value=10, value=2)
        num_support = col2.number_input("配角人数", min_value=1, max_value=50, value=5)
        
        btn_label = "🔄 重新生成人物设定" if has_chars else "生成人物设定"
        if st.button(btn_label, type="primary", disabled=is_generating or not prereq_ok):
            if not prereq_ok:
                st.error("❌ 请先完成世界观设定，人物需要基于世界观来设计")
            elif user_prompt:
                st.session_state["generating"] = "characters"
                st.session_state["gen_params"] = {"char_prompt": user_prompt, "num_main": num_main, "num_support": num_support, "max_tokens": st.session_state.get("mt_chars", 2000)}
                st.rerun()
        
        if has_chars:
            if st.button("🗑️ 清除人物设定", disabled=is_generating, key="clear_chars"):
                st.session_state.generated_content.pop("characters", None)
                st.session_state.generated_content.pop("characters_original", None)
                st.session_state.generated_content.pop("characters_prompt", None)
                workflow.vs.delete_section("character", "all_characters")
                workflow.vs.delete_extra_field("characters_original")
                workflow.vs.delete_extra_field("characters_prompt")
                workflow.novel_info.pop("characters", None)
                st.rerun()
        
        if "characters" in st.session_state.generated_content:
            # 显示原始生成 prompt
            original_prompt = st.session_state.generated_content.get("characters_prompt", "")
            if original_prompt:
                with st.expander("📝 原始生成需求", expanded=False):
                    st.info(original_prompt)
            # 显示原始生成文本
            original_text = st.session_state.generated_content.get("characters_original", "")
            if original_text and original_text != st.session_state.generated_content["characters"]:
                with st.expander("📄 原始生成文本（未被修改前的AI输出）", expanded=False):
                    st.text_area("AI原始输出", value=original_text, height=400, key="chars_original_view", disabled=True)
            result = st.session_state.generated_content["characters"]
            st.caption(f"📊 当前字数：**{len(result)}** 字")
            edited = st.text_area("✏️ 编辑人物设定（修改后自动保存）", value=result, height=400)
            if edited != result:
                st.session_state.generated_content["characters"] = edited
                workflow.vs.update_section("character", "all_characters", edited)
                workflow.novel_info["characters"] = edited
                downstream = get_downstream_steps("characters")
                if downstream:
                    st.warning(f"⚠️ 人物设定已更新！以下内容可能需要重新生成：{'、'.join(downstream)}")
                else:
                    st.success("✅ 已更新并保存到向量库")
            else:
                st.success("✅ 已保存到本地向量库")
    
    # 3. 小说大纲
    with tab3:
        st.subheader("第三步：创作小说大纲")
        st.caption("📋 大纲基于世界观和人物设定来规划情节走向，确保故事逻辑连贯。")
        
        # 前置步骤检查 - 严格约束
        prereq_ok = check_prerequisite("outline")
        if not prereq_ok:
            missing = get_missing_hints("outline")
            st.error(f"🚫 无法生成大纲！请先完成：{' → '.join(missing)}\n\n大纲必须基于世界观和人物来规划，否则情节可能与设定矛盾。")
        
        has_outline = bool(gc.get("outline"))
        if has_outline and prereq_ok:
            downstream = get_downstream_steps("outline")
            if downstream:
                st.warning(f"⚠️ 大纲已存在。修改大纲可能导致以下内容不一致：{'、'.join(downstream)}。建议修改后重新生成受影响的步骤。")
        
        user_prompt = st.text_area("补充对情节大纲的要求", 
            placeholder="例如：主角从底层开始，一路成长，最终都市封神...",
            height=80,
            key="outline_prompt")
        outline_cols = st.columns(2)
        with outline_cols[0]:
            total_chapters = st.text_input("总章节数规划", value="50", help="可输入具体数字如 50，或范围如 30-50")
        with outline_cols[1]:
            words_per_chapter = st.number_input("每章大概字数", min_value=500, max_value=20000, value=2000, step=500, help="规划每章的大致字数，供AI参考")
        
        btn_label = "🔄 重新生成大纲" if has_outline else "生成大纲"
        if st.button(btn_label, type="primary", disabled=is_generating or not prereq_ok):
            if not prereq_ok:
                st.error("❌ 请先完成世界观设定和人物设定")
            elif user_prompt:
                # 解析总章节数输入（支持范围如"30-50"）
                try:
                    tc_str = total_chapters.strip()
                    if "-" in tc_str:
                        parts = tc_str.split("-")
                        tc_val = int(parts[-1].strip())
                    else:
                        tc_val = int(tc_str)
                except (ValueError, IndexError):
                    tc_val = 50
                st.session_state["generating"] = "outline"
                st.session_state["gen_params"] = {"outline_prompt": user_prompt, "total_chapters": tc_val, "words_per_chapter": words_per_chapter, "max_tokens": st.session_state.get("mt_outline", 4000)}
                st.rerun()
        
        if has_outline:
            if st.button("🗑️ 清除大纲", disabled=is_generating, key="clear_outline"):
                st.session_state.generated_content.pop("outline", None)
                st.session_state.generated_content.pop("outline_original", None)
                st.session_state.generated_content.pop("outline_prompt", None)
                workflow.vs.delete_section("outline", "full_outline")
                workflow.vs.delete_extra_field("outline_original")
                workflow.vs.delete_extra_field("outline_prompt")
                workflow.novel_info.pop("outline", None)
                st.rerun()
        
        if "outline" in st.session_state.generated_content:
            # 显示原始生成 prompt
            original_prompt = st.session_state.generated_content.get("outline_prompt", "")
            if original_prompt:
                with st.expander("📝 原始生成需求", expanded=False):
                    st.info(original_prompt)
            # 显示原始生成文本
            original_text = st.session_state.generated_content.get("outline_original", "")
            if original_text and original_text != st.session_state.generated_content["outline"]:
                with st.expander("📄 原始生成文本（未被修改前的AI输出）", expanded=False):
                    st.text_area("AI原始输出", value=original_text, height=400, key="outline_original_view", disabled=True)
            result = st.session_state.generated_content["outline"]
            st.caption(f"📊 当前字数：**{len(result)}** 字")
            edited = st.text_area("✏️ 编辑大纲（修改后自动保存）", value=result, height=500)
            if edited != result:
                st.session_state.generated_content["outline"] = edited
                workflow.vs.update_section("outline", "full_outline", edited)
                workflow.novel_info["outline"] = edited
                downstream = get_downstream_steps("outline")
                if downstream:
                    st.warning(f"⚠️ 大纲已更新！以下内容可能需要重新生成：{'、'.join(downstream)}")
                else:
                    st.success("✅ 已更新并保存到向量库")
            else:
                st.success("✅ 已保存到本地向量库")
    
    # 4. 生成章节
    with tab4:
        st.subheader("第四步：生成单章正文")
        st.caption("📖 章节生成会自动检索世界观、人物和大纲作为RAG上下文，保持一致性。",
            help="章节生成 AI 上下文构建逻辑：\n"
                 "1️⃣ 核心设定直取：从已生成内容直接获取世界观(≤1500字)、人物(≤2000字)，比向量检索更完整精准\n"
                 "2️⃣ 相关大纲提取：只提取当前章节前后±2章的大纲内容，避免全文塞入\n"
                 "3️⃣ 前情回顾：仅取最近2章末尾800字作为衔接，不传全文\n"
                 "4️⃣ 语义搜索补充：用当前章节标题在向量库做语义搜索，返回最多3条语义最相关的结果；跳过已直接包含的设定/人物/大纲(避免重复)，其他章节内容每条截取前500字\n"
                 "5️⃣ 目标字数：Prompt 中指定目标字数，但受 max_tokens 硬限制\n\n"
                 "📍 代码：workflow/novel_workflow.py → generate_chapter()\n"
                 "辅助方法：_extract_relevant_outline()、_get_previous_chapters_summary()")
        
        # 前置步骤检查 - 严格约束
        prereq_ok = check_prerequisite("chapter")
        if not prereq_ok:
            missing = get_missing_hints("chapter")
            st.error(f"🚫 无法生成章节！请先完成：{' → '.join(missing)}\n\n章节需要引用世界观、人物和大纲来保持内容一致性。")
        
        # 展示已经生成的章节
        if "chapters" not in st.session_state.generated_content:
            st.session_state.generated_content["chapters"] = {}
        
        existing_chapters = st.session_state.generated_content.get("chapters", {})
        
        # 选择章节：已有章节可快速选择，也可手动输入新章节号
        if existing_chapters:
            sorted_keys = sorted(existing_chapters.keys(), key=lambda x: int(x))
            chapter_options = ["✏️ 手动输入新章节号"] + [f"第{k}章 {existing_chapters[k]['title']}" for k in sorted_keys]
            selected_chapter = st.selectbox("选择章节", options=chapter_options, key="chapter_select")
            
            if selected_chapter == "✏️ 手动输入新章节号":
                # 手动输入模式
                input_cols = st.columns(2)
                chapter_num = input_cols[0].number_input("章节号", min_value=1, max_value=1000, value=1, key="chap_num")
                chapter_title = input_cols[1].text_input("章节标题", placeholder="例如：初入都市", key="chap_title")
            else:
                # 选择了已有章节
                idx = chapter_options.index(selected_chapter) - 1
                selected_key = sorted_keys[idx]
                chapter_num = int(selected_key)
                chapter_title = existing_chapters[selected_key]["title"]
                st.info(f"📖 已选择第{selected_key}章「{chapter_title}」— 可在下方编辑或重新生成")
        else:
            # 没有已有章节，直接输入
            input_cols = st.columns(2)
            chapter_num = input_cols[0].number_input("章节号", min_value=1, max_value=1000, value=1, key="chap_num")
            chapter_title = input_cols[1].text_input("章节标题", placeholder="例如：初入都市", key="chap_title")
        
        # 章节目标字数设置
        target_words = st.number_input("📝 目标字数", min_value=500, max_value=20000, value=2000, step=500,
            help="期望AI生成的章节字数。注意：实际字数受 max_tokens 限制，建议 max_tokens ≥ 目标字数 × 1.5")
        
        # 检查此章节是否已存在
        key = f"{chapter_num}"
        chapter_exists = key in existing_chapters
        btn_label = "🔄 重新生成这一章" if chapter_exists else "生成这一章"
        
        if st.button(btn_label, type="primary", disabled=is_generating or not prereq_ok):
            if not chapter_title:
                st.error("❌ 请输入章节标题")
            elif not prereq_ok:
                st.error("❌ 请先完成世界观、人物和大纲设定")
            else:
                st.session_state["generating"] = "chapter"
                st.session_state["gen_params"] = {
                    "chapter_num": chapter_num, 
                    "chapter_title": chapter_title, 
                    "target_words": target_words,
                    "max_tokens": st.session_state.get("mt_chapter", 2500)
                }
                st.rerun()
        
        # 如果已经生成了，显示可编辑
        if "chapters" in st.session_state.generated_content and key in st.session_state.generated_content["chapters"]:
            chap = st.session_state.generated_content["chapters"][key]
            st.markdown(f"### 第 {chapter_num} 章 {chap['title']}")
            st.caption(f"📊 当前字数：**{len(chap['content'])}** 字")
            edited = st.text_area("✏️ 编辑章节内容（修改后自动保存）", value=chap["content"], height=600, key=f"chapter_edit_{key}")
            if edited != chap["content"]:
                st.session_state.generated_content["chapters"][key]["content"] = edited
                # 安全更新到向量库 - 使用chap中的标题而非输入框中的标题
                full_text = f"第{chapter_num}章 {chap['title']}\n{edited}"
                workflow.vs.update_section("chapter", f"chapter_{chapter_num}", full_text)
                st.success("✅ 已更新并保存到向量库")
            if st.button("🗑️ 删除这一章", disabled=is_generating, key=f"delete_chapter_{key}"):
                st.session_state.generated_content["chapters"].pop(key, None)
                workflow.vs.delete_section("chapter", f"chapter_{key}")
                st.rerun()
    
    # 5. 续写
    with tab5:
        st.subheader("✍️ 自由续写")
        st.caption("基于已有内容继续创作，AI会自动参考世界观、人物设定和前文来保持一致性。")
        
        # 续写不强制前置步骤，但提示效果可能受影响
        if not gc.get("world_setting") and not gc.get("characters"):
            st.warning("⚠️ 当前没有世界观和人物设定，续写可能缺少上下文，建议先完成前几步。")
        elif not gc.get("outline"):
            st.info("💡 没有小说大纲，续写时不会参考大纲走向，情节可能偏离预期。")
        
        # --- 选择续写来源 ---
        st.markdown("**选择续写起点**")
        source_option = st.radio(
            "从哪里开始续写？",
            options=["从已有章节续写", "自由输入文本续写"],
            horizontal=True,
            key="continue_source_option"
        )
        
        current_text = ""
        chapters = gc.get("chapters", {})
        if source_option == "从已有章节续写":
            if chapters:
                chapter_options = []
                for chap_num in sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    chap = chapters[chap_num]
                    chapter_options.append(f"第{chap_num}章 {chap.get('title', '')}")
                
                if chapter_options:
                    selected_idx = st.selectbox(
                        "选择要续写的章节",
                        range(len(chapter_options)),
                        format_func=lambda i: chapter_options[i],
                        key="continue_chapter_select"
                    )
                    # 取出选中章节的内容
                    sorted_keys = sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    selected_key = sorted_keys[selected_idx]
                    current_text = chapters[selected_key]["content"]
                    st.caption(f"📖 已自动加载第{selected_key}章内容（{len(current_text)}字）")
                else:
                    st.info("暂无已生成章节")
            else:
                st.info("暂无已生成章节，请先在「生成章节」中创建章节，或切换到「自由输入」模式。")
        else:
            current_text = st.text_area(
                "输入你要续写的内容", 
                placeholder="粘贴你已经写到这里的内容，AI会接着往下写...",
                height=300, 
                key="continue_text"
            )
        
        # --- 续写要求 ---
        user_prompt = st.text_input(
            "续写要求", 
            value="继续往下写",
            placeholder="例如：继续往下写 / 描写一场战斗 / 角色内心独白...",
            key="continue_prompt"
        )
        
        # 续写字数
        continue_length = st.slider("续写目标字数", min_value=500, max_value=3000, value=1500, step=500, key="continue_length")
        
        if st.button("✍️ 开始续写", type="primary", disabled=is_generating or not current_text):
            if not current_text:
                st.error("❌ 请先选择章节或输入续写内容")
            else:
                st.session_state["generating"] = "continue"
                st.session_state["gen_params"] = {
                    "continue_text": current_text, 
                    "continue_prompt": user_prompt,
                    "continue_length": continue_length,
                    "max_tokens": st.session_state.get("mt_continue", 2000)
                }
                st.rerun()
        
        # 显示续写结果
        cached_continue = st.session_state.get("continue_result")
        if cached_continue:
            st.divider()
            st.subheader("📝 续写结果")
            st.markdown(cached_continue)
            
            cached_merged = st.session_state.get("continue_merged", "")
            if cached_merged:
                with st.expander("📋 合并全文（原文 + 续写）", expanded=False):
                    st.text_area("合并后的全文，可复制使用：", value=cached_merged, height=400, key="continue_merged_view")
            
            if st.button("🗑️ 清除续写结果", key="clear_continue"):
                st.session_state.pop("continue_result", None)
                st.session_state.pop("continue_merged", None)
                st.rerun()
            
            # 如果是从章节续写，提供一键追加到章节的选项
            if source_option == "从已有章节续写" and chapters:
                st.markdown("---")
                st.markdown("**将续写结果追加到章节**")
                append_cols = st.columns([1, 1])
                with append_cols[0]:
                    append_chapter_options = []
                    for chap_num in sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                        chap = chapters[chap_num]
                        append_chapter_options.append(f"第{chap_num}章 {chap.get('title', '')}")
                    append_idx = st.selectbox(
                        "追加到哪个章节",
                        range(len(append_chapter_options)),
                        format_func=lambda i: append_chapter_options[i],
                        key="append_chapter_select"
                    )
                with append_cols[1]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("📎 追加到章节末尾", type="primary", use_container_width=True):
                        sorted_keys = sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                        target_key = sorted_keys[append_idx]
                        original_content = chapters[target_key]["content"]
                        new_content = original_content + "\n\n" + cached_continue
                        chapters[target_key]["content"] = new_content
                        chap_title = chapters[target_key].get("title", "")
                        chap_num_val = target_key
                        full_text = f"第{chap_num_val}章 {chap_title}\n{new_content}"
                        workflow.vs.update_section("chapter", f"chapter_{chap_num_val}", full_text)
                        st.success(f"✅ 已追加到第{target_key}章末尾")
                        st.rerun()
    
    # 6. 风格润色
    with tab6:
        st.subheader("🎨 风格润色")
        st.caption("模仿指定作品或作家的写作风格，对已有文本进行润色改写。保持情节不变，只改表达方式。")
        
        # --- 选择润色来源 ---
        st.markdown("**选择要润色的文本**")
        polish_source = st.radio(
            "从哪里获取文本？",
            options=["从已有章节选择", "自由输入文本"],
            horizontal=True,
            key="polish_source_option"
        )
        
        polish_text = ""
        polish_key = ""
        polish_chapters = gc.get("chapters", {})
        if polish_source == "从已有章节选择":
            if polish_chapters:
                polish_chapter_options = []
                for chap_num in sorted(polish_chapters.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    chap = polish_chapters[chap_num]
                    polish_chapter_options.append(f"第{chap_num}章 {chap.get('title', '')}")
                
                if polish_chapter_options:
                    polish_selected_idx = st.selectbox(
                        "选择要润色的章节",
                        range(len(polish_chapter_options)),
                        format_func=lambda i: polish_chapter_options[i],
                        key="polish_chapter_select"
                    )
                    sorted_keys = sorted(polish_chapters.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    polish_key = sorted_keys[polish_selected_idx]
                    polish_text = polish_chapters[polish_key]["content"]
                    st.caption(f"📖 已加载第{polish_key}章内容（{len(polish_text)}字）")
                else:
                    st.info("暂无已生成章节")
            else:
                st.info("暂无已生成章节，请先在「生成章节」中创建，或切换到「自由输入」模式。")
        else:
            polish_text = st.text_area(
                "输入要润色的文本",
                placeholder="粘贴需要润色的文本...",
                height=300,
                key="polish_text_input"
            )
        
        # --- 风格选择 ---
        st.markdown("**选择模仿风格**")
        style_type = st.radio(
            "模仿类型",
            options=["作品", "作家"],
            horizontal=True,
            key="polish_style_type"
        )
        
        if style_type == "作品":
            style_reference = st.text_input(
                "作品名称",
                placeholder="例如：红楼梦、百年孤独、三体、哈利波特...",
                key="polish_work_name"
            )
        else:
            style_reference = st.text_input(
                "作家名称",
                placeholder="例如：余华、村上春树、鲁迅、金庸...",
                key="polish_author_name"
            )
        
        if st.button("🎨 开始润色", type="primary", disabled=is_generating or not polish_text or not style_reference):
            if not polish_text:
                st.error("❌ 请先选择章节或输入要润色的文本")
            elif not style_reference:
                st.error("❌ 请输入要模仿的作品或作家名称")
            else:
                st.session_state["generating"] = "polish"
                st.session_state["gen_params"] = {
                    "polish_text": polish_text,
                    "style_reference": style_reference.strip(),
                    "style_type": style_type,
                    "polish_source": polish_source,
                    "polish_key": polish_key if polish_source == "从已有章节选择" and polish_chapters else "",
                    "max_tokens": st.session_state.get("mt_polish", 2000)
                }
                st.rerun()
        
        # 显示润色结果
        cached_polish = st.session_state.get("polish_result")
        if cached_polish:
            st.divider()
            
            col_old, col_new = st.columns(2)
            with col_old:
                st.markdown("**📝 原文**")
                original = st.session_state.get("polish_original", "")
                st.text_area("原文", value=original, height=400, key="polish_original_view", disabled=True)
            with col_new:
                st.markdown(f"**🎨 润色后（模仿{st.session_state.get('polish_style_label', '')}）**")
                st.text_area("润色后", value=cached_polish, height=400, key="polish_result_view")
            
            # 提供应用选项
            polish_src = st.session_state.get("polish_source", "")
            if polish_src == "从已有章节选择" and polish_chapters:
                pk = st.session_state.get("polish_key", "")
                if pk and pk in polish_chapters:
                    if st.button("✅ 用润色结果替换章节内容", type="primary"):
                        polish_chapters[pk]["content"] = cached_polish
                        chap_title = polish_chapters[pk].get("title", "")
                        full_text = f"第{pk}章 {chap_title}\n{cached_polish}"
                        workflow.vs.update_section("chapter", f"chapter_{pk}", full_text)
                        st.success(f"✅ 已替换第{pk}章内容")
                        st.session_state.pop("polish_result", None)
                        st.rerun()
            
            if st.button("🗑️ 清除润色结果", key="clear_polish"):
                st.session_state.pop("polish_result", None)
                st.session_state.pop("polish_original", None)
                st.session_state.pop("polish_style_label", None)
                st.session_state.pop("polish_source", None)
                st.session_state.pop("polish_key", None)
                st.rerun()
    
    # 7. 一致性检查
    with tab7:
        st.subheader("🔍 AI 一致性检查")
        st.caption("让AI审查所有已完成的设定，找出人名矛盾、设定冲突、逻辑不一致等问题。")
        
        # 检查是否至少有两个步骤完成
        done_steps = sum(1 for v in [gc.get("world_setting"), gc.get("characters"), gc.get("outline")] if v)
        if done_steps < 2:
            st.warning("⚠️ 至少需要完成两个步骤（世界观/人物/大纲）才能进行一致性检查。")
        else:
            # 显示当前已完成的步骤
            completed = []
            if gc.get("world_setting"):
                completed.append("🌍 世界观")
            if gc.get("characters"):
                completed.append("👤 人物")
            if gc.get("outline"):
                completed.append("📋 大纲")
            
            st.info(f"将检查以下内容的一致性：{' + '.join(completed)}")
            
            if st.button("🔍 开始一致性检查", type="primary", disabled=is_generating, use_container_width=True):
                st.session_state["generating"] = "consistency"
                st.session_state["gen_params"] = {
                    "max_tokens": st.session_state.get("mt_consistency", 4000)
                }
                st.rerun()
            
            # 显示持久化的检查结果（切换标签页回来仍可见）
            cached_result = st.session_state.get("consistency_result")
            if cached_result:
                st.markdown("### 📋 检查结果")
                st.markdown(cached_result)
                
                st.caption("💡 提示：发现矛盾后，请回到对应标签页手动修改，或重新生成受影响的步骤。点击「开始一致性检查」可重新检查。")
                
                if st.button("🗑️ 清除检查结果", key="clear_consistency"):
                    st.session_state.pop("consistency_result", None)
                    workflow.vs.delete_extra_field("consistency_result")
                    st.rerun()
            else:
                st.info("👆 点击上方按钮开始一致性检查，结果会保留在此标签页中。")
    
    # 8. 全局查找替换
    with tab8:
        st.subheader("🔎 全局查找替换")
        st.caption("在所有设定和章节中查找并替换文本，例如修改人物名字、地名等。")
        
        if not gc.get("world_setting") and not gc.get("characters") and not gc.get("outline") and not gc.get("chapters"):
            st.warning("⚠️ 当前小说没有任何内容，无法查找替换。")
        else:
            find_text = st.text_input("🔍 查找内容", placeholder="例如：李明", key="find_text")
            
            # 查找预览
            if find_text:
                find_results = workflow.global_find(find_text, gc.copy())
                if find_results:
                    total_count = sum(int(r.split("：找到 ")[1].split(" 处")[0]) for r in find_results)
                    st.success(f"✅ 找到 **{total_count}** 处匹配：")
                    for r in find_results:
                        st.markdown(f"- {r}")
                else:
                    st.info(f"未找到「{find_text}」")
            
            st.divider()
            
            replace_text = st.text_input("✏️ 替换为", placeholder="例如：张三", key="replace_text")
            
            if find_text and replace_text:
                # 确认操作
                if "confirm_replace" not in st.session_state:
                    st.session_state["confirm_replace"] = False
                
                if not st.session_state.get("confirm_replace"):
                    if st.button("🔄 执行替换", type="primary", use_container_width=True, disabled=is_generating):
                        # 先预览
                        find_results = workflow.global_find(find_text, gc.copy())
                        if find_results:
                            st.session_state["confirm_replace"] = True
                            st.rerun()
                        else:
                            st.warning("没有找到需要替换的内容")
                else:
                    st.warning(f"⚠️ 即将在所有内容中将「**{find_text}**」替换为「**{replace_text}**」，此操作不可撤销！")
                    confirm_cols = st.columns([1, 1])
                    with confirm_cols[0]:
                        if st.button("✅ 确认替换", type="primary", use_container_width=True):
                            result = workflow.global_find_replace(find_text, replace_text, gc.copy())
                            if result["changes"]:
                                # 更新 session_state
                                for key, value in result["updated_gc"].items():
                                    st.session_state.generated_content[key] = value
                                st.success("✅ 替换完成！")
                                for change in result["changes"]:
                                    st.markdown(f"- {change}")
                            else:
                                st.info("没有找到需要替换的内容")
                            st.session_state["confirm_replace"] = False
                            st.rerun()
                    with confirm_cols[1]:
                        if st.button("❌ 取消", use_container_width=True):
                            st.session_state["confirm_replace"] = False
                            st.rerun()
    
    # 9. 角色关系图谱
    with tab9:
        st.subheader("🕸️ 角色关系图谱")
        st.caption("AI自动提取角色关系，生成可视化关系图谱。")
        
        if not gc.get("characters"):
            st.warning("⚠️ 需要先完成人物设定才能生成角色关系图谱。")
        else:
            # 检查是否已有图谱数据
            if "relation_graph" not in st.session_state:
                st.session_state["relation_graph"] = None
            
            if st.button("🕸️ 生成角色关系图谱", type="primary", disabled=is_generating, use_container_width=True):
                st.session_state["generating"] = "relation_graph"
                st.session_state["gen_params"] = {"max_tokens": st.session_state.get("mt_relation", 2000)}
                st.rerun()
            
            # 清除图谱
            if st.session_state.get("relation_graph"):
                if st.button("🗑️ 清除角色图谱", key="clear_relation_graph"):
                    st.session_state.pop("relation_graph", None)
                    workflow.vs.delete_extra_field("relation_graph")
                    st.rerun()
            
            # 显示图谱
            graph_data = st.session_state.get("relation_graph")
            # 显示图谱原始错误
            if st.session_state.get("relation_graph_raw_error"):
                st.error(f"❌ 解析AI返回结果失败: {st.session_state['relation_graph_raw_error']}")
                raw = st.session_state.get("relation_graph_raw", "")
                if raw:
                    with st.expander("查看AI原始输出"):
                        st.text(raw)
                st.session_state["relation_graph_raw_error"] = None
                st.session_state["relation_graph_raw"] = None
            if graph_data:
                # 角色列表
                characters = graph_data.get("characters", [])
                relations = graph_data.get("relations", [])
                
                if characters:
                    st.markdown("### 📋 角色列表")
                    role_colors = {"主角": "#4fc3f7", "反派": "#ef5350", "配角": "#ffca28"}
                    role_bg_colors = {"主角": "#0d2740", "反派": "#3d1111", "配角": "#3d3010"}
                    for char in characters:
                        role = char.get("role", "配角")
                        accent = role_colors.get(role, "#aaaaaa")
                        bg = role_bg_colors.get(role, "#1a1a2e")
                        desc = char.get("desc", "")
                        st.markdown(
                            f'<div style="background:{bg};padding:10px 14px;border-radius:8px;'
                            f'border-left:4px solid {accent};margin:6px 0;display:flex;align-items:center;gap:10px;">'
                            f'<b style="color:{accent};font-size:15px;white-space:nowrap;">{char.get("name", "?")}</b>'
                            f'<span style="font-size:11px;color:{accent};opacity:0.8;background:{accent}22;'
                            f'padding:2px 8px;border-radius:4px;white-space:nowrap;">{role}</span>'
                            f'<span style="font-size:13px;color:#d0d0d0;overflow:hidden;text-overflow:ellipsis;'
                            f'white-space:nowrap;">{desc}</span>'
                            f'</div>', unsafe_allow_html=True
                        )
                    
                    # 关系图谱可视化
                    if relations:
                        st.markdown("### 🕸️ 关系图谱")
                        
                        # 关系类型颜色
                        rel_colors = {
                            "师徒": "#4fc3f7", "恋人": "#f48fb1", "敌人": "#ef5350",
                            "朋友": "#66bb6a", "主仆": "#ffca28", "同门": "#ab47bc",
                            "亲属": "#ff7043", "对手": "#ffa726", "盟友": "#26c6da"
                        }
                        
                        dot = graphviz.Digraph(comment="角色关系图谱")
                        dot.attr(rankdir="TB", bgcolor="transparent", dpi="150")
                        dot.attr("node", shape="box", style="rounded,filled", 
                                 fillcolor="#1a1a2e", fontcolor="white", fontsize="14",
                                 margin="0.2,0.1", width="1.2", height="0.5")
                        dot.attr("edge", fontsize="11", fontcolor="#cccccc", penwidth="1.5")
                        
                        # 添加节点
                        for char in characters:
                            name = char.get("name", "?")
                            role = char.get("role", "配角")
                            color = role_colors.get(role, "#aaaaaa")
                            dot.node(name, name, color=color, penwidth="2.5")
                        
                        # 添加边
                        for rel in relations:
                            from_name = rel.get("from", "")
                            to_name = rel.get("to", "")
                            rel_type = rel.get("type", "")
                            color = rel_colors.get(rel_type, "#888888")
                            dot.edge(from_name, to_name, label=rel_type, color=color, penwidth="1.5")
                        
                        # 渲染为SVG，支持拖动和缩放
                        try:
                            svg_data = dot.pipe(format="svg").decode("utf-8")
                            # 使用 components.html 注入可拖动缩放的SVG容器（st.markdown会移除script标签）
                            graph_html = f"""
                            <div style="background:#0d0d0d;padding:0;">
                                <div id="graph-container" style="
                                    width:100%; height:550px; border:1px solid #333;
                                    border-radius:8px; background:#0d0d0d; overflow:hidden;
                                    position:relative; cursor:grab;
                                ">
                                    <div id="graph-inner" style="
                                        display:inline-block; min-width:100%; transform-origin:0 0;
                                    ">
                                        {svg_data}
                                    </div>
                                </div>
                                <div style="display:flex;align-items:center;gap:12px;margin-top:6px;padding:0 8px;">
                                    <span style="font-size:12px;color:#888;">🖱️ 拖动平移 | 滚轮缩放</span>
                                    <button onclick="
                                        var inner=document.getElementById('graph-inner');
                                        inner.style.transform='scale(1) translate(0px,0px)';
                                    " style="font-size:12px;padding:2px 10px;border-radius:4px;
                                    border:1px solid #555;background:#1a1a2e;color:#ccc;cursor:pointer;">
                                        重置视图
                                    </button>
                                </div>
                            </div>
                            <script>
                            (function(){{
                                var container=document.getElementById('graph-container');
                                var inner=document.getElementById('graph-inner');
                                if(!container||!inner) return;
                                var scale=1,tx=0,ty=0,isDragging=false,startX,startY;
                                function updateTransform(){{
                                    inner.style.transform='scale('+scale+') translate('+tx+'px,'+ty+'px)';
                                }}
                                container.addEventListener('mousedown',function(e){{
                                    isDragging=true;startX=e.clientX;startY=e.clientY;
                                    var currentTx=tx,currentTy=ty;
                                    container.style.cursor='grabbing';
                                    function onMove(e){{
                                        if(!isDragging) return;
                                        tx=currentTx+(e.clientX-startX)/scale;
                                        ty=currentTy+(e.clientY-startY)/scale;
                                        updateTransform();
                                    }}
                                    function onUp(){{
                                        isDragging=false;container.style.cursor='grab';
                                        window.removeEventListener('mousemove',onMove);
                                        window.removeEventListener('mouseup',onUp);
                                    }}
                                    window.addEventListener('mousemove',onMove);
                                    window.addEventListener('mouseup',onUp);
                                }});
                                container.addEventListener('wheel',function(e){{
                                    e.preventDefault();
                                    var delta=e.deltaY>0?0.9:1.1;
                                    scale=Math.max(0.2,Math.min(5,scale*delta));
                                    updateTransform();
                                }});
                            }})();
                            </script>
                            """
                            components.html(graph_html, height=600, scrolling=True)
                        except Exception as e:
                            # fallback to st.graphviz_chart
                            st.graphviz_chart(dot, use_container_width=True)
                        
                        # 关系列表
                        st.markdown("### 📊 关系详情")
                        for rel in relations:
                            rel_type = rel.get("type", "")
                            color = rel_colors.get(rel_type, "#888888")
                            st.markdown(
                                f'- <span style="color:{color};">**{rel_type}**</span>：'
                                f'{rel.get("from", "")} ↔ {rel.get("to", "")} '
                                f'— {rel.get("desc", "")}',
                                unsafe_allow_html=True
                                )
    
    # 10. 导出
    with tab10:
        st.subheader("📤 导出完整小说")
        
        # 收集所有内容
        novel_content = []
        novel_title = st.session_state.get("novel_name", "") or st.session_state.get("novel_id", "小说")
        novel_content.append(f"# {novel_title}\n\n")
        
        # 添加元信息
        if "world_setting" in st.session_state.generated_content:
            novel_content.append("## 🌍 世界观设定\n\n")
            novel_content.append(st.session_state.generated_content["world_setting"])
            novel_content.append("\n\n---\n\n")
        
        if "characters" in st.session_state.generated_content:
            novel_content.append("## 👤 人物设定\n\n")
            novel_content.append(st.session_state.generated_content["characters"])
            novel_content.append("\n\n---\n\n")
        
        if "outline" in st.session_state.generated_content:
            novel_content.append("## 📋 故事大纲\n\n")
            novel_content.append(st.session_state.generated_content["outline"])
            novel_content.append("\n\n---\n\n")
        
        # 添加正文章节
        if "chapters" in st.session_state.generated_content and st.session_state.generated_content["chapters"]:
            novel_content.append("## 📖 正文\n\n")
            # 按章节号排序
            chapters = st.session_state.generated_content["chapters"]
            sorted_chap_keys = sorted(chapters.keys(), key=lambda x: int(x))
            for key in sorted_chap_keys:
                chap = chapters[key]
                novel_content.append(f"### 第 {key} 章 {chap['title']}\n\n")
                novel_content.append(chap["content"])
                novel_content.append("\n\n---\n\n")
        
        full_text = "".join(novel_content)
        
        # 显示统计
        word_count = len(full_text)
        st.info(f"当前总字数：**{word_count}** 字")
        
        if word_count > 0:
            st.text_area("预览完整Markdown", value=full_text[:2000] + ("..." if len(full_text) > 2000 else ""), height=300)
            
            # 保存到文件
            if st.button("💾 保存到文件", type="primary"):
                filename = save_novel_to_file(novel_title, full_text)
                st.success(f"✅ 已保存到: `{filename}`")
                
                # 提供下载按钮
                st.download_button(
                    label="⬇️ 下载Markdown文件",
                    data=full_text,
                    file_name=f"{novel_title}.md",
                    mime="text/markdown"
                )
        else:
            st.info("还没有生成任何内容，去前面标签页创作吧")

if __name__ == "__main__":
    main()
