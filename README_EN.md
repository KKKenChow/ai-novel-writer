# 📖 AI Full-Chain Novel Writing Tool

[中文](README.md) | English

> 🚀 Low-cost AI novel creation with full-process local vector memory, compatible with mainstream APIs

## ✨ Features

- 💰 **Ultra-low cost**: Using domestic LLM APIs, a 100K-character novel costs only **¥3-5**
- 🧠 **RAG Context Memory**: ChromaDB vector store is **completely local**, automatically retrieves relevant settings, solving AI "character breakdown" issues
- 🔗 **Full-chain creation**: World-building → Character design → Outline planning → Chapter generation → Continuation/Polishing, complete workflow
- ✏️ **Editable & adjustable**: Every step supports manual editing after generation, changes auto-update the vector store
- 💾 **Full data persistence**: All generated content, original text, prompts, consistency check results, and character graphs are auto-persisted to the vector store — no data loss when switching tabs or novels
- 🎯 **Chapter word count target**: Set target word count per chapter (500-8000 characters), AI generates accordingly
- 🔄 **Chapter re-editing**: Select generated chapters to re-edit or regenerate
- 🎨 **Style polishing**: Polish existing text by imitating the writing style of a specified work or author
- 🔍 **AI Consistency Check**: Automatically detect name contradictions, setting conflicts, and logical inconsistencies
- 🔎 **Global Find & Replace**: One-click find and replace across all settings and chapters (e.g., auto-sync name changes)
- 🕸️ **Character Relationship Graph**: AI auto-extracts character relationships, visualized with Graphviz
- 📤 **One-click Export**: Export complete novel in Markdown format with direct download
- 🔌 **Universal Compatibility**: Defaults to Volcengine Ark Doubao API (pay-per-use, fully compliant), also supports any OpenAI-format API
- 🌐 **China-friendly**: No VPN needed, stable access
- 🔒 **Privacy & Security**: Vector store and all creative data stay on your local machine, never uploaded to third parties
- 📚 **Multi-novel Management**: Manage multiple novels simultaneously with independent vector stores, one-click switching

---

## 🏗️ Technical Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Web UI (Streamlit)                         │
│  World|Characters|Outline|Chapters|Continue|Polish|Consistency   │
│           |Find&Replace|Character Graph|Export                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    Full-chain Creative Workflow                  │
│  Strict order: World → Characters → Outline → Chapters          │
│  Upstream edit → Conflict warning → Global find/replace         │
│  AI character relation extraction → Graphviz visualization      │
│  Chapter generation: Categorized context building               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│ 🔍 RAG Retrieval Flow                                           │
│  New chapter → Categorized context → Local vector store → AI    │
│  Direct settings + Relevant outline + Previous summary + Search │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                    │
┌───────▼───────┐                ┌───────────▼───────────┐
│ Local ChromaDB│                │   Cloud LLM API       │
│ (Vector Store)│                │ (Doubao/GPT/Other)    │
│  Free         │                │   Pay-per-token       │
└───────────────┘                └───────────────────────┘
```

### Core Principles

**Why use a local vector store?**
- AI context windows are limited; writing long novels causes AI to forget earlier settings, leading to character breakdown
- RAG (Retrieval-Augmented Generation) only feeds the **most relevant** context for the current chapter, keeping token count stable — cost-effective and precise
- Local vector store means no cloud vector service fees and better privacy

**Chapter Generation Context Logic**
- Instead of stuffing all content into the prompt, context is **categorized and controlled**:
  1. **Direct settings extraction**: Get world-building (≤1500 chars) and characters (≤2000 chars) directly — more complete and precise than vector retrieval
  2. **Relevant outline extraction**: Only extract outline content for ±2 chapters around the current chapter
  3. **Previous chapter summary**: Only the last 800 chars of the most recent 2 chapters for continuity
  4. **Semantic search supplement**: Use chapter title for semantic search in vector store, return up to 3 most relevant results; skip already-included settings/characters/outline to avoid duplication, other chapter content truncated to 500 chars each
  5. **Target word count control**: Specify target in prompt, but actual output is hard-limited by `max_tokens`

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Graphviz](https://graphviz.org/download/) system package (for character relationship graph rendering; macOS: `brew install graphviz`)
- An LLM API Key (Volcengine Ark pay-per-use recommended)

### 1. Clone the Project

```bash
git clone https://github.com/KKKenChow/ai-novel-writer.git
cd ai-novel-writer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Copy the environment template:
```bash
cp .env.example .env
```

Edit the `.env` file:
```env
# Your API Key
VOLC_API_KEY=your_actual_api_key_here

# API Base URL (defaults to Volcengine Ark, can be changed to any OpenAI-compatible API)
VOLC_API_BASE=https://ark.cn-beijing.volces.com/api/v3/chat/completions

# Model name
VOLC_MODEL=doubao-pro-32k
```

> 💡 You can also enter these in the sidebar after launching the app — no file editing needed.

#### ⚠️ Important: Compliant Usage

- **Do NOT** use CodingPlan discounted subscription API Keys for non-coding purposes — this violates the Terms of Service and may result in account suspension
- **Correct approach**: Use [Volcengine Ark](https://console.volcengine.com/ark) **pay-per-use** billing — all legitimate use cases are allowed, and prices are still very affordable

#### Other API Compatibility

You can switch to any OpenAI-compatible API provider:

| Provider | API Base URL Example | Price Reference |
|----------|---------------------|-----------------|
| Volcengine Ark (Recommended) | `https://ark.cn-beijing.volces.com/api/v1/chat/completions` | 100K chars ≈ ¥3-5 |
| Baidu Qianfan | `https://qianfan.baidubce.com/v2` | Similar |
| Alibaba Tongyi | `https://api.openai-proxy.org/v1` | Similar |
| DeepSeek | `https://api.deepseek.com/v1` | Cheaper |
| OpenAI | `https://api.openai.com/v1` | Slightly more expensive |

Just change the API Base and model name — no code changes needed.

### 4. Run

```bash
streamlit run app.py
```

The terminal will output a local address `http://localhost:8501` — open it in your browser.

> By default, only localhost is listened to — not exposed to the internet, safe and secure.

---

## 📖 User Guide

### Creation Flow (Strict Order)

The tool uses a **strict sequential creation flow** — each step depends on the previous step's output as context:

```
1️⃣ World-Building → 2️⃣ Character Design → 3️⃣ Novel Outline → 4️⃣ Chapter Generation → 5️⃣ Continue/Polish/Tools/Export
```

**Why must it be sequential?**
- Character power systems and social status must be based on the world setting
- Plot direction must be based on character relationships
- Chapter text must reference world, characters, and outline for consistency

When prerequisite steps are incomplete, the generate button is **disabled** with a clear message about what's missing.

#### Step 1 🌍 World-Building
1. Describe the genre and setting you want in the text box
   - Example: `A modern urban cultivation novel where the protagonist stumbles upon cultivation heritage while working in a big city`
2. Click "Generate World Setting"
3. After generation, you can directly edit — changes auto-save to the vector store
4. ⚠️ If downstream steps (characters/outline/chapters) already exist, editing the world setting will trigger a **conflict warning**
5. Supports "🗑️ Clear World Setting" to one-click clear and delete vector store data

#### Step 2 👤 Character Design
1. **Prerequisite**: Must complete world-building first
2. Add your special requirements for characters
3. Choose the number of protagonists and supporting characters
4. Click "Generate Character Settings"
5. Edit and adjust — auto-saves
6. Supports "🗑️ Clear Character Settings"

#### Step 3 📋 Novel Outline
1. **Prerequisite**: Must complete world-building + character design
2. Add requirements for plot direction
3. Set total chapter count (supports range input like "30-50")
4. Set approximate word count per chapter
5. Click "Generate Outline"
6. Edit and adjust — auto-saves
7. Supports "🗑️ Clear Outline"

#### Step 4 📖 Chapter Generation
1. **Prerequisite**: Must complete world-building + characters + outline
2. **Select chapter**:
   - Existing chapters can be quickly selected from dropdown for editing or regeneration
   - Or manually enter a new chapter number and title
3. **Set target word count**: 500-8000 characters, default 2000
4. Click "Generate This Chapter"
   - ✨ Auto-categorized context building (core settings + relevant outline + previous chapter summary + vector retrieval) ensures setting consistency
   - 💡 Click the **?** icon next to the chapter title to view AI context building details
5. After generation, edit — changes auto-update vector store
6. Generated chapters can be re-generated with "🔄 Regenerate This Chapter"
7. Supports "🗑️ Delete This Chapter" for individual chapter deletion

> ⚠️ **Target word count vs. max_tokens**: Chinese 1 character ≈ 1.5-2 tokens. If `max_tokens` is set below target word count × 1.5, AI output may be truncated before reaching the target. Recommended: `max_tokens ≥ target word count × 1.5`.

#### Step 5 ✍️ Continuation
Continuation is not bound by the sequential flow, but having world and character settings is recommended:
1. **Select continuation source**:
   - From existing chapter: dropdown select, auto-loads content
   - Free text input: paste any text
2. Enter continuation instructions (default "continue writing" works)
3. Set continuation target word count
4. Click "Start Continuation"
5. Continuation results support:
   - 📋 Merged full text preview (original + continuation)
   - 📎 One-click append to chapter end
   - 🗑️ Clear continuation result

#### Step 6 🎨 Style Polishing
Polish existing text by imitating a specified work or author's writing style:
1. **Select text source**:
   - From existing chapter
   - Free text input
2. **Select style**: Enter a work name (e.g., "Dream of the Red Chamber") or author name (e.g., "Yu Hua")
3. Click "Start Polishing"
4. Side-by-side comparison of original and polished text
5. Polished result supports:
   - ✅ One-click replace chapter content
   - 🗑️ Clear polished result

#### Step 7 🔍 Consistency Check
Worried about contradictions after changing character names or world settings? Use consistency check:
1. Available after completing at least two steps (world/characters/outline)
2. Click "Start Consistency Check"
3. AI cross-compares and flags:
   - 🔴 **Serious contradiction** (e.g., name inconsistency — must fix)
   - 🟡 **Potential issue** (recommended fix, may affect continuity)
   - ✅ **Consistent — no issues**
4. Check results auto-persist — still visible after switching tabs
5. Supports "🗑️ Clear Check Results"

#### Step 8 🔎 Global Find & Replace
No need to manually update each step after changing a character name:
1. Enter the text to find in "Find Content" (e.g., "Li Ming")
2. Real-time preview of matches (shows which sections and how many)
3. Enter replacement text in "Replace With" (e.g., "Zhang San")
4. Click "Execute Replace" → after confirmation, one-click replace across all content
5. Replacement scope includes: world-building, character settings, outline, all chapters (title + content)
6. Auto-syncs vector store after replacement

#### Step 9 🕸️ Character Relationship Graph
Visualize the relationship network between characters:
1. **Prerequisite**: Must complete character design first
2. Click "Generate Character Relationship Graph"
3. AI auto-extracts characters and relationships from settings
4. Generates visual graph:
   - 🔵 Protagonist, 🔴 Antagonist, 🟡 Supporting — different colors for role types
   - Edge labels show relationship types: mentor-student, lovers, enemies, friends, master-servant, fellow disciples, relatives, etc.
5. Character list and relationship details displayed below
6. Graph data auto-persists — supports "🗑️ Clear Character Graph"

#### Step 10 📤 Export Novel
After completing all creation:
1. Switch to the "Export Novel" tab
2. Preview, click "Save to File" to save in `output/` directory
3. Or click "Download Markdown File" to download directly

---

## 🔄 Editing & Conflict Management

### Editing Existing Content

All steps support manual editing after generation — edits auto-save to the vector store:

- World-building, characters, outline, and chapter content can all be edited at any time
- Changes auto-save on blur — no manual action needed

### Conflict Warning Mechanism

When editing an upstream step, the system automatically detects downstream content and issues warnings:

| Modified Step | Potentially Affected Downstream |
|--------------|-------------------------------|
| 🌍 World-Building | 👤 Characters, 📋 Outline, 📖 Chapters |
| 👤 Character Design | 📋 Outline, 📖 Chapters |
| 📋 Novel Outline | 📖 Chapters |

For example: if you change the protagonist's name from "Li Ming" to "Zhang San" in character settings, the outline and chapters won't auto-update. The system will warn you:
> ⚠️ Character settings updated! The following may need regeneration: 📋 Novel Outline, 📖 Chapters

**Recommended modification flow:**
1. Edit upstream content → receive conflict warning
2. Use "🔎 Global Find & Replace" to one-click sync old names
3. Use "🔍 Consistency Check" to confirm no remaining contradictions
4. Done!

### Regeneration & Deletion

If you're unsatisfied with a step's result:
- When content already exists, the generate button auto-changes to "🔄 Regenerate XXX"
- Regeneration overwrites current content — confirm before proceeding
- Each tab provides a "🗑️ Clear" button to one-click clear all data for that step (including persisted data in the vector store)
- Chapters support individual deletion

---

## 💾 Data Persistence

### Persistence Scope

All operation results auto-save to the local ChromaDB vector store — no data loss when switching tabs or novels:

| Data Type | Persisted Content | Storage Method |
|----------|------------------|----------------|
| World-Building | Generated content + original Prompt | Vector store section + extra_data |
| Character Design | Generated content + original Prompt | Vector store section + extra_data |
| Novel Outline | Generated content + original Prompt | Vector store section + extra_data |
| Chapters | Chapter content + title | Vector store section |
| Consistency Check | Check results | extra_data |
| Character Graph | Graph JSON data | extra_data |

### extra_data Mechanism

Non-core creative content (check results, original prompts, graph data, etc.) is stored in ChromaDB via the `extra_data` mechanism:
- Uses a single JSON document (`_extra_data`) to store all extra data
- Read/write via `save_extra_data(key, value)` / `load_extra_data(key)` / `delete_extra_field(key)`
- Managed independently from vector store sections

---

## 📚 Multi-Novel Management

### Sidebar Novel List

The left sidebar displays all novels — the currently selected one is shown as a **prominent blue gradient card**:
- 📖 + Blue highlight border + "Current" tag = novel being edited
- 📕 + Dark card = other novels

### Operations

| Operation | Description |
|-----------|------------|
| Create new novel | Enter name in the input box, auto-creates new vector store |
| Switch novel | Click "✏️ Open" button in the novel list |
| Delete novel | Click "🗑️ Delete", confirm to delete |
| Delete all novels | Click "💣️ Delete All", confirm to clear |

### Data Storage

- Each novel is saved in an independent Collection under the `chroma_db/` directory
- Novel ID format: `novel_timestamp` (no book name to avoid special character issues)
- Auto-loads corresponding vector store data when switching novels (including all persisted extra_data)
- Exported novels are saved in the `output/` directory

---

## 💡 Tips

### Cost Control

| Novel Length | Estimated Cost (Volcengine Ark) |
|-------------|-------------------------------|
| 10K chars | ¥0.3-0.5 |
| 100K chars | **¥3-5** |
| 500K chars | ¥15-25 |
| 1M chars | ¥30-50 |

The price of a cup of milk tea for a full-length novel.

### Money-Saving Tips

- **Plan with outline first**: Outlines use few tokens — refine the outline before writing chapters to avoid costly regeneration
- **Use editing**: AI-generated content can be directly modified — no need to regenerate every time
- **Continue instead of rewrite**: If only a section is unsatisfactory, use continuation — cheaper than regenerating a whole chapter
- **Set max_tokens wisely**: For chapter generation, set `max_tokens` to 1.5-2× the target word count — setting it too high wastes quota

### Writing Tips

- **Be specific with world descriptions**: The more detailed your description, the closer AI-generated settings will match your vision
- **Add key conflicts to character design**: Tell AI what character contradictions you want — the outline will be more compelling
- **Plan outline by volumes**: Set a larger chapter count (e.g., 50) — the outline will auto-divide into volumes for better pacing
- **Define chapter titles in advance**: Think of titles before generation — AI will generate content matching the title direction
- **Use global replace**: Change names with "Find & Replace" for one-click sync — much easier than manual edits
- **Use consistency check regularly**: Run a check after a few chapters to catch contradictions early before they compound
- **Leverage style polishing**: Generate content first, then polish with your favorite author's style for better results

---

## 📂 Project Structure

```
ai-novel-writer/
├── api/
│   └── api_client.py           # Universal LLM API client (OpenAI-compatible)
├── vector_store/
│   └── local_chroma.py         # Local ChromaDB vector store wrapper (with extra_data persistence)
├── workflow/
│   └── novel_workflow.py       # Full-chain creative workflow + consistency check + find/replace + character graph
├── app.py                      # Streamlit Web UI (main entry, 10 feature tabs)
├── output/                     # Exported novels saved here (auto-created)
├── chroma_db/                  # Chroma vector database files (auto-generated after running)
├── .streamlit/
│   └── config.toml             # Streamlit config (localhost only by default)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .env                        # Environment variables (create yourself, do NOT commit to Git)
├── .gitignore                  # Git ignore rules (excludes sensitive data and vector store)
├── LICENSE                     # MIT License
└── README.md                   # Chinese README
```

---

## ❓ FAQ

### Q: What is a vector store? What does "local" mean?
A: A vector store is a **smart search engine** that helps AI remember previously written settings. "Local" means the data is stored on your own hard drive — free and private.

### Q: Can I use any AI model?
A: Yes, as long as it's compatible with the OpenAI Chat API format. The vector store outputs plain text that any AI can understand.

### Q: Will prompts get longer and more expensive as I write?
A: No. Chapter generation uses **categorized context building**: direct settings extraction (with truncation) + relevant outline extraction (only nearby chapters) + previous chapter summary (only last 2 chapters' endings) + semantic search supplement (up to 3 results, skipping already-included settings, other chapters truncated to 500 chars). Prompt length stays stable, and so does cost.

### Q: What's the relationship between target word count and max_tokens?
A: Chinese 1 character ≈ 1.5-2 tokens. If `max_tokens` < target word count × 1.5, AI output will be truncated before reaching the target (e.g., target 2000 chars but max_tokens set to 1500 — might only output ~1000 chars). Recommended: `max_tokens ≥ target word count × 1.5`.

### Q: Will old data remain in the vector store after editing?
A: No. Each save uses fixed document IDs (e.g., `setting_world_setting`, `chapter_chapter_1`). Editing auto-overwrites old data — the vector store always has the latest version.

### Q: How do I sync a character name change across all content?
A: Use the "🔎 Global Find & Replace" feature! Enter the old name and new name, one-click replace across all settings and chapters, auto-syncs the vector store. No more manual updates one by one.

### Q: How is the character relationship graph generated?
A: AI analyzes your character settings and outline to auto-extract characters and their relationships (mentor-student, lovers, enemies, etc.), then renders a visual graph with Graphviz. Each generation re-analyzes to ensure consistency with the latest settings. Graph data is persisted — no loss when switching tabs.

### Q: Are consistency check and character graph results saved?
A: Yes! All results (including consistency check results, character graph data, and original prompts for each step) are persisted via ChromaDB's extra_data mechanism. Data persists across tab switches and novel switches. Manual clear via the "🗑️ Clear" button.

### Q: Is this project compliant? Will using Volcengine get my account suspended?
A: If you follow the instructions and use **Volcengine Ark pay-per-use**, it's fully compliant — no account suspension. Just don't use CodingPlan discounted subscriptions for non-coding purposes.

### Q: Is it secure? Can others access my app over the network?
A: The default config only listens on localhost (loopback) — not exposed externally. The `.env` file contains your API Key and is excluded in `.gitignore` — won't be committed to Git. The `chroma_db/` directory contains your creative data and is also excluded.

---

## 🛠️ Roadmap

Issues and PRs are welcome:

- [ ] Support auto-continuous multi-chapter generation
- [ ] Support plot branching / story trees
- [ ] Support PDF/EPUB export
- [ ] Support more vector store backends (FAISS, Milvus, etc.)
- [ ] Interactive character graph editing
- [ ] Writing style templates (ancient Chinese, sci-fi, romance, etc.)
- [ ] Custom context strategy for chapter generation

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

You are free to use, modify, and distribute this for personal or commercial projects.

---

## 🙏 Acknowledgments

- [ChromaDB](https://github.com/chroma-core/chroma) - Open-source lightweight vector database
- [Streamlit](https://streamlit.io/) - Rapid web framework
- [Graphviz](https://graphviz.org/) - Graph visualization tool
- Volcengine - Cost-effective Chinese LLM
