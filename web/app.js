/* AI小说创作 - 单页前端（原生JS） */
"use strict";

// ---------- 全局状态 ----------
const S = {
  novelId: localStorage.getItem("novel_id") || "",
  novel: null,          // 当前小说全量数据
  novels: [],
  activeTab: parseInt(localStorage.getItem("active_tab") || "0"),
  generating: false,
  memoryStatus: null,   // TODO 1.1：台账/摘要状态缓存（refreshNovel 时刷新）
};

const TABS = [
  ["🌍 世界观", "world"], ["👤 人物", "characters"], ["📋 大纲", "outline"],
  ["📖 章节", "chapter"], ["✍️ 续写", "continue"], ["🎨 润色", "polish"],
  ["🔍 一致性", "consistency"], ["🔎 查找替换", "findreplace"],
  ["🕸️ 角色图谱", "graph"], ["📤 导出", "export"], ["🧩 Skill管理", "skills"],
];

// ---------- 工具 ----------
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts = {}) {
  if (opts.body && typeof opts.body !== "string") {
    opts.body = JSON.stringify(opts.body);
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  }
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { msg = (await resp.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return resp.json();
}

function alertMsg(kind, text, ms = 6000) {
  const area = $("#alert-area");
  const d = el("div", `msg ${kind}`, text);
  area.appendChild(d);
  setTimeout(() => d.remove(), ms);
}

// ---------- 生成任务（SSE） ----------
const STEP_NAMES = {
  world_setting: "世界观", characters: "人物设定", outline: "大纲",
  chapter: "章节", chapter_beats: "场景节拍", golden_chapter: "黄金开篇",
  continue: "续写", polish: "润色", chapter_review: "章节评审", chapter_revise: "章节改写",
  style_fingerprint: "文风指纹", humanize: "去AI腔", batch_chapters: "批量生成",
  consistency: "一致性检查", relation_graph: "角色图谱", distill: "技能蒸馏",
  extend_outline: "扩展大纲", volume_chapters: "卷逐章概要", rewrite_outline: "局部改写大纲",
  migrate_cards: "迁移角色卡", memory_rebuild: "重建台账", rewrite_preview: "改写建议预览",
};

function runGeneration(step, params, { onToken, onDone, onError, streamTarget } = {}) {
  S.generating = true;
  renderTabs();
  const overlay = $("#gen-overlay");
  const statusEl = $("#gen-status");
  const streamEl = $("#gen-stream");
  const stageEl = $("#gen-stage");
  const metaEl = $("#gen-meta");
  statusEl.textContent = `正在生成：${STEP_NAMES[step] || step}…`;
  stageEl.classList.add("hidden");
  metaEl.classList.add("hidden");
  streamEl.textContent = "";
  streamEl.classList.toggle("hidden", !streamTarget && !onToken);
  overlay.classList.remove("hidden");

  api("/api/generate", { method: "POST", body: { novel_id: S.novelId, step, params } })
    .then(({ task_id }) => {
      const es = new EventSource(`/api/tasks/${task_id}/stream`);
      let streamed = "";
      es.addEventListener("progress", (e) => {
        const d = JSON.parse(e.data);
        if (typeof d === "string") { statusEl.textContent = d; return; }
        // 结构化进度：{msg, stage?, stage_total?, phase?}
        statusEl.textContent = d.msg || "";
        if (d.stage && d.stage_total > 1) {
          stageEl.textContent = `阶段 ${d.stage} / ${d.stage_total}`;
          stageEl.classList.remove("hidden");
        }
        if (d.phase === "requesting" && !streamed) {
          metaEl.textContent = "等待模型响应…";
          metaEl.classList.remove("hidden");
        }
      });
      es.addEventListener("token", (e) => {
        const tok = JSON.parse(e.data);
        streamed += tok;
        metaEl.textContent = `正在接收输出 · 已收到 ${streamed.length} 字`;
        metaEl.classList.remove("hidden");
        streamEl.classList.remove("hidden");
        streamEl.textContent = streamed;
        streamEl.scrollTop = streamEl.scrollHeight;
        if (onToken) onToken(tok, streamed);
        if (streamTarget) { streamTarget.value = streamed; }
      });
      es.addEventListener("chapter_done", (e) => {
        const d = JSON.parse(e.data);
        alertMsg("success", `✅ 第${d.num}章「${d.title}」生成完成`);
        refreshNovel(true);
      });
      es.addEventListener("done", (e) => {
        es.close(); finish();
        S.generating = false; renderTabs();
        const result = JSON.parse(e.data);
        if (result.warning) alertMsg("warn", result.warning, 10000);
        refreshNovel(true);
        if (onDone) onDone(result);
      });
      es.addEventListener("error", (e) => {
        es.close(); finish();
        S.generating = false; renderTabs();
        const msg = JSON.parse(e.data);
        alertMsg("error", msg, 12000);
        if (onError) onError(msg);
      });
      es.onerror = () => {
        // EventSource 自动重连；仅在任务已结束时关闭
      };
    })
    .catch((err) => {
      finish();
      S.generating = false; renderTabs();
      alertMsg("error", err.message, 12000);
      if (onError) onError(err.message);
    });

  function finish() { overlay.classList.add("hidden"); }
}

// ---------- 数据加载 ----------
async function refreshNovel(keepTab = false) {
  if (!S.novelId) { S.novel = null; S.memoryStatus = null; renderAll(); return; }
  try {
    S.novel = await api(`/api/novels/${encodeURIComponent(S.novelId)}`);
  } catch (e) {
    S.novel = null; S.novelId = ""; localStorage.removeItem("novel_id");
  }
  // TODO 1.1：顺带刷新台账/摘要状态（失败忽略，不影响主流程）
  try {
    S.memoryStatus = S.novel
      ? await api(`/api/novels/${encodeURIComponent(S.novelId)}/memory/status`) : null;
  } catch (e) { /* ignore */ }
  renderAll();
}

async function refreshNovelList() {
  const { novels } = await api("/api/novels");
  S.novels = novels;
  renderNovelManager();
  renderCurrentNovelCard();
}

async function refreshUsage() {
  try {
    const u = await api("/api/usage");
    const s = u.session, c = u.cumulative;
    const body = $("#usage-body");
    body.innerHTML = "";

    const mkRow = (html) => { const d = el("div", "caption"); d.innerHTML = html; return d; };
    body.appendChild(mkRow(`<b>本次会话</b>：${s.calls} 次调用 · 输入 ${s.prompt_tokens} / 输出 ${s.completion_tokens} tokens`));
    body.appendChild(mkRow(`字符量：入 ${s.total_chars_in} / 出 ${s.total_chars_out}`));
    body.appendChild(mkRow(`<b>历史总用量</b>：${c.calls} 次 · 输入 ${c.prompt_tokens} / 输出 ${c.completion_tokens} tokens`));

    // 单模型用量（拆分输入/输出 token）
    const models = Object.entries(u.by_model || {});
    if (models.length) {
      body.appendChild(mkRow(`<b>按模型统计</b>：`));
      for (const [model, m] of models) {
        const row = el("div", "caption");
        row.style.cssText = "display:flex;justify-content:space-between;align-items:center;gap:6px";
        const txt = el("span");
        txt.innerHTML = `${esc(model)}：${m.calls} 次 · 输入 ${m.prompt_tokens} / 输出 ${m.completion_tokens} · 合计 ${m.prompt_tokens + m.completion_tokens}`;
        const clr = el("button", "btn small shrink", "清除");
        clr.onclick = async () => {
          if (!confirm(`清除模型「${model}」的用量统计？（将从总用量中扣减）`)) return;
          await api("/api/usage/clear", { method: "POST", body: { model } });
          refreshUsage();
        };
        row.append(txt, clr);
        body.appendChild(row);
      }
    }

    const clearAll = el("button", "btn small", "🗑️ 清除全部用量统计");
    clearAll.style.marginTop = "6px";
    clearAll.onclick = async () => {
      if (!confirm("清除全部用量统计？（含本次会话、历史总用量和所有单模型用量）")) return;
      await api("/api/usage/clear", { method: "POST", body: {} });
      refreshUsage();
    };
    body.appendChild(clearAll);
  } catch (e) { $("#usage-body").innerHTML = `<div class="caption">${esc(e.message)}</div>`; }
}

async function refreshProviders() {
  const { providers, active } = await api("/api/providers");
  const body = $("#provider-config-body");
  body.innerHTML = "";
  const sel = el("select");
  sel.innerHTML = `<option value="">-- 新增自定义配置 --</option>` +
    providers.map(p => `<option value="${esc(p.name)}" ${p.name === active.name ? "selected" : ""}>${esc(p.name)}</option>`).join("");
  sel.onchange = async () => {
    if (sel.value) {
      await api("/api/providers/active", { method: "POST", body: { name: sel.value } });
      alertMsg("success", `已切换到配置「${sel.value}」，正在自动测试连接…`);
      refreshUsage();
      // 切换配置后自动测试连接
      const p = providers.find(x => x.name === sel.value);
      if (p) {
        const r = await testProviderConn(p);
        alertMsg(r.ok ? "success" : "error", `${r.msg}（${r.latency}ms）`);
      } else {
        renderConnStatus();
      }
    }
  };
  body.appendChild(sel);

  // 带说明文字的输入字段：label + hint + input
  const mkField = (labelText, hintText, placeholder) => {
    const wrap = el("div");
    wrap.style.marginTop = "8px";
    const lb = el("label", "field", labelText);
    lb.style.margin = "0";
    const input = el("input"); input.type = "text"; input.placeholder = placeholder;
    const hint = el("div", "field-hint", hintText);
    wrap.append(lb, input, hint);
    body.appendChild(wrap);
    return input;
  };
  const nameI = mkField("配置名称", "自定义名字，仅用于本地保存和区分多组配置（如：豆包-主用、DeepSeek-备用），与接口模型名无关", "如：豆包-主用");
  const keyI = mkField("API Key", "服务商后台获取的接口密钥，只保存在本地 user_config.json", "sk-...");
  const baseI = mkField("API Base URL", "接口地址（OpenAI 兼容格式），如火山方舟 https://ark.cn-beijing.volces.com/api/v3/chat/completions", "https://.../chat/completions");
  const modelI = mkField("接口模型名", "服务商要求填写的模型 ID（这才是发给接口的模型名），如 doubao-pro-32k、deepseek-chat", "如 doubao-pro-32k");
  // 选中配置时回填
  const fill = () => {
    const p = providers.find(x => x.name === sel.value);
    if (p) { nameI.value = p.name; keyI.value = p.api_key || ""; baseI.value = p.api_base || ""; modelI.value = p.model || ""; }
  };
  sel.addEventListener("change", fill); fill();

  const row = el("div", "row"); row.style.marginTop = "8px";
  const saveBtn = el("button", "btn primary small", "💾 保存并启用");
  saveBtn.onclick = async () => {
    try {
      const body = { name: nameI.value, api_key: keyI.value, api_base: baseI.value, model: modelI.value };
      await api("/api/providers", { method: "POST", body });
      alertMsg("success", "配置已保存并启用，正在自动测试连接…");
      refreshProviders(); refreshUsage();
      const r = await testProviderConn(body);
      alertMsg(r.ok ? "success" : "error", `${r.msg}（${r.latency}ms）`);
    } catch (e) { alertMsg("error", e.message); }
  };
  const testBtn = el("button", "btn small", "🔌 测试连接");
  testBtn.onclick = async () => {
    testBtn.disabled = true; testBtn.textContent = "测试中…";
    const r = await testProviderConn({ name: nameI.value, api_key: keyI.value, api_base: baseI.value, model: modelI.value });
    if (r.msg) alertMsg(r.ok ? "success" : "error", `${r.msg}（${r.latency}ms）`);
    testBtn.disabled = false; testBtn.textContent = "🔌 测试连接";
  };
  const delBtn = el("button", "btn small", "🗑️ 删除");
  delBtn.onclick = async () => {
    if (!sel.value || !confirm(`删除配置「${sel.value}」？`)) return;
    await api(`/api/providers/${encodeURIComponent(sel.value)}`, { method: "DELETE" });
    refreshProviders(); renderConnStatus();
  };
  row.append(saveBtn, testBtn, delBtn);
  body.appendChild(row);
}

async function renderConnStatus() {
  const wrap = $("#conn-status");
  const raw = localStorage.getItem("conn_status");
  let html = `<span style="color:var(--fg-dim)">⚪ 未测试连接</span>`;
  if (raw) {
    try {
      const r = JSON.parse(raw);
      if (r.testing) {
        html = `<span style="color:var(--yellow)">🟡 模型 ${esc(r.model || r.name || "")} 测试连接中…</span>`;
      } else {
        html = r.ok
          ? `<span style="color:var(--green)">🟢 ${esc(r.msg)}${r.latency ? `（${r.latency}ms）` : ""}</span>`
          : `<span style="color:var(--red)">🔴 ${esc(r.msg)}</span>`;
      }
    } catch (e) { /* ignore */ }
  }
  let activeHtml = "";
  try {
    const { active } = await api("/api/providers");
    if (active && active.name) {
      activeHtml = `<div class="caption">当前配置：<b>${esc(active.name)}</b>${active.model ? `<br>接口模型：<code>${esc(active.model)}</code>` : ""}</div>`;
    }
  } catch (e) { /* ignore */ }
  wrap.innerHTML = `<div class="caption">连接状态</div>${html}${activeHtml}`;
}

// 测试某个配置连通性：先置「测试连接中」状态，完成后更新结果
async function testProviderConn(p) {
  localStorage.setItem("conn_status", JSON.stringify({ testing: true, name: p.name, model: p.model }));
  renderConnStatus();
  try {
    const r = await api("/api/providers/test", { method: "POST", body: { name: p.name, api_key: p.api_key || "", api_base: p.api_base || "", model: p.model || "" } });
    localStorage.setItem("conn_status", JSON.stringify(r));
    renderConnStatus();
    return r;
  } catch (e) {
    localStorage.setItem("conn_status", JSON.stringify({ ok: false, msg: e.message, latency: 0 }));
    renderConnStatus();
    return { ok: false, msg: e.message, latency: 0 };
  }
}

// ---------- 侧边栏：小说管理 ----------
function renderCurrentNovelCard() {
  const wrap = $("#current-novel-card");
  if (!S.novelId) { wrap.innerHTML = `<div class="caption">尚未选择小说</div>`; return; }
  const name = S.novel?.name || S.novelId;
  wrap.innerHTML = `<h3>📖 ${esc(name)}</h3>`;
  const row = el("div", "row");
  const input = el("input"); input.type = "text"; input.placeholder = "输入新名称";
  const btn = el("button", "btn small shrink", "改名");
  btn.onclick = async () => {
    if (!input.value.trim()) return;
    await api(`/api/novels/${encodeURIComponent(S.novelId)}/rename`, { method: "POST", body: { name: input.value.trim() } });
    input.value = "";
    refreshNovel(); refreshNovelList();
  };
  row.append(input, btn);
  wrap.appendChild(row);
}

function renderNovelManager() {
  const body = $("#novel-manager-body");
  body.innerHTML = "";
  const row = el("div", "row");
  const input = el("input"); input.type = "text"; input.placeholder = "输入小说名称，点击创建";
  const btn = el("button", "btn primary small shrink", "➕ 创建");
  btn.onclick = async () => {
    if (!input.value.trim()) { alertMsg("error", "请输入小说名称"); return; }
    const dup = S.novels.find(n => n.name === input.value.trim());
    if (dup && !confirm(`已存在同名小说《${dup.name}》，仍要创建吗？`)) return;
    const r = await api("/api/novels", { method: "POST", body: { name: input.value.trim() } });
    input.value = "";
    S.novelId = r.id; localStorage.setItem("novel_id", r.id);
    await refreshNovelList(); refreshNovel();
  };
  row.append(input, btn);
  body.appendChild(row);

  for (const n of S.novels) {
    const item = el("div", `novel-item ${n.id === S.novelId ? "active" : ""}`);
    item.innerHTML = `<div>${n.id === S.novelId ? "📖 " : "📕 "}<b>${esc(n.name)}</b></div>
      <div class="meta">设定${n.type_counts.setting} · 人物${n.type_counts.character} · 大纲${n.type_counts.outline} · 章节${n.type_counts.chapter}</div>`;
    item.onclick = () => {
      S.novelId = n.id; localStorage.setItem("novel_id", n.id);
      refreshNovelList(); refreshNovel();
    };
    const ops = el("div", "ops");
    const del = el("button", "btn small", "🗑️ 删除");
    del.onclick = async (ev) => {
      ev.stopPropagation();
      if (!confirm(`确定删除《${n.name}》？此操作不可恢复！`)) return;
      await api(`/api/novels/${encodeURIComponent(n.id)}`, { method: "DELETE" });
      if (S.novelId === n.id) { S.novelId = ""; localStorage.removeItem("novel_id"); S.novel = null; }
      refreshNovelList(); refreshNovel();
    };
    ops.appendChild(del);
    item.appendChild(ops);
    body.appendChild(item);
  }
  if (S.novels.length > 1) {
    const delAll = el("button", "btn small", "🗑️ 删除全部小说");
    delAll.style.marginTop = "8px";
    delAll.onclick = async () => {
      if (!confirm(`确定删除全部 ${S.novels.length} 本小说？不可恢复！`)) return;
      if (!confirm("再次确认：真的要删除所有小说吗？")) return;
      await api("/api/novels", { method: "DELETE" });
      S.novelId = ""; localStorage.removeItem("novel_id"); S.novel = null;
      refreshNovelList(); refreshNovel();
    };
    body.appendChild(delAll);
  }
}

// ---------- 顶部进度 ----------
function renderProgress() {
  const wrap = $("#progress-bar-wrap");
  if (!S.novel) { wrap.innerHTML = ""; return; }
  const done = [S.novel.world_setting, S.novel.characters, S.novel.outline,
    Object.keys(S.novel.chapters || {}).length ? "y" : ""].filter(Boolean).length;
  const words = Object.values(S.novel.chapters || {}).reduce((a, c) => a + (c.content || "").length, 0);
  wrap.innerHTML = `创作进度 ${done}/4 步 · 总字数 ${words}
    <div class="progress-track"><div class="progress-fill" style="width:${done / 4 * 100}%"></div></div>`;
}

// ---------- Tabs ----------
function renderTabs() {
  const nav = $("#tabs");
  nav.innerHTML = "";
  TABS.forEach(([label], i) => {
    const b = el("button", i === S.activeTab ? "active" : "", label);
    b.onclick = () => { S.activeTab = i; localStorage.setItem("active_tab", i); renderTabContent(); renderTabs(); };
    nav.appendChild(b);
  });
}

function requireNovel() {
  if (!S.novel) {
    $("#tab-content").innerHTML = `<div class="msg info">请先在左侧「小说管理」中创建或选择一本小说。</div>`;
    return false;
  }
  return true;
}

function extra(key, dflt = "") { return (S.novel?.extra || {})[key] ?? dflt; }

function saveSection(type, title, content) {
  return api(`/api/novels/${encodeURIComponent(S.novelId)}/section`, { method: "PUT", body: { type, title, content } });
}
function delSection(type, title) {
  return api(`/api/novels/${encodeURIComponent(S.novelId)}/section?type=${encodeURIComponent(type)}&title=${encodeURIComponent(title)}`, { method: "DELETE" });
}
function saveExtra(key, value) {
  return api(`/api/novels/${encodeURIComponent(S.novelId)}/extra/${encodeURIComponent(key)}`, { method: "PUT", body: { value } });
}
function delExtra(key) {
  return api(`/api/novels/${encodeURIComponent(S.novelId)}/extra/${encodeURIComponent(key)}`, { method: "DELETE" });
}

/* 通用生成块：prompt 输入 + 按钮 + 参数 */
function genBlock({ label, placeholder, promptValue = "", btnText, step, extraFields = "", collectParams, disabled = false }) {
  const wrap = el("div", "box");
  wrap.innerHTML = `<label class="field">${esc(label)}</label>
    <textarea rows="3" placeholder="${esc(placeholder)}">${esc(promptValue)}</textarea>
    ${extraFields}
    <div style="margin-top:10px"><button class="btn primary" ${disabled ? "disabled" : ""}>${esc(btnText)}</button></div>`;
  const btn = wrap.querySelector("button");
  btn.onclick = () => {
    const params = collectParams(wrap);
    if (params === null) return;
    runGeneration(step, params);
  };
  return wrap;
}

/* 可编辑内容视图：显示 + 编辑自动保存 + 清除 + 原始对比 */
function editableContent({ title, content, type, sectionTitle, originalKey, promptKey, onCleared }) {
  const wrap = el("div", "box");
  const ta = el("textarea"); ta.rows = 14; ta.value = content;
  const status = el("div", "caption", "编辑后自动保存");
  let timer = null;
  ta.oninput = () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      await saveSection(type, sectionTitle, ta.value);
      status.textContent = "✅ 已自动保存 " + new Date().toLocaleTimeString();
    }, 800);
  };
  const ops = el("div", "row");
  const clearBtn = el("button", "btn small shrink", "🗑️ 清除");
  clearBtn.onclick = async () => {
    if (!confirm(`确定清除「${title}」？`)) return;
    await delSection(type, sectionTitle);
    if (originalKey) await delExtra(originalKey);
    if (promptKey) await delExtra(promptKey);
    refreshNovel(true);
    if (onCleared) onCleared();
  };
  ops.append(clearBtn);
  wrap.append(ta, status, ops);

  const original = originalKey ? extra(originalKey) : "";
  if (original && original !== content) {
    const det = el("details");
    det.innerHTML = `<summary class="caption">🔍 查看原始 AI 输出（与当前编辑版本不同）</summary><div class="content-view">${esc(original)}</div>`;
    wrap.appendChild(det);
  }
  const promptText = promptKey ? extra(promptKey) : "";
  if (promptText) {
    const det = el("details");
    det.innerHTML = `<summary class="caption">📝 生成时使用的需求描述</summary><div class="content-view">${esc(promptText)}</div>`;
    wrap.appendChild(det);
  }
  return wrap;
}

// ---------- 各 Tab 渲染 ----------

function tabWorld(root) {
  root.append(genBlock({
    label: "描述你想要的世界观（时代背景、力量体系、社会环境等）",
    placeholder: "例如：修仙世界，宗门林立，弱肉强食……",
    promptValue: extra("world_setting_prompt"),
    btnText: S.novel.world_setting ? "🔄 重新生成世界观" : "🚀 生成世界观",
    step: "world_setting",
    collectParams: (w) => {
      const p = w.querySelector("textarea").value.trim();
      if (!p) { alertMsg("error", "请输入世界观描述"); return null; }
      return { prompt: p };
    },
  }));
  if (S.novel.world_setting) {
    root.append(editableContent({
      title: "世界观设定", content: S.novel.world_setting,
      type: "setting", sectionTitle: "world_setting",
      originalKey: "world_setting_original", promptKey: "world_setting_prompt",
    }));
  }
}

function tabCharacters(root) {
  const ef = `<div class="row" style="margin-top:8px">
    <div><label class="field">主角人数</label><input type="number" id="num-main" min="1" max="10" value="${esc(extra("characters_num_main", "2"))}"></div>
    <div><label class="field">配角人数</label><input type="number" id="num-support" min="0" max="20" value="${esc(extra("characters_num_support", "5"))}"></div>
  </div>`;
  if (!S.novel.world_setting) {
    root.insertAdjacentHTML("beforeend", `<div class="msg warn">⚠️ 建议先生成世界观设定，再生成人物。</div>`);
  }
  root.append(genBlock({
    label: "描述人物要求（主角性格、背景、关系等）",
    placeholder: "例如：主角是冷静理智的年轻修士……",
    promptValue: extra("characters_prompt"),
    btnText: S.novel.characters ? "🔄 重新生成人物" : "🚀 生成人物设定",
    step: "characters", extraFields: ef,
    collectParams: (w) => {
      const p = w.querySelector("textarea").value.trim();
      if (!p) { alertMsg("error", "请输入人物设定要求"); return null; }
      return { prompt: p, num_main: +w.querySelector("#num-main").value, num_support: +w.querySelector("#num-support").value };
    },
  }));
  if (S.novel.characters) {
    root.append(editableContent({
      title: "人物设定", content: S.novel.characters,
      type: "character", sectionTitle: "all_characters",
      originalKey: "characters_original", promptKey: "characters_prompt",
    }));
  }
  // TODO 2.4：角色卡编辑器（放在既有生成/自由文本区之后，作为增强而非替换）
  renderCharacterCards(root);
}

/* TODO 2.4：角色卡编辑器（structured 模式）/ AI 迁移入口（freetext 模式） */
async function renderCharacterCards(root) {
  const box = el("div", "box");
  box.innerHTML = `<h4>🗂️ 角色卡</h4><div class="caption">加载中…</div>`;
  root.append(box);
  let data;
  try {
    data = await api(`/api/novels/${encodeURIComponent(S.novelId)}/character_cards`);
  } catch (e) {
    box.innerHTML = `<h4>🗂️ 角色卡</h4><div class="msg error">${esc(e.message)}</div>`;
    return;
  }
  const { cards, mode, has_freetext } = data;
  box.innerHTML = `<h4>🗂️ 角色卡（${mode === "structured" ? `共 ${cards.length} 张` : "自由文本模式"}）</h4>
    <div class="caption">角色卡按登场/退场章节精确注入章节生成；保存后自动同步渲染为自由文本人物设定。</div>`;

  if (mode !== "structured") {
    // freetext 模式：提示 + AI 迁移入口
    box.innerHTML += `<div class="msg info">当前为自由文本模式。可由 AI 一次性解析为角色卡（解析可能丢失细节，请核对预览）。</div>`;
    if (has_freetext) {
      const migBtn = el("button", "btn primary", "🔄 AI 迁移为角色卡");
      migBtn.style.marginTop = "8px";
      migBtn.onclick = () => {
        if (!confirm("AI 解析可能丢失细节，迁移前会展示预览供核对。继续？")) return;
        runGeneration("migrate_cards", {}, {
          onDone: (r) => {
            if (!r.preview) { alertMsg("error", "迁移解析失败，已保持自由文本模式"); return; }
            renderMigrationPreview(box, r.preview);
          },
        });
      };
      box.append(migBtn);
    }
    return;
  }

  // structured 模式：卡片编辑列表
  const list = el("div");
  const newCard = () => ({ name: "", role: "support", identity: "", personality: "",
    relationships: "", appearance_chapter: 1, exit_chapter: null, notes: "" });

  const fieldRow = (label, inputEl) => {
    const w = el("div");
    const lb = el("label", "field", label);
    w.append(lb, inputEl);
    return w;
  };

  const renderCards = () => {
    list.innerHTML = "";
    cards.forEach((c, i) => {
      const card = el("div", "box");
      card.style.marginTop = "8px";
      const head = el("div", "row");
      head.innerHTML = `<b class="shrink" style="align-self:center">#${i + 1} ${esc(c.name || "（未命名）")}</b>`;
      const delBtn = el("button", "btn small shrink", "🗑️ 删除");
      delBtn.onclick = () => { if (confirm(`删除角色卡「${c.name || "#" + (i + 1)}」？`)) { cards.splice(i, 1); renderCards(); } };
      head.append(delBtn);
      card.append(head);

      const nameI = el("input"); nameI.type = "text"; nameI.value = c.name || ""; nameI.placeholder = "角色姓名";
      nameI.oninput = () => { c.name = nameI.value; head.querySelector("b").textContent = `#${i + 1} ${c.name || "（未命名）"}`; };
      const roleSel = el("select");
      roleSel.innerHTML = `<option value="main" ${c.role === "main" ? "selected" : ""}>主角</option>
        <option value="support" ${c.role !== "main" ? "selected" : ""}>配角</option>`;
      roleSel.onchange = () => { c.role = roleSel.value; };
      const r1 = el("div", "row");
      r1.append(fieldRow("姓名", nameI), fieldRow("类型", roleSel));
      card.append(r1);

      const idI = el("input"); idI.type = "text"; idI.value = c.identity || ""; idI.placeholder = "如：青云宗外门弟子";
      idI.oninput = () => { c.identity = idI.value; };
      const persI = el("input"); persI.type = "text"; persI.value = c.personality || ""; persI.placeholder = "如：冷静理智、外冷内热";
      persI.oninput = () => { c.personality = persI.value; };
      const r2 = el("div", "row");
      r2.append(fieldRow("身份", idI), fieldRow("性格", persI));
      card.append(r2);

      const relI = el("input"); relI.type = "text"; relI.value = c.relationships || ""; relI.placeholder = "如：与张三为师徒，与李四为敌";
      relI.oninput = () => { c.relationships = relI.value; };
      const appI = el("input"); appI.type = "number"; appI.min = "1"; appI.value = c.appearance_chapter || 1;
      appI.oninput = () => { c.appearance_chapter = +appI.value || 1; };
      const exitI = el("input"); exitI.type = "number"; exitI.min = "1"; exitI.placeholder = "留空=不退场";
      if (c.exit_chapter) exitI.value = c.exit_chapter;
      exitI.oninput = () => { c.exit_chapter = exitI.value ? +exitI.value : null; };
      const r3 = el("div", "row");
      r3.append(fieldRow("人物关系", relI), fieldRow("登场章节", appI), fieldRow("退场章节", exitI));
      card.append(r3);

      const notesI = el("input"); notesI.type = "text"; notesI.value = c.notes || ""; notesI.placeholder = "其他备注（可选）";
      notesI.oninput = () => { c.notes = notesI.value; };
      card.append(fieldRow("备注", notesI));
      list.append(card);
    });
  };
  renderCards();
  box.append(list);

  const ops = el("div", "row");
  ops.style.marginTop = "8px";
  const addBtn = el("button", "btn small shrink", "➕ 新增角色");
  addBtn.onclick = () => { cards.push(newCard()); renderCards(); };
  const saveBtn = el("button", "btn primary small shrink", "💾 保存角色卡");
  saveBtn.onclick = async () => {
    const clean = cards.filter(c => (c.name || "").trim());
    if (clean.length !== cards.length && !confirm("存在未填写姓名的角色卡，将被忽略。继续保存？")) return;
    try {
      await api(`/api/novels/${encodeURIComponent(S.novelId)}/character_cards`, { method: "PUT", body: { cards: clean } });
      alertMsg("success", `已保存 ${clean.length} 张角色卡`);
      // TODO 4.1 联动：保存/新增主角后检查其登场章节是否在大纲中有安排
      for (const c of clean.filter(x => x.role === "main")) {
        try {
          const chk = await api(`/api/novels/${encodeURIComponent(S.novelId)}/characters/check_appearance`,
            { method: "POST", body: { name: c.name, chapter: c.appearance_chapter || 1 } });
          if (!chk.mentioned) {
            alertMsg("warn", `⚠️ 主角「${c.name}」：大纲第${c.appearance_chapter || 1}章附近未安排该角色登场，可到大纲 tab 使用局部改写补充`, 10000);
          }
        } catch (e) { /* ignore */ }
      }
      refreshNovel(true);
    } catch (e) { alertMsg("error", e.message); }
  };
  ops.append(addBtn, saveBtn);
  box.append(ops);

  const backBtn = el("button", "btn small", "↩️ 切换回自由文本模式");
  backBtn.style.marginTop = "6px";
  backBtn.onclick = async () => {
    if (!confirm("切换回自由文本模式？角色卡将被删除（已渲染的自由文本人物设定保留）。")) return;
    await api(`/api/novels/${encodeURIComponent(S.novelId)}/character_cards`, { method: "DELETE" });
    alertMsg("success", "已切换回自由文本模式");
    refreshNovel(true);
  };
  box.append(backBtn);
}

/* TODO 2.2：AI 迁移预览——确认后才 PUT 入库 */
function renderMigrationPreview(box, preview) {
  const old = box.querySelector("#migrate-preview");
  if (old) old.remove();
  const pv = el("div", "box");
  pv.id = "migrate-preview";
  pv.innerHTML = `<h4>🔍 迁移预览（共 ${(preview.cards || []).length} 张卡，请核对）</h4>
    <div class="content-view preview-scroll">${esc(preview.rendered || "")}</div>`;
  const row = el("div", "row");
  row.style.marginTop = "8px";
  const okBtn = el("button", "btn primary small shrink", "✅ 确认入库");
  okBtn.onclick = async () => {
    try {
      await api(`/api/novels/${encodeURIComponent(S.novelId)}/character_cards`, { method: "PUT", body: { cards: preview.cards } });
      alertMsg("success", "角色卡已入库");
      refreshNovel(true);
    } catch (e) { alertMsg("error", e.message); }
  };
  const cancelBtn = el("button", "btn small shrink", "取消");
  cancelBtn.onclick = () => pv.remove();
  row.append(okBtn, cancelBtn);
  pv.append(row);
  box.append(pv);
}

function tabOutline(root) {
  if (!S.novel.characters) {
    root.insertAdjacentHTML("beforeend", `<div class="msg warn">⚠️ 建议先生成人物设定，再生成大纲。</div>`);
  }
  const ef = `<div class="row" style="margin-top:8px">
    <div><label class="field">总章节数（可填范围如 30-50，取上限）</label><input type="text" id="total-chapters" value="${esc(extra("outline_total_chapters", "50"))}"></div>
    <div><label class="field">每章目标字数</label><input type="number" id="words-per-chapter" min="500" max="10000" step="100" value="${esc(extra("outline_words_per_chapter", "2000"))}"></div>
  </div>`;
  root.append(genBlock({
    label: "描述大纲要求（故事主线、节奏、结局走向等）",
    placeholder: "例如：主角从底层崛起，经历三次重大转折……",
    promptValue: extra("outline_prompt"),
    btnText: S.novel.outline ? "🔄 重新生成大纲" : "🚀 生成小说大纲",
    step: "outline", extraFields: ef,
    collectParams: (w) => {
      const p = w.querySelector("textarea").value.trim();
      if (!p) { alertMsg("error", "请输入大纲要求"); return null; }
      const tcRaw = w.querySelector("#total-chapters").value.trim();
      const m = tcRaw.match(/(\d+)\s*[-~—]\s*(\d+)/);
      const total = m ? parseInt(m[2]) : parseInt(tcRaw) || 50;
      return { prompt: p, total_chapters: total, words_per_chapter: +w.querySelector("#words-per-chapter").value || 2000 };
    },
  }));
  if (S.novel.outline) {
    root.append(editableContent({
      title: "小说大纲", content: S.novel.outline,
      type: "outline", sectionTitle: "full_outline",
      originalKey: "outline_original", promptKey: "outline_prompt",
    }));
  }
  // TODO 3.1 / 3.2 / 3.3 / 4.1：大纲扩展操作区
  renderOutlineExtra(root);
}

/* TODO 3.1/3.3/4.1：扩展大纲、卷细纲、插入卷标题、局部改写大纲 */
function renderOutlineExtra(root) {
  const box = el("div", "box");
  box.innerHTML = `<h4>🧩 大纲扩展操作</h4>`;

  // TODO 3.1：扩展大纲（末尾追加，原文逐字保留）
  const extRow = el("div", "row");
  extRow.innerHTML = `<label class="field shrink" style="align-self:center">📈 扩展大纲</label>
    <div><label class="field">新增章数</label><input type="number" id="ext-count" min="1" max="200" value="20" style="width:90px"></div>`;
  const extBtn = el("button", "btn small shrink", "🚀 在末尾追加");
  extBtn.style.alignSelf = "flex-end";
  extBtn.onclick = () => {
    const n = +extRow.querySelector("#ext-count").value;
    if (!n || n <= 0) { alertMsg("error", "请输入有效的新增章数"); return; }
    if (!confirm("将在现有大纲末尾追加，原有内容逐字保留。继续？")) return;
    runGeneration("extend_outline", { additional_chapters: n }, { onDone: () => refreshNovel(true) });
  };
  extRow.append(extBtn);
  box.append(extRow);

  // TODO 3.2：卷逐章概要手动提前生成（章节生成遇到无概要的卷也会自动生成）
  const vp = extra("volume_plan", null);
  if (Array.isArray(vp) && vp.length) {
    const volWrap = el("div");
    volWrap.innerHTML = `<label class="field" style="margin-top:10px">📚 卷逐章概要</label>
      <div class="caption">每卷各章的一句话概要。章节生成遇到无概要的卷时会自动生成，此处可提前手动生成。</div>`;
    for (const v of vp) {
      const row = el("div", "row");
      const done = !!(v.chapters_done || extra(`volume_chapters_${v.index}`));
      row.innerHTML = `<span class="caption shrink" style="align-self:center">${esc(v.name || `第${v.index}卷`)}（第${v.start}-${v.end}章）· ${done ? "✅ 已生成概要" : "⏳ 待生成"}</span>`;
      if (!done) {
        const b = el("button", "btn small shrink", "生成该卷概要");
        b.onclick = () => runGeneration("volume_chapters", { volume_index: v.index }, { onDone: () => refreshNovel(true) });
        row.append(b);
      }
      volWrap.append(row);
    }
    box.append(volWrap);
  }

  // TODO 3.3：插入卷标题（读取当前大纲末尾追加，不写 editableContent 内部）
  const vtBtn = el("button", "btn small", "🏷️ 插入卷标题");
  vtBtn.style.marginTop = "10px";
  vtBtn.onclick = async () => {
    const name = prompt("输入卷名（如：风起云涌）：");
    if (!name || !name.trim()) return;
    const outline = S.novel.outline || "";
    let maxV = 0;
    for (const m of outline.matchAll(/### 第(\d+)卷/g)) maxV = Math.max(maxV, +m[1]);
    const text = outline + `\n\n### 第${maxV + 1}卷 ${name.trim()}\n`;
    await saveSection("outline", "full_outline", text);
    alertMsg("success", `已在末尾插入「### 第${maxV + 1}卷 ${name.trim()}」`);
    refreshNovel(true);
  };
  box.append(vtBtn);

  // TODO 4.1：局部改写指定章节范围的大纲条目，其余逐字保留
  const rwWrap = el("div");
  rwWrap.innerHTML = `<label class="field" style="margin-top:10px">✂️ 局部改写大纲</label>
    <div class="row">
      <div><label class="field">起始章</label><input type="number" id="rw-start" min="1" value="1"></div>
      <div><label class="field">结束章</label><input type="number" id="rw-end" min="1" value="1"></div>
    </div>
    <textarea rows="3" id="rw-instruction" placeholder="改写要求，如：在第5章附近安排新角色「叶青」登场…"></textarea>
    <div style="margin-top:6px"><button class="btn primary small" id="rw-btn">🚀 局部改写</button></div>`;
  rwWrap.querySelector("#rw-btn").onclick = () => {
    const start = +rwWrap.querySelector("#rw-start").value;
    const end = +rwWrap.querySelector("#rw-end").value;
    const instruction = rwWrap.querySelector("#rw-instruction").value.trim();
    if (!instruction || !start || end < start) { alertMsg("error", "请填写改写要求，且章节范围有效"); return; }
    if (!confirm(`将改写第 ${start}-${end} 章的大纲条目，其余内容逐字保留。继续？`)) return;
    runGeneration("rewrite_outline", { start, end, instruction }, { onDone: () => refreshNovel(true) });
  };
  box.append(rwWrap);
  root.append(box);
}

// ---- 章节 Tab（最复杂） ----
const chUI = { page: 0, pageSize: 10, deleteMode: false, toDelete: new Set(), activeKey: "", addMode: "generate" };
// 场景节拍未保存草稿（按章号暂存，重渲染后恢复）
const beatsDrafts = {};

function tabChapter(root) {
  if (!S.novel.outline) {
    root.insertAdjacentHTML("beforeend", `<div class="msg warn">⚠️ 建议先生成大纲，再生成章节。</div>`);
  }
  renderChapterManager(root);
  renderChapterAdder(root);
  renderMemoryPanel(root);
  renderChapterEditor(root);
  renderImpactScan(root);
}

/* 新增章节区：AI 生成 / 空白章节 / 导入章节 三种方式平级切换 */
function renderChapterAdder(root) {
  const box = el("div", "box");
  box.innerHTML = `<h4>➕ 新增章节</h4>`;
  const tabsRow = el("div", "row");
  const modes = [["generate", "✍️ AI 生成"], ["blank", "➕ 空白章节"], ["import", "📥 导入章节"]];
  for (const [m, label] of modes) {
    const b = el("button", "btn small shrink", label);
    if (chUI.addMode === m) b.style.borderColor = "var(--accent)";
    b.onclick = () => { chUI.addMode = m; renderTabContent(); };
    tabsRow.append(b);
  }
  box.append(tabsRow);
  const content = el("div");
  content.style.marginTop = "10px";
  if (chUI.addMode === "blank") renderBlankChapterArea(content);
  else if (chUI.addMode === "import") content.append(renderImportArea());
  else renderChapterGenerator(content, true);
  box.append(content);
  root.append(box);
}

function renderChapterManager(root) {
  const chapters = S.novel.chapters || {};
  const keys = Object.keys(chapters).sort((a, b) => +a - +b);
  const box = el("div", "box");
  box.innerHTML = `<h4>📚 章节管理（共 ${keys.length} 章）</h4>`;

  const opsRow = el("div", "row");
  const modeBtn = el("button", "btn small shrink", chUI.deleteMode ? "退出批量删除" : "🗑️ 批量删除");
  modeBtn.onclick = () => { chUI.deleteMode = !chUI.deleteMode; chUI.toDelete.clear(); renderTabContent(); };
  opsRow.append(modeBtn);
  if (chUI.deleteMode && chUI.toDelete.size) {
    const doDel = el("button", "btn primary small shrink", `确认删除选中的 ${chUI.toDelete.size} 章`);
    doDel.onclick = async () => {
      if (!confirm(`确定删除 ${chUI.toDelete.size} 个章节？不可恢复！`)) return;
      for (const k of chUI.toDelete) await delSection("chapter", `chapter_${k}`);
      chUI.deleteMode = false; chUI.toDelete.clear();
      refreshNovel(true);
    };
    opsRow.append(doDel);
  }
  const sizeSel = el("select");
  sizeSel.className = "shrink"; sizeSel.style.width = "90px";
  [10, 15, 20, 50].forEach(n => sizeSel.innerHTML += `<option ${n === chUI.pageSize ? "selected" : ""}>${n}</option>`);
  sizeSel.onchange = () => { chUI.pageSize = +sizeSel.value; chUI.page = 0; renderTabContent(); };
  opsRow.append(sizeSel);
  box.append(opsRow);
  if (!keys.length) { root.append(box); return; }

  const pages = Math.max(1, Math.ceil(keys.length / chUI.pageSize));
  chUI.page = Math.min(chUI.page, pages - 1);
  const pageKeys = keys.slice(chUI.page * chUI.pageSize, (chUI.page + 1) * chUI.pageSize);
  const volPlan = extra("volume_plan", []) || [];
  const grid = el("div", "chapter-grid");
  for (const k of pageKeys) {
    const c = chapters[k];
    const words = (c.content || "").length;
    const colorCls = words >= 1500 ? "c-green" : words >= 500 ? "c-orange" : "c-red";
    const card = el("div", `chapter-card ${colorCls} ${k === chUI.activeKey ? "active" : ""} ${chUI.deleteMode ? "selecting" : ""} ${chUI.toDelete.has(k) ? "selected" : ""}`);
    // TODO 1.3：空白章（无正文）显示"待写"徽标，提示手写/填充意图
    const vol = volPlan.find(v => +k >= v.start && +k <= v.end);
    const volLabel = vol ? `第${vol.index}卷 · ` : "";
    card.innerHTML = `<div class="t">第${k}章 ${esc(c.title)}</div><div class="w">${volLabel}${words ? words + " 字" : "待写"}</div>`;
    card.onclick = () => {
      if (chUI.deleteMode) {
        chUI.toDelete.has(k) ? chUI.toDelete.delete(k) : chUI.toDelete.add(k);
        renderTabContent();
      } else {
        chUI.activeKey = k; renderTabContent();
        setTimeout(() => $("#chapter-editor")?.scrollIntoView({ behavior: "smooth" }), 50);
      }
    };
    grid.append(card);
  }
  box.append(grid);
  if (pages > 1) {
    const pg = el("div", "row");
    const prev = el("button", "btn small shrink", "◀ 上一页");
    prev.disabled = chUI.page === 0;
    prev.onclick = () => { chUI.page--; renderTabContent(); };
    const next = el("button", "btn small shrink", "下一页 ▶");
    next.disabled = chUI.page >= pages - 1;
    next.onclick = () => { chUI.page++; renderTabContent(); };
    pg.append(prev, el("span", "caption", `第 ${chUI.page + 1}/${pages} 页`), next);
    box.append(pg);
  }
  root.append(box);
}

/* TODO 1.2：导入外部章节区（粘贴文本或选择 .txt/.md 文件）——内嵌于"新增章节"区 */
function renderImportArea() {
  const chapters = S.novel.chapters || {};
  const wrap = el("div");
  wrap.innerHTML = `<div class="row">
      <div><label class="field">章节号</label><input type="number" id="imp-num" min="1" value="1"></div>
      <div><label class="field">标题（留空自动取内容首行或"第N章"）</label><input type="text" id="imp-title" placeholder="章节标题"></div>
    </div>
    <textarea rows="8" id="imp-content" placeholder="粘贴章节正文，或点击下方选择 .txt/.md 文件"></textarea>
    <div style="margin-top:6px"><input type="file" id="imp-file" accept=".txt,.md"></div>
    <div style="margin-top:8px"><button class="btn primary small" id="imp-btn">📥 导入</button></div>`;
  const numI = wrap.querySelector("#imp-num");
  const titleI = wrap.querySelector("#imp-title");
  const contentTa = wrap.querySelector("#imp-content");
  // 默认章节号取当前最大章号+1
  const keys = Object.keys(chapters).sort((a, b) => +a - +b);
  numI.value = keys.length ? +keys[keys.length - 1] + 1 : 1;

  wrap.querySelector("#imp-file").onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      contentTa.value = String(reader.result || "");
      // 自动填标题为文件名（去扩展名）
      if (!titleI.value.trim()) titleI.value = f.name.replace(/\.[^.]+$/, "");
    };
    reader.readAsText(f);
    e.target.value = "";
  };

  wrap.querySelector("#imp-btn").onclick = async () => {
    const chapter_num = +numI.value;
    const content = contentTa.value.trim();
    let title = titleI.value.trim();
    if (!chapter_num || !content) { alertMsg("error", "请提供有效的章节号与正文内容"); return; }
    if (!title) {
      // 留空自动取内容首行
      const firstLine = content.split("\n").map(s => s.trim()).filter(Boolean)[0] || "";
      title = firstLine.length && firstLine.length <= 30 ? firstLine : `第${chapter_num}章`;
    }
    const existing = chapters[String(chapter_num)];
    if (existing && (existing.content || "").trim()
        && !confirm(`第${chapter_num}章已有正文（${(existing.content || "").length} 字），导入将覆盖。确定？`)) return;
    try {
      const r = await api(`/api/novels/${encodeURIComponent(S.novelId)}/chapters/import`,
        { method: "POST", body: { chapter_num, title, content } });
      alertMsg("success", `✅ 已导入第${r.chapter_num}章「${r.title}」（${r.length} 字）`);
      if (r.ledger_stale) alertMsg("warn", "⚠️ 台账已标记待重建（可在下方长篇记忆面板重建）", 10000);
      refreshNovel(true);
    } catch (e) { alertMsg("error", e.message); }
  };
  return wrap;
}

/* TODO 1.3：新建空白章节（手写入口；标题可从大纲选择）——内嵌于"新增章节"区 */
function renderBlankChapterArea(container) {
  const chapters = S.novel.chapters || {};
  const keys = Object.keys(chapters).sort((a, b) => +a - +b);
  const wrap = el("div");
  wrap.innerHTML = `<div class="caption">创建无正文的章节占位，供手写；批量生成默认跳过空白章（尊重手写意图）。</div>
    <div class="row">
      <div><label class="field">章节号</label><input type="number" id="blank-num" min="1" value="${keys.length ? +keys[keys.length - 1] + 1 : 1}"></div>
      <div><label class="field">标题</label><input type="text" id="blank-title" placeholder="留空自动为「第N章」"></div>
      <button class="btn small shrink" id="blank-from-outline" style="align-self:flex-end">📋 从大纲选</button>
    </div>
    <div id="blank-title-sel"></div>
    <div style="margin-top:8px"><button class="btn primary small" id="blank-create">➕ 创建</button></div>`;
  wrap.querySelector("#blank-from-outline").onclick = async () => {
    try {
      const { titles } = await api(`/api/novels/${encodeURIComponent(S.novelId)}/outline_titles`);
      const sel = el("select");
      sel.innerHTML = `<option value="">-- 选择大纲标题 --</option>` +
        Object.keys(titles).sort((a, b) => +a - +b)
          .map(k => `<option value="${k}">第${k}章 ${esc(titles[k])}</option>`).join("");
      sel.onchange = () => {
        if (!sel.value) return;
        wrap.querySelector("#blank-num").value = sel.value;
        wrap.querySelector("#blank-title").value = titles[sel.value];
      };
      const holder = wrap.querySelector("#blank-title-sel");
      holder.innerHTML = ""; holder.append(sel);
    } catch (e) { alertMsg("error", e.message); }
  };
  wrap.querySelector("#blank-create").onclick = async () => {
    const chapter_num = +wrap.querySelector("#blank-num").value;
    const title = wrap.querySelector("#blank-title").value.trim();
    if (!chapter_num) { alertMsg("error", "请提供有效的章节号"); return; }
    try {
      const r = await api(`/api/novels/${encodeURIComponent(S.novelId)}/chapters/blank`,
        { method: "POST", body: { chapter_num, title } });
      alertMsg("success", `✅ 已创建空白章节：第${chapter_num}章「${r.title || title || "第" + chapter_num + "章"}」`);
      chUI.activeKey = String(chapter_num);   // 直接渲染编辑器供手写
      refreshNovel(true);
    } catch (e) { alertMsg("error", e.message); }
  };
  container.append(wrap);
}

function renderChapterGenerator(root, bare = false) {
  const box = bare ? el("div") : el("div", "box");
  if (!bare) box.innerHTML = `<h4>✍️ 生成章节</h4>`;
  const chapters = S.novel.chapters || {};
  const keys = Object.keys(chapters).sort((a, b) => +a - +b);

  const row1 = el("div", "row");
  // 章节号：可选已有章节或新章节
  const selWrap = el("div"); selWrap.innerHTML = `<label class="field">选择已有章节（或留空新建）</label>`;
  const sel = el("select");
  sel.innerHTML = `<option value="">-- 新建章节 --</option>` +
    keys.map(k => `<option value="${k}">第${k}章 ${esc(chapters[k].title)}</option>`).join("");
  selWrap.append(sel);
  const numWrap = el("div");
  numWrap.innerHTML = `<label class="field">章节号</label><input type="number" id="ch-num" min="1" value="${keys.length ? +keys[keys.length - 1] + 1 : 1}">`;
  const titleWrap = el("div");
  titleWrap.innerHTML = `<label class="field">章节标题</label><input type="text" id="ch-title" placeholder="留空则由 AI 拟定（优先取大纲）">`;
  const wordsWrap = el("div");
  wordsWrap.innerHTML = `<label class="field">目标字数</label><input type="number" id="ch-words" min="500" max="10000" step="100" placeholder="留空按大纲字数">`;
  row1.append(selWrap, numWrap, titleWrap, wordsWrap);
  box.append(row1);

  // 已有章节标题三选一
  const titleSrcWrap = el("div", "row hidden");
  titleSrcWrap.innerHTML = `<label class="field shrink" style="align-self:center">标题来源：</label>`;
  let titleSrc = "keep";
  [["keep", "保持原标题"], ["outline", "用大纲标题"], ["manual", "手动输入"]].forEach(([v, t]) => {
    const b = el("button", "btn small shrink", t);
    b.onclick = async () => {
      titleSrc = v;
      titleSrcWrap.querySelectorAll("button").forEach(x => x.style.borderColor = "");
      b.style.borderColor = "var(--accent)";
      if (v === "outline") {
        try {
          const { titles } = await api(`/api/novels/${encodeURIComponent(S.novelId)}/outline_titles`);
          const t2 = titles[sel.value];
          if (t2) box.querySelector("#ch-title").value = t2;
          else alertMsg("info", "大纲中未找到该章标题");
        } catch (e) { alertMsg("error", e.message); }
      }
    };
    titleSrcWrap.append(b);
  });
  box.append(titleSrcWrap);

  sel.onchange = () => {
    if (sel.value) {
      const c = chapters[sel.value];
      box.querySelector("#ch-num").value = sel.value;
      box.querySelector("#ch-title").value = c.title;
      titleSrcWrap.classList.remove("hidden");
    } else {
      titleSrcWrap.classList.add("hidden");
    }
  };

  // 场景节拍
  const beatsBox = el("div");
  beatsBox.innerHTML = `<label class="field">场景节拍（可选，提供时按场景逐段生成，质量更稳）</label>
    <div class="caption">未保存时自动预填大纲逐章概要；保存后独立存储，大纲更新时会提示可重新预填。</div>`;
  const beatsTa = el("textarea"); beatsTa.rows = 5;
  beatsTa.placeholder = "可留空；点击「生成场景节拍」让 AI 规划 3-6 个场景";
  // 大纲概要更新提示条（默认隐藏）
  const beatsNotice = el("div", "msg warn hidden");
  beatsNotice.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:4px";
  const beatsNoticeTxt = el("span", "", "📄 大纲该章概要已更新，与已保存节拍不一致");
  beatsNoticeTxt.style.flex = "1";
  const beatsRefill = el("button", "btn small shrink", "↻ 重新预填");
  const beatsDismiss = el("button", "btn small shrink", "忽略");
  beatsNotice.append(beatsNoticeTxt, beatsRefill, beatsDismiss);
  const beatsOps = el("div", "row");
  const beatsBtn = el("button", "btn small shrink", "🧩 生成场景节拍");
  beatsBtn.onclick = () => {
    const num = +box.querySelector("#ch-num").value;
    const title = box.querySelector("#ch-title").value.trim();
    runGeneration("chapter_beats", { chapter_num: num, chapter_title: title, target_words: +box.querySelector("#ch-words").value || 0 }, {
      onDone: (r) => { if (r.beats) { beatsTa.value = r.beats; beatsDrafts[num] = r.beats; } },
    });
  };
  const beatsSave = el("button", "btn small shrink", "💾 保存节拍");
  const getOutlineTitle = async (num) => {
    if (!outlineTitlesCache) {
      const { titles } = await api(`/api/novels/${encodeURIComponent(S.novelId)}/outline_titles`);
      outlineTitlesCache = titles || {};
    }
    return outlineTitlesCache[String(num)] || "";
  };
  beatsSave.onclick = async () => {
    const num = +box.querySelector("#ch-num").value;
    await saveExtra(`chapter_beats_${num}`, beatsTa.value);
    // 记录保存时的大纲概要基准，用于日后检测大纲是否更新
    let src = "";
    try { src = await getOutlineTitle(num); } catch (e) { /* ignore */ }
    await saveExtra(`chapter_beats_src_${num}`, src);
    delete beatsDrafts[num];
    beatsNotice.classList.add("hidden");
    alertMsg("success", "场景节拍已保存");
  };
  beatsOps.append(beatsBtn, beatsSave);
  beatsBox.append(beatsTa, beatsNotice, beatsOps);
  box.append(beatsBox);
  // 加载节拍：已保存 > 未保存草稿 > 大纲逐章概要预填
  let outlineTitlesCache = null;
  const loadBeats = async () => {
    const num = +box.querySelector("#ch-num").value;
    beatsNotice.classList.add("hidden");
    const saved = extra(`chapter_beats_${num}`, "");
    if (saved) {
      beatsTa.value = saved;
      // 弱双向：已保存节拍的大纲基准与当前概要不一致时提示可重新预填
      const src = extra(`chapter_beats_src_${num}`, null);
      if (src !== null) {
        try {
          const cur = await getOutlineTitle(num);
          if (cur && cur !== src) beatsNotice.classList.remove("hidden");
        } catch (e) { /* ignore */ }
      }
      return;
    }
    if (beatsDrafts[num] !== undefined) { beatsTa.value = beatsDrafts[num]; return; }
    try { beatsTa.value = await getOutlineTitle(num); } catch (e) { /* ignore */ }
  };
  beatsRefill.onclick = async () => {
    const num = +box.querySelector("#ch-num").value;
    try { beatsTa.value = await getOutlineTitle(num); } catch (e) { /* ignore */ }
    beatsNotice.classList.add("hidden");
  };
  beatsDismiss.onclick = () => beatsNotice.classList.add("hidden");
  beatsTa.addEventListener("input", () => {
    const num = +box.querySelector("#ch-num").value;
    if (!extra(`chapter_beats_${num}`, "")) beatsDrafts[num] = beatsTa.value;  // 只暂存未保存的章
  });
  box.querySelector("#ch-num").addEventListener("change", loadBeats);
  sel.addEventListener("change", loadBeats);
  loadBeats();

  // 黄金开篇
  const goldenRow = el("div", "row");
  goldenRow.innerHTML = `<label class="caption shrink" style="align-self:center"><input type="checkbox" id="use-golden"> 🏆 黄金开篇模式（第1-3章推荐：生成两个版本+AI评审择优，约4次API调用）</label>`;
  box.append(goldenRow);

  const genRow = el("div", "row");
  const genBtn = el("button", "btn primary", "🚀 生成本章");
  genBtn.onclick = () => {
    const num = +box.querySelector("#ch-num").value;
    const title = box.querySelector("#ch-title").value.trim();
    const params = {
      chapter_num: num, chapter_title: title,
      target_words: +box.querySelector("#ch-words").value || 0,
      beats: beatsTa.value.trim(),
    };
    runGeneration(box.querySelector("#use-golden").checked ? "golden_chapter" : "chapter", params, {
      onDone: (r) => { if (r.generated_title) box.querySelector("#ch-title").value = r.generated_title; },
    });
  };
  genRow.append(genBtn);
  box.append(genRow);

  // 批量生成
  const batch = el("details");
  batch.innerHTML = `<summary class="caption">📦 批量连续生成</summary>`;
  const bInner = el("div");
  bInner.innerHTML = `<div class="row">
    <div><label class="field">起始章节</label><input type="number" id="batch-start" min="1" value="1"></div>
    <div><label class="field">生成数量（≤20）</label><input type="number" id="batch-count" min="1" max="20" value="3"></div>
    <div><label class="field">目标字数</label><input type="number" id="batch-words" min="500" max="10000" step="100" placeholder="留空按大纲字数"></div>
  </div>
  <label class="caption"><input type="checkbox" id="batch-beats"> 每章自动生成场景节拍（更稳但多一次API调用）</label>
  <label class="caption"><input type="checkbox" id="batch-fill-blank"> 填充空白章（默认跳过空白章，尊重手写意图）</label>
  <div class="caption">已有章节会自动跳过；每章约2-6次API调用</div>
  <button class="btn primary" style="margin-top:8px">🚀 开始批量生成</button>`;
  bInner.querySelector("button").onclick = () => {
    runGeneration("batch_chapters", {
      start: +bInner.querySelector("#batch-start").value,
      count: Math.min(20, +bInner.querySelector("#batch-count").value),
      target_words: +bInner.querySelector("#batch-words").value || 0,
      auto_beats: bInner.querySelector("#batch-beats").checked,
      fill_blank: bInner.querySelector("#batch-fill-blank").checked,  // TODO 1.3：勾选才填充空白章
    }, { onDone: (r) => { if (r.summary) alertMsg("info", `批量生成完成：${r.summary}`, 15000); } });
  };
  batch.append(bInner);
  box.append(batch);
  root.append(box);
}

function renderMemoryPanel(root) {
  const ledger = extra("state_ledger");
  const summary = extra("rolling_summary");
  const ms = S.memoryStatus;
  // TODO 1.1：以 memory/status 为准判断是否渲染；status 不可用则回退旧逻辑
  if (ms && !ms.has_ledger && !ms.has_summary) return;
  if (!ms && !ledger && !summary) return;
  const det = el("details", "box");
  det.innerHTML = `<summary><b>🧠 长篇记忆（伏笔台账 / 滚动摘要）</b></summary>`;
  // TODO 1.1：有章节被重生成/编辑/导入时的 stale 提示
  if (ms && ms.ledger_stale) {
    det.innerHTML += `<div class="msg warn">⚠️ 有章节被重生成/编辑/导入，第 ${ms.ledger_stale_from ?? 1} 章起的台账与摘要已标记待重建。</div>`;
  }
  if (ledger) {
    det.innerHTML += `<h4>伏笔与角色状态台账</h4><div class="content-view">${esc(typeof ledger === "string" ? ledger : JSON.stringify(ledger, null, 2))}</div>`;
  }
  if (summary) {
    det.innerHTML += `<h4>滚动摘要</h4><div class="content-view">${esc(summary)}</div>`;
  }

  // TODO 1.1：手动重建台账与摘要（合并态零成本 / 逐章重算 delta 有 token 成本）
  const rb = el("div");
  rb.innerHTML = `<h4>🔧 重建台账与摘要</h4>
    <div class="row">
      <div><label class="field">从第几章起</label><input type="number" id="rb-from" min="1" value="${(ms && ms.ledger_stale_from) || 1}" style="width:90px"></div>
    </div>`;
  const rbRow = el("div", "row");
  const rbMerge = el("button", "btn small shrink", "🔧 仅重建合并态（零成本）");
  rbMerge.onclick = () => {
    const from_chapter = +rb.querySelector("#rb-from").value || 1;
    runGeneration("memory_rebuild", { from_chapter, regen: false }, { onDone: () => refreshNovel(true) });
  };
  const rbRegen = el("button", "btn small shrink", "🤖 逐章重算 delta（每章约2次API调用，有 token 成本）");
  rbRegen.onclick = () => {
    const from_chapter = +rb.querySelector("#rb-from").value || 1;
    if (!confirm(`将从第 ${from_chapter} 章起逐章重算 delta，每章约 2 次 API 调用，会产生 token 成本。确定？`)) return;
    runGeneration("memory_rebuild", { from_chapter, regen: true }, { onDone: () => refreshNovel(true) });
  };
  rbRow.append(rbMerge, rbRegen);
  rb.append(rbRow);
  det.append(rb);

  const clearBtn = el("button", "btn small", "🗑️ 清空长篇记忆");
  clearBtn.style.marginTop = "6px";
  clearBtn.onclick = async () => {
    if (!confirm("清空伏笔台账和滚动摘要？后续章节将失去长期一致性辅助。")) return;
    await delExtra("state_ledger"); await delExtra("rolling_summary");
    refreshNovel(true);
  };
  det.append(clearBtn);
  root.append(det);
}

function renderChapterEditor(root) {
  const chapters = S.novel.chapters || {};
  const key = chUI.activeKey;
  if (!key || !chapters[key]) return;
  const c = chapters[key];
  const box = el("div", "box");
  box.id = "chapter-editor";
  box.innerHTML = `<h4>📖 第${key}章 ${esc(c.title)}（${(c.content || "").length} 字）</h4>`;
  const ta = el("textarea"); ta.rows = 20; ta.value = c.content;
  const status = el("div", "caption", "编辑后自动保存");
  let timer = null;
  ta.oninput = () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      await saveSection("chapter", `chapter_${key}`, `第${key}章 ${c.title}\n${ta.value}`);
      status.textContent = "✅ 已自动保存 " + new Date().toLocaleTimeString();
    }, 800);
  };
  const ops = el("div", "row");
  const reviewBtn = el("button", "btn small shrink", "📝 AI 评审");
  reviewBtn.onclick = () => {
    runGeneration("chapter_review", { chapter_num: +key, chapter_title: c.title, content: ta.value }, {
      onDone: () => refreshNovel(true),
    });
  };
  const reviseBtn = el("button", "btn small shrink", "🔧 按评审改写");
  // TODO 4.2 第3步：融入新设定重写本章（人工确认后整章重写）
  const rewriteBtn = el("button", "btn small shrink", "🔁 融入新设定重写");
  rewriteBtn.onclick = () => {
    const instruction = prompt("输入要融入的新设定/变更说明（如：新角色「叶青」已于第3章登场）：");
    if (!instruction || !instruction.trim()) return;
    if (!confirm(`将按新设定重写第${key}章并覆盖原内容，确定？`)) return;
    runGeneration("chapter", {
      chapter_num: +key, chapter_title: c.title,
      target_words: 2000, extra_instruction: instruction.trim(),
    }, { onDone: () => refreshNovel(true) });
  };
  const delBtn = el("button", "btn small shrink", "🗑️ 删除本章");
  delBtn.onclick = async () => {
    if (!confirm(`确定删除第${key}章？不可恢复！`)) return;
    await delSection("chapter", `chapter_${key}`);
    chUI.activeKey = "";
    refreshNovel(true);
  };
  ops.append(reviewBtn, reviseBtn, rewriteBtn, delBtn);
  box.append(ta, status, ops);

  const review = extra(`chapter_review_${key}`);
  if (review) {
    const rBox = el("div", "box");
    rBox.innerHTML = `<h4>📝 评审结果</h4><div class="content-view">${esc(review)}</div>`;
    box.append(rBox);
    reviseBtn.onclick = () => {
      if (!confirm("将按评审意见改写本章并覆盖原内容，确定？")) return;
      runGeneration("chapter_revise", { chapter_num: +key, chapter_title: c.title, content: ta.value, review });
    };
  } else {
    reviseBtn.disabled = true;
    reviseBtn.title = "请先评审";
  }

  // 黄金开篇落选版本对比
  const golden = extra(`chapter_golden_${key}`);
  if (golden && golden.alt_content) {
    const det = el("details", "box");
    det.innerHTML = `<summary><b>🏆 黄金开篇：落选版本对比</b>（得分 ${esc(String(golden.scores?.[0]))} / ${esc(String(golden.scores?.[1]))}）</summary>
      <h4>落选版本正文</h4><div class="content-view">${esc(golden.alt_content)}</div>
      <h4>落选版本评审</h4><div class="content-view">${esc(golden.alt_review || "")}</div>`;
    const swap = el("button", "btn small", "🔄 换用落选版本");
    swap.onclick = async () => {
      if (!confirm("用落选版本替换当前章节内容？")) return;
      await saveSection("chapter", `chapter_${key}`, `第${key}章 ${c.title}\n${golden.alt_content}`);
      refreshNovel(true);
    };
    det.append(swap);
    box.append(det);
  }
  root.append(box);
}

/* TODO 4.2：新增设定回溯扫描——只提示+人工确认，不会自动批量重写 */
function renderImpactScan(root) {
  const chapters = S.novel.chapters || {};
  if (!Object.keys(chapters).length) return;
  const box = el("div", "box");
  box.innerHTML = `<h4>🔎 新增设定回溯扫描</h4>
    <div class="caption">新增角色/核心设定变更后，扫描已生成章节中受影响的章节。原则：只提示+人工确认，不会自动批量重写。</div>
    <div class="row">
      <div><label class="field">关键词</label><input type="text" id="scan-keywords" placeholder="多个关键词用逗号分隔，如新增主角名字"></div>
      <button class="btn primary small shrink" id="scan-btn" style="align-self:flex-end">🔍 扫描受影响章节</button>
    </div>
    <div id="scan-result"></div>`;
  const resultBox = box.querySelector("#scan-result");
  box.querySelector("#scan-btn").onclick = async () => {
    const raw = box.querySelector("#scan-keywords").value.trim();
    const keywords = raw.split(/[,，、\n]/).map(s => s.trim()).filter(Boolean);
    if (!keywords.length) { alertMsg("error", "请输入至少一个关键词"); return; }
    try {
      const { impacted } = await api(`/api/novels/${encodeURIComponent(S.novelId)}/impact_scan`,
        { method: "POST", body: { keywords } });
      resultBox.innerHTML = impacted.length
        ? `<div class="msg info">共 ${impacted.length} 章提及关键词：</div>`
        : `<div class="msg success">✅ 没有章节提及这些关键词，无需回溯。</div>`;
      for (const it of impacted) {
        const item = el("div", "box");
        item.style.marginTop = "6px";
        const hits = Object.entries(it.hits).map(([k, n]) => `${esc(k)}×${n}`).join("、");
        item.innerHTML = `<div class="row"><b class="shrink" style="align-self:center">第${it.chapter}章 ${esc(it.title)}</b>
          <span class="caption shrink" style="align-self:center">（命中：${hits}）</span></div>`;
        const pvBtn = el("button", "btn small shrink", "🤖 AI 改写建议预览");
        pvBtn.onclick = () => {
          const holder = item.querySelector("#rw-pv-" + it.chapter);
          holder.innerHTML = `<div class="caption">生成预览中…（完成后在此展开）</div>`;
          runGeneration("rewrite_preview", { chapter_num: it.chapter, instruction: keywords.join("，") }, {
            onDone: (r) => {
              holder.innerHTML = `<h4>改写建议预览</h4><div class="content-view preview-scroll">${esc(r.preview || "")}</div>`;
              const confirmBtn = el("button", "btn primary small", "✅ 确认重写本章");
              confirmBtn.style.marginTop = "6px";
              confirmBtn.onclick = () => {
                if (!confirm(`将按预览建议重写第${it.chapter}章并覆盖原内容，确定？`)) return;
                runGeneration("chapter", {
                  chapter_num: it.chapter, chapter_title: it.title,
                  extra_instruction: r.preview || keywords.join("，"),
                }, { onDone: () => refreshNovel(true) });
              };
              holder.append(confirmBtn);
            },
          });
        };
        item.querySelector(".row").append(pvBtn);
        const pvHolder = el("div"); pvHolder.id = "rw-pv-" + it.chapter;
        item.append(pvHolder);
        resultBox.append(item);
      }
    } catch (e) { alertMsg("error", e.message); }
  };
  root.append(box);
}

// ---- 续写 Tab ----
function tabContinue(root) {
  const chapters = S.novel.chapters || {};
  const keys = Object.keys(chapters).sort((a, b) => +a - +b);
  const box = el("div", "box");
  box.innerHTML = `<h4>✍️ 续写</h4>`;
  const srcRow = el("div", "row");
  const selWrap = el("div");
  selWrap.innerHTML = `<label class="field">从已有章节续写（或留空自由输入）</label>`;
  const sel = el("select");
  sel.innerHTML = `<option value="">-- 自由输入 --</option>` +
    keys.map(k => `<option value="${k}">第${k}章 ${esc(chapters[k].title)}</option>`).join("");
  selWrap.append(sel);
  srcRow.append(selWrap);
  box.append(srcRow);
  const ta = el("textarea"); ta.rows = 10; ta.placeholder = "输入或粘贴需要续写的文本";
  sel.onchange = () => { if (sel.value) ta.value = chapters[sel.value].content || ""; };
  box.append(ta);
  const row2 = el("div", "row");
  row2.innerHTML = `
    <div><label class="field">续写要求</label><input type="text" id="cont-prompt" value="继续往下写"></div>
    <div><label class="field">目标长度：约 <span id="cont-len-label">1500</span> 字</label>
      <input type="range" id="cont-len" min="500" max="3000" step="100" value="1500" style="width:100%"></div>`;
  box.append(row2);
  row2.querySelector("#cont-len").oninput = (e) => row2.querySelector("#cont-len-label").textContent = e.target.value;
  const btn = el("button", "btn primary", "🚀 开始续写");
  box.append(btn);
  const resultBox = el("div");
  box.append(resultBox);
  btn.onclick = () => {
    const text = ta.value.trim();
    if (!text) { alertMsg("error", "请输入需要续写的内容"); return; }
    runGeneration("continue", {
      continue_text: text,
      continue_prompt: row2.querySelector("#cont-prompt").value.trim() || "继续往下写",
      continue_length: +row2.querySelector("#cont-len").value,
    }, {
      onDone: (r) => {
        resultBox.innerHTML = `<div class="divider"></div><h4>续写结果（${(r.result || "").length} 字）</h4>
          <div class="content-view">${esc(r.result)}</div>`;
        if (sel.value) {
          const appendBtn = el("button", "btn primary small", `📎 追加到第${sel.value}章末尾`);
          appendBtn.onclick = async () => {
            const k = sel.value;
            const merged = (chapters[k].content || "") + "\n\n" + r.result;
            await saveSection("chapter", `chapter_${k}`, `第${k}章 ${chapters[k].title}\n${merged}`);
            alertMsg("success", "已追加到章节末尾");
            refreshNovel(true);
          };
          resultBox.append(appendBtn);
        }
      },
    });
  };
  root.append(box);
}

// ---- 润色 Tab ----
function tabPolish(root) {
  const chapters = S.novel.chapters || {};
  const keys = Object.keys(chapters).sort((a, b) => +a - +b);
  const box = el("div", "box");
  box.innerHTML = `<h4>🎨 风格润色</h4>`;
  const selWrap = el("div");
  selWrap.innerHTML = `<label class="field">润色来源（或自由输入）</label>`;
  const sel = el("select");
  sel.innerHTML = `<option value="">-- 自由输入 --</option>` +
    keys.map(k => `<option value="${k}">第${k}章 ${esc(chapters[k].title)}</option>`).join("");
  selWrap.append(sel);
  box.append(selWrap);
  const ta = el("textarea"); ta.rows = 8; ta.placeholder = "输入要润色的文本";
  sel.onchange = () => { if (sel.value) ta.value = chapters[sel.value].content || ""; };
  box.append(ta);

  const styleRow = el("div", "row");
  styleRow.innerHTML = `
    <div><label class="field">风格类型</label><select id="style-type">
      <option>作品</option><option>描述</option><option>作家</option></select></div>
    <div><label class="field">风格参考</label><input type="text" id="style-ref" placeholder="作品名 / 作家名 / 风格描述"></div>`;
  box.append(styleRow);

  const btn = el("button", "btn primary", "🚀 开始润色");
  box.append(btn);
  const resultBox = el("div");
  box.append(resultBox);
  btn.onclick = () => {
    const text = ta.value.trim();
    const ref = styleRow.querySelector("#style-ref").value.trim();
    if (!text || !ref) { alertMsg("error", "请输入要润色的文本和风格参考"); return; }
    runGeneration("polish", { polish_text: text, style_reference: ref, style_type: styleRow.querySelector("#style-type").value }, {
      onDone: (r) => {
        resultBox.innerHTML = `<div class="divider"></div><h4>润色对比（风格：${esc(r.style_label || "")}）</h4>
          <div class="two-col">
            <div><h4>原文</h4><div class="content-view">${esc(r.original)}</div></div>
            <div><h4>润色后</h4><div class="content-view">${esc(r.result)}</div></div>
          </div>`;
        if (sel.value) {
          const rep = el("button", "btn primary small", `🔄 替换第${sel.value}章内容`);
          rep.onclick = async () => {
            if (!confirm("用润色结果覆盖原章节内容？")) return;
            const k = sel.value;
            await saveSection("chapter", `chapter_${k}`, `第${k}章 ${chapters[k].title}\n${r.result}`);
            alertMsg("success", "已替换章节内容");
            refreshNovel(true);
          };
          resultBox.append(rep);
        }
      },
    });
  };
  root.append(box);

  // 文风指纹
  const fpBox = el("div", "box");
  fpBox.innerHTML = `<h4>🧬 文风指纹（提取后自动注入后续生成）</h4>`;
  const fp = extra("style_fingerprint");
  if (fp) {
    fpBox.innerHTML += `<div class="content-view">${esc(fp)}</div>`;
    const clear = el("button", "btn small", "🗑️ 清除文风指纹");
    clear.onclick = async () => { await delExtra("style_fingerprint"); refreshNovel(true); };
    fpBox.append(clear);
  } else {
    fpBox.innerHTML += `<textarea rows="4" id="fp-sample" placeholder="粘贴文风样例文本（可选）"></textarea>
      <input type="text" id="fp-desc" placeholder="或用文字描述目标文风（可选）" style="margin-top:6px">
      <button class="btn primary" style="margin-top:8px">🧬 提取文风指纹</button>`;
    fpBox.querySelector("button").onclick = () => {
      runGeneration("style_fingerprint", {
        sample: fpBox.querySelector("#fp-sample").value,
        description: fpBox.querySelector("#fp-desc").value,
      });
    };
  }
  root.append(fpBox);

  // 自定义套话黑名单
  const blBox = el("div", "box");
  blBox.innerHTML = `<h4>🚫 自定义套话黑名单</h4>
    <div class="caption">每行一个词，生成时会被规避，生成后会被检测</div>`;
  const blTa = el("textarea"); blTa.rows = 4;
  const bl = extra("custom_cliche_blacklist", []);
  blTa.value = Array.isArray(bl) ? bl.join("\n") : String(bl || "");
  const blSave = el("button", "btn small", "💾 保存黑名单");
  blSave.style.marginTop = "6px";
  blSave.onclick = async () => {
    const list = blTa.value.split("\n").map(s => s.trim()).filter(Boolean);
    await saveExtra("custom_cliche_blacklist", list);
    alertMsg("success", "黑名单已保存");
  };
  blBox.append(blTa, blSave);
  root.append(blBox);

  // 去AI腔
  const hmBox = el("div", "box");
  hmBox.innerHTML = `<h4>🧹 去 AI 腔</h4>`;
  const hmTa = el("textarea"); hmTa.rows = 6; hmTa.placeholder = "输入要去除 AI 腔的文本";
  const hmBtn = el("button", "btn primary", "🧹 开始改写");
  hmBtn.style.marginTop = "8px";
  const hmResult = el("div");
  hmBtn.onclick = () => {
    if (!hmTa.value.trim()) { alertMsg("error", "请输入文本"); return; }
    runGeneration("humanize", { humanize_text: hmTa.value }, {
      onDone: (r) => {
        hmResult.innerHTML = `<h4>改写结果</h4><div class="content-view">${esc(r.result)}</div>`;
      },
    });
  };
  hmBox.append(hmTa, hmBtn, hmResult);
  root.append(hmBox);
}

// ---- 一致性 Tab ----
function tabConsistency(root) {
  const done = [S.novel.world_setting, S.novel.characters, S.novel.outline].filter(Boolean).length
    + (Object.keys(S.novel.chapters || {}).length ? 1 : 0);
  if (done < 2) {
    root.innerHTML = `<div class="msg warn">⚠️ 至少需要完成两个步骤（世界观/人物/大纲/章节）才能进行一致性检查。</div>`;
    return;
  }
  const box = el("div", "box");
  box.innerHTML = `<h4>🔍 AI 一致性检查</h4>
    <div class="caption">检查设定之间的矛盾、章节正文与设定/大纲的冲突（章节分批送检）</div>`;
  const btn = el("button", "btn primary", "🚀 开始检查");
  btn.style.marginTop = "8px";
  btn.onclick = () => runGeneration("consistency", {});
  box.append(btn);
  const result = extra("consistency_result");
  if (result) {
    box.innerHTML += `<div class="divider"></div><h4>上次检查结果</h4><div class="content-view">${esc(result)}</div>`;
    const clear = el("button", "btn small", "🗑️ 清除结果");
    clear.onclick = async () => { await delExtra("consistency_result"); refreshNovel(true); };
    box.append(clear);
  }
  root.append(box);
}

// ---- 查找替换 Tab ----
function tabFindReplace(root) {
  const hasContent = S.novel.world_setting || S.novel.characters || S.novel.outline
    || Object.keys(S.novel.chapters || {}).length;
  if (!hasContent) {
    root.innerHTML = `<div class="msg warn">⚠️ 当前小说没有任何内容，无法查找替换。</div>`;
    return;
  }
  const box = el("div", "box");
  box.innerHTML = `<h4>🔎 全局查找替换</h4>
    <div class="caption">在所有设定和章节中查找并替换文本，例如修改人物名字、地名等。</div>
    <div class="row">
      <div><label class="field">🔍 查找内容</label><input type="text" id="find-text" placeholder="例如：李明"></div>
      <div><label class="field">✏️ 替换为</label><input type="text" id="replace-text" placeholder="例如：张三"></div>
    </div>`;
  const preview = el("div");
  box.append(preview);
  const findText = box.querySelector("#find-text");
  let debounce = null;
  const doFind = async () => {
    const q = findText.value.trim();
    if (!q) { preview.innerHTML = ""; return; }
    try {
      const { results } = await api(`/api/novels/${encodeURIComponent(S.novelId)}/find`, { method: "POST", body: { find_text: q } });
      if (results.length) {
        const total = results.reduce((a, r) => a + (parseInt((r.split("：找到 ")[1] || "0").split(" 处")[0]) || 0), 0);
        preview.innerHTML = `<div class="msg success">✅ 找到 ${total} 处匹配：</div>` +
          results.map(r => `<div class="caption">• ${esc(r)}</div>`).join("");
      } else {
        preview.innerHTML = `<div class="msg info">未找到「${esc(q)}」</div>`;
      }
    } catch (e) { preview.innerHTML = `<div class="msg error">${esc(e.message)}</div>`; }
  };
  findText.oninput = () => { clearTimeout(debounce); debounce = setTimeout(doFind, 500); };

  const repBtn = el("button", "btn primary", "🔄 执行替换");
  repBtn.style.marginTop = "10px";
  repBtn.onclick = async () => {
    const f = findText.value.trim();
    const r = box.querySelector("#replace-text").value;
    if (!f || !r) { alertMsg("error", "请输入查找内容和替换内容"); return; }
    const { results } = await api(`/api/novels/${encodeURIComponent(S.novelId)}/find`, { method: "POST", body: { find_text: f } });
    if (!results.length) { alertMsg("info", "没有找到需要替换的内容"); return; }
    if (!confirm(`⚠️ 即将在所有内容中将「${f}」替换为「${r}」，此操作不可撤销！确定？`)) return;
    const res = await api(`/api/novels/${encodeURIComponent(S.novelId)}/replace`, { method: "POST", body: { find_text: f, replace_text: r } });
    if (res.changes.length) {
      alertMsg("success", "✅ 替换完成！\n" + res.changes.join("\n"), 12000);
      refreshNovel(true);
    } else {
      alertMsg("info", "没有找到需要替换的内容");
    }
  };
  box.append(repBtn);
  root.append(box);
}

// ---- 角色图谱 Tab ----
function tabGraph(root) {
  const box = el("div", "box");
  box.innerHTML = `<h4>🕸️ 角色关系图谱</h4>
    <div class="caption">AI自动提取角色关系，生成可视化关系图谱。拖动平移，滚轮缩放。</div>`;
  const btn = el("button", "btn primary", "🚀 生成/更新图谱");
  btn.style.margin = "8px 0";
  btn.onclick = () => runGeneration("relation_graph", {}, {
    onDone: (r) => { if (r.graph) drawGraph(r.graph); refreshNovel(true); },
  });
  box.append(btn);
  const err = extra("relation_graph_raw_error");
  if (err) {
    box.innerHTML += `<div class="msg error">上次生成解析失败：${esc(err)}</div>
      <details><summary class="caption">AI 返回原文</summary><div class="content-view">${esc(extra("relation_graph_raw"))}</div></details>`;
  }
  root.append(box);

  const graph = extra("relation_graph");
  if (graph && graph.characters) {
    const wrap = el("div");
    wrap.id = "graph-container";
    root.append(wrap);
    setTimeout(() => drawGraph(graph), 0);
  } else {
    root.insertAdjacentHTML("beforeend", `<div class="msg info">还没有图谱数据，点击上方按钮生成。</div>`);
  }
}

const REL_COLORS = {
  "师徒": "#4fc3f7", "恋人": "#f06292", "敌人": "#ef5350", "朋友": "#48bb78",
  "主仆": "#ffca28", "同门": "#26c6da", "亲属": "#ab47bc", "对手": "#ed8936", "盟友": "#9ccc65",
};
const ROLE_COLORS = { "主角": "#4fc3f7", "反派": "#ef5350", "配角": "#ffca28" };

function drawGraph(graph) {
  const container = $("#graph-container") || $("#tab-content");
  container.querySelector("#graph-wrap")?.remove();
  container.querySelector(".char-cards")?.remove();
  container.querySelector(".rel-list")?.remove();

  const wrap = el("div"); wrap.id = "graph-wrap";
  const canvas = el("canvas"); canvas.id = "graph-canvas";
  canvas.width = Math.min(900, container.clientWidth || 900);
  canvas.height = 560;
  wrap.append(canvas);
  container.prepend(wrap);

  const chars = graph.characters || [];
  const rels = graph.relations || [];
  // 布局：主角居中，其余按角度环形
  const n = chars.length;
  const mainIdx = chars.findIndex(c => /主角/.test(c.role || ""));
  const pos = {};
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const R = Math.min(cx, cy) - 80;
  let ring = 0;
  chars.forEach((c, i) => {
    if (i === mainIdx) { pos[c.name] = { x: cx, y: cy }; return; }
    const angle = (ring / Math.max(1, n - 1)) * Math.PI * 2 - Math.PI / 2;
    pos[c.name] = { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) };
    ring++;
  });

  // 视图变换（拖动/缩放）
  const view = { x: 0, y: 0, scale: 1 };
  const ctx = canvas.getContext("2d");
  function draw() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(view.scale, 0, 0, view.scale, view.x, view.y);
    // 边
    for (const r of rels) {
      const a = pos[r.from], b = pos[r.to];
      if (!a || !b) continue;
      const color = REL_COLORS[r.type] || "#94a3b8";
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      // 箭头
      const ang = Math.atan2(b.y - a.y, b.x - a.x);
      const ex = b.x - 30 * Math.cos(ang), ey = b.y - 30 * Math.sin(ang);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 8 * Math.cos(ang - 0.4), ey - 8 * Math.sin(ang - 0.4));
      ctx.lineTo(ex - 8 * Math.cos(ang + 0.4), ey - 8 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fillStyle = color; ctx.fill();
      // 关系标签
      ctx.fillStyle = color; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(r.type || "", (a.x + b.x) / 2, (a.y + b.y) / 2 - 4);
    }
    // 节点
    for (const c of chars) {
      const p = pos[c.name];
      if (!p) continue;
      const border = ROLE_COLORS[c.role] || "#94a3b8";
      ctx.fillStyle = "#1a1a2e";
      ctx.strokeStyle = border; ctx.lineWidth = 2;
      const w = Math.max(64, c.name.length * 14 + 16), h = 34;
      ctx.beginPath();
      ctx.roundRect(p.x - w / 2, p.y - h / 2, w, h, 8);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#e2e8f0"; ctx.font = "13px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(c.name, p.x, p.y);
    }
  }
  draw();
  // 交互
  let drag = null;
  canvas.onmousedown = (e) => { drag = { x: e.offsetX, y: e.offsetY, vx: view.x, vy: view.y }; };
  canvas.onmousemove = (e) => {
    if (!drag) return;
    view.x = drag.vx + (e.offsetX - drag.x);
    view.y = drag.vy + (e.offsetY - drag.y);
    draw();
  };
  canvas.onmouseup = canvas.onmouseleave = () => { drag = null; };
  canvas.onwheel = (e) => {
    e.preventDefault();
    const k = e.deltaY < 0 ? 1.1 : 0.9;
    const ns = Math.min(5, Math.max(0.2, view.scale * k));
    view.x = e.offsetX - (e.offsetX - view.x) * (ns / view.scale);
    view.y = e.offsetY - (e.offsetY - view.y) * (ns / view.scale);
    view.scale = ns;
    draw();
  };
  const resetBtn = el("button", "btn small", "重置视图");
  resetBtn.style.cssText = "position:absolute;top:8px;right:8px";
  resetBtn.onclick = () => { view.x = 0; view.y = 0; view.scale = 1; draw(); };
  wrap.append(resetBtn);

  // 角色卡片
  const cards = el("div", "char-cards");
  for (const c of chars) {
    cards.innerHTML += `<div class="char-card">
      <div class="name">${esc(c.name)}</div>
      <div class="role">${esc(c.role || "")}</div>
      <div class="desc">${esc(c.desc || "")}</div></div>`;
  }
  container.append(cards);
  // 关系列表
  if (rels.length) {
    const list = el("div", "box rel-list");
    list.innerHTML = `<h4>关系详情</h4>` + rels.map(r =>
      `<div class="caption">• <b>${esc(r.from)}</b> → <b>${esc(r.to)}</b>
        <span style="color:${REL_COLORS[r.type] || "#94a3b8"}">[${esc(r.type || "")}]</span> ${esc(r.desc || "")}</div>`
    ).join("");
    container.append(list);
  }
}

// ---- 导出 Tab ----
function tabExport(root) {
  const chapters = S.novel.chapters || {};
  if (!Object.keys(chapters).length) {
    root.innerHTML = `<div class="msg info">还没有任何章节内容，无法导出。</div>`;
    return;
  }
  const box = el("div", "box");
  box.innerHTML = `<h4>📤 导出小说</h4>
    <div class="caption">Markdown 含世界观/人物/大纲/正文；docx 为投稿排版（仅正文）；EPUB 为上架格式（含目录）。</div>`;
  const row = el("div", "row");
  [["md", "⬇️ 下载 Markdown"], ["docx", "⬇️ 下载 docx"], ["epub", "⬇️ 下载 EPUB"]].forEach(([fmt, label]) => {
    const a = el("a", "btn primary shrink", label);
    a.href = `/api/novels/${encodeURIComponent(S.novelId)}/export/${fmt}`;
    a.style.textDecoration = "none"; a.style.textAlign = "center";
    row.append(a);
  });
  row.style.marginTop = "10px";
  box.append(row);
  root.append(box);
}

// ---- Skill 管理 Tab ----
function tabSkills(root) {
  // 注入设置（单技能注入正文的字数上限，全局生效）
  const cfgBox = el("div", "box");
  cfgBox.innerHTML = `<h4>⚙️ 注入设置</h4>
    <div class="caption">技能正文超过上限将被截断；总注入量 = 当前步骤匹配的技能数 × 上限。</div>
    <div class="row" style="margin-top:6px">
      <label class="field">单个技能注入字数上限 <input type="number" id="inject-chars" min="100" max="20000" step="100" style="width:110px"></label>
      <button class="btn small shrink" id="inject-save">💾 保存</button>
    </div>`;
  root.append(cfgBox);
  api("/api/skills/inject_limit").then(({ skill_inject_chars }) => {
    cfgBox.querySelector("#inject-chars").value = skill_inject_chars;
  }).catch(() => {});
  cfgBox.querySelector("#inject-save").onclick = async () => {
    const v = parseInt(cfgBox.querySelector("#inject-chars").value, 10);
    if (!v || v < 100 || v > 20000) { alertMsg("error", "请输入 100-20000 之间的数值"); return; }
    try {
      await api("/api/skills/inject_limit", { method: "PUT", body: { value: v } });
      alertMsg("success", `已保存：单个技能最多注入 ${v} 字`);
    } catch (e) { alertMsg("error", e.message); }
  };

  const box = el("div", "box");
  box.innerHTML = `<h4>🧩 Skill 技能包管理</h4>
    <div class="caption">技能包会在对应创作步骤自动注入 Prompt。启停状态按当前小说独立保存。</div>
    <div id="skills-list">加载中…</div>`;
  root.append(box);
  loadSkillsList();

  // 智能推荐
  const recBox = el("div", "box");
  recBox.innerHTML = `<h4>💡 智能推荐</h4>
    <div class="row"><input type="text" id="rec-query" placeholder="描述你的写作需求，如：如何写好打斗场面">
    <button class="btn small shrink" id="rec-btn">🔍 推荐</button></div>
    <div id="rec-result"></div>`;
  recBox.querySelector("#rec-btn").onclick = async () => {
    const q = recBox.querySelector("#rec-query").value.trim();
    if (!q) return;
    const { skills } = await api("/api/skills/recommend", { method: "POST", body: { query: q, novel_id: S.novelId } });
    $("#rec-result").innerHTML = skills.length
      ? skills.map(s => `<div class="caption">• <b>${esc(s.name)}</b>（${esc(s.dir)}，适用：${(s.apply_to || []).join("/")}）</div>`).join("")
      : `<div class="caption">没有匹配的技能包</div>`;
  };
  root.append(recBox);

  // 蒸馏
  const disBox = el("div", "box");
  disBox.innerHTML = `<h4>🧪 从文章蒸馏技能</h4>
    <textarea rows="5" id="distill-articles" placeholder="粘贴一篇或多篇写作方法论文章…"></textarea>
    <button class="btn primary" style="margin-top:8px">🧪 开始蒸馏</button>
    <div id="distill-result"></div>`;
  disBox.querySelector("button").onclick = () => {
    const articles = disBox.querySelector("#distill-articles").value.trim();
    if (!articles) { alertMsg("error", "请粘贴参考文章内容"); return; }
    runGeneration("distill", { articles, max_tokens: 6000 }, {
      onDone: (r) => {
        $("#distill-result").innerHTML = `<h4>蒸馏结果（确认后保存）</h4>
          <textarea rows="10" id="distill-text">${esc(r.result)}</textarea>
          <button class="btn primary small" id="distill-save" style="margin-top:6px">💾 保存为新技能</button>`;
        $("#distill-save").onclick = async () => {
          const text = $("#distill-text").value;
          const nameM = text.match(/name:\s*(.+)/);
          const dir = (nameM ? nameM[1].trim() : `distilled_${Date.now()}`).replace(/[\\/:*?"<>|\s]/g, "_");
          await api("/api/skills", { method: "POST", body: { dir_name: dir, meta: parseSkillMeta(text), body: stripSkillMeta(text) } });
          alertMsg("success", `技能已保存到 ${dir}`);
          loadSkillsList();
        };
      },
    });
  };
  root.append(disBox);

  // 导入 + 新建
  const impBox = el("div", "box");
  impBox.innerHTML = `<h4>📥 导入 / 新建技能</h4>
    <input type="file" id="skill-file" accept=".zip,.md">
    <div class="caption">支持 .zip（含 SKILL.md）或单个 .md 文件</div>
    <div class="divider"></div>
    <div class="row">
      <input type="text" id="new-skill-dir" placeholder="目录名（英文/数字）">
      <input type="text" id="new-skill-name" placeholder="技能名称">
    </div>
    <textarea rows="3" id="new-skill-body" placeholder="技能正文（注入 Prompt 的指令内容）" style="margin-top:6px"></textarea>
    <button class="btn primary small" id="new-skill-btn" style="margin-top:6px">➕ 新建技能</button>`;
  impBox.querySelector("#skill-file").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const resp = await fetch("/api/skills/import", { method: "POST", body: fd });
      const r = await resp.json();
      if (!resp.ok) throw new Error(r.detail || "导入失败");
      alertMsg("success", `已导入到 ${r.dir}`);
      loadSkillsList();
    } catch (err) { alertMsg("error", err.message); }
    e.target.value = "";
  };
  impBox.querySelector("#new-skill-btn").onclick = async () => {
    const dir = impBox.querySelector("#new-skill-dir").value.trim();
    const name = impBox.querySelector("#new-skill-name").value.trim();
    const body = impBox.querySelector("#new-skill-body").value;
    if (!dir || !name) { alertMsg("error", "请填写目录名和技能名称"); return; }
    await api("/api/skills", { method: "POST", body: { dir_name: dir, meta: { name, enabled: true, apply_to: ["chapter", "continue", "polish"] }, body } });
    alertMsg("success", "技能已创建");
    loadSkillsList();
  };
  root.append(impBox);
}

function parseSkillMeta(text) {
  const m = text.match(/^---\s*\n([\s\S]*?)\n---/);
  const meta = { enabled: true, apply_to: ["chapter", "continue", "polish"] };
  if (m) {
    m[1].split("\n").forEach(line => {
      const mm = line.match(/^(\w+):\s*(.+)$/);
      if (mm) meta[mm[1]] = mm[2].trim();
    });
  }
  return meta;
}
function stripSkillMeta(text) {
  return text.replace(/^---\s*\n[\s\S]*?\n---\s*\n?/, "").trim();
}

async function loadSkillsList() {
  const wrap = $("#skills-list");
  try {
    const { skills, step_labels } = await api(`/api/skills?novel_id=${encodeURIComponent(S.novelId)}`);
    if (!skills.length) { wrap.innerHTML = `<div class="caption">暂无技能包</div>`; return; }
    wrap.innerHTML = "";
    const groups = [["✅ 已启用", skills.filter(s => s.effective_enabled)], ["⏸️ 未启用", skills.filter(s => !s.effective_enabled)]];
    for (const [title, list] of groups) {
      if (!list.length) continue;
      wrap.insertAdjacentHTML("beforeend", `<h4 style="margin-top:12px">${title}</h4>`);
      for (const s of list) {
        const item = el("div", "skill-item");
        const overrideBadge = s.novel_override !== null && s.novel_override !== undefined
          ? `<span class="badge">本小说${s.novel_override ? "启用" : "停用"}</span>` : "";
        item.innerHTML = `<div class="head">
            <span class="name">${esc(s.name)}</span>
            <span class="badge ${s.effective_enabled ? "on" : "off"}">${s.effective_enabled ? "生效中" : "未生效"}</span>
            ${overrideBadge}
            <span class="badge">适用：${(s.apply_to || []).map(a => step_labels[a] || a).join("、") || "未指定"}</span>
            <span class="badge">${s.chars} 字</span>
          </div>
          <div class="caption">${esc(s.description || "")}</div>`;
        const ops = el("div", "row");
        const mkBtn = (label, fn) => { const b = el("button", "btn small shrink", label); b.onclick = fn; return b; };
        ops.append(
          mkBtn("✅ 本小说启用", () => toggleSkill(s.dir, true)),
          mkBtn("⏸️ 本小说停用", () => toggleSkill(s.dir, false)),
          mkBtn("↩️ 跟随全局", () => toggleSkill(s.dir, null)),
        );
        const editBtn = mkBtn("✏️ 编辑", () => {
          const editArea = el("div");
          editArea.innerHTML = `<label class="field">适用步骤（逗号分隔：world,characters,outline,chapter,continue,polish,consistency,relations）</label>
            <input type="text" value="${esc((s.apply_to || []).join(","))}" id="edit-apply">
            <label class="field">正文</label><textarea rows="8" id="edit-body">${esc(s.body)}</textarea>
            <button class="btn primary small" style="margin-top:6px">💾 保存</button>`;
          editArea.querySelector("button").onclick = async () => {
            await api(`/api/skills/${encodeURIComponent(s.dir)}`, {
              method: "PUT",
              body: {
                meta: { name: s.name, description: s.description, enabled: s.enabled, source: s.source, keywords: s.keywords, phases: s.phases, apply_to: editArea.querySelector("#edit-apply").value.split(",").map(x => x.trim()).filter(Boolean) },
                body: editArea.querySelector("#edit-body").value,
              },
            });
            alertMsg("success", "已保存");
            loadSkillsList();
          };
          item.append(editArea);
          editBtn.remove();
        });
        ops.append(editBtn);
        ops.append(mkBtn("🗑️ 删除", async () => {
          if (!confirm(`删除技能「${s.name}」？`)) return;
          await api(`/api/skills/${encodeURIComponent(s.dir)}`, { method: "DELETE" });
          loadSkillsList();
        }));
        item.append(ops);
        wrap.append(item);
      }
    }
  } catch (e) { wrap.innerHTML = `<div class="msg error">${esc(e.message)}</div>`; }
}

async function toggleSkill(dir, enabled) {
  await api(`/api/skills/${encodeURIComponent(dir)}/toggle`, { method: "POST", body: { novel_id: S.novelId, enabled } });
  loadSkillsList();
}

// ---------- 渲染入口 ----------
function renderTabContent() {
  const root = $("#tab-content");
  root.innerHTML = "";
  if (!requireNovel()) return;
  const key = TABS[S.activeTab][1];
  ({
    world: tabWorld, characters: tabCharacters, outline: tabOutline, chapter: tabChapter,
    continue: tabContinue, polish: tabPolish, consistency: tabConsistency,
    findreplace: tabFindReplace, graph: tabGraph, export: tabExport, skills: tabSkills,
  })[key](root);
}

function renderAll() {
  renderProgress();
  renderTabs();
  renderTabContent();
  renderCurrentNovelCard();
}

// ---------- 启动 ----------
$("#sidebar-toggle").onclick = () => $("#sidebar").classList.toggle("collapsed");

// 主题切换（白/黑），持久化到 localStorage（元素可能因缓存旧页面缺失，做防御判断）
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $("#theme-toggle");
  if (btn) btn.textContent = theme === "light" ? "☀️" : "🌙";
  try { localStorage.setItem("theme", theme); } catch (e) { /* ignore */ }
}
{
  const themeBtn = $("#theme-toggle");
  if (themeBtn) themeBtn.onclick = () =>
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
}
applyTheme(localStorage.getItem("theme") || "dark");

// 右侧悬浮面板（模型配置/连接状态/API用量）展开/收起，状态持久化
function applyRightbar(open) {
  $("#rightbar").classList.toggle("collapsed", !open);
  document.body.classList.toggle("rightbar-open", open);
  try { localStorage.setItem("rightbar_open", open ? "1" : "0"); } catch (e) { /* ignore */ }
}
$("#rightbar-toggle").onclick = () =>
  applyRightbar($("#rightbar").classList.contains("collapsed"));
applyRightbar(localStorage.getItem("rightbar_open") === "1");
(async function init() {
  renderTabs();
  renderConnStatus();
  await refreshNovelList();
  await refreshNovel();
  refreshProviders();
  refreshUsage();
  setInterval(refreshUsage, 30000);
})();
