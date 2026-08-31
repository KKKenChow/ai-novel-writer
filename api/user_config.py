"""
用户配置管理：所有含敏感信息（API Key等）的配置统一存放在 user_config.json
该文件已加入 .gitignore，不会被提交到仓库
"""
import os
import json
import threading
from typing import List, Dict, Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_config.json")

_lock = threading.Lock()

DEFAULT_CONFIG = {
    "active_provider": "",      # 当前启用的模型配置名称
    "providers": [],            # [{"name","api_key","api_base","model","max_output","reasoning","reasoning_effort_options","reasoning_effort"}]
    "max_tokens_overrides": {}, # 每步 max_tokens 用户覆盖 {step: int}，空值表示用默认
    "cumulative_usage": {       # 跨会话累计用量
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    },
    "skill_inject_chars": 2000,  # 单个技能注入正文的字数上限
}


def load_config() -> Dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:
            pass
    return cfg


def save_config(cfg: Dict):
    with _lock:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------- 模型配置（providers） ----------

def list_providers() -> List[Dict]:
    return load_config().get("providers", [])


def get_provider(name: str) -> Optional[Dict]:
    for p in list_providers():
        if p.get("name") == name:
            return p
    return None


def upsert_provider(provider: Dict):
    """新增或覆盖同名配置"""
    cfg = load_config()
    providers = cfg.get("providers", [])
    for i, p in enumerate(providers):
        if p.get("name") == provider.get("name"):
            providers[i] = provider
            break
    else:
        providers.append(provider)
    cfg["providers"] = providers
    save_config(cfg)


def delete_provider(name: str):
    cfg = load_config()
    cfg["providers"] = [p for p in cfg.get("providers", []) if p.get("name") != name]
    if cfg.get("active_provider") == name:
        cfg["active_provider"] = cfg["providers"][0]["name"] if cfg["providers"] else ""
    save_config(cfg)


def get_active_provider() -> Optional[Dict]:
    cfg = load_config()
    name = cfg.get("active_provider", "")
    p = get_provider(name) if name else None
    if p is None and cfg.get("providers"):
        p = cfg["providers"][0]
    return p


def set_active_provider(name: str):
    cfg = load_config()
    cfg["active_provider"] = name
    save_config(cfg)


def _read_env_file(path: str) -> Dict:
    """极简解析 .env 文件（KEY=VALUE 行），仅用于一次性迁移，不依赖 python-dotenv"""
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return result


def ensure_env_migrated():
    """
    老用户无感迁移：若 user_config.json 中还没有任何模型配置，
    且项目根目录存在旧版 .env（含 VOLC_* 配置），则自动迁移为一个 provider。
    迁移后 .env 不再被读取，可自行删除。
    """
    cfg = load_config()
    if cfg.get("providers"):
        return
    env_path = os.path.join(os.path.dirname(CONFIG_PATH), ".env")
    env = _read_env_file(env_path)
    api_key = env.get("VOLC_API_KEY", "")
    api_base = env.get("VOLC_API_BASE", "")
    model = env.get("VOLC_MODEL", "")
    if api_key or api_base or model:
        p = {
            "name": "默认配置(迁移自.env)",
            "api_key": api_key,
            "api_base": api_base,
            "model": model,
        }
        cfg["providers"] = [p]
        cfg["active_provider"] = p["name"]
        save_config(cfg)


# ---------- 累计用量 ----------

def add_cumulative_usage(prompt_tokens: int, completion_tokens: int, calls: int = 1, model: str = ""):
    cfg = load_config()
    u = cfg.setdefault("cumulative_usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    u["calls"] = u.get("calls", 0) + calls
    u["prompt_tokens"] = u.get("prompt_tokens", 0) + prompt_tokens
    u["completion_tokens"] = u.get("completion_tokens", 0) + completion_tokens
    if model:
        by_model = cfg.setdefault("usage_by_model", {})
        m = by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        m["calls"] = m.get("calls", 0) + calls
        m["prompt_tokens"] = m.get("prompt_tokens", 0) + prompt_tokens
        m["completion_tokens"] = m.get("completion_tokens", 0) + completion_tokens
    save_config(cfg)


def get_cumulative_usage() -> Dict:
    return load_config().get("cumulative_usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})


def get_usage_by_model() -> Dict:
    return load_config().get("usage_by_model", {})


def clear_usage(model: str = ""):
    """清除用量统计。model 为空 → 清除全部（总用量+所有单模型）；
    指定 model → 只删该模型统计，并从总用量中扣减"""
    cfg = load_config()
    if not model:
        cfg["cumulative_usage"] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        cfg["usage_by_model"] = {}
    else:
        by_model = cfg.get("usage_by_model", {})
        m = by_model.pop(model, None)
        cfg["usage_by_model"] = by_model
        if m:
            u = cfg.setdefault("cumulative_usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            u["calls"] = max(0, u.get("calls", 0) - m.get("calls", 0))
            u["prompt_tokens"] = max(0, u.get("prompt_tokens", 0) - m.get("prompt_tokens", 0))
            u["completion_tokens"] = max(0, u.get("completion_tokens", 0) - m.get("completion_tokens", 0))
    save_config(cfg)


# ---------- Skill 注入设置 ----------

def get_skill_inject_chars() -> int:
    try:
        return max(100, int(load_config().get("skill_inject_chars", 2000) or 2000))
    except (ValueError, TypeError):
        return 2000


def set_skill_inject_chars(value: int):
    cfg = load_config()
    cfg["skill_inject_chars"] = max(100, int(value))
    save_config(cfg)


# ---------- 每步 max_tokens 覆盖 ----------

def get_max_tokens_overrides() -> Dict:
    """返回 {step: int}，只含有效正整数项"""
    raw = load_config().get("max_tokens_overrides", {}) or {}
    out = {}
    for k, v in raw.items():
        try:
            v = int(v)
            if v > 0:
                out[k] = v
        except (ValueError, TypeError):
            continue
    return out


def set_max_tokens_overrides(overrides: Dict):
    """整体写回覆盖表；无效值（非正整数）自动剔除"""
    cfg = load_config()
    cleaned = {}
    for k, v in (overrides or {}).items():
        try:
            v = int(v)
            if v > 0:
                cleaned[str(k)] = v
        except (ValueError, TypeError):
            continue
    cfg["max_tokens_overrides"] = cleaned
    save_config(cfg)


def update_provider_fields(name: str, fields: Dict):
    """只更新 provider 的指定字段（如探测结果），不触碰其他字段"""
    cfg = load_config()
    for i, p in enumerate(cfg.get("providers", [])):
        if p.get("name") == name:
            p.update(fields)
            cfg["providers"][i] = p
            save_config(cfg)
            return
