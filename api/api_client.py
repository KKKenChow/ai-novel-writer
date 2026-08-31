"""
通用大模型 API 客户端
兼容所有 OpenAI 格式 API（火山方舟/豆包/DeepSeek/GPT等）
"""
import os
import time
import requests
import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class LLMAPIClient:
    # 本次会话累计用量统计（token）
    session_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_chars_in": 0,
        "total_chars_out": 0,
    }

    def __init__(self, api_key=None, api_base=None, model="doubao-pro-32k", timeout=180, max_retries=2,
                 max_output=None, reasoning_effort=None, thinking_disabled=False):
        self.api_key = api_key
        self.api_base = api_base or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.model = model
        self.api_url = self.api_base
        self.timeout = timeout
        self.max_retries = max_retries
        # 单次输出 token 上限（推理模型思考也占用该额度），由 provider 配置的 max_output 决定
        self.MAX_TOKENS_LIMIT = int(max_output) if max_output else 65536
        # 思考强度（OpenAI 兼容参数 reasoning_effort）；None/空串 = 不传该参数
        self.reasoning_effort = reasoning_effort or None
        # 关闭思考模式（DeepSeek/豆包格式 thinking.type=disabled）；仅用户显式开启时注入
        self.thinking_disabled = bool(thinking_disabled)
        # 取消检查回调：fn() -> bool，返回 True 表示任务已被用户取消；由外部注入
        self.cancel_check = None
        # 当前进行中的 HTTP 响应（供 cancel() 强断阻塞中的连接）
        self._active_response = None
        # 最近一次调用的 finish_reason（stop/length/None），供上层判断截断
        self.last_finish_reason = None
        # 最近一次调用是否在请求关闭思考后仍输出了思考内容（参数未生效信号）
        self.thinking_disable_ignored = False
        # skill 注入回调：由外部注入，签名 fn(prompt: str, step: str) -> str
        self.skill_provider = None
        # 实例级用量统计（避免多客户端/多会话共享类属性串数据）
        self.session_usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_chars_in": 0,
            "total_chars_out": 0,
        }
    
    def _apply_extra_params(self, data: Dict):
        """按 provider 配置注入额外请求参数（思考强度 / 关闭思考）"""
        if self.reasoning_effort:
            data["reasoning_effort"] = self.reasoning_effort
        if self.thinking_disabled:
            data["thinking"] = {"type": "disabled"}

    def cancel(self):
        """外部请求取消：强断当前阻塞中的 HTTP 连接，使正在进行的请求立即抛错返回"""
        resp = self._active_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def _check_cancelled(self):
        if self.cancel_check and self.cancel_check():
            raise GenerationCancelled("已被用户取消")

    @staticmethod
    def _est_tokens(chars: int) -> int:
        """中文约 0.6-0.7 token/字的粗略估算，仅用于日志可读性"""
        return int(chars * 0.7)

    def chat(self, messages: List[Dict], temperature=0.7, max_tokens=2000, stream_callback=None,
             reasoning_callback=None) -> str:
        """调用大模型聊天接口，支持自动重试。
        stream_callback: 可选，fn(chunk: str)。提供时优先使用流式接口逐段回调；
        若 API 不支持流式或流式请求在收到任何内容前失败，自动降级为非流式。
        reasoning_callback: 可选，fn(chunk: str)，流式时回调思考过程（reasoning_content）。
        """
        self.last_finish_reason = None
        self.thinking_disable_ignored = False
        self._check_cancelled()
        if stream_callback:
            try:
                return self._chat_stream(messages, temperature, max_tokens, stream_callback,
                                         reasoning_callback=reasoning_callback)
            except _StreamNotSupported as e:
                logger.info(f"流式请求不可用，自动降级为非流式: {e}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                # 网络类错误走非流式的重试逻辑（但取消导致的断连直接抛取消）
                self._check_cancelled()
                logger.info("流式请求网络异常，降级为非流式重试")
        return self._chat_plain(messages, temperature, max_tokens)

    def _chat_plain(self, messages: List[Dict], temperature=0.7, max_tokens=2000) -> str:
        """非流式聊天请求（原 chat 实现）"""
        # 防御性截断：max_tokens 不得超过模型最大输出限制
        if max_tokens > self.MAX_TOKENS_LIMIT:
            logger.warning(f"max_tokens={max_tokens} 超过模型上限 {self.MAX_TOKENS_LIMIT}，自动截断")
            max_tokens = self.MAX_TOKENS_LIMIT

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        self._apply_extra_params(data)
        
        # 打印API调用信息到控制台
        msg_count = len(messages)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"API调用 → model={self.model}, url={self.api_url}, messages={msg_count}条, "
                    f"输入≈{total_chars}字(~{self._est_tokens(total_chars)}token), "
                    f"temperature={temperature}, max_tokens={max_tokens}")
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self._check_cancelled()
                response = requests.post(self.api_url, headers=headers, json=data, timeout=self.timeout)
                self._active_response = response
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    self.last_finish_reason = choice.get("finish_reason")
                    content = choice.get("message", {}).get("content") or ""
                    usage = result.get("usage", {})
                    reasoning_len = len(choice.get("message", {}).get("reasoning_content") or "")
                    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                    if self.thinking_disabled and (reasoning_len or reasoning_tokens):
                        self.thinking_disable_ignored = True
                        logger.warning("已请求关闭思考，但响应仍包含思考内容——该服务商可能不支持 thinking 参数")
                    if self.last_finish_reason == "length":
                        logger.warning(f"输出达到 max_tokens={max_tokens} 上限被截断（finish_reason=length）")
                    if not content:
                        hint = ""
                        if self.last_finish_reason == "length" and reasoning_len:
                            hint = ("【可能原因】该模型为推理模型，思考(reasoning)耗尽了全部 max_tokens 额度，"
                                    "未输出正文。请到「模型配置 → 高级设置」调大对应步骤的 max_tokens，或调低思考强度。")
                        raise Exception(f"API返回空内容{hint}: {str(result)[:300]}")
                    # 打印API响应摘要
                    logger.info(f"API响应 ← model={result.get('model', self.model)}, tokens={usage.get('total_tokens', '?')}"
                                f"{f'(思考{reasoning_tokens})' if reasoning_tokens else ''}, 内容长度={len(content)}")
                    # 累计用量统计
                    self._record_usage(usage, total_chars, len(content))
                    return content
                else:
                    raise Exception(f"API返回异常: {result}")
            except GenerationCancelled:
                raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if self.cancel_check and self.cancel_check():
                    raise GenerationCancelled("已被用户取消")
                if attempt < self.max_retries:
                    wait_time = (attempt + 1) * 5  # 递增等待: 5s, 10s
                    time.sleep(wait_time)
                    continue
                raise Exception(f"API请求失败（已重试{self.max_retries}次）: {str(e)}")
            except requests.exceptions.HTTPError as e:
                # 4xx 错误不重试（如认证失败、参数错误）
                if response.status_code < 500:
                    raise Exception(f"API错误({response.status_code}): {response.text[:300]}")
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(3)
                    continue
                raise Exception(f"API服务器错误（已重试{self.max_retries}次）: {str(e)}")
            finally:
                self._active_response = None
        
        raise Exception(f"API请求失败: {str(last_error)}")

    def _record_usage(self, usage: Dict, chars_in: int, chars_out: int):
        """累计实例级会话用量 + 跨会话持久化用量（失败静默）"""
        u = self.session_usage
        u["calls"] += 1
        u["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        u["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        u["total_chars_in"] += chars_in
        u["total_chars_out"] += chars_out
        try:
            from api.user_config import add_cumulative_usage
            add_cumulative_usage(usage.get("prompt_tokens", 0) or 0,
                                 usage.get("completion_tokens", 0) or 0,
                                 model=self.model)
        except Exception:
            pass

    def _chat_stream(self, messages: List[Dict], temperature, max_tokens, stream_callback,
                     reasoning_callback=None) -> str:
        """流式聊天请求（SSE）。收到任何内容前失败则抛 _StreamNotSupported 触发降级；
        收到部分内容后失败则直接抛异常（避免降级重试导致内容重复）。
        reasoning_callback: 推理模型的思考过程（reasoning_content）逐段回调。"""
        if max_tokens > self.MAX_TOKENS_LIMIT:
            max_tokens = self.MAX_TOKENS_LIMIT
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        self._apply_extra_params(data)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"API调用(流式) → model={self.model}, messages={len(messages)}条, "
                    f"输入≈{total_chars}字(~{self._est_tokens(total_chars)}token), max_tokens={max_tokens}")

        self._check_cancelled()
        response = requests.post(self.api_url, headers=headers, json=data,
                                 timeout=self.timeout, stream=True)
        if response.status_code != 200:
            # 未收到任何内容，可以安全降级
            raise _StreamNotSupported(f"HTTP {response.status_code}: {response.text[:200]}")
        # 部分网关 SSE 响应不带 charset，requests 会对 text/* 默认按 ISO-8859-1 解码导致中文乱码
        response.encoding = "utf-8"
        self._active_response = response

        parts = []
        usage = {}
        reasoning_len = 0
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                self._check_cancelled()
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0]
                if choice0.get("finish_reason"):
                    self.last_finish_reason = choice0["finish_reason"]
                delta = choice0.get("delta") or {}
                rc = delta.get("reasoning_content") or ""
                if rc:
                    reasoning_len += len(rc)
                    if reasoning_callback:
                        try:
                            reasoning_callback(rc)
                        except Exception:
                            pass
                delta = delta.get("content") or ""
                if delta:
                    parts.append(delta)
                    try:
                        stream_callback(delta)
                    except Exception:
                        pass
        finally:
            self._active_response = None
            try:
                response.close()
            except Exception:
                pass
        content = "".join(parts)
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        if self.thinking_disabled and (reasoning_len or reasoning_tokens):
            self.thinking_disable_ignored = True
            logger.warning("已请求关闭思考，但流式响应仍包含思考内容——该服务商可能不支持 thinking 参数")
        if self.last_finish_reason == "length":
            logger.warning(f"输出达到 max_tokens={max_tokens} 上限被截断（finish_reason=length）")
        if not content:
            hint = ""
            if self.last_finish_reason == "length" and reasoning_len:
                hint = "（思考耗尽了全部 max_tokens 额度，未输出正文，请调大 max_tokens 或调低思考强度）"
            raise _StreamNotSupported(f"流式响应无内容{hint}")
        logger.info(f"API响应(流式) ← model={self.model}, 内容长度={len(content)}"
                    f"{f', 思考长度={reasoning_len}' if reasoning_len else ''}")
        self._record_usage(usage, total_chars, len(content))
        return content

    def test_connection(self, timeout: int = 30, probe: bool = True) -> Tuple[bool, str, float, Dict]:
        """
        测试API连通性：发送最小请求；可选探测模型思考能力。
        返回 (是否成功, 描述信息, 延迟毫秒, 能力信息dict)
        能力信息: {"reasoning": bool, "reasoning_effort_options": [...]|None}
        """
        caps = {"reasoning": False, "reasoning_effort_options": None, "thinking_disable": None}
        if not self.api_key:
            return False, "未填写 API Key", 0.0, caps
        if not self.api_url:
            return False, "未填写 API Base URL", 0.0, caps
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False
        }
        start = time.time()
        try:
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=timeout)
            latency = (time.time() - start) * 1000
            if resp.status_code != 200:
                if resp.status_code in (401, 403):
                    return False, f"认证失败({resp.status_code})：API Key 无效或无权限", latency, caps
                if resp.status_code == 404:
                    return False, "接口不存在(404)：请检查 API Base URL 或模型名称", latency, caps
                return False, f"HTTP {resp.status_code}: {resp.text[:150]}", latency, caps
            result = resp.json()
            if "choices" not in result:
                return False, f"返回格式异常: {str(result)[:100]}", latency, caps
            msg = f"连接成功（模型: {result.get('model', self.model)}）"
            if probe:
                caps = self._probe_reasoning(headers, timeout)
                if caps["reasoning"]:
                    opts = caps.get("reasoning_effort_options")
                    msg += " 🧠推理模型"
                    msg += f"（支持思考强度: {'/'.join(opts)}）" if opts else "（不支持调整思考强度）"
                    if caps.get("thinking_disable"):
                        msg += "（支持关闭思考）"
            return True, msg, latency, caps
        except requests.exceptions.Timeout:
            return False, f"连接超时（>{timeout}s）", (time.time() - start) * 1000, caps
        except requests.exceptions.ConnectionError as e:
            return False, f"网络连接失败：{str(e)[:120]}", (time.time() - start) * 1000, caps
        except Exception as e:
            return False, f"测试失败：{str(e)[:120]}", (time.time() - start) * 1000, caps

    def _probe_reasoning(self, headers: Dict, timeout: int) -> Dict:
        """探测是否为推理模型及思考强度支持情况。所有失败静默降级为"非推理"。"""
        caps = {"reasoning": False, "reasoning_effort_options": None}
        base = {
            "model": self.model,
            "messages": [{"role": "user", "content": "1+1=?"}],
            "max_tokens": 2000,
            "stream": False
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=base, timeout=timeout)
            if resp.status_code != 200:
                return caps
            result = resp.json()
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = result.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            if message.get("reasoning_content") or (details.get("reasoning_tokens") or 0) > 0:
                caps["reasoning"] = True
        except Exception:
            return caps
        if not caps["reasoning"]:
            return caps
        # 探测是否支持 reasoning_effort 参数（不支持的 API 通常返回 400）
        try:
            resp = requests.post(self.api_url, headers=headers,
                                 json={**base, "reasoning_effort": "low"}, timeout=timeout)
            if resp.status_code == 200:
                caps["reasoning_effort_options"] = ["low", "medium", "high"]
        except Exception:
            pass
        # 探测是否支持关闭思考（thinking.type=disabled，DeepSeek/豆包系格式）
        try:
            resp = requests.post(self.api_url, headers=headers,
                                 json={**base, "thinking": {"type": "disabled"}}, timeout=timeout)
            if resp.status_code == 200:
                caps["thinking_disable"] = True
        except Exception:
            pass
        return caps

    def generate(self, prompt: str, step: str = "", stream_callback=None,
                 reasoning_callback=None, **kwargs) -> str:
        """单轮生成；step 用于 skill 注入（标识当前创作步骤）"""
        if self.skill_provider and step:
            prompt = self.skill_provider(prompt, step)
        return self.chat([{"role": "user", "content": prompt}], stream_callback=stream_callback,
                         reasoning_callback=reasoning_callback, **kwargs)


class GenerationCancelled(Exception):
    """用户手动取消生成（取消按钮/连接被强断）"""
    pass


class _StreamNotSupported(Exception):
    """流式请求在收到任何内容前失败，可安全降级为非流式"""
    pass
