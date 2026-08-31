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

    def __init__(self, api_key=None, api_base=None, model="doubao-pro-32k", timeout=180, max_retries=2):
        self.api_key = api_key
        self.api_base = api_base or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.model = model
        self.api_url = self.api_base
        self.timeout = timeout
        self.max_retries = max_retries
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
    
    # 模型最大输出token限制（大多数32k模型的硬上限）
    MAX_TOKENS_LIMIT = 32768

    def chat(self, messages: List[Dict], temperature=0.7, max_tokens=2000, stream_callback=None) -> str:
        """调用大模型聊天接口，支持自动重试。
        stream_callback: 可选，fn(chunk: str)。提供时优先使用流式接口逐段回调；
        若 API 不支持流式或流式请求在收到任何内容前失败，自动降级为非流式。
        """
        if stream_callback:
            try:
                return self._chat_stream(messages, temperature, max_tokens, stream_callback)
            except _StreamNotSupported as e:
                logger.info(f"流式请求不可用，自动降级为非流式: {e}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                # 网络类错误走非流式的重试逻辑
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
        
        # 打印API调用信息到控制台
        msg_count = len(messages)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"API调用 → model={self.model}, url={self.api_url}, messages={msg_count}条, 总字数≈{total_chars}, temperature={temperature}, max_tokens={max_tokens}")
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"].get("content") or ""
                    if not content:
                        raise Exception(f"API返回空内容: {str(result)[:300]}")
                    # 打印API响应摘要
                    usage = result.get("usage", {})
                    logger.info(f"API响应 ← model={result.get('model', self.model)}, tokens={usage.get('total_tokens', '?')}, 内容长度={len(content)}")
                    # 累计用量统计
                    self._record_usage(usage, total_chars, len(content))
                    return content
                else:
                    raise Exception(f"API返回异常: {result}")
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
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

    def _chat_stream(self, messages: List[Dict], temperature, max_tokens, stream_callback) -> str:
        """流式聊天请求（SSE）。收到任何内容前失败则抛 _StreamNotSupported 触发降级；
        收到部分内容后失败则直接抛异常（避免降级重试导致内容重复）。"""
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
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"API调用(流式) → model={self.model}, messages={len(messages)}条, 总字数≈{total_chars}, max_tokens={max_tokens}")

        response = requests.post(self.api_url, headers=headers, json=data,
                                 timeout=self.timeout, stream=True)
        if response.status_code != 200:
            # 未收到任何内容，可以安全降级
            raise _StreamNotSupported(f"HTTP {response.status_code}: {response.text[:200]}")

        parts = []
        usage = {}
        for raw_line in response.iter_lines(decode_unicode=True):
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
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                parts.append(delta)
                try:
                    stream_callback(delta)
                except Exception:
                    pass
        content = "".join(parts)
        if not content:
            raise _StreamNotSupported("流式响应无内容")
        logger.info(f"API响应(流式) ← 内容长度={len(content)}")
        self._record_usage(usage, total_chars, len(content))
        return content

    def test_connection(self, timeout: int = 30) -> Tuple[bool, str, float]:
        """
        测试API连通性：发送最小请求
        返回 (是否成功, 描述信息, 延迟毫秒)
        """
        if not self.api_key:
            return False, "未填写 API Key", 0.0
        if not self.api_url:
            return False, "未填写 API Base URL", 0.0
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
            if resp.status_code == 200:
                result = resp.json()
                if "choices" in result:
                    return True, f"连接成功（模型: {result.get('model', self.model)}）", latency
                return False, f"返回格式异常: {str(result)[:100]}", latency
            if resp.status_code in (401, 403):
                return False, f"认证失败({resp.status_code})：API Key 无效或无权限", latency
            if resp.status_code == 404:
                return False, "接口不存在(404)：请检查 API Base URL 或模型名称", latency
            return False, f"HTTP {resp.status_code}: {resp.text[:150]}", latency
        except requests.exceptions.Timeout:
            return False, f"连接超时（>{timeout}s）", (time.time() - start) * 1000
        except requests.exceptions.ConnectionError as e:
            return False, f"网络连接失败：{str(e)[:120]}", (time.time() - start) * 1000
        except Exception as e:
            return False, f"测试失败：{str(e)[:120]}", (time.time() - start) * 1000

    def generate(self, prompt: str, step: str = "", stream_callback=None, **kwargs) -> str:
        """单轮生成；step 用于 skill 注入（标识当前创作步骤）"""
        if self.skill_provider and step:
            prompt = self.skill_provider(prompt, step)
        return self.chat([{"role": "user", "content": prompt}], stream_callback=stream_callback, **kwargs)


class _StreamNotSupported(Exception):
    """流式请求在收到任何内容前失败，可安全降级为非流式"""
    pass
