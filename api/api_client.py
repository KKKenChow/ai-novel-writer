"""
通用大模型 API 客户端
兼容所有 OpenAI 格式 API（火山方舟/豆包/DeepSeek/GPT等）
"""
import os
import time
import requests
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class LLMAPIClient:
    def __init__(self, api_key=None, api_base=None, model="doubao-pro-32k", timeout=180, max_retries=2):
        self.api_key = api_key or os.getenv("VOLC_API_KEY")
        self.api_base = api_base or os.getenv("VOLC_API_BASE", "https://ark.cn-beijing.volces.com/api/v1/chat/completions")
        self.model = model
        self.api_url = self.api_base
        self.timeout = timeout
        self.max_retries = max_retries
    
    # 模型最大输出token限制（大多数32k模型的硬上限）
    MAX_TOKENS_LIMIT = 32768

    def chat(self, messages: List[Dict], temperature=0.7, max_tokens=2000) -> str:
        """调用大模型聊天接口，支持自动重试"""
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
                    content = result["choices"][0]["message"]["content"]
                    # 打印API响应摘要
                    usage = result.get("usage", {})
                    logger.info(f"API响应 ← model={result.get('model', self.model)}, tokens={usage.get('total_tokens', '?')}, 内容长度={len(content)}")
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
    
    def generate(self, prompt: str, **kwargs) -> str:
        """单轮生成"""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
