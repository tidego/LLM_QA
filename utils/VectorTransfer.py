import os
import time
import asyncio
from typing import Optional
from openai import AsyncOpenAI, OpenAIError

from config.Config import DASHSCOPE_API_KEY


class TokenBucket:
    """
    异步令牌桶限速器
    """

    def __init__(self, rate: float, capacity: int):
        """
        :param rate: 每秒生成的令牌数量（QPS）
        :param capacity: 最大可积累的令牌数（应略大于 rate）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp

            # 添加新令牌
            new_tokens = elapsed * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.timestamp = now

            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.timestamp = time.monotonic()
                return


class VectorTransfer:
    """
    使用 DashScope获取文本向量
    """

    def __init__(self, qps: Optional[int] = 15, capacity: Optional[int] = None):
        """
        初始化向量器

        :param qps: 每秒最大请求数（默认 15）
        :param capacity: 令牌桶最大容量（默认 qps * 2）
        """
        if DASHSCOPE_API_KEY is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        else:
            api_key = DASHSCOPE_API_KEY
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        final_capacity = capacity if capacity is not None else qps * 2
        self.limiter = TokenBucket(rate=qps, capacity=final_capacity)

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """
        获取单条文本的嵌入向量

        :param text: 待编码文本
        :return: Embedding 向量，或 None 表示失败
        """
        max_retries = 3

        for attempt in range(max_retries):
            await self.limiter.acquire()  # 限速

            try:
                response = await self.client.embeddings.create(
                    model="text-embedding-v4",
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding

            except OpenAIError as e:
                if attempt < max_retries - 1:
                    print(f"第 {attempt + 1} 次请求失败，错误信息: {e}，正在进行重试...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"第 {attempt + 1} 次请求失败，错误信息: {e}，已达到最大重试次数。")
        return None

if  __name__ == "__main__":
    async def main():
        vector_transfer = VectorTransfer()
        embedding = await vector_transfer.get_embedding("Hello, World!")
        print(embedding)

    asyncio.run(main())