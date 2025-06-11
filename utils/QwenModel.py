import os
import asyncio
from typing import Optional, List, Union
from dashscope import Generation
from aiolimiter import AsyncLimiter
from asyncio import Semaphore
from dashscope.common.error import TimeoutException, ServiceUnavailableError, DashScopeException
from config.Config import DASHSCOPE_API_KEY
from utils.Logger import Logger
from tqdm.asyncio import tqdm_asyncio


class AsyncQwenModel:
    """
    异步Qwen模型类，支持流式输出和非流式输出,支持reasoning。
    """

    def __init__(self,
                 model="qwen-turbo",
                 timeout=1000,
                 rate_limit_per_second=6,
                 concurrency=5,
                 stream: bool = False,
                 enable_thinking: bool = False,
                 **kwargs):
        """
        初始化Qwen模型，支持流式输出和非流式输出。
        :param model: 模型名称，默认为"qwen-turbo"
        :param timeout: 超时时间，默认为1000秒
        :param rate_limit_per_second: 速率限制，默认为3
        :param concurrency: 并发数，默认为5
        :param stream: 流式输出，默认为False
        :param enable_thinking: 思考模式，默认为False,仅DeepSeek、Qwen3系列模型支持
        :param kwargs: 模型API参数，如temperature、top_p等,也可以在调用时传入
        """
        self.model = model
        self.timeout = timeout
        self.stream = stream
        self.enable_thinking = enable_thinking
        self.api_key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        self.rate_limiter = AsyncLimiter(max_rate=rate_limit_per_second, time_period=1)  # 限速器，每秒允许3个请求
        self.semaphore = Semaphore(concurrency)  # 信号量，控制并发数
        self.logger = Logger()
        self.extra_params = kwargs

    async def get_response(self, prompt: Union[str, List], **kwargs) -> Optional[str]:
        """
        异步获取模型响应。
        如果发生错误，会自动重试3次。
        :param prompt:
        :param kwargs:
        :return:
        """
        retries = 3
        for attempt in range(retries):
            try:
                async with self.rate_limiter:
                    async with self.semaphore:
                        return await asyncio.wait_for(self._send(prompt, **kwargs), timeout=self.timeout)
            except (ServiceUnavailableError, TimeoutException, DashScopeException) as e:
                self.logger.error(f"API 调用时出错: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

    async def _send(self, messages: Union[str, List], only_content=True, stream_callback=print, **kwargs) -> Optional[
        str]:
        """
        发送请求到DashScope API。
        :param messages:文本消息列表，每个消息是一个字典，包含role和content字段。
        :param only_content:如果为True，只返回content字段，否则返回整个响应。
        :param kwargs:其他参数，如temperature、top_p等。
        :return:
        """
        all_kwargs = {
            **self.extra_params,
            **kwargs
        }
        loop = asyncio.get_event_loop()

        def call_api():
            return Generation.call(
                model=self.model,
                api_key=self.api_key,
                messages=messages,
                result_format='message',
                stream=self.stream,
                **all_kwargs
            )

        response = await loop.run_in_executor(None, call_api)

        if self.stream:
            resp = {"reasoning_content": "", "answer_content": ""}
            prev_reasoning = ""
            prev_content = ""

            for chunk in response:
                try:
                    choice = chunk.get("output", {}).get("choices", [{}])[0]
                    reasoning = choice.get("message", {}).get("reasoning_content", "")
                    content = choice.get("message", {}).get("content", "")

                    # reasoning 部分增量输出
                    if reasoning:
                        delta = reasoning[len(prev_reasoning):]
                        if delta:
                            stream_callback(delta)
                            resp["reasoning_content"] += delta
                            prev_reasoning = reasoning

                    # content 部分增量输出
                    if content:
                        delta = content[len(prev_content):]
                        if delta:
                            stream_callback(delta)
                            resp["answer_content"] += delta
                            prev_content = content

                except Exception as e:
                    self.logger.warning(f"chunk结构异常：{chunk}，错误：{e}")
                    continue

            if self.enable_thinking:
                return resp if only_content else response
            else:
                return resp["answer_content"] if only_content else response

        if not response.output:
            raise Exception("DashScope空输出。")
        return response.output.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def batch_response(self, prompts: List[dict], **kwargs) -> List[Optional[str]]:
        """
        批量处理请求。
        :param prompts: 请求列表，每个列表内是字典
        :param kwargs:
        :return:
        """
        tasks = [self.get_response(prompt, **kwargs) for prompt in prompts]
        try:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc="处理进度",
                unit="条",
                total=len(tasks),
            )
        except Exception as e:
            self.logger.error(f"批量响应 gather 异常：{e}")
            raise

        return results

    @staticmethod
    def format_messages(prompt: Union[str, List]) -> List[dict]:
        if isinstance(prompt, str):
            """
            格式化用户消息,将其转换为DashScope API支持的消息格式。
            :param prompt: 用户输入Prompt列表，默认每个列表的Prompt转为User消息。
            :return: 格式化后的消息列表
            """

            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            formatted = []
            for item in prompt:
                if isinstance(item, str):
                    formatted.append([{"role": "user", "content": item}])
                elif isinstance(item, dict):
                    formatted.append([item])
                elif isinstance(item, list):
                    formatted.append(item)
                else:
                    raise ValueError(f"不支持的消息格式： {item}")
            return formatted
        else:
            raise ValueError("Prompt 必须是一个字符串或字符串/dict 列表.")
