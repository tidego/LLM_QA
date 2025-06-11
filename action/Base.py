import time
from typing import Union, Any, Optional, Callable
from utils.Logger import Logger
from utils.QwenModel import AsyncQwenModel


class Action():
    """
    动作类，定义了一个动作的基本结构，包括名称和描述。
    """

    def __init__(self, name: str = None, desc: str = None, prompt: str = None, **kwargs):
        self.name = name
        self.desc = desc
        self.model = AsyncQwenModel(**kwargs)
        self.logger = Logger()
        self.prompt = prompt
        self._last_action_time = None  # 记录上一次动作时间（秒）
        self._last_result = None  # 记录上一次动作的结果

    async def execute(self, *args, **kwargs) -> Any:
        """默认的执行逻辑，子类应该重写此方法。"""
        if self.prompt:
            result = await self.timed_ask(self.prompt)
            self._last_result = result
            return result
        else:
            self.logger.warning("未定义prompt，execute 没有执行任何操作。")
            return None

    def update_prompt(self, prompt: str):
        """
        更新prompt
        :param prompt:
        :return:
        """
        self.prompt = prompt
        return self

    async def ask(self, prompts: Union[str, list, dict], stream_callback: Optional[Callable[[str], None]] = None,
                  **kwargs) -> list[str | None] | str | None:
        """
        实现action的self.ask方法，参考MetaGPT的_ask方法:
        :param prompts:
        :param stream_callback
        :return:
        """
        self.logger.info(f"\033[96m当前执行的 Action: 【{self.name}】\033[0m")
        try:
            messages = self.model.format_messages(prompts)
            stream_callback = stream_callback or self.default_stream_printer
            if len(messages) == 1:
                return await self.model.get_response(messages, stream_callback=stream_callback, **kwargs)
            else:
                return await self.model.batch_response(messages, stream_callback=stream_callback, **kwargs)
        except Exception as e:
            self.logger.warning(f"Error: {e}")
            return None

    async def timed_ask(self, prompts: Union[str, list, dict], **kwargs) -> list[
                                                                                str | None] | str | None:
        """
        单条执行动作并记录耗时（秒）
        :param stream_callback:
        :param self:
        :param prompts:
        :return:
        """
        start = time.perf_counter()
        result = await self.ask(prompts, **kwargs)
        end = time.perf_counter()
        self._last_action_time = end - start
        return result

    @property
    def time(self) -> float:
        """
        获取上一次动作的耗时（秒）
        :return:
        """
        return self._last_action_time

    @property
    def result(self) -> Any:
        """
        获取上一次动作的结果
        :return:
        """
        return self._last_result

    @staticmethod
    def default_stream_printer(text: str):
        print(f"\033[95;1m{text}\033[0m", end="", flush=True)


if __name__ == '__main__':
    import asyncio


    async def main():
        action = Action(name="测试动作", desc="测试输入的各种格式", stream=True)

        # 测试 prompt：字符串输入
        prompt1 = "什么是叠加原理？"

        # 测试 prompt：列表字符串输入
        prompt2 = ["什么是叠加原理？", "动态电阻（或称交流电阻）表示的是什么？"]

        # 测试 prompt：对话历史格式
        prompt3 = [
            {"role": "system", "content": "什么是叠加原理。"},
            {"role": "user", "content": "动态电阻（或称交流电阻）表示的是什么？"}
        ]

        # 测试 prompt：多个对话作为批量输入
        prompt4 = [
            [{"role": "user", "content": "什么是叠加原理"}],
            [{"role": "user", "content": "动态电阻（或称交流电阻）表示的是什么？"}]
        ]

        print("测试单条输入：")
        result1 = await action.timed_ask(prompt1)
        print("返回结果:", result1)
        print("耗时:", action.time)
        print("\n\n")
        print("\n测试列表输入：")
        result2 = await action.timed_ask(prompt2, max_tokens=10)
        print("耗时:", action.time)
        print("返回结果:", result2)
        print("\n\n")
        print("\n测试对话格式输入：")
        result3 = await action.timed_ask(prompt3, top_k=5)
        print("耗时:", action.time)
        print("返回结果:", result3)
        print("\n\n")
        print("\n测试批量对话输入：")
        result4 = await action.timed_ask(prompt4, temperature=0.8)
        print("耗时:", action.time)
        print("返回结果:", result4)


    asyncio.run(main())
