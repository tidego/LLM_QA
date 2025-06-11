from typing import Union, List, Iterable, Any
from action.Base import Action
from collections import deque
from utils.Logger import Logger


class Agent:
    def __init__(self, name: str, desc: str = None, actions: list = None, memory_size: int = 20):
        self.name = name
        self.desc = desc
        self.actions = deque(actions or [])
        self.memory = Memory(memory_size)  # 长期记忆
        self.logger = Logger()
        self.is_idle = True
        self._send_memory_callback = None  # 注册环境通信回调

    @property
    def current_action(self):
        """
        获取当前动作
        :return:
        """
        return self.actions[0] if self.actions else None

    @property
    def current_actions(self):
        """
        获取当前动作列表
        :return:
        """
        return list(self.actions)

    def add_action(self, action: Union[Action, List[Action]], to_front: bool = False):
        """
        添加一个新动作，to_front=True 插入队首（优先执行）
        :param action:
        :param to_front:
        :return:
        """
        if isinstance(action, Iterable) and not isinstance(action, Action):
            for act in action:
                self.add_action(act, to_front=to_front)
        else:
            if to_front:
                self.actions.appendleft(action)
            else:
                self.actions.append(action)

    def publish_memory(self) -> list[dict]:
        """
        发布当前记忆
        :return:
        """
        return self.memory.get_all()


    def register_sender(self, send_memory_callback):
        """
        注册环境通信回调，用于发送记忆给其他智能体
        :param send_memory_callback:
        :return:
        """
        self._send_memory_callback = send_memory_callback

    def send_memory_to(self, target_agent: 'Agent', persistent: bool = False):
        """
        调用环境的 send_memory_to 方法，将记忆发送给目标智能体
        :param target_agent:
        """
        if self._send_memory_callback:
            memory = self.publish_memory()
            self._send_memory_callback(target_agent, memory, persistent)
        else:
            raise Exception("请先注册环境通信回调")

    async def run(self, *args, **kwargs) -> None:
        """
        执行当前所有动作
        :param args:
        :param kwargs:
        :return:
        """
        if args or kwargs:
            self.logger.warning(
                f"\033[33mAgent {self.name} 收到了额外的参数：{args} {kwargs},请确认是否重写Action的execute"
                f"方法，如果想要添加动作并下一轮执行，请使用add_run方法。\033[0m")
        self.is_idle = False
        while self.actions:
            await self.run_step(*args, **kwargs)
        self.is_idle = True

    async def run_step(self, *args, **kwargs):
        """只执行一个动作"""
        if self.actions:
            self.is_idle = False
            action = self.actions.popleft()
            result = await action.execute(*args, **kwargs)
            self.memory.add({"action": action.name, "result": result})
            self.logger.info(f"\033[96mAgent {self.name} 执行了动作 {action.name} 用时 {action.time:.4f}秒\033[0m")
            self.logger.info(f"\033[94m结果为：\n {result}\033[0m")
            self.is_idle = True
        if not self.actions:
            self.is_idle = True

    async def add_run(self, action: Union[Action, List[Action]], *args, **kwargs) -> None:
        """
        添加动作或动作列表并立即执行，不影响原有动作队列
        :param action: 单个 Action 或多个 Action
        """
        self.is_idle = False

        # 判断是否为可迭代的动作列表
        new_actions = list(action) if isinstance(action, Iterable) and not isinstance(action, Action) else [action]

        self.logger.info(f"\033[96mAgent {self.name} 添加了 {len(new_actions)} 个动作并立即执行：\033[0m")

        for act in new_actions:
            result = await act.execute(*args, **kwargs)
            self.memory.add({"action": act.name, "result": result})
            self.logger.info(f"\033[96mAgent {self.name} 执行了动作 {act.name} 用时 {act.time:.4f}秒\033[0m")
            self.logger.info(f"\033[94m结果为：\n{result}\033[0m")

        self.is_idle = True

    def get_last_memory(self) -> Any:
        """
        获取最近一次执行的动作和结果
        :return:
        """
        return self.memory.peek()

    def get_last_result(self) -> Any:
        """
        获取最近一次执行的动作的结果
        :return:
        """
        return self.memory.peek()["result"]


    def get_last_k_memory(self, k: int = 1) -> list[dict]:
        """
        获取最近k次执行的动作和结果
        :param k:
        :return:
        """
        return self.memory.get_all()[-k:]

    def get_last_k_result(self, k: int = 1) -> list[dict]:
        """
        获取最近k次执行的动作和结果
        :param k:
        :return:
        """

        return [memory["result"]for memory in self.memory.get_all()[-k:]]
class Memory:
    def __init__(self, max_size: int = 50):
        """
        使用 deque 实现栈结构，自动清除旧条目
        :param max_size:
        """

        self._memory = deque(maxlen=max_size)

    def add(self, item: dict) -> None:
        """
        添加记忆条目（后进先出）
        :param item:
        :return:
        """
        self._memory.append(item)  # 加到尾部，即栈顶

    def pop(self) -> dict:
        """
        弹出最近一条记忆（栈顶）
        :return:
        """
        return self._memory.pop() if self._memory else None

    def peek(self) -> dict:
        """
        查看最近一条记忆（不移除）
        :return:
        """
        return self._memory[-1] if self._memory else None

    def get_all(self) -> list[dict]:
        """
        获取所有记忆（从旧到新）
        :return:
        """
        return list(self._memory)

    def clear(self) -> None:
        """
        清空记忆
        :return:
        """
        self._memory.clear()

    def __len__(self):
        return len(self._memory)

    def __repr__(self):
        return f"Memory(size={len(self._memory)}, max_size={self._memory.maxlen})"
