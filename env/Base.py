import asyncio
from typing import List
from agent.Base import Agent
from utils.Logger import Logger


class Environment:
    def __init__(self):
        self.agents: List[Agent] = []
        self.logger = Logger()

    def add_agent(self, agent: Agent):
        """
        添加一个智能体到环境
        :param agent:
        :return:
        """
        agent.register_sender(self.send_memory_to)
        self.agents.append(agent)
        self.logger.info(f"[Env] 添加智能体：{agent.name}")

    def get_agent(self, agent_name: str) -> Agent:
        """
        获取指定名称的智能体
        :param agent_name:
        :return:
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent

    def register_agent(self, agent: Agent):
        """
        注册智能体（等价于 add_agent）,别名
        :param agent:
        :return:
        """
        self.add_agent(agent)

    def broadcast_memory(self, sender: Agent, persistent: bool = False):
        """将 sender 的记忆广播给其他 agent"""
        for agent in self.agents:
            if agent != sender:
                sender.sent_to(agent, self, persistent=persistent)

    def send_memory_to(self, target_agent: Agent, memory: List[dict], persistent: bool = False):
        """将记忆传给目标 agent"""
        self.logger.info(f"[Env] 向 {target_agent.name} 发送记忆：{memory}")
        if persistent:
            for item in memory:
                target_agent.memory.add(item)

    def show_status(self):
        """展示所有 Agent 状态"""
        self.logger.info("[Agent 状态]")
        for agent in self.agents:
            status = "空闲" if agent.is_idle else "忙碌"
            self.logger.info(f"\033[32m{agent.name}: {status}\033[0m")

    def all_idle(self) -> bool:
        """
        判断环境中所有 Agent 是否都处于空闲状态
        :return:
        """
        return all(agent.is_idle for agent in self.agents)

    async def run(self):
        """单步运行每个 Agent（只执行一个动作）"""
        for agent in self.agents:
            if agent.actions:
                await agent.run_step()
                self.logger.info(f"[Env] {agent.name} 执行了动作{agent.current_action}")

    async def run_all(self):
        """逐个运行每个 Agent的所有动作"""
        for agent in self.agents:
            if agent.actions:
                await agent.run()
                self.logger.info(f"[Env] {agent.name} 执行了动作{agent.current_action}")

    def clear(self):
        self.agents = []
        self.logger.info("[Env] 所有 agent 已清空。")
