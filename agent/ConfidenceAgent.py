from agent.Base import Agent


class ConfidenceAgent(Agent):
    NAME = "置信评估器"
    DESC = "根据问题类型和知识库的匹配程度，评估当前智能体的置信水平"

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)
