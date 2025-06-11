from agent.Base import Agent
class SingleChoiceAgent(Agent):
    NAME = "单选回答器"
    DESC = "回答单选问题"
    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)