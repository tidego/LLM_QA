from agent.Base import Agent


class MultiChoiceAgent(Agent):
    NAME = "多选回答器"
    DESC = "回答多选问题"

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)
