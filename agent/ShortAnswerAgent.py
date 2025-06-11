from agent.Base import Agent


class ShortAnswerAgent(Agent):
    NAME = "简答器"
    DESC = "回答简答问题"

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)
