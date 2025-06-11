from agent.Base import Agent


class ClassifierAgent(Agent):
    NAME = "分类器"
    DESC = "调用Action[问题分类]，根据问题的类型，选择合适的智能体回答问题"

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)






