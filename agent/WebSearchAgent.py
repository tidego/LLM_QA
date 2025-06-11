from agent.Base import Agent


class WebSearchAgent(Agent):
    NAME = "Web搜索器"
    DESC = "根据问题，从互联网上搜索答案"

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)
