import time

from action.Base import Action
from utils.SearchAPI import AppBuilderSearch


class WebSearch(Action):
    NAME = "Web搜索器"
    DESC = "根据问题，从互联网上搜索答案"
    SEARCH = AppBuilderSearch()

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)

    async def execute(self, *args, **kwargs):
        """
        重写execute方法，实现Web搜索，搜索是要将Prompt作为关键词传入
        :param args:
        :param kwargs:
        :return:
        """
        start = time.perf_counter()
        result = await self.SEARCH.run(self.prompt)
        end = time.perf_counter()
        self._last_action_time = end - start
        self._last_result = result
        return result

    def query(self, query: str) -> "WebSearchAgent":
        """
        构建prompt
        :param query:
        :return:
        """
        self.prompt = query
        return self

