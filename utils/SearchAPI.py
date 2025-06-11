import warnings

warnings.filterwarnings("ignore", message=".*protected namespace.*")  # 这是第三方库内部警告，忽略即可
import asyncio
import os
from typing import Optional
from appbuilder.mcp_server.client import MCPClient
from config.Config import BAIDU_AppBuilder_API


class AppBuilderSearch:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BAIDU_AppBuilder_API") or BAIDU_AppBuilder_API
        if not self.api_key:
            raise ValueError("请配置 BAIDU_AppBuilder_API")
        self.client = MCPClient()
        self.service_url = (
            "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?api_key=Bearer+<AppBuilder API Key>"
        ).replace("<AppBuilder API Key>", self.api_key)
        self.connected = False

    async def connect(self):
        if not self.connected:
            await self.client.connect_to_server(service_url=self.service_url)
            self.connected = True
            print(f"已连接工具: {[tool.name for tool in self.client.tools]}")

    async def search(self, query: str):
        if not self.connected:
            await self.connect()
        result = await self.client.call_tool("AIsearch", {"query": query})
        return result

    async def run(self, query: str) -> str:
        """
        调用工具搜索并返回第结果文本
        :param query:
        :return:
        """
        result = await self.search(query)
        if result and result.content and len(result.content) > 0:
            return result.content[0].text
        else:
            return "[未获取到有效内容]"


if __name__ == "__main__":
    async def main():
        searcher = AppBuilderSearch()
        answer = await searcher.run("成都美食有哪些")
        print(answer)


    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
