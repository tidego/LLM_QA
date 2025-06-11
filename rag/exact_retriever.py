import asyncio
import pickle
import re
import time
from abc import ABC, abstractmethod
from typing import List, Union, Any, Tuple
from config.Config import CHUNK_DB_PATH
from utils.Logger import Logger


class BaseExactMatcher(ABC):
    """
    精确匹配器的基类，所有子类需要实现 _match_single 方法
    """

    def __init__(self, chunks: List[str]):
        """
        初始化匹配器，传入待匹配的文本列表(文本块）
        :param documents:
        """
        self.chunks = chunks
        self.logger = Logger()
        self._last_retrieve_time = None  # 记录上一次检索时间（秒）

    @abstractmethod
    def _match_single(self, query: str) -> Any:
        """
        子类需实现的单条精确匹配逻辑
        """
        pass

    def _timed_single(self, query: str) -> Any:
        """
        单条执行检索并记录耗时（秒）
        """
        start = time.perf_counter()
        result = self._match_single(query)
        end = time.perf_counter()
        self._last_match_time = end - start
        return result

    async def retrieve(self, query: Union[str, List[str]]) -> List[str | bool]:
        """
        查询函数，支持单条（同步）与多条（异步）输入
        :param query: str 或 List[str]
        :return: List[str | bool]
        """
        if isinstance(query, str):
            return self._timed_single(query)

        elif isinstance(query, list):
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, self._match_single, q) for q in query]
            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            end = time.perf_counter()
            self._last_retrieve_time_list = end - start
            return results

        else:
            raise TypeError("query 必须为 str 或 List[str]")

    @property
    def retrieve_time(self) -> float:
        """
        获取最近一次检索耗时（秒）
        :return:
        """

        return self._last_match_time

    @property
    def retrieve_time_list(self) -> float:
        """
        获取最近一次批量查询耗时（秒）
        :return:
        """

        return self._last_retrieve_time_list


class SlidingSubstringMatcher(BaseExactMatcher):
    def __init__(
            self,
            chunks: List[str],
            window_ratio: float = 0.5,
            step_ratio: float = 0.125
    ):
        """
        滑动窗口子串匹配器，用于在文本中查找子串
        :param chunks:文本块列表
        :param window_ratio:滑动窗口的比例，取值范围[0,1]
        :param step_ratio:步长的比例，取值范围[0,1]
        """
        super().__init__(chunks)
        self.window_ratio = window_ratio
        self.step_ratio = step_ratio

    def _clean_text(self, text: str) -> str:
        """
        文本清洗，去除标点符号、空格和转义字符
        :param text:
        :return:
        """
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)

    def _generate_substrings(self, text: str) -> List[str]:
        """
        生成子串列表，根据窗口大小和步长生成
        :param text:
        :return:
        """
        text = self._clean_text(text)
        n = len(text)
        if n == 0:
            return []

        win_size = max(1, int(n * self.window_ratio))
        step_size = max(1, int(n * self.step_ratio))
        substrings = []

        for i in range(0, min(n - win_size + 1, step_size * 4), step_size):
            substrings.append(text[i:i + win_size])
            if len(substrings) == 4:
                break

        return substrings

    def _match_single(self, query: str) -> tuple[str, bool]:
        """
        判断查询子串是否存在于任意文本块中
        :param query: 查询字符串
        :return: 是否匹配
        """
        substrings = self._generate_substrings(query)
        for chunk in self.chunks:
            clean_chunk = self._clean_text(chunk)
            if any(sub in clean_chunk for sub in substrings):
                return chunk, True
        return chunk, False




if __name__ == "__main__":
    with open(CHUNK_DB_PATH, "rb") as f:
        chunks = pickle.load(f)

    matcher = SlidingSubstringMatcher(chunks)


    async def run():
        # 单条
        query = "叠加原理是指什么？"
        _,result = await matcher.retrieve(query)
        print(f"\n[单条] 匹配结果: {result}，耗时: {matcher.retrieve_time:.4f} 秒")

        # 多条
        querys = ["叠加原理是什么？",

                  "以下措施（）可作为减少接触电压与跨步电压的措施。A: 接地网的布置应尽量使电位分布均匀，B:设置临时遮拦，C:增设水平均压带,D:带安全帽",

                  "在爆炸性气体危险区域为1区的场所内，电缆线路（）有中间接头。A.—般不得；B严禁；C.不宜；D.可以"
                  ]
        results = await matcher.retrieve(querys)
        print(f"\n[多条] 匹配耗时: {matcher.retrieve_time_list:.4f} 秒")
        for q, r in zip(querys, results):
            print(f"查询：{q}\n匹配：{r[0]}\n")


    asyncio.run(run())
