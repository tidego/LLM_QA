import asyncio
import pickle
import time
from abc import ABC, abstractmethod
from typing import List, Union
from rank_bm25 import BM25Okapi
from config.Config import TOP_K, TOKENIZED_DB_PATH, BM25_MODEL_PATH, CHUNK_DB_PATH
from utils.Logger import Logger
import numpy as np
import jieba


class BaseFuzzyRetriever(ABC):
    """
    模糊检索，相当于精确检索更具鲁棒性，比如“压与跨步电压的措施。A: 接地网的布置应尽量使电位分布均匀，B:设置临时遮拦，C:增设水平均压带,D:带安全帽”这种残缺的问题，精确检索为失败，而模糊检索Top1分数为1
    """

    def __init__(self, chunks: List[str], do_tokenize: bool = True):
        """
        初始化检索器
        :param chunks: 原始文本块
        :param tokenized_chunks: 如果提供，则使用外部分词结果
        :param do_tokenize: 是否需要进行分词（若为False则需提供tokenized_chunks）
        """
        if not do_tokenize:
            # 传入的 chunks 实际是 tokenized_chunks，合并为可读字符串
            self.tokenized_chunks = chunks
            self.chunks = ["".join(tokens) for tokens in chunks]
        else:
            self.chunks = chunks
            self.tokenized_chunks = [self._tokenize(doc) for doc in chunks]

        self.retriever = self._init_retriever()  # 初始化检索器
        self._last_retrieve_time = None  # 检索耗时
        self._last_retrieve_time_list = None  # 批量检索耗时
        self.logger = Logger()  # 日志

    def _tokenize(self, text: str) -> List[str]:
        """
        分词器，可根据需要替换为其他分词器
        :param text:
        :return:
        """
        return list(jieba.cut(text))

    @abstractmethod
    def _init_retriever(self) -> object:
        """
        必须由子类实现，返回一个具有 get_scores 方法的检索器
        """
        pass

    def _retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[tuple[str, float]]:
        """
        检索函数，返回与查询最相关的文本块及其 Softmax 分数
        :param query: 查询文本
        :param top_k: 返回条数
        :param score_threshold: 最小分数阈值
        :return: List of (text, softmax_score)
        """
        tokens = self._tokenize(query)
        scores = self.retriever.get_scores(tokens)

        # Softmax 归一化
        scores_np = np.array(scores)
        exp_scores = np.exp(scores_np - np.max(scores_np))  # 防溢出
        softmax_scores = exp_scores / np.sum(exp_scores)

        # 排序 + 过滤
        ranked_indices = sorted(range(len(scores)), key=lambda i: softmax_scores[i], reverse=True)
        results = []
        for i in ranked_indices:
            if softmax_scores[i] >= score_threshold:
                results.append((self.chunks[i], softmax_scores[i]))
            if len(results) >= top_k:
                break

        return results

    def _timed_single(self, query: str, top_k: int, score_threshold: float) -> List[str]:
        """
        执行检索并记录耗时（秒）,其余参数与retrieve一致
        :param query:
        :param top_k:
        :param score_threshold:
        :return:
        """
        start = time.perf_counter()
        results = self._retrieve(query, top_k, score_threshold)
        end = time.perf_counter()
        self._last_retrieve_time = end - start
        return results

    async def retrieve(
            self, query: Union[str, List[str]], top_k: int = 5, score_threshold: float = 0.0
    ) -> Union[List[str], List[List[str]]]:
        """
        支持同步查询（单条字符串）与异步批量查询（字符串数组）
        """
        if isinstance(query, str):
            return self._timed_single(query, top_k, score_threshold)

        elif isinstance(query, list):
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, self._timed_single, q, top_k, score_threshold)
                for q in query
            ]
            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            end = time.perf_counter()
            self._last_retrieve_time_list = end - start
            return results

        else:
            raise TypeError("query 必须是 str 或 List[str]")

    @property
    def retrieve_time(self) -> float:
        """
        获取最近一次 timed_retrieve 的耗时（秒）
        """
        return self._last_retrieve_time

    @property
    def retrieve_time_list(self) -> float:
        """
        获得多条查询的总耗时（秒）
        :return:
        """
        return self._last_retrieve_time_list


class BM25FuzzyRetriever(BaseFuzzyRetriever):
    """
    BM25 是一种基于词频与文档长度归一化的打分排序算法，属于信息检索中“概率相关模型（Probabilistic Retrieval Model）”的一种，建模查询词与相关文档和相关文档的概率关系，它的核心思想是：文档中出现越多查询词（词频），且词越罕见（逆文档频率），该文档相关性越高，但也需考虑文档长度的影响。
    不过使用BM25进行检索时，一些“是指什么”的后缀会对检索结果产生影响，这是因为BM25词粒度切分很敏感，使用时用过前去除这些后缀，如“是指什么”等。
    """

    def _init_retriever(self):
        return BM25Okapi(self.tokenized_chunks)


class PrebuiltBM25FuzzyRetriever(BM25FuzzyRetriever):
    """
    使用预构建的 BM25 模型进行检索，加载模型和原始 chunks，无需重新训练。
    """

    def __init__(self, bm25_model_path: str):
        """
        直接从 pickle 文件加载 BM25 模型与原始 chunks
        :param bm25_model_path: 保存模型的路径
        """
        with open(bm25_model_path, "rb") as f:
            bundle = pickle.load(f)
            self.retriever = bundle["model"]
            self.chunks = bundle["chunks"]

        self._last_retrieve_time = None
        self._last_retrieve_time_list = None
        self.logger = Logger()

    def _init_retriever(self):
        return self.retriever


if __name__ == "__main__":


    retriever = PrebuiltBM25FuzzyRetriever(BM25_MODEL_PATH)


    async def run():
        # 单条
        query = "叠加原理是什么？"
        results = await retriever.retrieve(query, top_k=TOP_K)
        print("result[0][1]:", results[0][1])
        print("result[1]:", results[1])
        print(f"\n[单条] 检索耗时: {retriever.retrieve_time:.4f} 秒")
        for text, score in results:
            print(f"{score:.4f} | {text}")

        # 多条
        querys = ["叠加原理是什么？",

                  "以下措施（）可作为减少接触电压与跨步电压的措施。A: 接地网的布置应尽量使电位分布均匀，B:设置临时遮拦，C:增设水平均压带,D:带安全帽",

                  "在爆炸性气体危险区域为1区的场所内，电缆线路（）有中间接头。A.—般不得；B严禁；C.不宜；D.可以"
                  ]
        results = await retriever.retrieve(querys, top_k=TOP_K)
        print(f"\n[多条] 总耗时: {retriever.retrieve_time_list:.4f} 秒")
        for i, rset in enumerate(results):
            print(f"\n查询{i + 1}:")
            for text, score in rset:
                print(f"{score:.4f} | {text}")


    asyncio.run(run())
