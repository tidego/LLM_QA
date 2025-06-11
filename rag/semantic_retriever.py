"""
    FAISS 检索器，语义向量召回
"""
import asyncio
import json
import time
from typing import Union, Optional
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.Logger import Logger
from config.Config import Base_DIR, TOP_K
from utils.VectorTransfer import VectorTransfer


class FAISSRetriever:
    def __init__(
            self,
            vecdb_path: Path,
            embedding_dim: int = 1024,
            index_type: str = "flat",  # flat/flat_ip, ivf, ivfpq, hnsw, lsh
            use_gpu: bool = False
    ) -> None:
        """
        初始化检索器，支持多种 FAISS 索引类型。

        :param vecdb_path: 向量数据库路径（每行为 {"text": ..., "embedding": ...}）
        :param embedding_dim: 向量维度（例如 BGE-Large 为 1024，text-embedding-3-large 为 1536）
        :param index_type: 索引类型，支持以下几种：

            1. "flat/flat_ip"（精确检索 / 暴力检索）：
                - 原理：将所有向量存入内存，计算每个查询与所有向量的距离（L2 或 inner product），返回最相似者。
                - 优点：结果完全准确，性能稳定。
                - 缺点：速度慢，内存消耗大。
                - 适用：小型知识库（<50k 条），或需要精确检索的任务。

            2. "ivf"（倒排索引 IVF: Inverted File）：
                - 原理：先用 kmeans 对向量聚类，将向量分配到多个“桶”中；查询时只在前几个相关桶中搜索。
                - 优点：比 flat 快很多，占用内存少。
                - 缺点：可能会略微牺牲精度（依赖 nlist 聚类数 和 nprobe 检索桶数）。
                - 适用：中等规模数据集（50k ~ 5M），对精度容忍度较高的检索任务。

            3. "ivfpq"（IVF + Product Quantization）：
                - 原理：在 IVF 基础上，进一步压缩每个向量（分块 + 量化编码）。
                - 优点：大大减少内存占用，可扩展到千万量级数据。
                - 缺点：需要训练（训练时间较长），精度受影响。
                - 适用：超大规模数据集（百万以上），对内存或速度有较高要求的场景。

            4. "hnsw"（图索引：Hierarchical Navigable Small World）：
                - 原理：构建一个多层图结构，通过图遍历高效找到近似最近邻。
                - 优点：高精度、高性能；不需要训练。
                - 缺点：构建时间略久，内存略大。
                - 适用：中等到大规模数据集（10k~百万），推荐首选近似检索方法。

            5. "lsh"（局部敏感哈希：Locality-Sensitive Hashing）：
                - 原理：将高维向量通过哈希函数映射到低维空间，近似保留相似度。
                - 优点：实现简单，训练速度快。
                - 缺点：精度较低，一般不建议用于精确召回任务。
                - 适用：对精度要求较低、速度要求极高的场景（如初步粗筛）。

        :param use_gpu: 是否启用 GPU 加速（需安装 faiss-gpu 版本），仅对部分索引类型支持（如 flat, ivf）
        """

        self.vecdb_path = vecdb_path  # 向量数据库路径
        self.embedding_dim = embedding_dim  # 向量维度
        self.index_type = index_type.lower()  # 索引类型
        self.use_gpu = use_gpu  # 是否使用GPU加速
        self.logger = Logger()  # 日志记录器
        self.index = None  # 索引对象
        self.texts = []  # 文本列表
        self.embeddings = None  # 嵌入向量
        self._last_retrieve_time = None  # 记录单条检索时间
        self._last_retrieve_time_list = None  # 记录批量检索时间

        self.index_path = Base_DIR / "vector_db" / f"{vecdb_path.stem}_{self.index_type}.index"
        # 检查index_type是否合法
        if self.index_type not in ["flat", "flat_ip", "ivf", "ivfpq", "hnsw", "lsh"]:
            raise ValueError(f"未知索引类型: {self.index_type}")

        if self.index_path.exists():
            self._load_index()
        else:
            self.index = self._build_index()
            self._load_data()

            if not self.index.is_trained:
                self.index.train(self.embeddings)
                self.logger.info("索引已训练完成")

            self._save_index()

    def _load_index(self) -> None:
        """
        加载已有FAISS索引和对应的文本。
        """
        self.index = faiss.read_index(str(self.index_path))
        self.logger.info(f"已加载索引文件：{self.index_path}")
        self._load_texts_only()

    def _load_texts_only(self) -> None:
        """
        仅加载文本内容，与索引顺序保持一致。
        """
        texts = []
        with open(self.vecdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'text' in data:
                    texts.append(data['text'])
        self.texts = texts

    def _load_data(self) -> None:
        """
        加载嵌入向量与文本
        :return:
        """
        embeddings = []
        texts = []

        with open(self.vecdb_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="正在加载向量数据库"):
                data = json.loads(line.strip())
                if 'text' in data and 'embedding' in data:
                    texts.append(data['text'])
                    vec = np.array(data['embedding'], dtype='float32')
                    if self.index_type == "flat_ip":
                        vec /= np.linalg.norm(vec) + 1e-10
                    embeddings.append(vec)

        self.texts = texts
        self.embeddings = np.vstack(embeddings).astype('float32')

        if not self.index.is_trained:
            self.logger.info("索引未训练，正在训练中...")
            self.index.train(self.embeddings)
            self.logger.info("索引已训练完成")

        self.index.add(self.embeddings)
        self.logger.info(f"{len(self.texts)} 个数据块已建立索引")

    def _save_index(self) -> None:
        """
        保存FAISS索引到自动推断的路径。
        """
        self.logger.warning(f"确保当前路径不包含中文，否则会报错,当前路径为：{self.index_path.parent}")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.logger.info(f"索引已保存至：{self.index_path}")

    def _build_index(self):
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "flat_ip":
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            index.nprobe = 10
        elif self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, 100, 16, 8)
            index.nprobe = 10
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        elif self.index_type == "lsh":
            index = faiss.IndexLSH(self.embedding_dim, 128)
        else:
            raise ValueError(f"未知索引类型: {self.index_type}")

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                self.logger.info("已启用GPU索引")
            except Exception as e:
                self.logger.warning(f"GPU 初始化失败，回退至 CPU: {e}")

        return index

    async def retrieve(
            self,
            query: Union[str, list[str]],
            top_k: int = 5,
            score_threshold: float = 0.0,
            vector_transfer: Optional[VectorTransfer] = None,
    ) -> Union[tuple[str, float], list]:
        """
        根据查询（单条或多条）返回相似文本和相似度

        :param query: 单条字符串或字符串列表
        :param top_k: 返回的最大相似文本条数
        :param score_threshold: 最低相似度阈值（距离越小，相似度越高）
        :param vector_transfer: 外部传入的 VectorTransfer 实例
        :return: 检索结果，格式为(text, score),或[(text, score),...]
        """
        vector_transfer = vector_transfer or VectorTransfer()

        if isinstance(query, str):
            start = time.perf_counter()
            embedding = await vector_transfer.get_embedding(query)
            if embedding is None:
                raise ValueError("向量化失败，请检查当前网络环境")
            vec = np.array([embedding], dtype='float32')
            if self.index_type == "flat_ip":
                vec /= np.linalg.norm(vec) + 1e-10
            result = self._search_by_vector(vec, top_k, score_threshold)
            self._last_retrieve_time = time.perf_counter() - start
            return result

        elif isinstance(query, list):
            start = time.perf_counter()
            tasks = [vector_transfer.get_embedding(q) for q in query]
            embeddings = await asyncio.gather(*tasks)
            results = []
            for emb in embeddings:
                if emb is not None:
                    vec = np.array([emb], dtype='float32')
                    if self.index_type == "flat_ip":
                        vec /= np.linalg.norm(vec) + 1e-10
                    result = self._search_by_vector(vec, top_k, score_threshold)
                else:
                    result = []
                results.append(result)
            self._last_retrieve_time_list = time.perf_counter() - start
            return results

        else:
            raise TypeError("query 必须为 str 或 list[str]")

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

    def _search_by_vector(self, vec: np.ndarray, top_k: int, score_threshold: float) -> list[tuple[str, float]]:
        """
        给定向量执行 FAISS 检索
        """
        D, I = self.index.search(vec, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            sim_score = self._to_similarity(dist)
            if sim_score >= score_threshold:
                results.append((self.texts[idx], sim_score))
        return results

    def _to_similarity(self, distance: float) -> float:
        """
        将 FAISS 返回的距离转换为相似度分数。

        - 对于 L2（欧氏距离），使用 sim = 1 / (1 + distance)
        - 对于 IP（内积，越大越好），使用 sim = distance,需确保输入向量已归一化

        :param distance: FAISS 返回的距离或得分
        :return: 相似度分数（值越大越相似）
        """

        if self.index_type == "flat_ip":
            return distance  # 内积
        else:
            return 1 / (1 + distance)  # L2 距离，越小越相似


if __name__ == "__main__":
    vecdb_path = Base_DIR / "vector_db" / "vecdb_1024_250609.jsonl"
    index_types = ["flat", "flat_ip", "ivf", "ivfpq", "hnsw", "lsh"]


    async def run():
        query = "叠加原理是指什么？"
        querys = ["叠加原理是什么？",

                  "以下措施（）可作为减少接触电压与跨步电压的措施。A: 接地网的布置应尽量使电位分布均匀，B:设置临时遮拦，C:增设水平均压带,D:带安全帽",

                  "在爆炸性气体危险区域为1区的场所内，电缆线路（）有中间接头。A.—般不得；B严禁；C.不宜；D.可以"]
        for idx_type in index_types:
            print(f"\n【索引类型: {idx_type}】")
            retriever = FAISSRetriever(vecdb_path, index_type=idx_type)
            result = await retriever.retrieve(query, top_k=TOP_K)

            print(f"\n[单条] 检索耗时: {retriever.retrieve_time:.4f} 秒")
            for text, score in result:
                print(f"{score:.4f} | {text}")

            results = await retriever.retrieve(querys, top_k=TOP_K)
            print(f"\n[多条] 总耗时: {retriever.retrieve_time_list:.4f} 秒")
            for i, rset in enumerate(results):
                print(f"\n查询{i + 1}:")
                for text, score in rset:
                    print(f"{score:.4f} | {text}")


    asyncio.run(run())
