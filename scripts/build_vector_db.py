"""
将分段后的文本进行向量化，使用Qwen-Embedding
"""
from pathlib import Path
from typing import List, Dict
from utils.Jsonl import read_jsonl, write_jsonl
from utils.VectorTransfer import VectorTransfer
from config.Config import Base_DIR, EQ_SPLIT_PATH, KG_SPLIT_PATH
from tqdm.asyncio import tqdm_asyncio

OUTPUT_PATH = Base_DIR / "vector_db" / 'vecdb_1024_250609.jsonl'
vectorTransfer = VectorTransfer(qps=15, capacity=30)  # 阿里云限流每秒30请求,每分钟1.2M tokens
async def embed_record(record: Dict) -> Dict:
    """
    chunk进行向量化
    :param record:jsonl中的一条记录
    :return:向量数据库一条记录
    """

    embedding = await vectorTransfer.get_embedding(record["text"])
    return {
        "id": record["id"],
        "embedding": embedding,
        "text": record["text"],
        "source": record["source"],
        "metadata": {}
    }


async def build_vector_db(input_paths: List[Path]) -> List[Dict]:
    """
    构建向量数据库
    :param input_path:试题型、知识型jsonl文件路径
    :param output_path:向量数据库jsonl文件路径
    :return:json数组
    """
    all_records = []
    for path in input_paths:
        all_records.extend(read_jsonl(path))  # 合并所有记录

    tasks = [embed_record(record) for record in all_records]
    results = await tqdm_asyncio.gather(*tasks, desc="Embedding", total=len(tasks),unit="chunk")
    data = results
    return data


async def build_and_save() -> None:
    """
    构建向量数据库并保存
    :return:
    """
    input_paths = [EQ_SPLIT_PATH, KG_SPLIT_PATH]
    results = await build_vector_db(input_paths)
    write_jsonl(results, OUTPUT_PATH)


if __name__ == '__main__':
    import asyncio

    asyncio.run(build_and_save())
    print("Done")
