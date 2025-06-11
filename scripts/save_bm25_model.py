import pickle
from rank_bm25 import BM25Okapi
from config.Config import TOKENIZED_DB_PATH, BM25_MODEL_PATH, CHUNK_DB_PATH

# 加载分词后的文本块和原始文本块
with open(TOKENIZED_DB_PATH, "rb") as f:
    tokenized_chunks = pickle.load(f)

with open(CHUNK_DB_PATH, "rb") as f:
    chunks = pickle.load(f)

# 构建 BM25 模型
bm25_model = BM25Okapi(tokenized_chunks)


# 保存模型和原始文本块
with open(BM25_MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": bm25_model,
        "chunks": chunks  #
    }, f)

print(f"BM25 模型已保存至: {BM25_MODEL_PATH}")
