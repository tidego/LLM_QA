from config.Config import EQ_SPLIT_PATH, KG_SPLIT_PATH, Base_DIR
from utils.Jsonl import read_jsonl
import jieba
import pickle

OUTPUT_PATH = Base_DIR / "chunk_db" / 'chunk_tokenization_16183_250609.pkl'

# 读取原始文本块
input_paths = [EQ_SPLIT_PATH, KG_SPLIT_PATH]
texts = []
for input_path in input_paths:
    texts.extend(read_jsonl(input_path))
chunks = [text["text"] for text in texts]

# 仅保留分词列表
tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(tokenized_chunks, f)

