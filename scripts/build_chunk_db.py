from config.Config import EQ_SPLIT_PATH, KG_SPLIT_PATH, Base_DIR
from utils.Jsonl import read_jsonl
import pickle

OUTPUT_PATH = Base_DIR / "chunk_db" / 'chunk_16183_250609.pkl'

input_paths = [EQ_SPLIT_PATH, KG_SPLIT_PATH]
texts = []
# 读取原始文本块
for input_path in input_paths:
    texts.extend(read_jsonl(input_path))
chuncks = [text["text"] for text in texts]

with  open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(chuncks, f)
