""" 
用于将知识库的内容进行二分类,方便后续分段
"""
import shutil
from pathlib import Path

# 原始数据目录
Base_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = Base_DIR / "kg_data" / 'raw'
EQ_DIR = Base_DIR / "kg_data" / 'eq'
KG_DIR = Base_DIR / "kg_data" / 'kg'
# 如果不存在创建文件夹
EQ_DIR.mkdir(parents=True, exist_ok=True)
KG_DIR.mkdir(parents=True, exist_ok=True)
for file in RAW_DIR.iterdir():
    if any(kw in file.name for kw in ["题", "卷"]):
        shutil.copy2(file, EQ_DIR)
    else:
        shutil.copy2(file, KG_DIR)

print("Done")