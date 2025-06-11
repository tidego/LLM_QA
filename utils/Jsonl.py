import json
from pathlib import Path
from typing import List, Dict


def read_jsonl(path: Path) -> List[Dict]:
    """
    读取jsonl文件
    :param path:jsonl文件路径
    :return:json数组
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl(data: List[Dict], path: Path):
    """
    写入jsonl文件
    :param data:json数组
    :param path:jsonl文件路径
    :return:
    """
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')