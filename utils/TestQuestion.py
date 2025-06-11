import json

from config.Config import TEST_QUESTION_PATH


def process_question_item(sample: dict) -> dict:
    """
    处理单个题目字典，输出包含 id, 合并后的问题文本，以及题型。
    """
    qid = sample.get("id")
    qtype = sample.get("type", "")
    question = sample.get("question", "")
    options = []
    has_options = any(sample.get(key) for key in ["A", "B", "C", "D"])
    if has_options:
        for key in ["A", "B", "C", "D"]:
            if sample.get(key):
                options.append(f"{key}:{sample[key]}")
        merged_question = question + " " + " ".join(options)
    else:
        merged_question = question

    return {
        "id": qid,
        "question": merged_question,
        "question_type": qtype
    }

