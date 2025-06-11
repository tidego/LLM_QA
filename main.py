"""
**电力问答系统**

作者: 顾兆林,曲阜师范大学，软件工程
日期: 2025-06-11
描述: 用于处理与电力相关的问答任务，包括文本分块、RAG、Multi-Agents等功能，详见readme.md。
邮箱：zhaolin_gu@163.com
GitHub:github.com/tidego
个人网页：tidego.cn
"""

import asyncio
import json
import pickle
import time
from typing import Union
from action.Classifier import Classifier
from action.Confidence import Confidence
from action.MultiChoiceAnswer import MultiChoiceAnswer
from action.ShortAnswer import ShortAnswer
from action.SingleChoiceAnswer import SingleChoiceAnswer
from action.WebSearch import WebSearch
from agent.ClassifierAgent import ClassifierAgent
from agent.ConfidenceAgent import ConfidenceAgent
from agent.MultiChoiceAgent import MultiChoiceAgent
from agent.ShortAnswerAgent import ShortAnswerAgent
from agent.SingleChoiceAgent import SingleChoiceAgent
from agent.WebSearchAgent import WebSearchAgent
from config.Config import BM25_MODEL_PATH, TOKENIZED_DB_PATH, \
    INDEX_TYPE, WINDOW_RATIO, STEP_RATIO, USE_GPU, VECTOR_DB_PATH, MODEL_NAME, TEMPERATURE, TOP_P, TOP_K, STREAM, \
    PRESENCE_PENALTY, FREQUENCY_PENALTY, STOP, ENABLE_THINKING, CHUNK_DB_PATH, TEST_QUESTION_PATH, FUZZY_THRESHOLD, \
    SCORE_THRESHOLD, TIME_RECORDS_PATH, CONFIDENCE_RECORDS_PATH, PREDICT_ANSWER_PATH, REASONING, RATE_LIMIT_PER_SECOND, \
    CONCURRENCY, ONLY_CONTENT
from env.Base import Environment
from rag.exact_retriever import SlidingSubstringMatcher
from rag.fuzzy_retriever import PrebuiltBM25FuzzyRetriever
from rag.semantic_retriever import FAISSRetriever
from utils.Logger import Logger
from utils.TestQuestion import process_question_item

# ======================= 记录信息 ============================= #
time_records = {}            # "问题id": "总时间"
confidence_records = {}      # "问题id": "置信度"
predict_answer = {}          # "问题id": "预测答案"
logger = Logger()            # 日志记录器
# ======================= 数据预加载 ============================ #
with open(TOKENIZED_DB_PATH, "rb") as f:
    tokenized_chunks = pickle.load(f)                       # 加载分词后的数据

with open(CHUNK_DB_PATH, "rb") as f:                        # 加载分块后的数据
    chunks = pickle.load(f)

with open(TEST_QUESTION_PATH, "r", encoding="utf-8") as f:  # 加载测试集
    test_questions = json.load(f)

# ======================= RAG模块初始化 =========================== #
exact_retriever = SlidingSubstringMatcher \
    (chunks=chunks, window_ratio=WINDOW_RATIO, step_ratio=STEP_RATIO)  # 精确检索
fuzzy_retriever = PrebuiltBM25FuzzyRetriever(BM25_MODEL_PATH)          # 模糊检索
semantic_retriever = FAISSRetriever \
    (VECTOR_DB_PATH, index_type=INDEX_TYPE, use_gpu=USE_GPU)           # 语义检索

# ===================== Multi-Agents模块初始化 =================== #
classifierAgent = ClassifierAgent()      # 问题分类器
confidenceAgent = ConfidenceAgent()      # 置信评估器
multiChoiceAgent = MultiChoiceAgent()    # 多选回答器
singleChoiceAgent = SingleChoiceAgent()  # 单选回答器
shortAnswerAgent = ShortAnswerAgent()    # 简答回答器
webSearchAgent = WebSearchAgent()        # 网络搜索器

# ==================== 动作参数引入 ============================== #
KWARGS = {
    "model_name": MODEL_NAME,                          # 模型名称
    "temperature": TEMPERATURE,                        # 温度
    "stream": STREAM,                                  # 流式输出
    "presence_penalty": PRESENCE_PENALTY,              # 模型重复性
    "frequency_penalty": FREQUENCY_PENALTY,            # 模型频率
    "stop": STOP,                                      # 停止词
    "enable_thinking": ENABLE_THINKING ,               # 推理
    "rate_limit_per_second": RATE_LIMIT_PER_SECOND,    # 速率
    "concurrency": CONCURRENCY,                        # 并发数
    "only_content": ONLY_CONTENT,                      # 仅输出内容
}

if TOP_K:
    KWARGS["top_k"] = TOP_K                  # top_k多样性
else:
    KWARGS["tok_p"] = TOP_P                  # top_p多样性

# =================== 环境初始化 ================================ #
env = Environment()
Agents = [classifierAgent, confidenceAgent, multiChoiceAgent, singleChoiceAgent, shortAnswerAgent, webSearchAgent]
for agent in Agents:
    env.register_agent(agent)  # 注册智能体


# =================== 调度逻辑 ================================== #
# Question输入后的时间线:
# 问题分类 -> 等待检索结果-> 单选/多选/问答
#                     -> 置信评估 -> 置信等级低(终止回答) -> 网络搜索 -> 单选/多选/问答 -> 置信评估
#                     ↑(不执行，精准匹配or模糊检索Top1分数>0.8)                -> 单选多选问答
# 精确检索/模糊检索 -> 语义检索
# 详细可以参见readme.md文档末尾
async def dispatch(id: int, question: str, question_type: str = None):
    """
    **最核心部分**
    调度逻辑，后期可以引入异步中断，进一步优化时间线
    :param question:
    :param question_type:
    :return:
    """
    # logger.info(f"\033[1;36m问题: {question}\033[0m") 如果需要输出问题，请取消注释
    # 分类器与精确检索并发
    task_classifier = asyncio.create_task(classifierAgent.add_run(Classifier.with_type(question_type)))
    task_exact = asyncio.create_task(exact_retriever.retrieve(question))

    exact_result, _ = await asyncio.gather(task_exact, task_classifier)
    q_type = classifierAgent.get_last_result()
    # logger.info(f"\033[1;36m问题: {question}，分类结果: {q_type}\033[0m") 由于问题分类给了，这里就不输出了
    answer_agent = {"单选": singleChoiceAgent, "多选": multiChoiceAgent, "问答": shortAnswerAgent}[q_type]

    if exact_result[1]:
        logger.info(f"题目{id},精确匹配成功，匹配结果：{exact_result[0]}，置信度：2")
        await answer_agent.add_run(build_prompt_for_type(q_type, info=str(exact_result[0]), question=question))
        answer = answer_agent.get_last_result()
        return {"answer": answer, "confidence": 2}

    # 模糊检索
    fuzzy_result = await fuzzy_retriever.retrieve(question, top_k=TOP_K)
    if fuzzy_result and fuzzy_result[0][1] > FUZZY_THRESHOLD:
        logger.info(f"题目{id},模糊匹配成功，匹配结果：{fuzzy_result[0][0]}，置信度：2")
        await answer_agent.add_run(build_prompt_for_type(q_type, info=fuzzy_result[0][0], question=question))
        answer = answer_agent.get_last_result()
        return {"answer": answer, "confidence": 2}

    # 语义检索
    semantic_result = await semantic_retriever.retrieve(question, top_k=TOP_K)
    if semantic_result and semantic_result[0][1] > SCORE_THRESHOLD:
        logger.info(f"题目{id},语义匹配成功，匹配结果：{semantic_result[0][0]}，置信度：2")
        await answer_agent.add_run(build_prompt_for_type(q_type, info=semantic_result[0][0], question=question))
        answer = answer_agent.get_last_result()
        return {"answer": answer, "confidence": 2}

    # 置信评估与答题并发执行
    info_text = semantic_result[0][0] if semantic_result else ""
    task_answer = asyncio.create_task(
        answer_agent.add_run(build_prompt_for_type(q_type, info=info_text, question=question))
    )
    task_confidence = asyncio.create_task(
        confidenceAgent.add_run(Confidence(**KWARGS).build_prompt(info=info_text, question=question))
    )
    await asyncio.wait([task_answer, task_confidence])
    logger.info(f"题目{id}，置信度评估中...{confidenceAgent.get_last_result()}")
    conf_score = confidenceAgent.get_last_result().get("confidence")

    if conf_score == 1:
        logger.info(f"题目{id},置信度评估成功，置信度：1")
        return {"answer": answer_agent.get_last_result(), "confidence": 1}

    # 网络搜索、新答题任务 、新置信评估
    await webSearchAgent.add_run(WebSearch().query(question))
    web_info = webSearchAgent.get_last_result()
    new_answer = asyncio.create_task(
        answer_agent.add_run(build_prompt_for_type(q_type, info=web_info, question=question))
    )
    new_confidence = asyncio.create_task(
        confidenceAgent.add_run(Confidence(**KWARGS).build_prompt(info=web_info, question=question))
    )
    await asyncio.wait([new_answer, new_confidence])
    final_score = confidenceAgent.get_last_result().get("confidence")
    logger.info(f"网络搜索成功，题目{id}置信度：{final_score}")

    return {
        "answer": answer_agent.get_last_result(),
        "confidence": 1 if final_score == 1 else 0
    }


def build_prompt_for_type(q_type: str, info: Union[str, tuple], question: str):
    if q_type == "单选":
        return SingleChoiceAnswer(**KWARGS).build_prompt(info=info, question=question, reasoning=REASONING)
    elif q_type == "多选":
        return MultiChoiceAnswer(**KWARGS).build_prompt(info=info, question=question, reasoning=REASONING)
    else:
        return ShortAnswer(**KWARGS).build_prompt(info=info, question=question, reasoning=REASONING)


async def handle_question(q: dict):
    """
    处理单个问题的异步函数
    :param q:
    :return:
    """
    qid = q["id"]
    question = q["question"]
    qtype = q["question_type"]
    start_time = time.time()
    result = await dispatch(qid, question, qtype)
    end_time = time.time()
    time_records[qid] = end_time - start_time
    confidence_records[qid] = result["confidence"]
    predict_answer[qid] = result["answer"]



async def main():
    """
    主函数
    :return:
    """
    start = time.time()

    for q in test_questions:
        await handle_question(process_question_item(q))
        await asyncio.sleep(0.5)  # 每个问题之间间隔 0.5 秒，防止限流

    end = time.time()
    logger.info(f"{len(test_questions)}条任务执行完毕！,总耗时: {end - start:.2f}秒")
    logger.info(f"{len(test_questions)}条任务执行完毕！,总耗时: {end - start:.2f}秒")
    # 写入之前字典排一下序号
    sorted_time = dict(sorted(time_records.items(), key=lambda x: int(x[0])))
    sorted_confidence = dict(sorted(confidence_records.items(), key=lambda x: int(x[0])))
    sorted_answer = dict(sorted(predict_answer.items(), key=lambda x: int(x[0])))

    # 处理 predict_answer 写入
    with open(PREDICT_ANSWER_PATH, 'a', encoding='utf-8') as f:
        for qid, ans in sorted_answer.items():
            pred = "、".join(ans["answer"]) if isinstance(ans["answer"], list) else ans["answer"]
            f.write(json.dumps({"id": int(qid), "predict": pred}, ensure_ascii=False) + '\n')

    # 处理 confidence 写入
    with open(CONFIDENCE_RECORDS_PATH, 'a', encoding='utf-8') as f:
        for qid, conf in sorted_confidence.items():
            f.write(json.dumps({"id": int(qid), "confidence": conf}, ensure_ascii=False) + '\n')

    # 处理 time 写入
    with open(TIME_RECORDS_PATH, 'a', encoding='utf-8') as f:
        for qid, duration in sorted_time.items():
            f.write(json.dumps({"id": int(qid), "time": round(duration, 4)}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
