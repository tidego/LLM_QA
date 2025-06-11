"""
  ！Config文件路径不能移动
"""
from pathlib import Path

# =============================================#
#                检索参数配置                     #
# =============================================#
TOP_K = 3              # 检索结果数量,仅对模糊匹配和语义检索有效
SCORE_THRESHOLD = 0.7  # 语义相似度TOP1阈值
FUZZY_THRESHOLD = 1    # 模糊匹配阈值TOP1，这里用的是sofamax分数
WINDOW_RATIO = 0.5     # 精确检索：滑动窗口比例
STEP_RATIO = 0.125     # 精确检索：步长比例
INDEX_TYPE = "flat"    # faiss检索类型, 可选值：flat, flat_ip, ivf等,详见fuzzy_retriever.py
USE_GPU = False        # faiss是否使用GPU加速，加速需要下载模型faiss-gpu

# =============================================#
#                LLM相关参数                     #
# =============================================#
MODEL_NAME = "qwen-turbo"  # 模型名称，可替换为阿里云其他模型
RATE_LIMIT_PER_SECOND = 6  # 速率限制，每秒允许请求
CONCURRENCY = 3            # 并发数，控制并发数
# 这里说明一下，可能不是因为触发了QPM而限流，而是Token限流(TPM)
MAX_TOKENS = 512           # 最大生成token数
TEMPERATURE = 0.5          # 温度
TOP_P = 0.8                # 多样性
# TOP_K = 3                # 多样性,与TOP_P 二选一
STREAM = False             # 是否流式输出,注意流式输出请不要并发，否则内容交错
PRESENCE_PENALTY = 0.0     # 模型生成结果中已有的答案的惩罚
FREQUENCY_PENALTY = 0.0    # 模型生成结果中重复内容的惩罚
STOP = None                # 停止词 可以自定义str/List[str]
ENABLE_THINKING = False    # 是否开启 reasoning,仅DeepSeek、Qwen3系列模型支持
REASONING = False          # 这里是Prompt级控制，回答问答题时建议为True
ONLY_CONTENT = False       # 仅输出内容，不输出字段信息
# =============================================#
#                 API_KEY                      #
# =============================================#
DASHSCOPE_API_KEY = None     # 替换为实际DashScope API KEY,否则从环境变量中获取
BAIDU_AppBuilder_API = None  # 替换为实际BAIDU_AppBuilder_API,否则从环境变量获取

# =============================================#
#                 知识库路径                     #
# =============================================#
# 默认路径
Base_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
RAW_DIR = Base_DIR / "kg_data" / 'raw'             # 原始MD文件
EQ_DIR = Base_DIR / "kg_data" / 'eq'               # 问答型
KG_DIR = Base_DIR / "kg_data" / 'kg'               # 知识点型
# 分块后
EQ_SPLIT_PATH = Base_DIR / "kg_data" / 'eq_split' / 'eq.jsonl'
KG_SPLIT_PATH = Base_DIR / "kg_data" / 'kg_split' / 'kg.jsonl'

# ============== 数据库路径 ==================== #
CHUNK_DB_PATH = Base_DIR / "chunk_db" / "chunk_16183_250609.pkl"                   # 分块数据库
TOKENIZED_DB_PATH = Base_DIR / "chunk_db" / "chunk_tokenization_16183_250609.pkl"  # 分词数据库
VECTOR_DB_PATH = Base_DIR / "vector_db" / "vecdb_1024_250609.jsonl"                # 向量数据库

# ================BM25模型====================== #
BM25_MODEL_PATH = Base_DIR / "model" / "bm25_model.pkl"                     # BM25模型

# ================ 测试集 ====================== #
TEST_QUESTION_PATH = Base_DIR / "notebooks" / "question.json"               # 测试集

# =============== 输出 ========================= #
TIME_RECORDS_PATH = Base_DIR / "output" / "time_records.jsonl"              # 时间记录
CONFIDENCE_RECORDS_PATH = Base_DIR / "output" / "confidence_records.jsonl"  # 置信度记录
PREDICT_ANSWER_PATH = Base_DIR / "output" / "predict_answer.jsonl"          # 预测答案
