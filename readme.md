# 项目简介
1.本项目是一项简易、可拓展、稳定性好、配置方便的问答系统，只需要收集md文件，便可以迁移到其他领域的问答系统。

2.经测试本项目在Python 3.10.15按照requirements.txt安装依赖，可以运行

3.由于Qwen模型有TQM和TQM限流，运行时注意

4.数据暂时不开源,如需要请联系zhaolin_gu@163.com



# action
> 存放了multi_action模块，每个动作继承BaseAction类，并实现execute方法。action作为Agent执行的基本单位，可以在Agent定义每个智能体动作的调度逻辑。

Action类:
- name，desc，用于描述动作，
- model，用于指定动作使用的模型。
- logger，用于记录日志。
- prompt，用于定义动作的提示词。
- _last_action_time，用于记录上一次动作执行时间，用于计算总耗时
- _last_result，用于记录上一次动作的返回结果
- 各类方法参考源码

**acton目录**
- Base.py 定义了BaseAction类，用于定义动作的基本结构
- Classifier.py 定义了问题分类动作
- Confidence.py 定义了置信度评估动作
- MultiChoiceAnswer.py 定义了多选问题回答动作
- SingleChoiceAnswer.py 定义了单选问题回答动作
- ShortAnswer.py 定义了简答问题回答动作
- WebSearch.py 定义了网络搜索动作

# agent
> 存放了Agent类和Memory类，用于定义智能体的行为和记忆。

Agent类:
- name，desc，用于描述智能体
- actions 自身的动作列表，默认先进先出，即先进来的动作先执行。
- memory，用于记录智能体的记忆，默认为Memory类
- longger，用于记录智能体的日志
- is_idle，用于记录智能体的空闲状态
- _send_memory_callback 用于环境回调，防止循环引用
- 各类方法参考源码

Memory类:
> 存放了Memory类，用于记录智能体的记忆。先进后出的栈结构，方便获取最近的记忆，同时栈满清除最旧的记忆

**agent目录**
> agent作为动作执行者，如果后续需要支持复杂逻辑可以拓展

- Agent.py 定义了Agent类与Memory类
- ClassifierAgent.py 定义了问题分类智能体
- ConfidenceAgent.py 定义了置信度评估智能体
- MultiChoiceAnswerAgent.py 定义了多选问题回答智能体
- SingleChoiceAnswerAgent.py 定义了单选问题回答智能体
- ShortAnswerAgent.py 定义了简答问题回答智能体，由于问答消耗的Prompt更多，每个动作执行sleep(30)秒，避免TPM限流
- WebSearchAgent.py 定义了网络搜索智能体

# chunk_db
**chunk_db目录**
> chunk_db目录下存放了分块数据库及分词之后的数据库
- chunk_16183_250609.pkl 分块数据库，用于精确检索
- chunk_tokenization_16183_250609.pkl 用于模糊检索，基于BM25算法实现

# config
**config目录**
> config目录下存放了项目配置文件，可以根据自己的需要进行修改
- Config.py 定义了RAG、LLM相关参数及项目路径配置

# env
Environment类:
> 是所有智能体运行的环境，智能体可以在环境中注册，然后由环境调度智能体的行为，和信息之间的交互
- agent 用于记录智能体的列表
- logger 用于记录日志
- Env价值在于方法的实现，用于智能体调度、信息交互等。


# kg_data
**kg_data** 
> 目录下存放了知识库原始数据及分块后的数据。
- eq目录下存放了试题型数据
- kp目录下存放了知识点型数据
- eq_split目录下存放了试题型数据分块后的数据
- kp_split目录下存放了知识点型数据分块后的数据
- noise目录下存放了噪声数据
- raw目录下存放了原始数据
- merge目录下存放了用于快速测试的知识库(未经数据处理）,比如全文ctrlF查找是否有原题

# logs
> 存放了项目日志文件,所有使用logger记录器的动作，都会记录在logs目录下

# model
> 存放了BM25模型，这样就不必每次运行时进行各种计算，如TF，DF，IDF等

# netebooks
> 用于存储一些数据，测试文件

# output
> 存放了项目运行结果
- confidence_records.jsonl 置信度记录文件
- predict_answer.jsonl     预测答案文件
- time_records.jsonl       运行时间记录文件

# rag
> 存放检索模块，包含精确检索、概率模糊检索、语义检索
- exact_retriever.py 精确检索，基于滑动窗口子串匹配实现
- fuzzy_retriever.py 模糊检索，基于BM25算法实现
- semantic_retriever.py 语义检索，基于FAISS实现，支持多种索引方式

# scripts
> 存放了项目运行脚本
- build_chunk_db.py 构建分块数据库
- build_chunk_tokenization_db.py 构建分词之后的分块数据库
- build_vector_db.py 构建向量数据库
- merge_file.py 将原始数据集进行合并，用于结合Dify快速原型测试
- save_bm25_model.py 保存BM25模型，下一次直接读取，不用重新计算
- split_chunks.py 分块**核心脚本**，对试题型、知识点型数据进行分块
- split_files_by_name.py 根据文件名将知识库分为**试题型**和**知识点型**

# utils
> 存放了项目工具类
- JSONExtractor.py 定义了JSON提取器,增加鲁棒性
- Jsonl.py 定义了相关jsonl工具
- Logger.py 定义了日志工具
- QwenModel.py 本项目使用的模型，支持Qwen系列，可支持推理
- SearchAPI.py 接入了百度MCP API,用于检索
- TestQuestion.py 将测试问题转换为所需要的格式
- VectorTransfer.py 基于令牌桶算法的向量转换工具

# vector_db
> 向量数据库及其索引所在目录
命名规则为：{vecdb}+{维度}+{时间}
若是索引文件，则命名规则为：{vecdb}+{维度}+{时间}+{索引方式}
索引方式参加semantic_retriever.py

# main.py
> 项目的入口文件，启动项目，进行测试


# 调度逻辑
> 多智能体的调度逻辑如下

系统目标是在保证准确性前提下，尽可能高效完成检索与答题流程，输出：
* 答案
* 置信等级（0/1/2）
* 运行时间

第一阶段：**精确检索与分类**
1. 并行启动两个任务：

   * **问题分类器**（确定是单选/多选/问答）
   * **精确检索**（Sliding Window 检索，匹配原文）

2. 等待两个任务完成后：

   * 如果**精确检索成功（命中）**

     * 立即开始答题任务
     * 设置置信等级为 2
     * 流程结束
   * 否则：

     * 继续执行 **模糊检索（BM25）**



第二阶段：**模糊检索与条件判断**

3. 模糊检索返回后，获取 **Top1 分数**：

   * 若 Top1 > 0.8

     * 开始答题任务
     * 设置置信等级为 2
     * 流程结束
   * 否则：

     * 执行语义检索（向量召回）



第三阶段：**语义检索与置信评估处理**

4. 等待语义检索返回，获取 **Top1 相似度分数**：

   * 若 Top1 > 0.7
     * 开始答题任务
     * 设置置信等级为 2
     * 流程结束
   * 否则：
     * 执行**第一次置信评估任务**与**答题任务**（并发）
     * 等待置信结果：
       * 若置信 = 1
         *  流程结束，设置等级为 1
       * 若置信 = 0
         *  中止当前答题任务（若仍在执行）
         *  启动 网络搜索
         *  基于搜索结果，再次执行：
           * 答题任务（新）
           * 置信评估任务（新）
         * 设置置信等级：
           * 1：新置信评估认为可回答
           * 0：新置信评估仍认为无关
           * 流程结束


2:精确匹配命中 / 模糊检索Top1>0.8 / 语义检索Top1>0.7（跳过置信评估） 
1:来自置信评估，认为背景能支持回答问题（可能来源于语义/网络检索）             
0:两轮置信评估都判断背景无关，网络搜索无法提供有效信息                   

