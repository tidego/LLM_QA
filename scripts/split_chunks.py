import json
import re
from typing import List, Iterator
from pathlib import Path
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer,util
from config.Config import Base_DIR, EQ_DIR, KG_DIR
import spacy


def split_eq_text(text: str) -> List[str]:
    """
    将输入文本按段落分块，段落由一个及以上换行符分隔。
    特殊处理以 A/B/C/答案 开头的段落，合并入前一段。
    :param text:原始文本
    :return:试题分块列表
    """

    # 以两个或以上换行符分块
    raw_chunks = re.split(r'\n{2,}', text)
    chunks: List[str] = []

    for para in raw_chunks:
        para = para.strip()
        if not para:
            continue

        # 如果是选项或“答案：”开头，合并到上一段
        if re.match(r'^(A|B|C|D)[、．.,，：:）)]?\s*|^\s*[【]?\s*答案\s*[】]?\s*[:：]?\s*'
                , para):
            if chunks:
                chunks[-1] += ' ' + para
            else:
                chunks.append(para)
        else:
            chunks.append(para)

    return chunks


def split_kg_text_rule(text: str)->List[str]:
    """
    基于规则分割文本：
    :param text:
    :return:
    """
    pass
def split_kg_text_semantic(
    text: str,
    spacy_model_name: str = "zh_core_web_lg", # 性能强劲，速度更快
    bert_model_name: str = "shibing624/text2vec-base-chinese", # 生产环境可用模型
    sim_threshold: float = 0.7,
    window_size: int = 6,
    stride: int = 4
) -> List[str]:
    """
    按照语义聚合，滑动窗口拼接，补尾的方式，将文本分块，适用于规则处理之后
    ！并不适用于含数学公式的文本，因为公式可能会被误判为语义聚合的一部分
    这个函数可以用于后期精细调优知识库，但是前期简单的方法是基于LainChain进行分块
    :param text:原始文本
    :param spacy_model_name:sp模型
    :param bert_model_name:bert模型
    :param sim_threshold:相似度阈值
    :param window_size:窗口大小
    :param stride:步长
    :return:分块列表
    """
    # SpaCy 分句
    nlp = spacy.load(spacy_model_name)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return []
    print(sentences)
    # 句向量生成
    model = SentenceTransformer(bert_model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # 语义聚合
    chunks = []
    current_chunk = [sentences[0]]
    current_vecs = [embeddings[0]]

    for i in range(1, len(sentences)):
        mean_vec = torch.mean(torch.stack(current_vecs), dim=0, keepdim=True)
        sim = util.cos_sim(mean_vec, embeddings[i]).item()
        if sim > sim_threshold:
            current_chunk.append(sentences[i])
            current_vecs.append(embeddings[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_vecs = [embeddings[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # 滑动窗口拼接
    final_chunks = []
    for i in range(0, len(chunks) - window_size + 1, stride):
        window = chunks[i:i + window_size]
        final_chunks.append(" ".join(window))

    # 补尾
    if len(chunks) > 0 and (len(chunks) - window_size) % stride != 0:
        final_chunks.append(" ".join(chunks[-window_size:]))

    return final_chunks

def langchain_chunk_split(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128
) -> List[str]:
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 对文本进行分块
    :param text: 原始文本
    :param chunk_size: 每个块的最大 token 长度
    :param chunk_overlap: 块之间的重叠 token 数
    :return: 分块后的文本列表
    """

    # 设定分割符的有限顺序
    separators = ["\n\n", "\n", "。", "；", "$", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len  # 基于字符长度的近似Tokens计算
    )

    chunks = splitter.split_text(text)
    return chunks

def is_eq_choice_doc(text: str) -> bool:
    """
    判断整篇文档是否为选择题文档：依据是至少出现一次连续的 A、B、C 选项。
    :param text:文档内容
    :return:是否是选择题文档||除了选择题文档外，其他简单、判断类型均为False
    """
    # 匹配 A B C D 出现，之间不跨越超过80个字符，允许无标点
    pattern = r'A[：:、．.,，]?.{0,80}?B[：:、．.,，]?.{0,80}?C[：:、．.,，]?.{0,80}?'
    return bool(re.search(pattern, text, flags=re.DOTALL))


def clean_invisible_whitespace(text: str) -> str:
    """
    清除文本中的不可见空白字符，如不间断空格（NBSP）、零宽空格、制表符等。

    :param text:待去除空白字符的文本
    :return:清洗后的文本
    """
    invisible_chars = [
        '\u00A0',
        '\u2002',
        '\u2003',
        '\u2009',
        '\u202F',
        '\u200B',
        '\uFEFF',
        '\t',
    ]
    for ch in invisible_chars:
        text = text.replace(ch, ' ')
    # 替换多空格为单个空格
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def is_eq_noise_chunk(chunk: str, noise_path: Path) -> bool:
    """
    在选择题文档中，判断某个段落是否为噪声段。
    :param chunk:分块后的段落
    :return:是否为噪声段，是则返回True，否则返回False
    """

    def _looks_like_not_noise(text: str) -> bool:
        """
        判断段落是否应被保留（非噪声）的三条规则：
        1. 题干型：数字编号开头：如 1. xxx / 1．xxx / 1、xxx
        2. 正文型：中文字符数量 >= 20
        3. 判断型：含“正确”“错误”“×”“√”
        4. 工厂供电试题库填空题.md 需单独处理可将其划分为kg中
        """
        # ---------- 题干型 ----------
        # 将可能出现的转义形式 1\. 统一为 1.
        text_norm = text.replace('\\.', '.')
        if re.match(r'^\s*\d+\s*[．、\.]\s*\S+', text_norm):
            return True

        # --------- 正文型 ----------
        # 统计中文字符（含中文标点）数量
        chinese_chars = re.findall(r'[\u4e00-\u9fff，。；：“”‘’？！…（）]', text)
        if len(chinese_chars) >= 20:
            return True

        # ----------  判断型 ----------
        if re.search(r'正确|错误|×|√|x', text):
            return True

        return False

    if is_eq_choice_doc(chunk) or _looks_like_not_noise(chunk):  # 逻辑相同直接复用即可，存在ABCD即为非噪声
        return False
    else:
        with noise_path.open('a', encoding='utf-8') as f:
            f.write(chunk.strip() + '\n\n')
    return True


def is_kg_noise_chunk(chunk: str, noise_path: Path) -> bool:

    pass


def split_eq_md(paths: Iterator[Path], output_path: Path, noise_path=None) -> None:
    """
    读取多个 Markdown 文件，将其按段落分块，并保存为 JSONL 格式。
    :param paths:Path对象的迭代器，包含多个Markdown文件路径
    :param output_path:输出文件路径
    :param noise_path:可选，是否进行噪声处理，默认不处理
    :return:
    """
    with output_path.open('w', encoding='utf-8') as outfile:
        for path in paths:
            if path.name == '工厂供电试题库填空题.md':
                continue
            with path.open('r', encoding='utf-8') as infile:
                text = infile.read()
            chunks = split_eq_text(text)
            for idx, chunk in enumerate(chunks):
                clean_chunk = clean_invisible_whitespace(chunk)
                if noise_path and is_eq_choice_doc(text) and is_eq_noise_chunk(clean_chunk, noise_path):
                    # 后续可以添加逻辑
                    continue

                record = {
                    'id': f'{path.stem}_{idx}',
                    'text': clean_chunk,
                    'source': path.name
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    # 噪声处理路径
    if noise_path:
        print(f'保存噪声文件为：{noise_path}')


def split_kg_md(paths: Iterator[Path], output_path: Path, noise_path=None) -> None:
    """
    读取多个 Markdown 文件，将其按段落分段，并保存为 JSONL 格式,默认这里不进行噪声处理的实现。
    :param paths:Path对象的迭代器，包含多个Markdown文件路径
    :param output_path:输出文件路径
    :param noise_path:噪声文件路径，可选，默认为None
    :return:
    """
    with  output_path.open('w', encoding='utf-8') as outfile:
        for path in paths:
            with path.open('r', encoding='utf-8') as infile:
                text = infile.read()
            chunks = langchain_chunk_split(text)
            for idx, chunk in enumerate(chunks):
                if noise_path and is_kg_noise_chunk(chunk, noise_path):
                    # 后续可以添加逻辑
                    continue
                record = {
                    "id": f"{path.stem}_{idx}",
                    "text": clean_invisible_whitespace(chunk),
                    "source": path.name
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    if noise_path:
        print(f"保存噪声文件为：{noise_path}")


if __name__ == '__main__':
    EQ_SPLIT_PATH = Base_DIR / "kg_data" / 'eq_split' / 'eq.jsonl'
    KG_SPLIT_PATH = Base_DIR / "kg_data" / 'kg_split' / 'kg.jsonl'
    EQ_NOISE_PATH = Base_DIR / "kg_data" / 'noise' / 'eq_choice_noise.txt'
    KG_NOISE_PATH = Base_DIR / "kg_data" / 'noise' / 'kg_noise.txt'
    # test_eq_files = list(EQ_DIR.iterdir())[:3]
    # test_kg_files = list(KG_DIR.iterdir())[:3]
    split_eq_md(EQ_DIR.iterdir(), EQ_SPLIT_PATH, EQ_NOISE_PATH)
    split_kg_md(KG_DIR.iterdir(), KG_SPLIT_PATH) # 知识点类不设置噪声处理
    print(f'Done!')
