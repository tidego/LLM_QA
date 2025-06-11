import json
import re
from typing import Union


class JSONExtractor:
    """
    从文本中提取 JSON 格式的内容，支持三种策略：
    1. ```json {...} ```
    2. ``` {...} ```
    3. 无代码块，直接是 JSON 字符串
    4. 注意问答题可能存在{}，这会导致解析失败，需要在简答Action上补充相应逻辑
    """
    def extract_json_from_text(self, text: str) -> Union[dict, list]:
        text = text.strip()

        # 匹配所有 ```...``` 中的内容
        code_block_pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
        matches = re.findall(code_block_pattern, text)

        if matches:
            # 优先取最后一个代码块
            json_str = matches[-1].strip()
            try:
                return self._parse_json(json_str)
            except ValueError:
                raise ValueError("最终代码块不是有效的 JSON。")

        # 直接尝试整个文本解析为 JSON
        try:
            return self._parse_json(text)
        except ValueError:
            raise ValueError("未找到有效的 JSON 格式内容。")

    def _parse_json(self, json_str: str) -> Union[dict, list]:
        try:
            # 自动转义单个反斜杠，防止 \m \r 等非法转义
            json_str = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', json_str)
            data = json.loads(json_str)
            if isinstance(data, (dict, list)):
                return data
            raise ValueError("JSON 内容不是对象或数组。")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 格式无效: {e}")



def sanitize_brackets(text: str) -> tuple[str, dict]:
    """
    替换文本中非 JSON 的嵌套花括号内容为占位符，避免干扰 JSON 解析。
    支持嵌套结构。

    算法：
    1. 首先找到最外层 JSON 的花括号区间（首个完整的 {...}）
    2. 遍历文本，使用栈来记录花括号的位置。
    3. 忽略最外层 JSON 范围，其它花括号用占位符替换。
    4. 返回替换后的文本和占位符映射表。

    :param text: 原始文本，可能含有干扰 JSON 的大括号内容
    :return: 替换后的文本，以及占位符映射表
    """
    placeholder_map = {}
    stack = []
    placeholder_count = 0
    result = []
    skip_ranges = []

    # ---------- 找到最外层 JSON 的 {...} ----------
    def find_outer_json_range(t: str) -> tuple[int, int] | None:
        s = []
        for idx, ch in enumerate(t):
            if ch == '{':
                if not s:
                    start = idx
                s.append('{')
            elif ch == '}':
                if s:
                    s.pop()
                    if not s:
                        return start, idx
        return None

    json_range = find_outer_json_range(text)
    if json_range:
        skip_ranges.append(json_range)

    # ---------------逐字符扫描，替换其它嵌套 {...} ----------
    i = 0
    while i < len(text):
        if any(start <= i <= end for start, end in skip_ranges):
            # 当前字符在跳过区域内，直接原样加入
            result.append(text[i])
            i += 1
            continue

        char = text[i]
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(i)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    end_idx = i
                    original = text[start_idx:end_idx + 1]
                    placeholder = f"__PLACEHOLDER_{placeholder_count}__"
                    placeholder_map[placeholder] = original
                    result.append(placeholder)
                    placeholder_count += 1
                    i = end_idx + 1
                    continue
        elif not stack:
            result.append(char)
        i += 1

    return "".join(result), placeholder_map


def restore_placeholders(result: Union[dict, list], placeholder_map: dict) -> Union[dict, list]:
    """
    将占位符替换回原始的大括号内容，恢复最终结构
    :param result: JSON 结构（dict 或 list）
    :param placeholder_map: 占位符映射表
    :return: 替换还原后的 JSON 结构
    """
    result_str = json.dumps(result, ensure_ascii=False)
    for placeholder, original in placeholder_map.items():
        result_str = result_str.replace(f'"{placeholder}"', original)
        result_str = result_str.replace(placeholder, original)
    return json.loads(result_str)

