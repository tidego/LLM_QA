from action.Base import Action
from config.Config import Base_DIR, REASONING
from utils.JSONExtractor import JSONExtractor


class MultiChoiceAnswer(Action):
    NAME = "多选回答器"
    DESC = "回答多选问题"
    PROMPT_TEMPLATE = (Base_DIR / 'prompt' / 'MultiChoicePrompt.txt').read_text(encoding="utf-8")
    PARSER = JSONExtractor()

    def __init__(self, **kwargs):
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)


    def build_prompt(self, info: str, question: str, reasoning: bool = REASONING) -> "MutiChoiceAnswer":
        """
        构建prompt
        :param retrieved_chunks: 检索到的背景知识
        :param user_question: 用户提问，包含问题与选项
        :param reasoning: 是否Prompt级分析
        :return:
        """
        self.prompt = self.PROMPT_TEMPLATE \
            .replace("<<<retrieved_chunks>>>", info) \
            .replace("<<<user_question>>>", question) \
            .replace("<<<reasoning>>>", str(reasoning))
        return self
    async def execute(self, *args, **kwargs) -> dict | list | None:
        """
        重写execute方法，实现多选回答
        :param args:
        :param kwargs:
        :return:
        """
        if self.prompt:
            result = await self.timed_ask(self.prompt,**kwargs)
            try:
                result = self.PARSER.extract_json_from_text(result)
            except Exception as e:
                self.logger.error(f"JSON解析失败: {e}")
                result = None
            self._last_result = result
            return result
        else:
            self.logger.warning("未定义prompt，execute 没有执行任何操作。")
            return None
