from action.Base import Action


class Classifier(Action):
    NAME = "问题分类"
    DESC = "根据输入的问题将其分为单选、多选、问答三种题型"

    def __init__(self, **kwargs):
        """
        问题分类
        由于测试题目中已经给了问题类型，所以直接传入即可，后续可拓展为LLM分类
        :param question_type:
        :param kwargs:
        """
        super().__init__(name=self.NAME, desc=self.DESC, **kwargs)

    async def execute(self) -> str:
        self._last_action_time = 0  # 近似为0
        self._last_result = self.prompt
        return self.prompt

    @classmethod
    def with_type(cls, question_type: str, **kwargs) -> "Classifier":
        """
        传入问题类型，创建一个Classifier实例
        :param question_type:
        :param kwargs:
        :return:
        """
        instance = cls(**kwargs)
        instance.update_prompt(question_type)
        return instance

    @staticmethod
    def classify(question_type: str) -> "Classifier":
        return Classifier.with_type(question_type=question_type)
