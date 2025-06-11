import asyncio
from action.Base import Action
from action.Classifier import Classifier
from action.Confidence import Confidence
from action.MultiChoiceAnswer import MultiChoiceAnswer
from action.ShortAnswer import ShortAnswer
from action.SingleChoiceAnswer import SingleChoiceAnswer
from action.WebSearch import WebSearch
from agent.Base import Agent
from config.Config import REASONING
from env.Base import Environment


class SayHelloAction(Action):
    def __init__(self, **kwargs):
        super().__init__(name="打招呼", desc="打招呼", prompt="你好，请用一句话打招呼", **kwargs)


class AskQuestionAction(Action):
    def __init__(self, **kwargs):
        super().__init__(name="提问", desc="提问一个简单的问题", prompt="什么是人工智能？", **kwargs)


async def main():
    # 创建两个智能体
    alice = Agent(name="Alice")
    bob = Agent(name="Bob")

    # 给他们分配动作
    alice.add_action(SayHelloAction())
    alice.add_action(AskQuestionAction(max_tokens=1))

    bob.add_action(AskQuestionAction(max_tokens=1))
    bob.add_action(SayHelloAction())

    # 初始化环境并注册智能体
    env = Environment()
    env.register_agent(alice)
    env.register_agent(bob)

    print("\n==== 环境开始运行 ====\n")

    # 触发一次初始动作
    await env.run()
    env.show_status()

    # 持续轮询，直到全部空闲，这里Alice不会执行提问，因为alice执行SayHello很快完成，is_side为True
    # 然后Bob执行完毕提问动作，均为空闲，故Bob也不会执行SayHello
    while not env.all_idle():
        await env.run()
        env.show_status()
    #====================== 查看未执行动作 ======================#
    print("\nAlice未执行动作:", alice.current_actions)
    print("\nBob 未执行动作:",  bob.current_actions)
    # ===================== 执行剩余动作 =========================#
    print("\n==== 开始执行剩余动作 ====\n")
    await alice.run()
    await bob.run()
    # ===================== 分类器 ============================#
    classifier_list = [Classifier.with_type(q_type) for q_type in ["单选", "多选", "问答"]]
    await bob.add_run(classifier_list)
    print("\nBob 问题分类:")
    print(bob.get_last_k_result(3))
    # =================== 置信度评估 ===========================#
    await bob.add_run(Confidence().build_prompt(info="bob 10岁了", question="bob 多大了", reasoning=True))
    print("\nBob 多大了置信度评估:")
    print(bob.get_last_result()["confidence"])

    # =================== 单选 ================================#
    await bob.add_run(
        SingleChoiceAnswer().build_prompt(info="bob 10岁了", question="bob 多大了,A:11,B:13,C:10", reasoning=True))
    print("\nBob 多大了单选答案:")
    print(bob.get_last_memory())

    # =================== 多选 =================================#
    await bob.add_run(MultiChoiceAnswer().build_prompt(info="bob 10岁了,bob姐姐13岁了",
                                                      question="bob和姐姐分别多大了,A:11,B:13,C:10,D:12",
                                                      reasoning=REASONING))
    print("\nBob 和姐姐多大了多选答案:")
    print(bob.get_last_memory())

    # ==================== 问答 ================================#
    await bob.add_run(ShortAnswer(max_tokens=100).build_prompt(info="bob 10岁了,bob姐姐13岁了",question="bob和姐姐分别多大了",reasoning=REASONING))
    print("\nBob 和姐姐多大了问答答案:")
    print(bob.get_last_memory())


    # =================== 搜索 =================================#
    await bob.add_run(WebSearch().query("成都有哪些景点？"))
    print("\n成都有哪些景点:")
    print(bob.get_last_result())


    # 打印记忆
    for agent in env.agents:
        print(f"\n{agent.name} 的记忆:")
        for mem in agent.memory.get_all():
            print(mem)





if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
