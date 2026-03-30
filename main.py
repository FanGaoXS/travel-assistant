from langchain_openai import ChatOpenAI

from agent.agent import Agent
from environment import environment
from tools.retriever.retriever import Retriever
from tools.searcher.searcher import Searcher

if __name__ == '__main__':
    env = environment.get()
    llm = ChatOpenAI(
        model=env.model_name,
        base_url=env.base_url,
        api_key=env.api_key,
        temperature=env.temperature,
        streaming=env.streaming,
    )
    tools = [
        Retriever(env).as_tool(),
        Searcher(env).as_tool(),
    ]
    agent = Agent(env, llm, tools)
    username = ""
    while True:
        change_username = False
        if username != "":
            if input("Do you need to change your username? (y/n): \n").strip().lower().startswith("y"):
                change_username = True
        if username == "" or change_username:
            username = input("Press Enter your username to continue: \n")

        question = input("Press Enter question: (Press exit to exit)\n")
        if question == "exit":
            break

        for chunk in agent.stream(question, username):
            # 情况 1：Agent 正在决定调用哪个工具
            if "actions" in chunk:
                for action in chunk["actions"]:
                    print(f"[思考] 正在调用: {action.tool} (参数: {action.tool_input})")

            # 情况 2：工具返回了结果
            if "steps" in chunk:
                print(f"[观察] 已获取数据，正在整理回答...")

            # 情况 3：最终输出内容
            if "output" in chunk:
                print("\n[回答]: ", end="")
                # 实现打字机效果
                for char in chunk["output"]:
                    print(char, end="", flush=True)
                    import time

                    time.sleep(0.01)  # 极短延迟模拟打字
                print("\n")  # 换行
