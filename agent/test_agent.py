import unittest

from langchain_openai import ChatOpenAI

from agent.agent import Agent
from environment import environment
from tools.retriever.retriever import Retriever
from tools.searcher.searcher import Searcher


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_agent():
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
        print(agent.invoke("人工智能的应用是什么", "1"))


if __name__ == '__main__':
    unittest.main()
