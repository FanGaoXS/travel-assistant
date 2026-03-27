import unittest

from environment import environment
from tools.retriever.retriever import Retriever


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_retriever():
        env = environment.get()
        r = Retriever(env)
        print(r.invoke("人工智能的历史"))
        print(r.invoke("人工智能的应用"))


if __name__ == '__main__':
    unittest.main()
