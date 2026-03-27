import unittest

from environment import environment
from tools.searcher.searcher import Searcher


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_searcher():
        env = environment.get()
        s = Searcher(env)
        print(s.invoke("北京今天天气怎么样"))


if __name__ == '__main__':
    unittest.main()
