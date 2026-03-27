from typing import Any

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.tools import Tool

from environment.environment import Env


class Searcher:
    searcher: DuckDuckGoSearchResults

    def __init__(self, env: Env):
        self.searcher = DuckDuckGoSearchResults(
            max_results=env.searcher_max_results,
        )
        pass

    def invoke(self, text: str, **kwargs) -> Any:
        return self.searcher.invoke(text, **kwargs)

    def as_tool(self) -> Tool:
        return Tool(
            func=self.searcher.run,
            name="browser_search",
            description="检索互联网信息",
        )
