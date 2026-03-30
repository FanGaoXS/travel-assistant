from typing import Iterator

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import Output
from langchain_core.tools import Tool

from environment.environment import Env


class Agent:
    executor: RunnableWithMessageHistory
    histories: dict[str, BaseChatMessageHistory]

    def __init__(self, env: Env, llm: ChatOpenAI, tools: list[Tool]):
        prompt_template = hub.pull("hwchase17/openai-functions-agent")
        agent = AgentExecutor(
            agent=create_tool_calling_agent(llm, tools, prompt_template),
            tools=tools,
            verbose=env.verbose,
        )
        self.executor = RunnableWithMessageHistory(
            runnable=agent,
            get_session_history=self.__get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self.histories = {}

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
        return self.histories[session_id]

    def invoke(self, text: str, session_id: str, **kwargs) -> Output:
        return self.executor.invoke(
            input={"input": text},
            config={"configurable": {"session_id": session_id}},
        )

    def stream(self, text: str, session_id: str, **kwargs) -> Iterator[Output]:
        return self.executor.stream(
            input={"input": text},
            config={"configurable": {"session_id": session_id}},
        )
