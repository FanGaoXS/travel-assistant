from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.tools import StructuredTool
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores.base import (
    VectorStore,
    VectorStoreRetriever,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from environment.environment import Env


class Retriever:
    retriever: VectorStoreRetriever

    def __init__(self, env: Env):
        loaded = WebBaseLoader(
            web_path=env.retriever_web_path,
            show_progress=True,
        ).load()
        docs = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            add_start_index=True,
        ).split_documents(loaded)
        embeddings = DashScopeEmbeddings(
            model=env.text_embedding_model_name,
            dashscope_api_key=env.text_embedding_api_key,
        )
        db: VectorStore = FAISS.from_documents(
            documents=docs,
            embedding=embeddings,
        )
        self.retriever = db.as_retriever()

    def invoke(self, text: str, **kwargs) -> List[Document]:
        return self.retriever.invoke(text, **kwargs)

    def as_tool(self) -> StructuredTool:
        return create_retriever_tool(
            retriever=self.retriever,
            name="db_search",
            description="搜索本地数据库"
        )
