import os
from dataclasses import dataclass

import dotenv


@dataclass
class Env:
    verbose: bool
    base_url: str
    api_key: str
    model_name: str
    temperature: float
    streaming: bool
    text_embedding_api_key: str
    text_embedding_model_name: str
    retriever_web_path: str
    searcher_max_results: int


def get() -> Env:
    dotenv.load_dotenv()  # load environment from .env file
    return Env(
        verbose=True if os.getenv("VERBOSE", "true").upper() == "TRUE" else False,
        base_url=os.getenv("BASE_URL", ""),
        api_key=os.getenv("API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", ""),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        streaming=True if os.getenv("STEAMING", "true").upper() == "TRUE" else False,
        text_embedding_api_key=os.getenv("TEXT_EMBEDDING_API_KEY", ""),
        text_embedding_model_name=os.getenv("TEXT_EMBEDDING_MODEL_NAME", ""),
        retriever_web_path=os.getenv("RETRIEVER_WEB_PATH", ""),
        searcher_max_results=int(os.getenv("SEARCHER_MAX_RESULTS", "5")),
    )
