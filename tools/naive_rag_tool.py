from langchain.tools.retriever import create_retriever_tool
import os

from utils import get_embedding_model


def get_naive_rag_tool(vectorstore_name):
    from langchain_chroma import Chroma

    # Add to vectorDB
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=get_embedding_model(model="quentinz/bge-large-zh-v1.5:latest"),
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),
    )

    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        f"{vectorstore_name}_knowledge_base_tool",
        f"search and return information about {vectorstore_name}",
    )
    return retriever_tool