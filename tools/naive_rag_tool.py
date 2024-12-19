from langchain.tools.retriever import create_retriever_tool
import os

from utils import get_embedding_model


def get_naive_rag_tool(vectorstore_name):
    from langchain_chroma import Chroma

    # Add to vectorDB
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=get_embedding_model(platform_type="Ollama", model="quentinz/bge-large-zh-v1.5:latest"),
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.15,
        }
    )

    retriever_tool = create_retriever_tool(
        retriever,
        f"{vectorstore_name}_knowledge_base_tool",
        f"search and return information about {vectorstore_name}",
    )
    retriever_tool.response_format = "content"
    retriever_tool.func = lambda query: {
        f"已知内容 {inum+1}": doc.page_content.replace(doc.metadata["source"] + "\n\n", "")
        for inum, doc in enumerate(retriever.invoke(query))
    }
    return retriever_tool


if __name__ == "__main__":
    retriever_tool = get_naive_rag_tool("personal_information")
    print(retriever_tool.invoke("刘虔"))