import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import base64
from io import BytesIO
import streamlit as st


PLATFORMS = ["Ollama", "Xinference", "GLM"] # ["fastchat", "openai"]


def get_llm_models(platform_type: Literal[tuple(PLATFORMS)], base_url: str="", api_key: str="EMPTY"):
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            llm_models = [model["model"] for model in client.list()["models"] if "bert" not in model.details.families]
            return llm_models
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            llm_models = client.list_models()
            return [k for k,v in llm_models.items() if v.get("model_type") == "LLM"]
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    elif platform_type == "GLM":
        # from zhipuai import ZhipuAI
        #
        # client = ZhipuAI(
        #     api_key="",  # 填写您的 APIKey
        # )
        # client.list_models()
        return [
            'glm-4-alltools',
            'glm-4-plus',
            'glm-4-0520',
            'glm-4',
            'glm-4-air',
            'glm-4-airx',
            'glm-4-long',
            'glm-4-flashx',
            'glm-4-flash'
        ]


def get_embedding_models(platform_type: Literal[tuple(PLATFORMS)], base_url: str="", api_key: str="EMPTY"):
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            embedding_models = [model["model"] for model in client.list()["models"] if "bert" in model.details.families]
            return embedding_models
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            embedding_models = client.list_models()
            return [k for k,v in embedding_models.items() if v.get("model_type") == "embedding"]
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []


def get_chatllm(
        platform_type: Literal[tuple(PLATFORMS)],
        model: str,
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0.1
):
    if platform_type == "Ollama":
        if not base_url:
            base_url = "http://127.0.0.1:11434"
        return ChatOllama(
            temperature=temperature,
            # streaming=True,
            model=model,
            base_url=base_url
        )
    elif platform_type == "Xinference":
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        if not api_key:
            api_key = "EMPTY"
    elif platform_type == "GLM":
        if not base_url:
            base_url = "https://open.bigmodel.cn/api/paas/v4"
        if not api_key:
            api_key = "EMPTY"

    return ChatOpenAI(
        temperature=temperature,
        model_name=model,
        streaming=True,
        base_url=base_url,
        api_key=api_key,
    )

    # if platform_type == "ollama":
    #     # from langchain_ollama import ChatOllama
    #     # return ChatOllama
    #     return ChatOpenAI(
    #         temperature=temperature,
    #         model_name=model,
    #         streaming=True,
    #         base_url=base_url,
    #         api_key=api_key,
    #     )
    # elif platform_type == "xinference":
    #     # from langchain_community.llms import Xinference
    #     # return Xinference
    #     return ChatOpenAI(
    #         temperature=temperature,
    #         model_name=model,
    #         streaming=True,
    #         base_url=base_url,
    #         api_key=api_key,
    #     )



def show_graph(graph):
    flow_state = StreamlitFlowState(
                       nodes=[StreamlitFlowNode(
                           id=node.id,
                           pos=(0,0),
                           data={"content": node.id},
                           node_type="input" if node.id == "__start__"
                                             else "output" if node.id == "__end__"
                                             else "default",
                       ) for node in graph.nodes.values()],
                       edges=[StreamlitFlowEdge(
                           id=str(enum),
                           source=edge.source,
                           target=edge.target,
                           animated=True,
                       ) for enum, edge in enumerate(graph.edges)],
                   )
    streamlit_flow('example_flow',
                   flow_state,
                   layout=TreeLayout(direction='down'), fit_view=True
    )


def get_kb_names():
    kb_root = os.path.join(os.path.dirname(__file__), "kb")
    if not os.path.exists(kb_root):
        os.mkdir(kb_root)
    kb_names = [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]
    return kb_names

def get_embedding_model(
        platform_type: Literal[tuple(PLATFORMS)] = "Ollama",
        model: str = "quentinz/bge-large-zh-v1.5",
        base_url: str = "",
        api_key: str = "EMPTY",
):
    if platform_type == "Ollama":
        # from langchain_ollama import ChatOllama
        # return ChatOllama
        if not base_url:
            base_url = "http://127.0.0.1:11434/"
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=base_url, model=model)
    elif platform_type == "Xinference":
        from langchain_community.embeddings.xinference import XinferenceEmbeddings
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        return XinferenceEmbeddings(server_url=base_url, model_uid=model)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)


def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    # 读取图片
    with open(image_path, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base_str}"

if __name__ == "__main__":
    print(get_embedding_model(platform_type="Ollama"))
