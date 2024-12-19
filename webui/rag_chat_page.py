import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import get_naive_rag_tool
import json

RAG_PAGE_INTRODUCTION = "你好，我是你的 Chatchat 智能助手，当前页面为`RAG 对话模式`，可以在对话让大模型基于左侧所选知识库进行回答，有什么可以帮助你的吗？"


def get_rag_graph(platform, model, temperature, selected_kbs, KBS):
    tools = [KBS[k] for k in selected_kbs]
    tool_node = ToolNode(tools)

    def call_model(state):
        llm = get_chatllm(platform, model, temperature=temperature)
        llm_with_tools = llm.bind_tools(tools, tool_choice="any")
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

def graph_response(graph, input):
    for event in graph.stream(
        {"messages": input},
        config={"configurable": {"thread_id": 42}},
        stream_mode="messages",
    ):
        # st.write(event)
        # st.write(graph.get_state_history(config={"configurable": {"thread_id": 42}},))

        if type(event[0]) == AIMessageChunk:
            if len(event[0].tool_calls):
                # st.write(event[0].tool_calls)
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "正在查询...",
                        "knowledge_base": event[0].tool_calls[0]["name"].replace("_knowledge_base_tool", ""),
                        "query": str(event[0].tool_calls[0]["args"]["query"]),
                    }
                )
            yield event[0].content
        elif type(event[0]) == ToolMessage:
            status_placeholder = st.empty()
            with (status_placeholder.status("正在查询...", expanded=True) as s):
                st.write("已调用 `", event[0].name.replace("_knowledge_base_tool", ""), "` 知识库进行查询")  # Show which tool is being called
                continue_save = False
                if len(st.session_state["rag_tool_calls"]):
                    if "content" not in st.session_state["rag_tool_calls"][-1].keys() \
                            and event[0].name.replace("_knowledge_base_tool", "") == \
                        st.session_state["rag_tool_calls"][-1]["knowledge_base"]:
                        continue_save = True
                        st.write("知识库检索输入: ")
                        st.code(st.session_state["rag_tool_calls"][-1]["query"],
                                wrap_lines=True)  # Display the input data sent to the tool
                st.write("知识库检索结果：")
                for k, content in json.loads(event[0].content).items():
                    st.write(f"- {k}:")
                    st.code(content, wrap_lines=True) # Placeholder for tool output that will be updated later below
                s.update(label="已完成知识库检索！", expanded=False)
                if continue_save:
                    st.session_state["rag_tool_calls"][-1]["status"] = "已完成知识库检索！"
                    st.session_state["rag_tool_calls"][-1]["content"] = json.loads(event[0].content)
                else:
                    st.session_state["rag_tool_calls"].append(
                        {
                            "status": "已完成知识库检索！",
                            "knowledge_base": event[0].name.replace("_knowledge_base_tool", ""),
                            "content": json.loads(event[0].content)
                        })


def get_rag_chat_response(platform, model, temperature, input, selected_tools, KBS):
    app = get_rag_graph(platform, model, temperature, selected_tools, KBS)
    return graph_response(graph=app, input=input)

def display_chat_history():
    for message in st.session_state["rag_chat_history_with_tool_call"]:
        with st.chat_message(message["role"], avatar=get_img_base64("chatchat_avatar.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message.keys():
                for tool_call in message["tool_calls"]:
                    with st.status(tool_call["status"], expanded=False):
                        st.write("已调用 `", tool_call["knowledge_base"], "` 知识库进行查询")
                        if "query" in tool_call.keys():
                            st.write("知识库检索输入: ")
                            st.code(tool_call["query"], wrap_lines=True)  # Display the input data sent to the tool
                        st.write("知识库检索结果：")
                        for k, content in tool_call["content"].items():
                            st.write(f"- {k}:")
                            st.code(content,
                                    wrap_lines=True)  # Placeholder for tool output that will be updated later below

            st.write(message["content"])

def clear_chat_history():
    st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    st.session_state["rag_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    st.session_state["rag_tool_calls"] = []


def rag_chat_page():
    kbs = get_kb_names()
    KBS = dict()
    for k in kbs:
        KBS[f"{k}"] = get_naive_rag_tool(k)

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    if "rag_chat_history_with_tool_call" not in st.session_state:
        st.session_state["rag_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []


    with st.sidebar:
        selected_kbs = st.multiselect("请选择对话中可使用的知识库", kbs, default=kbs)

    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_llm_models(platform))
            temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["rag_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        stream_response = get_rag_chat_response(
            platform,
            model,
            temperature,
            st.session_state["rag_chat_history"][-history_len:],
            selected_kbs,
            KBS
        )

        with st.chat_message("assistant", avatar=get_img_base64("chatchat_avatar.png")):
            response = st.write_stream(stream_response)
        st.session_state["rag_chat_history"] += [{"role": 'assistant', "content": response}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'assistant', "content": response, "tool_calls": st.session_state["rag_tool_calls"]}]
        st.session_state["rag_tool_calls"] = []