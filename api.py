import json
import time
from typing import *

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from utils import get_chatllm, get_llm_models
from tools import (
    weather_search_tool,
    get_naive_rag_tool,
    get_duckduckgo_search_tool,
    arxiv_search_tool,
    wikipedia_search_tool,
    daily_ai_papers_tool,
)


class OpenAIOutput(BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    model: Optional[str] = None
    object: Literal[
        "chat.completion", "chat.completion.chunk"
    ] = "chat.completion.chunk"
    role: Literal["assistant"] = "assistant"
    finish_reason: Optional[str] = None
    created: int = Field(default_factory=lambda: int(time.time()))
    tool_calls: List[Dict] = []

    class Config:
        extra = "allow"

    def model_dump(self) -> dict:
        result = {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "created": self.created,
            **(self.model_extra or {}),
        }

        if self.object == "chat.completion.chunk":
            result["choices"] = [
                {
                    "delta": {
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    },
                    "role": self.role,
                }
            ]
        elif self.object == "chat.completion":
            result["choices"] = [
                {
                    "message": {
                        "role": self.role,
                        "content": self.content,
                        "finish_reason": self.finish_reason,
                        "tool_calls": self.tool_calls,
                    }
                }
            ]
        return result

    def model_dump_json(self):
        return json.dumps(self.model_dump(), ensure_ascii=False)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # print(last_message)
    if last_message.tool_calls:
        return "tools"
    return END


def get_agent_graph(platform, model, temperature, tools):
    tool_node = ToolNode(tools)

    def call_model(state, config):
        messages = state['messages']
        llm = get_chatllm(platform, model, temperature=temperature, base_url="http://192.168.8.68:9997/v1")
        llm = llm.bind_tools(tools, parallel_tool_calls=False)
        response = llm.invoke(messages, config=config)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app



app = FastAPI()

@app.post("/v1/chat/completions")
def chat(
    query: str = "北京今天天气如何？",
    platform: str = "Xinference",
    model: str | None = None,
    temperature: float = 0.01,
):
    def graph_response():
        default_llm = list(get_llm_models(platform, base_url="http://192.168.8.68:9997").keys())[0]
        graph = get_agent_graph(
            platform=platform,
            model=model or default_llm,
            temperature=temperature,
            tools=[weather_search_tool],
        )
        for msg, meta in graph.stream(
            {"messages": query},
            config={"configurable": {"thread_id": 42}},
            stream_mode="messages",
        ):
            yield msg # 这里需要把msg的内容转换成OpenAIOutput
    return StreamingResponse(graph_response())


# for x in chat():
#     from rich import print
#     print(x)

