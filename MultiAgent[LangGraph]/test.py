from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage

import os



class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-b5433580f43948fcbd4a1b84171f7835",
    base_url="https://api.deepseek.com/v1"
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 添加节点和边
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 打印图结构
print(graph.get_graph().draw_mermaid())

def stream_graph_updates(user_input: str, config: dict):
    """
    创建一个初始状态，其中包含用户输入作为消息
    调用图的stream方法来流式处理图的执行
    对于每个事件，打印出助手的响应
    """
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values"
        ):
        for value in event.values():
            # print("DEBUG:", value)
            if isinstance(value[-1], AIMessage):
                print("Assistant:", value[-1].content)
        # event["messages"][-1].pretty_print()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        config = {"configurable": {"thread_id": "1"}}
        stream_graph_updates(user_input, config)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
