from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


class State(MessagesState):
    pass


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [add]

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


def approval(state: State):

    last_message = state["messages"][-1]

    tool_call = last_message.tool_calls[0]

    answer = interrupt(
        f"Approve tool call {tool_call['name']} "
        f"with arguments {tool_call['args']}? (yes/no)"
    )

    if answer != "yes":

        return {
            "messages": [
                ToolMessage(
                    content="Tool execution rejected by human.",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            ]
        }

    return {}


def should_continue(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "approval"

    return END


graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("approval", approval)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    should_continue
)

graph.add_edge("approval", "tools")

graph.add_edge("tools", "chatbot")


checkpointer = InMemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)


config = {
    "configurable": {
        "thread_id": "agent-hitl-1"
    }
}


result = app.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is 10 + 5?"
            )
        ]
    },
    config=config
)

print(result)


result = app.invoke(
    Command(resume="yes"),
    config=config
)

print(result["messages"][-1].content)