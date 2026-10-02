
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END ,MessagesState
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


class State(MessagesState):
  pass


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [multiply, add]

tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


def should_continue(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    should_continue
)

graph.add_edge("tools", "chatbot")

with SqliteSaver.from_conn_string("checkPoints.db") as checkpointer:

    app = graph.compile(
        checkpointer=checkpointer
    )

    config1 = {
        "configurable": {
            "thread_id": "user1"
        }
    }

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Rahul.")
            ]
        },
        config=config1
    )

    print(result["messages"][-1].content)

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content="What is my name?")
            ]
        },
        config=config1
    )

    print(result["messages"][-1].content)
    config2 = {
    "configurable": {
        "thread_id": "user2"
    }
}

    result = app.invoke(
    {
        "messages": [
            HumanMessage(content="What is my name?")
        ]
    },
    config=config2
)

print(result["messages"][-1].content)