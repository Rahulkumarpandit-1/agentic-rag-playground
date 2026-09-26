from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class State(TypedDict):
    a: int
    b: int
    result: int
    
def ask_approval(state: State):

    answer = interrupt(
        f"Approve adding {state['a']} + {state['b']}? (yes/no)"
    )

    if answer != "yes":
        return {
            "result": "Action rejected."
        }

    result = add.invoke({
        "a": state["a"],
        "b": state["b"]
    })

    return {
        "result": result
    }
    
graph = StateGraph(State)

graph.add_node("ask_approval", ask_approval)

graph.add_edge(START, "ask_approval")
graph.add_edge("ask_approval", END)

checkpointer = InMemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": "tool-approval-1"
    }
}

result = app.invoke(
    {
        "a": 10,
        "b": 5
    },
    config=config
)

print(result)

        
result = app.invoke(
    Command(resume="no"),
    config=config
)

print(result)        