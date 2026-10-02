from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    approved:str
    result:str

def human_approval(state: State):

    answer = interrupt("Do you approve this action?")

    return {
        "message": answer
    }

def ask_Human(state: State):
    answer=interrupt("Do you approve this action?(yes/no)")
    
    return {
        "approved": answer
    }
    
def check_approval(state: State):
    if state["approved"] == "yes":
        return "approved"
    return "rejected"   

def approved_node(state: State):

    return {
        "result": "Action approved and performed."
    }


def rejected_node(state: State):

    return {
        "result": "Action rejected."
    } 
graph = StateGraph(State)

graph.add_node("ask_Human", ask_Human)
graph.add_node("approved", approved_node)    
graph.add_node("rejected", rejected_node)

graph.add_edge(START, "ask_Human")
graph.add_conditional_edges(
    "ask_Human",
    check_approval,{
        "approved": "approved", 
        "rejected": "rejected"
    }
)
graph.add_edge("approved", END)
graph.add_edge("rejected", END)

checkpointer = InMemorySaver()

app = graph.compile(checkpointer=checkpointer)


config = {
    "configurable": {
        "thread_id": "hitl-1"
    }
}


result = app.invoke(
    {
        "message": "I want to perform an action."
    },
    config=config
)
print(result)
result = app.invoke(
    Command(resume="no"),
    config=config
)

print(result)
