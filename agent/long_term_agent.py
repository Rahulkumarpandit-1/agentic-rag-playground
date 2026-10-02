from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState,StateGraph, START, END
from langgraph.store.memory import InMemoryStore
import langgraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
import inspect
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore
import os

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

store = InMemoryStore()


class State(MessagesState):
    memory:dict
class UserMemory(BaseModel):
    name: str | None = Field(default=None)
    language: str | None = Field(default=None)       
class MemoryDecision(BaseModel):
    save_memory: bool
        
router_llm=llm.with_structured_output(MemoryDecision)   

def should_continue(state:State):
    last_message=state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"
    return END
    
def save_memory(state:State):
    user_message=state["messages"][-1].content
    extracted=memory_llm.invoke(user_message)   
    
    if extracted.name:
        store.put(
            ("user","rahul"),
            "name",
            {"name":extracted.name}
        )    
        
    if extracted.language:
        store.put(
            ("user", "rahul"),
            "language",
            {"language": extracted.language}
        )
        memories = store.search(
    ("user", "rahul")
)
 
        for memory in memories:
          print(memory)
    return state    
   
def agent(state:State):
    memory=state["memory"]
    
    response=llm_with_tools.invoke(
            [
            SystemMessage(
                content=f"""
                You are a helpful AI assistant.

                Long-term memory:
                {memory}

                Use tools when necessary.
                """
            )
        ] + state["messages"]
    )
    return {
            "messages":[response]
    }
    
memory_llm = llm.with_structured_output(UserMemory)        
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)
def  chatbot(state:State):
    memory=state["memory"]
    
    response = llm.invoke(
        f"""
        Answer the user's question.

        Long-term memory:
        {memory}

        User:
        {state["messages"][-1].content}
        """
    )
    return {
        "messages":[response]
    }
def read_memory(state: State):

    memories = store.search(
        ("user", "rahul")
    )

    memory = {}

    for item in memories:
        memory[item.key] = item.value

    return {
        "memory": memory
    }
    
def retrieve_memory(state:State):
    memories=store.search(
        ("user","rahul")
    )    
    memory={}
    for item in memories:
        memory[item.key]=item.value 
        
    return{
        "memory":memory
    }    
def memory_router(state:State):
    user_message=state["messages"][-1].content
    decision=router_llm.invoke(user_message)    
    if decision.save_memory:
        return "save_memory"
    return "read_memory"
graph = StateGraph(State)
graph.add_node("agent",agent)
graph.add_node("tools",ToolNode(tools))
graph.add_node("chatbot", chatbot)
graph.add_node("memory_router", memory_router)
graph.add_node("retrieve_memory", retrieve_memory)
graph.add_edge(START, "retrieve_memory")

graph.add_edge("retrieve_memory","agent")
graph.add_conditional_edges(
    "agent",
    should_continue
)
graph.add_edge("tools", END)


app = graph.compile()


result = app.invoke({
    "messages": [
        HumanMessage(content="What is 10 + 5?")
    ]
})

print(result["messages"][-1].content)
