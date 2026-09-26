from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState       
from langgraph.prebuilt import ToolNode         

import os
from dotenv import load_dotenv
load_dotenv()
store=InMemoryStore()   
llm=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
    
)
store.put( 
    ("user","rahul"),
    "name",
    {
        "name":"rahul"
    }
)
class MemoryDecision:
    save_memory:bool
    
router_llm=llm.with_structured_output(MemoryDecision)    
class State(MessagesState):
    memory:dict

def read_memory(state:State):
    memories=store.search("user","rahul")
    
    memory={}
    for item in memories:
        memory[item.key]=item.value 
        
    return {
        "memory":memory 
    }    
def memory_router(state:State):
    user_message=state["messages"][-1]
    
    decision=router_llm.invoke([user_message]) 
    
    if decision.save_memory:
        return "save_memory"    
    
    return "agent"
@tool
def add(a:int ,b:int)->int:
    """add two number:"""
    return a+b
@tool
def multiply(a:int,b:int)->int:
    """multiply two number"""
    return a*b

tools=[add,multiply]
llm_with_tools=llm.bind_tools(tools)
tool_node=ToolNode(tools)

def agent(state:State):
    memory=state["memory"]
    
    messages=[
        SystemMessage(
            content=f"""user Memory: {memory}"""    
        )
    ]+state["messages"]
    
    response=llm_with_tools.invoke(messages)    
    return {
        "messages":[response]
    }
    
def should_continue(state:State):
    last_message=state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"
    
    return END

graph=StateGraph(State)
graph.add_node("agent",agent)
graph.add_node("read_memory",read_memory)
graph.add_node("tools",tool_node)

graph.add_edge(START,"read_memory")
graph.add_edge("read_memory","agent")
graph.add_conditional_edges(
    "agent",
    should_continue
)   
graph.add_edge("tools","agent")
app=graph.compile()

result=app.invoke({
    "messages":[
        HumanMessage(content="what is my name?"),
    ]
})

print(result["messages"][-1].content)