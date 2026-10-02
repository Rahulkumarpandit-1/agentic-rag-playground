
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq 
from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState  
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import InMemorySaver   
import os
from dotenv import load_dotenv  
import sqlite3
load_dotenv()
conn=sqlite3.connect(
    "memory.db"
)
cursor=conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        thread_id TEXT,
        key TEXT,
        value TEXT
)"""

)
conn.commit() 

def save_memory(thread_id,key,value):
    cursor.execute("""
        INSERT INTO memory
        VALUES(?,?,?)
        """,(thread_id,key,value))
    conn.commit()
    
save_memory(
    "1",
    "name",
    "Rahul"
)

def get_memory(thread_id,key):
    cursor.execute("""
                   SELECT value FROM memory
                   WHERE thread_id=? AND key=?
                   """,
                   (thread_id,key)
    )
    result=cursor.fetchone()
    if result:
        return result[0]
    else:
        return None
memory=InMemorySaver()

loader=PyPDFLoader("llm.pdf")
docs=loader.load()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200)

chunk=text_splitter.split_documents(docs)
vectorstore=FAISS.from_documents(chunk,embeddings)  
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
retrieved_docs=retriever.invoke("StateGraph START END nodes edges LangGraph")

    
llm=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)   
class MemoryItem(BaseModel):
    key: str
    value: str
    found: bool
                 
class State(MessagesState):
    rewritten_question:str
    context:str
    relevance:str
    memory:dict
    pages:list[str]
class RouteDecision(BaseModel):
    route:Literal[
        "memory",
        "tools",
        "retrieval",
        "direct"
    ] 

class  MemoryQuery(BaseModel):
    key:str
memory_query_llm = llm.with_structured_output(
    MemoryQuery
)        

router_llm =llm.with_structured_output(RouteDecision)    


def route_question(state):
    user_message = state["messages"][-1].content

    decision = router_llm.invoke(
        f"""
You are a routing assistant.

Choose exactly one route.

memory:
Questions about the user's personal information,
preferences, profile, name, memory.

tools:
Math, arithmetic, calculations,word_count.

retrieval:
Questions about AI, Machine Learning,
Deep Learning, LLMs, Transformers,
RAG, LangChain, LangGraph,
Embeddings, Vector Databases,
Prompt Engineering, Agents,
or anything that should be answered
from the PDF knowledge base.

direct:
Everything else.

Question:
{user_message}
"""
    )

    print(decision)

    return decision.route

@tool
def current_time():
    """Get current Time"""
    
@tool
def word_count(text:str):
    """Count words in text  """       
@tool
def add(a:int,b:int)->int:
    """add two number:"""
    return a+b  
@tool
def multiply(a:int,b:int)->int: 
    """multiply two number"""
    return a*b
@tool
def divide(a:int,b:int)->int:
    "division of number"
    return a/b
@tool
def search_pdf(question:str)->str:
    """search the pdf knowledge based"""
    docs=retriever.invoke(question)
    
    context="\n\n".join(
        doc.page_content
        for doc in docs
    )
    return context
def rewrite_question(state:State):  
    
    user_message=state["messages"][-1].content  
    history_text = "\n".join(
    msg.content
    for msg in state["messages"][:-1]
)
    prompt = f"""
You are a query rewriter.

Given the conversation history and
the latest user question, rewrite
the latest question into a complete
standalone question.

Do not answer.

If the question is already clear,
return it unchanged.

Chat History:
{history_text}

Latest Question:
{user_message}
"""
    response = llm.invoke(prompt)
    return {
            "rewritten_question":response.content}

def relevance_checker(state:State):
    question=state["rewritten_question"]
    context=state["context"]
    response=f"""You are a relevance checker.

Determine whether the provided context
contains enough information to answer
the question.

Return ONLY:

YES
or
NO

Question:
{question}

Context:
{context}
"""
    response = llm.invoke(response)

    return {
    "relevance":response.content.strip().upper()
}
def memory_node(state: State):

    user_message = state["messages"][-1].content
    response = llm.invoke(
    f"""
Determine which memory key the user is asking for.

Possible keys:
name
country
college

Examples:
What is my name? -> name
Where am I from? -> country
Which college do I study in? -> college

Question:
{user_message}

Return ONLY one word:
name
country
college
"""
)

    key = response.content.strip().lower()
    value=get_memory("1",key)
    print("VALUE:", value)

    if value:
     return {
            "messages": [
                AIMessage(
                    content=value
                )
            ]
        }
tools=[add,multiply,word_count,search_pdf,divide]
llm_with_tools=llm.bind_tools(tools)


def retrieval_node(state:State):
    question=state["rewritten_question"]    
    
    
    retrieved_docs=retriever.invoke(question)
    context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)
    pages=[]
    
    for doc in retrieved_docs:
      pages.append(doc.metadata["page_label"])
    return{ 
           "context":context,
           "pages":pages}
    
def answer_node(state:State):
    question=state["rewritten_question"]
    context=state["context"]
    prompt = f"""
You are a helpful assistant.

Use ONLY the information found in the context.

If the answer is not present in the context,
reply exactly:

I don't know based on the provided document.

Question:
{question}

Context:
{context}

Answer:
"""
    response = llm.invoke(prompt)
    answer = f"""
{response.content}

Source Pages: {state['pages']}
"""

    return {
    "messages": [
        AIMessage(content=answer)
    ]
}

def direct_node(state:State):
    print("DIRECT")
    return{}    
def route_relevance(state: State):
    if state["relevance"] == "YES":
        return "answer"

    return "fallback"
def fallback_node(state: State):
    return {
        "messages": [
            AIMessage(
                content="I don't know based on the provided document."
            )
        ]
    }
    
def chatbot(state: State):
    response = llm_with_tools.invoke(
        state["messages"]
    )

    print("CONTENT:", response.content)
    print("TOOL CALLS:", response.tool_calls)
    return {
        "messages": [response]
    }    
tools_node=ToolNode(tools)    
llm_with_tools=llm.bind_tools(tools)
graph=StateGraph(State)

graph.add_node("memory",memory_node)
graph.add_node("tools",tools_node)
graph.add_node("retrieved_document",retrieval_node)
graph.add_node("answer",answer_node)       
graph.add_node("chatbot", chatbot)
graph.add_node("rewrite_question",rewrite_question)
graph.add_node("relevance_checker",relevance_checker)
graph.add_node("fallback",fallback_node)    
graph.add_node("direct",direct_node)
graph.add_conditional_edges(
    START,
    route_question,
    {
        "memory":"memory",
        "tools":"chatbot",
        "retrieval":"chatbot",
        "direct":"direct"
    }
)
graph.add_edge("memory",END)
@tool
def word_count(text: str) -> int:
    """Count words in text"""
    return len(text.split())
graph.add_edge("tools","chatbot")  
graph.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
) 
graph.add_edge("rewrite_question","retrieved_document")
graph.add_edge("retrieved_document","relevance_checker")
graph.add_conditional_edges(
    "relevance_checker",
    route_relevance,
    {
        "answer":"answer",
        "fallback":"fallback"
    }
)
graph.add_edge("direct",END)
graph.add_edge("answer", END)
graph.add_edge("fallback", END)
app=graph.compile(
    checkpointer=InMemorySaver()    
)
config={
    "configurable":{
        "thread_id":"1"
    }
}
tool_map = {
    "add": add,
    "multiply": multiply,
    "word_count": word_count    
}
print(get_memory("1","name"))
while True:
    question=input("you: ")
    
    if question.lower()=="exit":
       break
    result=app.invoke(
    {
        "messages":[
            HumanMessage(content=question)
        ]
    },
    config=config
    )  

  
    print(result["messages"][-1].content
)
    
                     


