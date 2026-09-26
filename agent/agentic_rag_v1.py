
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
from datetime import datetime
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import InMemorySaver   
import os
from dotenv import load_dotenv  
import sqlite3
load_dotenv()

class State(MessagesState):
    pass
    
loader=PyPDFLoader("llm.pdf");
docs=loader.load();
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

llm=ChatGroq(model="openai/gpt-oss-20b",
             api_key=os.getenv("GROQ_API_KEY"))   
          
@tool
def current_time():
    """Get current time"""
    return str(datetime.now())
    
@tool
def word_count(text:str) -> int:
    """Count words in text"""
    return len(text.split())    
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
def summarize(text:str)->str:
    """ summarize the given task in one sentence"""
    
@tool
def search_pdf(question:str)->str:
    """Search the PDF knowledge base for information
about AI, Machine Learning, Deep Learning,
LLMs, Transformers, Embeddings, RAG,
LangChain, LangGraph, Agents,
Prompt Engineering, and Vector Databases.
"""
    docs=retriever.invoke(question)
    
    context="\n\n".join(
        doc.page_content
        for doc in docs
    )
    return context    
def chatbot(state: State):

    system_prompt = """
You are an AI agent.

Use tools whenever needed.

IMPORTANT:
If the user asks for a word count,
you MUST use the word_count tool.
Never count words manually.

If information must come from the PDF,
use search_pdf first.

You may use multiple tools before answering.
"""

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    print("CONTENT:", response.content)
    print("TOOL CALLS:", response.tool_calls)

    return {
        "messages": [response]
    } 
    
    
tools=[add,multiply,word_count,search_pdf,divide]
llm_with_tools=llm.bind_tools(tools)             
tool_node=ToolNode(tools)

graph=StateGraph(State)
graph.add_node("tools",tool_node)
graph.add_node("chatbot",chatbot)

graph.add_edge(START,"chatbot")
graph.add_conditional_edges("chatbot",
                            tools_condition,
                            {"tools":"tools",
                            "__end__":END
                            }
                            )
graph.add_edge("tools","chatbot")


app=graph.compile()

result=app.invoke({
    "messages":[
        HumanMessage(content="Find the PDF definition of embeddings and count its words.")]
})
print(result["messages"][-1].content)