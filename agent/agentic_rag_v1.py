
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq 
from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState  
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.sqlite import SqliteSaver 
from datetime import datetime
import wikipedia
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import InMemorySaver   
import os
from dotenv import load_dotenv  
import sqlite3
load_dotenv()

conn=sqlite3.connect(
        "memory.db",
        check_same_thread=False
)
memory=SqliteSaver(conn)
class State(MessagesState):
    relevance:str
    rewritten_question:str
    context:str
    attempts:str
    
    
class RouteDecision(BaseModel):
    route: Literal["retrieval", "tool"]  
    
   
loader=PyPDFLoader("llm.pdf");
docs=loader.load();
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
text_splitter=RecursiveCharacterTextSplitter(
  chunk_size=1500,
  chunk_overlap=200)   

chunk=text_splitter.split_documents(docs)
vectorstore=FAISS.from_documents(chunk,embeddings)  
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}
) 

llm=ChatGroq(model="openai/gpt-oss-20b",
             api_key=os.getenv("GROQ_API_KEY"))   
router_llm=llm.with_structured_output(RouteDecision)           
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
def search_wikipedia(text: str) -> str:
   
    """
    Search Wikipedia for general knowledge,
    history, science, geography, people,
    programming languages, companies,
    and topics not covered by the PDF.
    """
    
    response=llm.invoke(f"""
    summarize the given text in one sentence 
        Text:
        {text}
        """
    )
    return response.content
def route_question(state: State):     
    question = state["messages"][-1].content

    decision = router_llm.invoke(
        f"""
Choose exactly one route.

retrieval = questions that should be answered using the PDF/document.
This includes concepts, definitions, explanations, or topics that may be
covered in the PDF.

tool = calculations, current time, word count, or tasks that require
an actual tool.

If the question could reasonably be answered from the PDF,
choose retrieval.

Question:
{question}
"""
    )

    print("ROUTE =", decision.route)

    return decision.route
def retrieve_documents(state:State):    
    
    attempts=state["attempts"]+1 
    question=state["rewritten_question"]
    
    retrieved_docs=retriever.invoke(question)
    print("retrieveDocuments - attempt", state["attempts"] + 1)
    
    pairs=[
        [question,doc.page_content]
        for doc in retrieved_docs       
    ]
    
    scores=reranker.predict(pairs)
    
    ranked_docs=sorted(
        zip(scores,retrieved_docs),
        key=lambda x:x[0],
        reverse=True
    )
    top_docs=[
        doc
        for score,doc in ranked_docs[:3]
    ]
    
    context="\n\n".join(
        doc.page_content
        for doc in top_docs    
    )
    return {
        "context":context,
        "attempts":attempts
    }
    
def relevance_checker(state: State):


    question = state["rewritten_question"]
    context = state["context"]

    prompt = f"""
Determine whether the retrieved context is relevant to the question.

Return ONLY:

YES

or

NO

Choose YES if the context contains information related to the
topic of the question, even if it does not contain the complete answer.

Choose NO only if the context is unrelated to the question.

Question:
{question}

Context:
{context}
"""

    response = llm.invoke(prompt)

    print("RELEVANCE =", response.content)

    return {
        "relevance": response.content.strip().upper()
    }

def rewritten_question(state:State):
    print("rewriteNode")
    user_question=state["messages"][-1].content  
    
    prompt = f"""
Rewrite the user's question so it is clear,
complete, and optimized for document retrieval.

Do not answer the question.

Only return the rewritten question.

Question:
{user_question}
    """   
    response=llm.invoke(prompt)
    return {
        "rewritten_question":response.content       
    }
    
    
def improve_query(state:State):
    print("improve query")
    
    question=state["messages"][-1].content
    old_query=state["rewritten_question"]
    context=state["context"]
    
    prompt = f"""
The previous retrieval did not provide enough relevant information.

Create a better search query for retrieving the answer from the PDF.

Original question:
{question}

Previous retrieval query:
{old_query}

Previous retrieved context:
{context}
Rewrite the retrieval query to focus on the most important
keywords and concepts.

Return ONLY the new retrieval query.
"""
    

    response=llm.invoke(prompt)
    return{
        "rewritten_question":response.content.strip()
    }

def route_retry(state:State):   
    if state["attempts"]<2:
        return "retry"
    return "fallback"       
def chatbot(state: State):

    
    system_prompt = """
You are an AI assistant with access to tools.

Use tools whenever they help answer the question.

If a tool result is insufficient,
you may use another relevant tool.

Do not call irrelevant tools.

Answer only after you have enough information.
"""

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

  
    print("TOOL CALLS:", response.tool_calls)

    return {
        "messages": [response]
    } 
def answer_node(state: State):
    print("answerNode")
    question=state["rewritten_question"]    
    context=state["context"]    
    
    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say exactly:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response]
    }

def route_relevance(state: State):  
    if state["relevance"] == "YES":
        return"answer"
    
    return route_retry(state) 
def fallback_node(state: State):

    return {
        "messages": [
            AIMessage(
                content="I don't know based on the provided document."
            )
        ]
    }
tools=[add,multiply,word_count,divide,
       current_time,summarize,search_wikipedia]
llm_with_tools=llm.bind_tools(tools)             
tool_node=ToolNode(tools)
graph=StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_node("rewrite", rewritten_question)
graph.add_node("retrieval", retrieve_documents)
graph.add_node("relevance", relevance_checker)
graph.add_node("answer", answer_node)
graph.add_node("improve_query", improve_query)  
graph.add_node("fallback", fallback_node)   

graph.add_conditional_edges(
    START,
    route_question,
    {
        "retrieval": "rewrite",
        "tool": "chatbot"
    }
)
graph.add_edge("rewrite", "retrieval")
graph.add_edge("retrieval", "relevance")
graph.add_conditional_edges("relevance",
                            route_relevance,
                            {
                                "answer":"answer",
                                "retry":"improve_query",
                                "fallback":"fallback"  
                            })

graph.add_edge("improve_query","retrieval")
graph.add_conditional_edges("chatbot",
                            tools_condition,
                            {
                                "tools": "tools",
                                "__end__": END  
                            })
graph.add_edge("tools", "chatbot")
graph.add_edge("answer", END)
graph.add_edge("fallback", END)

app=graph.compile(
    checkpointer=memory
)
config={
    "configurable":{
        "thread_id":"rahul"
    }
}
while True:
    question=input("You: ")
    
    if question.lower()=="exit":
        break
    
    result=app.invoke({
    "messages":[
        HumanMessage(
            content=question
        )
    ],"attempts":0
},    config=config)
    print(result["messages"][-1].content)