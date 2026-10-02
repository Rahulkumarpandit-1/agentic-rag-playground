from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
load_dotenv()


llm=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)
@tool
def multiply (a:int,b:int) -> int:
    """multiply two numbers"""
    return a*b  
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    return a / b
tools={
    "multiply":multiply,    
    "add":add,
    "divide":divide 
}

llm_with_tools=llm.bind_tools([multiply,add,divide])
question = "What is 10 divided by 0?"
messages = [
    HumanMessage(content=question)
]
try: 
 divide.invoke({"a": 10, "b": 0})
except Exception as e:
    print("tool failed",e)
while True:
  response=llm_with_tools.invoke(messages)
  
  if not response.tool_calls:
      final_response=response
      break
  messages.append(response)
  tool_messages=[]
  for tool_call in response.tool_calls:
    print(tool_call)
    tool_args=tool_call["args"]
    tool_name=tool_call["name"]
    
    tool=tools[tool_name]
    try:
     tool_result=tool.invoke(tool_args)
     print(tool_result)
     print(tool_name,tool_args)
     tool_message = ToolMessage(
         content=str(tool_result),
         tool_call_id=tool_call["id"]
     )
    except Exception as e:
     print(tool_name,tool_args)
     tool_message = ToolMessage(
         content=f"Tool failed: {e}",
         tool_call_id=tool_call["id"]
     )
    
     
    tool_messages.append(tool_message)
  messages.extend(tool_messages)


print("Tool calls:", final_response.tool_calls)
print(final_response.content)