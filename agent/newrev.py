from pydantic import BaseModel  
from langchain_groq import ChatGroq

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)
class RouteDecision(BaseModel):
   route=str
   decision=str
   
   
router_llm=llm.with_structured_output(RouteDecision)     

def route_question(state:State):
    user_message=state["messages"][-1].content    
    decision=router_llm.invoke(user_message)    
    
    return decision.route