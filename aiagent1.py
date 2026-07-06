from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    messages: list[ChatCompletionUserMessageParam]    

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",    
)

def process(state: State) -> State:
    """Processes the conversation with the user."""    
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=state["messages"]    
    )
    print(response.choices[0].message.content)
    return state

graph = StateGraph(State)
graph.add_node("process", process)
 
graph.add_edge(START, "process") 
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("You: ")
while user_input.lower() != "exit":    
    agent.invoke({"messages": [ChatCompletionUserMessageParam(content=user_input, role="user")]})
    user_input = input("You: ")



