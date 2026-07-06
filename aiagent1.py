from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    messages: list[BaseMessage]

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def process(state: State) -> State:
    """Processes the conversation with the user."""
    response = llm.invoke(state["messages"])
    print(response.content)
    return state

graph = StateGraph(State)
graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("You: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("You: ")
