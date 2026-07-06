from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import BadRequestError
import os
import requests

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.7,
)

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers a+b and return addition."""
    return(a + b)

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers a-b and return subtraction."""
    return(a - b)

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers a*b and return multiplication."""
    return(a * b)

@tool
def weather(latitude: float, longitude: float) -> str:
    """tells the current weather at the given latitude and longitude."""
    api_url = "https://api.open-meteo.com/v1/forecast"
    response = requests.get(api_url, params={
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
    })
    response.raise_for_status()
    data = response.json()
    current = data["current_weather"]
    temp = current["temperature"]
    windspeed = current["windspeed"]
    return f"The current weather at ({latitude}, {longitude}) is {temp}°C with a wind speed of {windspeed} km/h."

@tool
def fetch_external_data(query: str) -> str:
    """Fetches data from an external API for the given query."""
    api_url = "https://api.example.com/data"  # TODO: replace with the real API URL
    response = requests.get(api_url, params={"query": query})
    response.raise_for_status()
    return response.text

tools = [add, subtract, multiply, weather, fetch_external_data]
model = llm.bind_tools(tools)

def model_call(state: State) -> State:
    """Processes the conversation with the user."""
    system_prompt = SystemMessage(content="You are my AI assistant, anster my query based on your best ability")
    messages = [system_prompt] + list(state["messages"])
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = model.invoke(messages)
            return {"messages": [response]}
        except BadRequestError:
            if attempt == max_attempts:
                raise

def condition_call(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

tool_node = ToolNode(tools=tools)

agent = StateGraph(State)
agent.add_node("model_call", model_call)
agent.add_node("tool_node", tool_node)
agent.add_edge(START, "model_call")
agent.add_conditional_edges("model_call", condition_call, {"continue": "tool_node", "end": END})
agent.add_edge("tool_node", "model_call")
app = agent.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [HumanMessage(content="add 50 to 32 and then multiply the with 2 and what is weather in mumbai city")]}
print_stream(app.stream(inputs, stream_mode="values"))
