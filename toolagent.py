from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionMessageParam, ChatCompletionToolMessageParam
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import os

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[ChatCompletionMessageParam], add_messages]

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
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

tools = [add, subtract, multiply]

def model_call(state: State) -> State:
    """Processes the conversation with the user."""    
    system_prompt = ChatCompletionSystemMessageParam(content="You are my AI assistant, anster my query based on your best ability", role="system")
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages= [system_prompt] + state["messages"]
    )  
    return {"messages": [response]}

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
        message = s.choices[0].message
        if isinstance(message, tuple):
            print(message)
        else: 
            message.preety_print()

inputs = {"messages": [ChatCompletionUserMessageParam(content="add 40 to 32", role="user")]}
app.invoke({"messages": [ChatCompletionUserMessageParam(content="add 40 to 32", role="user")]})
        



