from typing import TypedDict, Union
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    messages: list[Union[ChatCompletionSystemMessageParam,ChatCompletionUserMessageParam]]    

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def process(state: State) -> State:
    """Processes the conversation with the user."""    
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=state["messages"]
    )
    print(response.choices[0].message.content)
    state["messages"].append(ChatCompletionUserMessageParam(content=response.choices[0].message.content, role="assistant"))
    return state

graph = StateGraph(State)
graph.add_node("process", process)
 
graph.add_edge(START, "process") 
graph.add_edge("process", END)
agent = graph.compile()

conversation_state = []
user_input = input("You: ")
while user_input.lower() != "exit":    
    conversation_state.append(ChatCompletionUserMessageParam(content=user_input, role="user"))
    result = agent.invoke({"messages": conversation_state})
    conversation_state = result["messages"]
    user_input = input("You: ")



