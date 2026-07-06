from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import BadRequestError
import os
import requests

load_dotenv()

document_content = ""

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_email(content: str) -> str:
    """Updates the email  with the provided content."""
    global document_content
    document_content = content
    return f"Email has been updated successfully. Current content:\n{document_content}"

@tool
def save_email(filename: str) -> str:
    """Saves the current email to text file and finish the process.
    Args: 
        filename (str): The name of the file to save the email content to.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:        
        with open(filename, "w") as f:
            f.write(document_content)
        print(f"Email saved to '{filename}' successfully.")
        return f"Email has been successfully saved to '{filename}'."
    except Exception as e:
        return f"An error occurred while saving the email: {str(e)}"
    
tools = [update_email, save_email]
model = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1").bind_tools(tools)

def emaildrafter(state: State) -> State:
    system_message = SystemMessage(content=f"""You are Drafter, a helpful writing assistant. You are going to help the user update and modify email.
                                   
                                   - If user wants to update the email, you will call the 'update_email' tool with the content provided by user.
                                   - If user wants to save the email, you will call the 'save_email' tool with the filename provided by user.
                                   - Make sure to always show the content of the email after updating it. 

                                   The current email content is:{document_content}                                     
                                   """)
    
    if not state["messages"]:
        user_input = "I am ready to help with email, what you want to do?"
        user_message = HumanMessage(content=user_input)
    
    else:
        user_input = input("\n What would you like to do with email?: ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_message] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: State) -> str:
    """Determines whether the conversation should continue or end based on the last message."""

    messages = state["messages"]

    if not messages:
        return "continue"  # Continue if there are no messages yet

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
        "saved" in message.content.lower()):
            return "end"  # End if the last message indicates the email has been saved
    
    return "continue"  # Continue otherwise
            
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(State)

graph.add_node("emaildrafter", emaildrafter)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "emaildrafter")
graph.add_edge("emaildrafter", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "emaildrafter", "end": END})
app = graph.compile()

def run_email_drafter():
    """Runs the email drafter application."""

    print("\n=======Email Drafter========")
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n=======Drafter finished========")

if __name__ == "__main__":
    run_email_drafter()