from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0
    )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pdf_path = "Roadmap_Q1_Q2_2026.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

pdf_loader = PyPDFLoader(pdf_path)


try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = "chroma_db"
collection_name = "stockmarket"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Chroma vector store created and persisted at '{persist_directory}' with collection name '{collection_name}'")

except Exception as e:
    print(f"Error creating Chroma vector store: {e}")
    raise


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retrieve_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Roadmap Q1 Q2 2026 document.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Roadmap Q1 Q2 2026 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retrieve_tool]

llm = llm.bind_tools(tools)

class RagAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: RagAgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Roadmap Q1 Q2 2026 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the Roadmap Q1 Q2 2026 data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tool_dict = {our_tool.name: our_tool for our_tool in tools}

def llm_agent(state: RagAgentState) -> RagAgentState:    
    """call llm with current state and return new state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages": [messages]}

def retriver_agent(state: RagAgentState) -> RagAgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tool_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tool_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(RagAgentState)
graph.add_node("llm", llm_agent)
graph.add_node("retriever", retriver_agent)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever", False: END}
)
graph.add_edge("retriever", "llm")
ragagent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = ragagent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()
