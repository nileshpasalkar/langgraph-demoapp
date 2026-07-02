from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    name: str
    message: str


def greet_node(state: State) -> State:
    print(state)
    """Greets the user with a personalized message."""
    name = state["name"]
    state["message"] = f"Hello, {name}! Welcome to LangGraph!"
    return state
def condition()->str:
    return "greeter"


def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("greeter", greet_node)
    graph.add_edge(START, "greeter")        
    graph.add_edge("greeter", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    name = input("Enter your name: ")
    result = app.invoke({"name": name})

    print(result["message"])
