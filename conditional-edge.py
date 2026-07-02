from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    number1: int
    number2: int
    operation: str
    final_result: int


def addition_node(state: State) -> State:    
    """Adds two numbers and stores the result in the state."""
    state["final_result"] = state["number1"] + state["number2"]
    return state

def substraction_node(state: State) -> State:    
    """Subtracts two numbers and stores the result in the state."""
    state["final_result"] = state["number1"] - state["number2"]
    return state

def decide_next_node(state: State) -> State:
    """Routes to the appropriate operation based on the condition."""    
    if state["operation"] == "+":
        return "addition_operation"
    elif state["operation"] == "-":
        return "substraction_operation"

def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("addition_node", addition_node)
    graph.add_node("substraction_node", substraction_node)
    graph.add_node("router_node", lambda state: state)    

    graph.add_edge(START, "router_node")        
    graph.add_conditional_edges("router_node",
                                decide_next_node,
                                {
                                    "addition_operation": "addition_node",
                                    "substraction_operation": "substraction_node"
                                })
    graph.add_edge("addition_node", END)    
    graph.add_edge("substraction_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print("Available operations: +, -")
    number1 = int(input("Enter first number: "))
    number2 = int(input("Enter second number: "))
    operation = input("Enter operation: ")  
    state = State(number1=number1, number2=number2, operation=operation, final_result=0)
    result = app.invoke(state)

    print(result)
