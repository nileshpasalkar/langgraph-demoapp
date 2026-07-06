from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    name: str
    guessedNumber: int
    actualNumber: int    

def greet_user_node(state: State) -> State:
    """Greets the user and prompts them to guess a number."""
    state["name"] = input("Please enter your name: ")
    print(f"Welcome to the Number Guessing Game, {state['name']}!")
    print("Try to guess the number between 1 and 100.")
    return state

def guess_number_node(state: State) -> State:
    """Checks if the guessed number is correct and updates the state accordingly."""
    state["guessedNumber"] = int(input("Enter your guess: "))
    return state

def check_guess_node(state: State) -> State:
    if state["guessedNumber"] == state["actualNumber"]:
        print("Congratulations! You've guessed the correct number.")
        return "correct"
    elif state["guessedNumber"] < state["actualNumber"]: 
        print("Your guess is too low. Try again.")
        return "wrong"
    elif state["guessedNumber"] > state["actualNumber"]:
        print("Your guess is too high. Try again.")
        return "wrong"
    
def build_graph() -> StateGraph:
    graph = StateGraph(State)    

    graph.add_node("guess_number_node", guess_number_node)    
    graph.add_edge(START, "guess_number_node")
    graph.add_conditional_edges("guess_number_node",
                                check_guess_node,
                                {"correct": END, "wrong": "guess_number_node"})    
    graph.add_edge("guess_number_node", END)

    return graph.compile()

if __name__ == "__main__":
    import random

    app = build_graph()
    actual_number = random.randint(1, 100)
    state = State(name="", guessedNumber=0, actualNumber=actual_number)
    
    print("Welcome to the Number Guessing Game!")
    print("Try to guess the number between 1 and 100.")
    
    result = app.invoke(state)