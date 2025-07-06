from langchain.chat_models import init_chat_model
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import yaml
import os

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDe82ID5jmROVw7ps-iv-2NZvBbJiw5mxg"  # Replace with your actual key

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 200.3,
        "AAPL": 100.4,
        "AMZN": 150.0,
        "RIL": 87.6
    }.get(symbol.upper(), 0.0)
@tool
def document_qa(question: str) -> str:
    '''Answer questions from documents'''
    return f"Document response for: {question}"

@tool
def option_price(symbol: str )-> float:
    '''Return the current price of a option given the option symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 220.0,
        "AAPL": 200.4,
        "AMZN": 250.0,
        "RIL": 87.6
    }.get(symbol.upper(), 0.0)
    
    
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def get_enabled_tools(config):
    """Return enabled tools based on YAML config"""
    tool_map = {
        "stock_price": get_stock_price,
        "document_qa": document_qa,
        "option_price" : option_price
    }
    return [tool_map[name] for name, enabled in config["agent"]["tools"].items() if enabled]
def main():
    config = load_config()
    
    # Initialize model
    # llm = init_chat_model(config["agent"]["llm"]["model"])
    llm = init_chat_model(
        config["agent"]["llm"]["model"],
        model_provider=config["agent"]["llm"].get("model_provider")  # Or any default
    )

    # Initialize tools
    # llm_with_tools = llm.bind_tools(tools)
    
    tools =get_enabled_tools(config)
    print(f"Enabled tools: {[t.name for t in tools]}")
    
    # if tools:
    #     llm_with_tools = llm.bind_tools(tools)
    # else:
    #     llm_with_tools = llm
    
    
    # langgraph toolnode
    llm_with_tools = llm
    
    
    # Define chatbot node
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Build graph
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))
    
    # Set edges
    builder.set_entry_point("chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    
    graph = builder.compile()
    
    # Test queries
    queries = [
        "What is the price of AAPL stock right now?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        state = graph.invoke({"messages": [{"role": "user", "content": query}]})
        print(f"Response: {state['messages'][-1].content}")

if __name__ == "__main__":
    main()