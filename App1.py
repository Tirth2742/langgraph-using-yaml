from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatOllama
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import yaml
import os
import json

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDe82ID5jmROVw7ps-iv-2NZvBbJiw5mxg"

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol'''
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
def option_price(symbol: str) -> float:
    '''Return the current price of an option given the option symbol'''
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
        "option_price": option_price
    }
    return [tool_map[name] for name, enabled in config["agent"]["tools"].items() if enabled]

def init_llm(config):
    """Initialize LLM based on config"""
    provider = config["agent"]["llm"]["provider"]
    model = config["agent"]["llm"]["model"]
    
    if provider == "google_genai":
        return init_chat_model(f"{provider}:{model}")
    elif provider == "ollama":
        return ChatOllama(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def tool_condition(state: State):
    """Custom tool condition that works with both Gemini and Ollama"""
    last_message = state["messages"][-1].content
    
    # Check for tool usage patterns
    tool_triggers = {
        "get_stock_price": ["stock", "price", "AAPL", "MSFT", "AMZN", "RIL"],
        "option_price": ["option", "derivative", "call", "put"],
        "document_qa": ["document", "policy", "benefit", "vacation"]
    }
    
    for tool_name, triggers in tool_triggers.items():
        if any(trigger in last_message.lower() for trigger in triggers):
            return "call_tool"
    
    return "end"

def main():
    config = load_config()
    
    # Initialize model
    llm = init_llm(config)
    
    # Initialize tools
    tools = get_enabled_tools(config)
    print(f"Enabled tools: {[t.name for t in tools]}")
    
    # Define chatbot node
    def chatbot(state: State):
        # For Gemini, we can use native tool calling
        if config["agent"]["llm"]["provider"] == "google_genai":
            llm_with_tools = llm.bind_tools(tools)
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        # For Ollama, use a custom prompt
        prompt = (
            "You are a helpful assistant. If the user asks about stocks, "
            "respond with: TOOL_CALL:get_stock_price:<symbol>. "
            "If they ask about options, respond with: TOOL_CALL:option_price:<symbol>. "
            "Otherwise, provide a normal response."
        )
        messages = state["messages"] + [{"role": "system", "content": prompt}]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Build graph
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))
    
    # Set edges
    builder.set_entry_point("chatbot")
    builder.add_conditional_edges("chatbot", tool_condition)
    builder.add_edge("tools", "chatbot")
    
    graph = builder.compile()
    
    # Test queries
    queries = [
        "What is the price of AAPL stock right now?",
        "What's the current option price for MSFT?",
        "Who invented the telephone?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        state = graph.invoke({"messages": [{"role": "user", "content": query}]})
        
        # For Ollama, parse tool responses
        if config["agent"]["llm"]["provider"] == "ollama":
            last_message = state["messages"][-1].content
            if "TOOL_CALL:" in last_message:
                tool_call = last_message.split("TOOL_CALL:")[1].strip()
                tool_name, input_val = tool_call.split(":")
                tool = next(t for t in tools if t.name == tool_name)
                result = tool.invoke(input_val)
                print(f"Response: {result}")
                continue
        
        print(f"Response: {state['messages'][-1].content}")

if __name__ == "__main__":
    main(
