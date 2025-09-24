# pip install langgraph langchain-google-genai python-dotenv

"""
LangGraph Tool Calling - Integrating Tools with StateGraph

This script demonstrates:
- Creating and binding tools to LLMs
- Using ToolNode for automatic tool execution
- Handling tool calls in StateGraph
- Managing state with tool results
- Error handling for tool execution

Documentation: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tools
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

print("=== LangGraph Tool Calling ===\n")

# Section 1: Define Tools
print("1. Define Tools")
print("-" * 16)


@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> str:
    """Calculate tip amount and total bill including tip.

    Args:
        bill_amount: The original bill amount
        tip_percentage: Tip percentage (default 18%)

    Returns:
        String with tip calculation details
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount

    return f"Bill: ${bill_amount:.2f}, Tip ({tip_percentage}%): ${tip_amount:.2f}, Total: ${total_amount:.2f}"


@tool
def convert_temperature(temperature: float, from_unit: str = "celsius", to_unit: str = "fahrenheit") -> str:
    """Convert temperature between Celsius and Fahrenheit.

    Args:
        temperature: Temperature value to convert
        from_unit: Source unit ("celsius" or "fahrenheit")
        to_unit: Target unit ("celsius" or "fahrenheit")

    Returns:
        String with conversion result
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        converted = (temperature * 9/5) + 32
        return f"{temperature}°C = {converted:.1f}°F"
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        converted = (temperature - 32) * 5/9
        return f"{temperature}°F = {converted:.1f}°C"
    else:
        return f"Invalid conversion: {from_unit} to {to_unit}"


@tool
def word_count(text: str) -> str:
    """Count words, characters, and lines in text.

    Args:
        text: Text to analyze

    Returns:
        String with text statistics
    """
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    lines = len(text.split("\n"))

    return f"Words: {words}, Characters: {chars}, Characters (no spaces): {chars_no_spaces}, Lines: {lines}"


# Create tool list
tools = [calculate_tip, convert_temperature, word_count]
print(
    f"✓ Created {len(tools)} tools: calculate_tip, convert_temperature, word_count")
print()

# Section 2: Initialize LLM with Tools
print("2. Initialize LLM with Tools")
print("-" * 30)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)
print("✓ Bound tools to Gemini model")
print()

# Section 3: Define State Schema
print("3. Define State Schema")
print("-" * 24)


class AgentState(TypedDict):
    """State schema for tool-calling agent"""
    messages: Annotated[list, add_messages]


print("✓ Created AgentState with messages using add_messages reducer")
print()

# Section 4: Define Node Functions
print("4. Define Node Functions")
print("-" * 25)


def call_model(state: AgentState) -> AgentState:
    """Node that calls the LLM with tool capabilities"""
    print(f"  → Calling model with {len(state['messages'])} messages")

    response = llm_with_tools.invoke(state["messages"])

    # Log if tools were called
    if response.tool_calls:
        print(f"  → Model requested {len(response.tool_calls)} tool calls")
        for tool_call in response.tool_calls:
            print(f"    - {tool_call['name']}: {tool_call['args']}")
    else:
        print("  → Model provided direct response (no tools)")

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Conditional edge to determine next step"""
    last_message = state["messages"][-1]

    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("  → Tool calls detected, routing to tools")
        return "tools"
    else:
        print("  → No tool calls, ending conversation")
        return "end"


print("✓ Defined call_model and should_continue functions")
print()

# Section 5: Create StateGraph with ToolNode
print("5. Create StateGraph with ToolNode")
print("-" * 35)

# Create the tool node (automatically handles tool execution)
tool_node = ToolNode(tools)

# Initialize workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

print("✓ Created StateGraph with agent and tools nodes")
print()

# Section 6: Add Edges
print("6. Add Edges")
print("-" * 15)

# Add edges
workflow.add_edge(START, "agent")

# Conditional edge from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tools, go back to agent
workflow.add_edge("tools", "agent")

print("✓ Added edges: START → agent → tools → agent (loop)")
print("✓ Added conditional edge: agent → (tools/end)")
print()

# Section 7: Compile and Test
print("7. Compile and Test")
print("-" * 20)

app = workflow.compile()
print("✓ Graph compiled successfully!")
print()

# Test cases
test_cases = [
    "Calculate a 20% tip on a $45.50 bill",
    "Convert 25 degrees Celsius to Fahrenheit",
    "Count the words in this sentence: 'LangGraph makes building AI agents much easier and more reliable.'",
    "What's the weather like today?",  # No tool needed
]

for i, query in enumerate(test_cases, 1):
    print(f"Test Case {i}: {query}")
    print("-" * 40)

    # Create initial state
    initial_state = {"messages": [HumanMessage(content=query)]}

    try:
        # Run the graph
        result = app.invoke(initial_state)

        # Print conversation flow
        print("Conversation:")
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"  Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(
                        f"  AI: [Tool calls: {[tc['name'] for tc in msg.tool_calls]}]")
                else:
                    print(f"  AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"  Tool ({msg.name}): {msg.content}")

        print()

    except Exception as e:
        print(f"  Error: {e}")
        print()

# Section 8: Understanding Tool Integration
print("8. Understanding Tool Integration")
print("-" * 35)

print("Key concepts demonstrated:")
print("• Tools are defined using @tool decorator with type hints")
print("• LLM.bind_tools() enables the model to call tools")
print("• ToolNode automatically executes tool calls")
print("• Messages flow: Human → AI (with tool calls) → Tool results → AI (final response)")
print("• State management handles the entire conversation history")
print("• Conditional edges route based on whether tools are needed")
print()

print("=== Tool Calling Complete! ===")

# TODO: Student Exercise
print("\n" + "="*50)
print("TODO: Student Exercise - Math & File Operations Agent")
print("="*50)
print("""
Create a tool-calling agent with mathematical and text processing capabilities:

1. Define these tools:
   - calculate_average: Takes a list of numbers and returns the average
   - find_prime_numbers: Takes a number n and returns all primes up to n
   - text_statistics: Takes text and returns detailed stats (words, sentences, avg word length)
   - generate_password: Takes length parameter and returns a secure password

2. Create a StateGraph that:
   - Uses these tools when appropriate
   - Can handle multi-step calculations (e.g., "Find primes up to 20, then calculate their average")
   - Provides helpful responses even when no tools are needed

3. Test with queries like:
   - "Find all prime numbers up to 30"
   - "Calculate the average of 15, 23, 8, 42, 19"
   - "Analyze this text: 'The quick brown fox jumps over the lazy dog.'"
   - "Generate a 12-character secure password"
   - "What is the capital of France?" (no tools needed)

4. Observe how the agent decides when to use tools vs direct responses

Hint: Use isinstance() to check message types and hasattr() to check for tool_calls
""")
