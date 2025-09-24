# pip install langgraph langchain-google-genai python-dotenv

"""
LangGraph Basics - Understanding StateGraph Fundamentals

This script introduces the core concepts of LangGraph:
- Creating a StateGraph with typed state
- Defining nodes and their functions
- Adding edges and conditional routing
- Compiling and running the graph
- Understanding state persistence across nodes

Documentation: https://langchain-ai.github.io/langgraph/concepts/low_level/
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

print("=== LangGraph Basics: StateGraph Fundamentals ===\n")

# Section 1: Define State Schema
print("1. Defining State Schema")
print("-" * 30)


class ChatState(TypedDict):
    """State schema for our chat graph"""
    messages: list[str]
    user_name: str
    conversation_count: int
    mood: Literal["happy", "neutral", "sad"]


print("✓ Created ChatState with messages, user_name, conversation_count, and mood")
print()

# Section 2: Initialize LLM
print("2. Initialize LLM")
print("-" * 20)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7
)
print("✓ Initialized Gemini 2.0 Flash model")
print()

# Section 3: Define Node Functions
print("3. Define Node Functions")
print("-" * 25)


def greet_user(state: ChatState) -> ChatState:
    """First node: Greet the user and analyze mood"""
    print(f"  → Greeting user: {state['user_name']}")

    # Analyze mood from the last message if available
    if state["messages"]:
        last_message = state["messages"][-1]
        if any(word in last_message.lower() for word in ["sad", "upset", "angry", "frustrated"]):
            mood = "sad"
        elif any(word in last_message.lower() for word in ["happy", "great", "awesome", "excited"]):
            mood = "happy"
        else:
            mood = "neutral"
    else:
        mood = "neutral"

    greeting = f"Hello {state['user_name']}! Nice to meet you."

    return {
        "messages": state["messages"] + [greeting],
        "mood": mood,
        "conversation_count": state["conversation_count"] + 1
    }


def generate_response(state: ChatState) -> ChatState:
    """Second node: Generate AI response based on mood"""
    print(f"  → Generating response for mood: {state['mood']}")

    # Create context-aware prompt based on mood
    if state["mood"] == "happy":
        system_prompt = "You are an enthusiastic and cheerful assistant. Match the user's positive energy!"
    elif state["mood"] == "sad":
        system_prompt = "You are a compassionate and supportive assistant. Be gentle and understanding."
    else:
        system_prompt = "You are a helpful and professional assistant."

    # Get the last user message (skip our greeting)
    user_messages = [msg for msg in state["messages"]
                     if not msg.startswith("Hello")]
    if user_messages:
        last_message = user_messages[-1]
        response = llm.invoke(
            f"{system_prompt}\n\nUser message: {last_message}")
        ai_response = response.content
    else:
        ai_response = "How can I help you today?"

    return {
        **state,
        "messages": state["messages"] + [ai_response],
        "conversation_count": state["conversation_count"] + 1
    }


def should_continue(state: ChatState) -> Literal["continue", "end"]:
    """Conditional edge: Decide whether to continue conversation"""
    # End after 4 exchanges to keep demo short
    if state["conversation_count"] >= 4:
        print("  → Conversation limit reached, ending...")
        return "end"
    else:
        print("  → Continuing conversation...")
        return "continue"


print("✓ Defined three functions: greet_user, generate_response, should_continue")
print()

# Section 4: Create StateGraph
print("4. Create StateGraph")
print("-" * 20)

# Initialize the StateGraph with our schema
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("greet", greet_user)
workflow.add_node("respond", generate_response)

print("✓ Created StateGraph and added nodes")
print()

# Section 5: Add Edges
print("5. Add Edges")
print("-" * 15)

# Add edges to connect nodes
workflow.add_edge(START, "greet")
workflow.add_edge("greet", "respond")

# Add conditional edge
workflow.add_conditional_edges(
    "respond",
    should_continue,
    {
        "continue": "respond",  # Loop back to respond
        "end": END
    }
)

print("✓ Added edges: START → greet → respond")
print("✓ Added conditional edge: respond → (continue/end)")
print()

# Section 6: Compile Graph
print("6. Compile Graph")
print("-" * 18)

app = workflow.compile()
print("✓ Graph compiled successfully!")
print()

# Section 7: Run the Graph
print("7. Run the Graph")
print("-" * 18)

# Initial state
initial_state = {
    "messages": ["I'm feeling great today! Can you help me with Python?"],
    "user_name": "Alice",
    "conversation_count": 0,
    "mood": "neutral"
}

print("Initial state:")
print(f"  User: {initial_state['user_name']}")
print(f"  Messages: {len(initial_state['messages'])}")
print(f"  Count: {initial_state['conversation_count']}")
print()

# Run the graph
print("Running graph...")
result = app.invoke(initial_state)

print("\nFinal result:")
print(f"  Total exchanges: {result['conversation_count']}")
print(f"  Final mood: {result['mood']}")
print(f"  Messages exchanged: {len(result['messages'])}")
print()

print("Conversation flow:")
for i, message in enumerate(result["messages"]):
    speaker = "User" if i == 0 else ("Assistant" if i % 2 == 1 else "AI")
    print(f"  {speaker}: {message[:100]}{'...' if len(message) > 100 else ''}")
print()

# Section 8: Understanding State Persistence
print("8. Understanding State Persistence")
print("-" * 35)

print("Key concepts demonstrated:")
print("• State is automatically passed between nodes")
print("• Each node can read and modify the shared state")
print("• StateGraph ensures type safety with TypedDict")
print("• Conditional edges enable dynamic routing")
print("• The graph maintains state throughout execution")
print()

print("=== LangGraph Basics Complete! ===")

# TODO: Student Exercise
print("\n" + "="*50)
print("TODO: Student Exercise - Simple Chatbot with Logic")
print("="*50)
print("""
Create your own StateGraph that implements a simple customer support chatbot:

1. Create a state schema with:
   - user_query: str
   - category: Literal["technical", "billing", "general"]
   - priority: Literal["low", "medium", "high"]
   - resolved: bool

2. Create nodes for:
   - classify_query: Analyze the query and set category/priority
   - handle_technical: Handle technical issues
   - handle_billing: Handle billing issues
   - handle_general: Handle general questions
   - escalate: For high priority or unresolved issues

3. Add conditional routing:
   - From classify_query to appropriate handler based on category
   - From handlers to escalate if high priority or unresolved
   - Otherwise to END

4. Test with different types of queries and observe the routing

Hint: Use the user_query to determine category (keywords like "password", "login" = technical,
"payment", "refund" = billing, etc.)
""")
