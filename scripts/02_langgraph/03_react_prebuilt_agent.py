# pip install langgraph langchain-google-genai python-dotenv

"""
LangGraph ReAct Prebuilt Agent - Using create_react_agent

This script demonstrates:
- Using prebuilt ReAct agent with create_react_agent
- Configuring agent with custom tools and memory
- Understanding ReAct (Reasoning + Acting) pattern
- Managing conversation history and context
- Customizing agent behavior with prompts

Documentation: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-agent
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

print("=== LangGraph ReAct Prebuilt Agent ===\n")

# Section 1: Define Comprehensive Tools
print("1. Define Comprehensive Tools")
print("-" * 32)


@tool
def search_knowledge_base(query: str) -> str:
    """Search a simulated knowledge base for information.

    Args:
        query: Search query

    Returns:
        Relevant information or indication that no results were found
    """
    # Simulated knowledge base
    knowledge_base = {
        "python": "Python is a high-level programming language known for its simplicity and readability. Created by Guido van Rossum in 1991.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It's built on top of LangChain and enables complex agent workflows.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
        "machine learning": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "react": "ReAct (Reasoning + Acting) is a paradigm that combines reasoning and acting in language models to solve complex tasks through iterative thought and action cycles."
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"Found information: {value}"

    return f"No specific information found for '{query}'. The knowledge base contains information about: {', '.join(knowledge_base.keys())}"


@tool
def calculate_compound_interest(principal: float, rate: float, time: float, compound_frequency: int = 1) -> str:
    """Calculate compound interest.

    Args:
        principal: Initial amount
        rate: Annual interest rate (as percentage)
        time: Time in years
        compound_frequency: How many times interest is compounded per year

    Returns:
        Detailed calculation results
    """

    # Convert percentage to decimal
    rate_decimal = rate / 100

    # Compound interest formula: A = P(1 + r/n)^(nt)
    amount = principal * (1 + rate_decimal /
                          compound_frequency) ** (compound_frequency * time)
    compound_interest = amount - principal

    return (f"Principal: ${principal:,.2f}, Rate: {rate}%, Time: {time} years, "
            f"Compound Frequency: {compound_frequency}/year | "
            f"Final Amount: ${amount:,.2f}, Interest Earned: ${compound_interest:,.2f}")


@tool
def analyze_text_sentiment(text: str) -> str:
    """Analyze sentiment of given text using simple keyword analysis.

    Args:
        text: Text to analyze

    Returns:
        Sentiment analysis results
    """
    positive_words = ['good', 'great', 'excellent', 'amazing',
                      'wonderful', 'fantastic', 'happy', 'love', 'perfect', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'horrible',
                      'hate', 'worst', 'sad', 'angry', 'frustrated', 'disappointed']

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        sentiment = "Positive"
    elif negative_count > positive_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return (f"Sentiment: {sentiment} | "
            f"Positive indicators: {positive_count}, Negative indicators: {negative_count} | "
            f"Text length: {len(text)} characters")


@tool
def get_programming_tip(language: str = "python") -> str:
    """Get a programming tip for specified language.

    Args:
        language: Programming language (default: python)

    Returns:
        Programming tip for the specified language
    """
    tips = {
        "python": [
            "Use list comprehensions for concise and readable code: [x**2 for x in range(10)]",
            "Use f-strings for string formatting: f'Hello {name}, you are {age} years old'",
            "Use 'with' statements for file operations to ensure proper cleanup",
            "Use enumerate() when you need both index and value: for i, value in enumerate(list)",
            "Use .get() method for dictionaries to avoid KeyError: dict.get(key, default_value)"
        ],
        "javascript": [
            "Use const for variables that won't be reassigned, let for variables that will",
            "Use arrow functions for concise function syntax: const add = (a, b) => a + b",
            "Use destructuring for cleaner code: const {name, age} = person",
            "Use template literals for string interpolation: `Hello ${name}`",
            "Use .map(), .filter(), and .reduce() for functional programming"
        ]
    }

    import random
    language_tips = tips.get(language.lower(), tips["python"])
    return f"{language.title()} tip: {random.choice(language_tips)}"


# Create tools list
tools = [search_knowledge_base, calculate_compound_interest,
         analyze_text_sentiment, get_programming_tip]
print(f"✓ Created {len(tools)} tools for the ReAct agent")
print()

# Section 2: Initialize LLM
print("2. Initialize LLM")
print("-" * 18)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1
)
print("✓ Initialized Gemini model for ReAct agent")
print()

# Section 3: Create ReAct Agent with Memory
print("3. Create ReAct Agent with Memory")
print("-" * 35)

# Create memory for persistent conversations
memory = MemorySaver()

# Custom system prompt for the ReAct agent
system_prompt = """You are a helpful AI assistant that uses tools to provide accurate and comprehensive answers.

When answering questions:
1. Think step by step about what information you need
2. Use available tools when they can provide relevant information
3. Combine tool results with your knowledge to give complete answers
4. If you need to use multiple tools, do so systematically
5. Always explain your reasoning and the tools you're using

Available tools:
- search_knowledge_base: Search for factual information
- calculate_compound_interest: Financial calculations
- analyze_text_sentiment: Text sentiment analysis
- get_programming_tip: Programming advice

Be conversational and helpful while being thorough in your responses."""

# Create the ReAct agent
agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
    prompt=system_prompt
)

print("✓ Created ReAct agent with memory and custom system prompt")
print()

# Section 4: Test Agent with Various Queries
print("4. Test Agent with Various Queries")
print("-" * 36)

# Configuration for conversation
config = {"configurable": {"thread_id": "react_demo_thread"}}

test_queries = [
    "What is LangGraph and why is it useful?",
    "Calculate the compound interest on $5000 invested for 3 years at 5% annual rate, compounded quarterly",
    "Analyze the sentiment of this review: 'This product is absolutely terrible and I hate it!'",
    "Give me a Python programming tip",
    "Can you search for information about machine learning and then give me a programming tip for JavaScript?"
]

for i, query in enumerate(test_queries, 1):
    print(f"Query {i}: {query}")
    print("-" * 60)

    try:
        # Invoke the agent
        response = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        # Print the final response
        final_message = response["messages"][-1]
        print(f"Agent Response: {final_message.content}")
        print()

        # Show reasoning process (intermediate steps)
        print("Reasoning Process:")
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"  → Tool called: {tool_call['name']}")
                    print(f"    Args: {tool_call['args']}")
            elif hasattr(msg, 'name') and msg.name:  # Tool response
                print(
                    f"  ← Tool result ({msg.name}): {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"Error: {e}")
        print()

# Section 5: Demonstrate Conversation Memory
print("5. Demonstrate Conversation Memory")
print("-" * 35)

print("Follow-up question (uses conversation history):")
follow_up_query = "Based on our previous conversation, can you give me another programming tip for the same language you mentioned before?"

response = agent.invoke(
    {"messages": [HumanMessage(content=follow_up_query)]},
    config=config
)

print(f"Query: {follow_up_query}")
print(f"Agent Response: {response['messages'][-1].content}")
print()

# Section 6: Understanding ReAct Pattern
print("6. Understanding ReAct Pattern")
print("-" * 32)

print("ReAct (Reasoning + Acting) Pattern:")
print("• Thought: Agent thinks about what it needs to do")
print("• Action: Agent uses tools to gather information")
print("• Observation: Agent processes tool results")
print("• Repeat: Agent continues until it can provide a final answer")
print()

print("Key features of create_react_agent:")
print("• Automatic tool selection and execution")
print("• Built-in conversation memory with checkpointer")
print("• Customizable system prompts")
print("• Handles multi-step reasoning automatically")
print("• Maintains conversation context across interactions")
print()

print("=== ReAct Prebuilt Agent Complete! ===")

# TODO: Student Exercise
print("\n" + "="*60)
print("TODO: Student Exercise - Personal Assistant ReAct Agent")
print("="*60)
print("""
Create a personal assistant ReAct agent with the following capabilities:

1. Define these tools:
   - schedule_reminder: Takes date, time, and message
   - weather_lookup: Takes city name (simulate with predefined responses)
   - unit_converter: Converts between different units (length, weight, temperature)
   - calculate_tip_split: Calculate tip and split bill among multiple people
   - generate_random_fact: Returns interesting random facts

2. Create a ReAct agent with:
   - Custom system prompt that makes it act as a helpful personal assistant
   - Memory to remember user preferences and conversation history
   - Appropriate reasoning for complex multi-step tasks

3. Test scenarios:
   - "Set a reminder for tomorrow at 2 PM to call mom"
   - "What's the weather like in Paris and convert 20°C to Fahrenheit?"
   - "I had dinner with 3 friends, the bill was $85.50, we want to tip 18%. How much does each person pay?"
   - "Tell me a random fact about space"
   - "Remember that I prefer Celsius for temperatures" (then ask about weather again)

4. Observe how the agent:
   - Decides which tools to use
   - Combines information from multiple tools
   - Remembers context from previous interactions
   - Provides comprehensive, helpful responses

Advanced: Add error handling for when tools fail or return unexpected results.
""")
