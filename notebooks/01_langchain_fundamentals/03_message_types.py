# !pip install -q langchain-google-genai python-dotenv langchain-core
"""
Message Types in LangChain
===========================

In this notebook, we learn:
- Different message types: SystemMessage, HumanMessage, AIMessage
- How to construct structured conversations
- Best practices for message sequencing
- Managing conversation history

Official documentation:
- LangChain Messages: https://python.langchain.com/docs/concepts/messages/
- Chat History: https://python.langchain.com/docs/concepts/chat_history/
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000,
)

print("✅ Gemini 2.5 Flash model initialized")

# =============================================================================
# 1. SYSTEM MESSAGE - SETTING BEHAVIOR
# =============================================================================

print("\n🔧 1. System Message - Setting AI Behavior")

# System message defines the AI's role and behavior
system_msg = SystemMessage(
    content="You are a helpful Python programming tutor. Always provide clear explanations and working code examples.")

# Human message with a question
human_msg = HumanMessage(content="How do I create a list in Python?")

# Send both messages together
messages = [system_msg, human_msg]
response = llm.invoke(messages)

print(f"System message: {system_msg.content}")
print(f"Human question: {human_msg.content}")
print(f"AI response: {response.content}")

# =============================================================================
# 2. CONVERSATION WITH HISTORY
# =============================================================================

print("\n💬 2. Building a Conversation with History")

# Start a conversation with context
conversation = [
    SystemMessage(
        content="You are a helpful cooking assistant. Give practical cooking advice."),
    HumanMessage(content="I want to make pasta. What ingredients do I need?"),
]

# Get first response
first_response = llm.invoke(conversation)
print(f"Human: {conversation[1].content}")
print(f"AI: {first_response.content}")

# Add the AI response to conversation history
conversation.append(AIMessage(content=first_response.content))

# Continue the conversation
conversation.append(HumanMessage(content="How long should I cook the pasta?"))

# Get second response with full context
second_response = llm.invoke(conversation)
print(f"\nHuman: {conversation[3].content}")
print(f"AI: {second_response.content}")

# =============================================================================
# 3. DIFFERENT SYSTEM MESSAGE STYLES
# =============================================================================

print("\n🎭 3. Different System Message Styles")

# Define different system personas
personas = {
    "teacher": "You are a patient teacher. Explain concepts step by step with simple examples.",
    "expert": "You are a technical expert. Provide detailed, accurate information with technical depth.",
    "friend": "You are a friendly companion. Be casual, encouraging, and use everyday language.",
    "critic": "You are a constructive critic. Point out potential issues and suggest improvements."
}

question = "Should I use Python or JavaScript for my first programming project?"

for persona_name, persona_description in personas.items():
    print(f"\n--- {persona_name.upper()} PERSONA ---")

    messages = [
        SystemMessage(content=persona_description),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    print(f"Response: {response.content}")

# =============================================================================
# 4. MESSAGE ATTRIBUTES AND METADATA
# =============================================================================

print("\n📊 4. Message Attributes and Metadata")

# Create messages with additional metadata
system_msg = SystemMessage(
    content="You are a helpful assistant.",
    additional_kwargs={"role_description": "system"}
)

human_msg = HumanMessage(
    content="What's the weather like?",
    additional_kwargs={"user_id": "user123", "timestamp": "2025-01-01"}
)

print(f"System message content: {system_msg.content}")
print(f"System message type: {type(system_msg).__name__}")
print(f"System message kwargs: {system_msg.additional_kwargs}")

print(f"\nHuman message content: {human_msg.content}")
print(f"Human message type: {type(human_msg).__name__}")
print(f"Human message kwargs: {human_msg.additional_kwargs}")

# =============================================================================
# 7. KEY INFORMATION
# =============================================================================

print("\n📋 Key Information:")
print("✅ SystemMessage: Sets AI behavior and role")
print("✅ HumanMessage: Represents user input")
print("✅ AIMessage: Represents AI responses")
print("✅ Messages are processed in sequence")
print("✅ Conversation history provides context")
print("✅ System messages should be clear and specific")
print("✅ Use additional_kwargs for metadata")


# =============================================================================
# STUDENT TODO EXERCISE
# =============================================================================

print("\n" + "=" * 60)
print("📝 STUDENT TODO EXERCISE")
print("=" * 60)

"""
TODO: Create a customer service chatbot conversation

Requirements:
1. Create a SystemMessage that defines a helpful customer service agent
2. Create a conversation where:
   - Customer asks about return policy
   - Agent responds
   - Customer asks a follow-up question about refund timing
   - Agent provides final response

3. Print the complete conversation history
4. Make sure to include AIMessage for the agent's responses

Your code here:
"""

# TODO: Define your customer service system message that answers
# - Always in English
# - Doing joke
# - Friendly tone
# - Provides clear answers always with a table in the answer
# system_message = SystemMessage(
#     content="Your system prompt here..."
# )

# TODO: Create the conversation flow
# conversation = [
#     system_message,
#     # Add HumanMessage for first question
#     # Get response and add AIMessage
#     # Add HumanMessage for follow-up
#     # Get final response
# ]

# TODO: Print the complete conversation
# for i, message in enumerate(conversation):
#     print(f"{i+1}. {type(message).__name__}: {message.content}")

print("💡 Hint: Remember to add each AI response as an AIMessage to maintain conversation history!")
