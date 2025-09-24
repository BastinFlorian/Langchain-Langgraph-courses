# !pip install -q langchain-google-genai python-dotenv langchain-core
"""
First LLM Call with Google Gemini and LangChain
===============================================

In this notebook, we learn:
- Setting up Google Gemini API keys
- Initializing a LangChain chat model
- Making first simple call with invoke()
- Comparing different Gemini models

Official documentation:
- LangChain: https://python.langchain.com/docs/
- Google Gemini: https://ai.google.dev/gemini-api/docs/
- LangChain Google GenAI: https://python.langchain.com/docs/integrations/chat/google_generative_ai/

Your task:
- Execute the cells step by step
- Understand each part of the code and how to call a LLM
- Debug the empty answer by understanding the cause and fixing it
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Check if API key is configured
if not os.environ.get("GOOGLE_API_KEY"):
    print("⚠️  GOOGLE_API_KEY is not configured in .env file")
    print("Please add your Google Gemini API key to the .env file:")
    print("GOOGLE_API_KEY=your_api_key_here")
    exit(1)

print("✅ Configuration detected")

# =============================================================================
# 1. INITIALIZE GEMINI 2.5 FLASH MODEL
# =============================================================================

print("\n🤖 Initializing Gemini 2.5 Flash model...")

# Initialize model with basic parameters
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Latest and fastest model
    # Controls creativity (0 = deterministic, 1 = creative)
    temperature=0.7,
    max_tokens=1000,          # Response length limit
)

print(f"✅ Model {llm_flash.model} initialized")

# =============================================================================
# 2. FIRST SIMPLE CALL
# =============================================================================

print("\n📝 First call with a simple question...")

# Method 1: Direct call with a string
response = llm_flash.invoke(
    "Hello! Can you introduce yourself in one sentence?")

print(f"🤖 Response: {response.content}")
print(f"📊 Metadata: {response.response_metadata}")

# =============================================================================
# 3. COMPARISON WITH GEMINI 2.0 FLASH
# =============================================================================

print("\n🔄 Comparison with Gemini 2.0 Flash...")

# Initialize Gemini 2.0 model
llm_2_0 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1000,
)

# More complex question to see the difference
question = "Explain the difference between artificial intelligence and machine learning in one simple sentence."

print(f"\n❓ Question: {question}")

# Response with Gemini 2.5
response_2_5 = llm_flash.invoke(question)
print(f"\n🟦 Gemini 2.5 Flash: {response_2_5.content}")

# Response with Gemini 2.0
response_2_0 = llm_2_0.invoke(question)
print(f"\n🟩 Gemini 2.0 Flash: {response_2_0.content}")

# =============================================================================
# 4. PARAMETER TESTING
# =============================================================================

print("\n⚙️  Testing with different parameters...")

# More creative model
llm_creative = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,      # Maximum creativity
    max_tokens=500,
)

# More deterministic model
llm_precise = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,      # More consistent responses
    max_tokens=500,
)

creative_question = "Tell me a short and original story about a robot learning to cook."

print(f"\n❓ Creative question: {creative_question}")

# Creative response
response_creative = llm_creative.invoke(creative_question)
print(f"\n🎨 Creative mode (temp=1.0): {response_creative.content}")

# Precise response
response_precise = llm_precise.invoke(creative_question)
print(f"\n🎯 Precise mode (temp=0.0): {response_precise.content}")

# =============================================================================
# 5. KEY INFORMATION
# =============================================================================

print("\n📋 Important information:")
print("✅ invoke(): Main method to call an LLM")
print("✅ temperature: Controls creativity (0.0 to 1.0)")
print("✅ max_tokens: Limits response length")
print("✅ response.content: Contains the response text")
print("✅ response.response_metadata: Contains metadata (tokens used, etc.)")

# =============================================================================
# STUDENT TODO EXERCISE
# =============================================================================

print("\n" + "=" * 60)
print("📝 STUDENT TODO EXERCISE")
print("=" * 60)

"""
TODO: Create and test your own LLM model configurations

Requirements:
1. Initialize a Gemini model with these specific parameters:
   - model: "gemini-2.5-flash"
   - temperature: 0.5 (balanced creativity)
   - max_tokens: 200

2. Test your model with this question:
   "What are 3 benefits of learning programming?"

3. Compare the response with a second model using temperature=0.9

Your code here:
"""

# TODO: Create your first model configuration
# my_llm = ChatGoogleGenerativeAI(
#     # Add your parameters here
# )

# TODO: Create your second model with higher temperature
# my_creative_llm = ChatGoogleGenerativeAI(
#     # Add your parameters here
# )

# TODO: Test both models with the same question
# question = "What are 3 benefits of learning programming?"
#
# response1 = my_llm.invoke(question)
# response2 = my_creative_llm.invoke(question)
#
# print(f"Balanced response: {response1.content}")
# print(f"Creative response: {response2.content}")
