# !pip install -q langchain-google-genai python-dotenv langchain-core
"""
Simple Chains with LangChain Expression Language (LCEL)
======================================================

In this notebook, we learn:
- Chaining prompt templates with LLMs using the | operator
- Understanding LangChain Expression Language (LCEL)
- Creating sequential processing workflows
- Passing data between chain components

Official documentation:
- LCEL: https://python.langchain.com/docs/concepts/lcel/
- Chains: https://python.langchain.com/docs/how_to/sequence/
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
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
# 1. BASIC CHAIN: PROMPT + LLM
# =============================================================================

print("\n🔗 1. Basic Chain: Prompt + LLM")

# Create a simple prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for a beginner."
)

# Create a chain using the | operator (pipe)
basic_chain = prompt | llm

# Use the chain
result = basic_chain.invoke({"topic": "blockchain"})

print("Input: blockchain")
print(f"Output: {result.content}")

# =============================================================================
# 2. CHAIN WITH OUTPUT PARSER
# =============================================================================

print("\n📝 2. Chain with Output Parser")

# Create a chain that parses the output to just a string
string_parser = StrOutputParser()
parsing_chain = prompt | llm | string_parser

# Compare outputs
print("--- Without parser ---")
result_without_parser = (prompt | llm).invoke({"topic": "machine learning"})
print(f"Type: {type(result_without_parser)}")
print(f"Content: {result_without_parser.content}")

print("\n--- With parser ---")
result_with_parser = parsing_chain.invoke({"topic": "machine learning"})
print(f"Type: {type(result_with_parser)}")
print(f"Content: {result_with_parser}")

# =============================================================================
# 3. CHAT PROMPT CHAIN
# =============================================================================

print("\n💬 3. Chat Prompt Chain")

# Create a chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding instructor. Provide clear, practical examples."),
    ("human", "Show me how to {task} in Python with a simple example.")
])

# Create a chat chain
chat_chain = chat_prompt | llm | string_parser

# Use the chat chain
coding_result = chat_chain.invoke({"task": "read a JSON file"})
print(f"Coding example: {coding_result}")

# =============================================================================
# 4. MULTI-STEP CHAIN
# =============================================================================

print("\n🔄 4. Multi-step Chain")

# Step 1: Generate a topic
topic_prompt = PromptTemplate(
    input_variables=["subject"],
    template="Suggest an interesting subtopic related to {subject}. Respond with just the subtopic name."
)

# Step 2: Explain the topic
explain_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in 2-3 sentences."
)

# Create individual chains
topic_chain = topic_prompt | llm | string_parser
explain_chain = explain_prompt | llm | string_parser

# Combine them manually


def multi_step_process(subject):
    # Step 1: Get a subtopic
    subtopic = topic_chain.invoke({"subject": subject})
    print(f"Generated subtopic: {subtopic}")

    # Step 2: Explain the subtopic
    explanation = explain_chain.invoke({"topic": subtopic})
    return explanation


# Use the multi-step process
result = multi_step_process("artificial intelligence")
print(f"Explanation: {result}")

# =============================================================================
# 5. CHAIN WITH CUSTOM FUNCTIONS
# =============================================================================

print("\n⚙️ 5. Chain with Custom Functions")


def format_input(input_dict):
    """Custom function to preprocess input"""
    topic = input_dict["topic"]
    formatted_topic = topic.upper().replace(" ", "_")
    return {"formatted_topic": formatted_topic, "original": topic}


def format_output(text):
    """Custom function to postprocess output"""
    return f"📚 EXPLANATION: {text.strip()}"


# Create custom runnables
input_formatter = RunnableLambda(format_input)
output_formatter = RunnableLambda(format_output)

# Create a prompt that uses the formatted input
custom_prompt = PromptTemplate(
    input_variables=["formatted_topic", "original"],
    template="Topic code: {formatted_topic}\nProvide a brief explanation of {original}:"
)

# Create the full chain with custom functions
custom_chain = input_formatter | custom_prompt | llm | string_parser | output_formatter

# Use the custom chain
custom_result = custom_chain.invoke({"topic": "neural networks"})
print(f"Custom chain result: {custom_result}")

# =============================================================================
# 6. CONDITIONAL CHAINS
# =============================================================================

print("\n🎯 6. Conditional Chains")


def route_by_difficulty(input_dict):
    """Route to different prompts based on difficulty level"""
    difficulty = input_dict.get("difficulty", "beginner")
    topic = input_dict["topic"]

    if difficulty == "beginner":
        template = "Explain {topic} in very simple terms, like you're talking to a child."
    elif difficulty == "intermediate":
        template = "Explain {topic} with some technical details, but keep it accessible."
    else:  # advanced
        template = "Provide a detailed technical explanation of {topic} with advanced concepts."

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=template
    )

    return prompt.format(topic=topic)


# Create conditional chain
conditional_chain = RunnableLambda(route_by_difficulty) | llm | string_parser

# Test with different difficulty levels
for level in ["beginner", "intermediate", "advanced"]:
    print(f"\n--- {level.upper()} LEVEL ---")
    result = conditional_chain.invoke({
        "topic": "quantum computing",
        "difficulty": level
    })
    print(f"Result: {result}")

# =============================================================================
# 8. KEY INFORMATION
# =============================================================================

print("\n📋 Key Information:")
print("✅ | operator: Chains components together (pipe operator)")
print("✅ LCEL: LangChain Expression Language for building chains")
print("✅ RunnableLambda: Wrap custom functions in chains")
print("✅ StrOutputParser: Extracts string content from LLM responses")
print("✅ Chains process data sequentially")
print("✅ Custom functions enable preprocessing and postprocessing")
print("✅ Conditional logic allows dynamic chain behavior")

print("\n🎯 Next step: Learn intermediate LCEL features")
