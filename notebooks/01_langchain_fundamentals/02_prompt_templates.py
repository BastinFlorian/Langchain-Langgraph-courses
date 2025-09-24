# !pip install -q langchain-google-genai python-dotenv langchain-core
"""
Prompt Templates with LangChain and Google Gemini
================================================

In this notebook, we learn:
- Creating reusable prompt templates
- Using variables in prompts
- Difference between PromptTemplate and ChatPromptTemplate
- Formatting and using templates effectively

Official documentation:
- LangChain Prompt Templates: https://python.langchain.com/docs/concepts/prompt_templates/
- LangChain Templates Guide: https://python.langchain.com/docs/how_to/prompts_composition/
"""

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=10000,
)

print("✅ Gemini 2.5 Flash model initialized")

# =============================================================================
# 1. BASIC PROMPT TEMPLATE
# =============================================================================

print("\n📝 1. Basic Prompt Template")

# Create a simple template with one variable
basic_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for a beginner."
)

# Format the template
formatted_prompt = basic_template.format(topic="machine learning")
print(f"Formatted prompt: {formatted_prompt}")

# Use with LLM
response = llm.invoke(formatted_prompt)
print(f"Response: {response.content}")

# =============================================================================
# 2. MULTI-VARIABLE PROMPT TEMPLATE
# =============================================================================

print("\n📝 2. Multi-variable Prompt Template")

# Create template with multiple variables
multi_template = PromptTemplate(
    input_variables=["subject", "audience", "length"],
    template="""Write a {length} explanation of {subject} for {audience}.
Make sure to use appropriate language and examples."""
)

# Format with multiple variables
formatted_multi = multi_template.format(
    subject="artificial intelligence",
    audience="high school students",
    length="short"
)

print(f"Formatted prompt: {formatted_multi}")

response = llm.invoke(formatted_multi)
print(f"Response: {response.content}")

# =============================================================================
# 3. CHAT PROMPT TEMPLATE
# =============================================================================

print("\n💬 3. Chat Prompt Template")

# Create a chat template with system and human messages
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Always provide clear, well-commented code examples."),
    ("human", "Show me how to {task} in Python.")
])

# Format the chat template
formatted_chat = chat_template.format_messages(task="read a CSV file")
print(f"Formatted messages: {formatted_chat}")

# Use with LLM
response = llm.invoke(formatted_chat)
print(f"Response: {response.content}")

# =============================================================================
# 4. FEW-SHOT PROMPT TEMPLATE
# =============================================================================

print("\n🎯 4. Few-shot Prompt Template")

# Define examples for few-shot learning
examples = [
    {
        "input": "What is 2+2?",
        "output": "2+2 equals 4"
    },
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris"
    },
    {
        "input": "What color is the sky?",
        "output": "The sky is typically blue during clear weather"
    },
]

# Create example template
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Create few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="You are a helpful assistant. Answer questions clearly and concisely.",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# Test the few-shot template
test_input = "What is the capital of Japan?"
formatted_prompt = few_shot_prompt.format(input=test_input)
print(f"Formatted few-shot prompt:\n{formatted_prompt}")
print("\n" + "-" * 50)

response = llm.invoke(formatted_prompt)
print(f"Response: {response.content}")

# =============================================================================
# 5. CONDITIONAL TEMPLATE
# =============================================================================

print("\n🔄 5. Conditional Template Usage")

# Create templates for different use cases
templates = {
    "explain": PromptTemplate(
        input_variables=["concept"],
        template="Provide a clear explanation of {concept} with real-world examples."
    ),
    "summarize": PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 2-3 sentences: {text}"
    ),
    "translate": PromptTemplate(
        input_variables=["text", "language"],
        template="Translate the following text to {language}: {text}"
    )
}

# Function to use appropriate template


def process_request(request_type, **kwargs):
    if request_type in templates:
        template = templates[request_type]
        formatted_prompt = template.format(**kwargs)
        return llm.invoke(formatted_prompt)
    else:
        return "Unknown request type"


# Test different templates
explain_response = process_request("explain", concept="neural networks")
print(f"Explanation: {explain_response.content}")

summarize_response = process_request(
    "summarize",
    text="Artificial intelligence is a rapidly growing field that focuses on creating machines capable of performing tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, and understanding natural language."
)
print(f"Summary: {summarize_response.content}")

# =============================================================================
# 6. TEMPLATE COMPOSITION
# =============================================================================

print("\n🔧 6. Template Composition")

# Create modular templates
context_template = PromptTemplate(
    input_variables=["domain"],
    template="You are an expert in {domain}."
)

task_template = PromptTemplate(
    input_variables=["task", "details"],
    template="Your task is to {task}. Here are the details: {details}"
)

# Combine templates


def create_composed_prompt(domain, task, details):
    context = context_template.format(domain=domain)
    task_prompt = task_template.format(task=task, details=details)
    return f"{context}\n\n{task_prompt}"


# Use composed template
composed_prompt = create_composed_prompt(
    domain="data science",
    task="analyze a dataset",
    details="The dataset contains customer purchase data with columns: date, product, price, customer_id"
)

print(f"Composed prompt: {composed_prompt}")

response = llm.invoke(composed_prompt)
print(f"Response: {response.content}")

# =============================================================================
# 7. KEY INFORMATION
# =============================================================================

print("\n📋 Key Information:")
print("✅ PromptTemplate: For simple string templates")
print("✅ ChatPromptTemplate: For structured conversations")
print("✅ input_variables: Define template variables")
print("✅ .format(): Fill in template variables")
print("✅ .format_messages(): For chat templates")
print("✅ Templates enable reusability and consistency")
print("✅ Few-shot prompting improves response quality")

# =============================================================================
# STUDENT TODO EXERCISE
# =============================================================================

print("\n" + "=" * 60)
print("📝 STUDENT TODO EXERCISE")
print("=" * 60)

"""
TODO: Create a FewShotPromptTemplate for a sentiment analysis task

Requirements:
1. Create at least 4 examples showing different sentiments (positive, negative, neutral, mixed)
2. Each example should have:
   - "text": The input text to analyze
   - "sentiment": The sentiment classification
   - "confidence": A confidence score (high/medium/low)

3. Use an appropriate prefix that explains the task
4. Test your template with a new sentence

Example structure to get you started:
sentiment_examples = [
    {
        "text": "I love this product! It's amazing!",
        "sentiment": "positive",
        "confidence": "high"
    },
    # Add more examples here...
]

Hint: Think about edge cases like sarcasm or mixed feelings!
"""

# Your code here:
# sentiment_examples = [
#     # TODO: Add your examples
# ]

# sentiment_example_template = PromptTemplate(
#     # TODO: Define your template
# )

# sentiment_few_shot_prompt = FewShotPromptTemplate(
#     # TODO: Configure your few-shot template
# )

# Test your template:
# test_sentence = "The movie was okay, not great but watchable"
# response = llm.invoke(sentiment_few_shot_prompt.format(text=test_sentence))
# print(f"Sentiment analysis: {response.content}")
