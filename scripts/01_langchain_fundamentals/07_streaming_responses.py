# !pip install -q langchain-google-genai python-dotenv langchain-core
"""
Streaming Responses with LangChain and Google Gemini
===================================================

In this notebook, we learn:
- Streaming LLM responses in real-time
- Working with AIMessageChunk objects
- Aggregating streamed responses
- Building streaming chains

Official documentation:
- LangChain Streaming: https://python.langchain.com/docs/concepts/streaming/
- Streaming Guide: https://python.langchain.com/docs/how_to/streaming/
"""


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=3000,
)

print("✅ Gemini 2.5 Flash model initialized for streaming")

# =============================================================================
# 1. BASIC STREAMING
# =============================================================================

print("\n📡 1. Basic Streaming")


def demonstrate_basic_streaming():
    """Show basic streaming functionality"""
    prompt = "Write a short story about a robot learning to paint. Make it engaging and creative."

    print("🔄 Streaming response:")
    print("=" * 50)

    try:
        # Stream the response
        has_content = False
        for chunk in llm.stream(prompt):
            if chunk.content:
                has_content = True
                print(chunk.content, end="", flush=True)

        if not has_content:
            print("No streaming content received. Using regular invoke instead.")
            response = llm.invoke(prompt)
            print(response.content)

    except Exception as e:
        print(f"Streaming error: {e}")
        print("Falling back to regular invoke:")
        response = llm.invoke(prompt)
        print(response.content)

    print("\n" + "=" * 50)
    print("✅ Streaming demonstration complete")


demonstrate_basic_streaming()

# =============================================================================
# 2. UNDERSTANDING CHUNKS
# =============================================================================

print("\n🧩 2. Understanding AIMessageChunk")


def analyze_chunks():
    """Analyze the structure of streaming chunks"""
    prompt = "Explain quantum computing in three sentences."

    print("📊 Chunk analysis:")
    chunks = []

    try:
        for i, chunk in enumerate(llm.stream(prompt)):
            chunks.append(chunk)
            print(
                f"Chunk {i}: '{chunk.content}' (type: {type(chunk).__name__})")

            if i >= 5:  # Limit output for demonstration
                print("... (more chunks)")
                break
    except Exception as e:
        print(f"Chunk analysis error: {e}")
        print("Using regular invoke for demonstration:")
        response = llm.invoke(prompt)
        # Create a mock chunk for demonstration
        mock_chunk = type('MockChunk', (), {'content': response.content})()
        chunks = [mock_chunk]
        print(f"Mock chunk: '{response.content[:50]}...' (type: AIMessage)")

    return chunks


chunk_samples = analyze_chunks()

# =============================================================================
# 3. AGGREGATING STREAMING RESPONSES
# =============================================================================

print("\n🔗 3. Aggregating Streaming Responses")


def aggregate_streaming_response(prompt):
    """Aggregate chunks into a complete response"""
    print(f"🔄 Processing: {prompt}")

    # Method 1: Manual aggregation
    full_response = ""
    chunk_count = 0

    for chunk in llm.stream(prompt):
        if chunk.content:
            full_response += chunk.content
            chunk_count += 1

    print(f"📊 Received {chunk_count} chunks")
    print(f"📝 Complete response: {full_response}")

    # Method 2: Using chunk addition
    print("\n--- Using chunk addition ---")
    aggregated_chunk = None

    for chunk in llm.stream(prompt):
        if aggregated_chunk is None:
            aggregated_chunk = chunk
        else:
            aggregated_chunk = aggregated_chunk + chunk

    print(f"📝 Aggregated content: {aggregated_chunk.content}")


aggregate_streaming_response("What are the benefits of renewable energy?")

# =============================================================================
# 4. STREAMING WITH PROMPT TEMPLATES
# =============================================================================

print("\n📝 4. Streaming with Prompt Templates")

# Create a prompt template
story_template = PromptTemplate(
    input_variables=["character", "setting", "challenge"],
    template="""Write a short story with these elements:
- Character: {character}
- Setting: {setting}
- Challenge: {challenge}

Make it engaging and creative."""
)


def stream_with_template(character, setting, challenge):
    """Stream response using a prompt template"""
    formatted_prompt = story_template.format(
        character=character,
        setting=setting,
        challenge=challenge
    )

    print(f"🎭 Creating story with {character} in {setting}")
    print("=" * 60)

    for chunk in llm.stream(formatted_prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n" + "=" * 60)


stream_with_template(
    character="a young inventor",
    setting="a floating city in the clouds",
    challenge="fixing the city's failing gravity generator"
)

# =============================================================================
# 5. STREAMING CHAINS
# =============================================================================

print("\n🔗 5. Streaming Chains")

# Create a streaming chain
streaming_template = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writing assistant. Write engaging, vivid content."),
    ("human", "Write about {topic} in an {style} style.")
])

# Create streaming chain
streaming_chain = streaming_template | llm


def demonstrate_streaming_chain():
    """Show how to stream with chains"""
    print("🔄 Streaming chain example:")

    for chunk in streaming_chain.stream({
        "topic": "time travel",
        "style": "mysterious"
    }):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n✅ Chain streaming complete")


demonstrate_streaming_chain()


# =============================================================================
# 10. KEY INFORMATION
# =============================================================================

print("\n📋 Key Information:")
print("✅ .stream(): Stream responses chunk by chunk")
print("✅ AIMessageChunk: Individual streaming response pieces")
print("✅ Chunk aggregation: Combine chunks into complete responses")
print("✅ Real-time processing: Process content as it streams")

print("\n🎯 Next step: Learn about error handling and debugging")
print("📚 See: 08_error_handling.py")

# =============================================================================
# STUDENT TODO EXERCISE
# =============================================================================

print("\n" + "=" * 60)
print("📝 STUDENT TODO EXERCISE")
print("=" * 60)

"""
TODO: Create a Streaming Progress Tracker

Requirements:
1. Create a function called `stream_with_progress` that:
   - Takes a prompt and tracks the streaming progress
   - Counts words and characters as they stream
   - Prints a progress indicator every 10 words
   - Returns the complete response and final statistics

2. Test your function with this prompt:
   "Write a detailed explanation of how photosynthesis works, including
   the light and dark reactions, and why it's important for life on Earth."

3. Your function should print something like:
   - "[10 words]..."
   - "[20 words]..."
   - "Final stats: 87 words, 542 characters"

Your code here:
"""

# TODO: Implement your streaming progress tracker
# def stream_with_progress(prompt):
#     # Your implementation here
#     word_count = 0
#     char_count = 0
#     complete_response = ""
#
#     for chunk in llm.stream(prompt):
#         if chunk.content:
#             # Add your progress tracking logic
#             pass
#
#     return complete_response, {"words": word_count, "chars": char_count}

# TODO: Test your function
# test_prompt = "Write a detailed explanation of how photosynthesis works..."
# response, stats = stream_with_progress(test_prompt)
# print(f"Final response length: {len(response)} characters")
# print(f"Statistics: {stats}")

print("\n💡 Hint: Use chunk.content to get text, len(text.split()) for word count!")
