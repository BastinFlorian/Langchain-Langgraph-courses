# pip install langgraph langchain-google-genai python-dotenv langfuse

"""
LangGraph + Langfuse Callback Handler - Advanced Observability

This script demonstrates:
- Setting up Langfuse for LangGraph observability
- Tracking agent conversations and tool usage
- Understanding traces, spans, and generations
- Monitoring agent performance and costs
- Debugging agent behavior through detailed logs

Documentation:
- LangGraph: https://langchain-ai.github.io/langgraph/concepts/observability/
- Langfuse: https://langfuse.com/docs/integrations/langchain/get-started

IMPORTANT: You'll need to set up Langfuse API keys!
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Langfuse imports


# Load environment variables
load_dotenv()

print("=== LangGraph + Langfuse Observability ===\n")

# Section 1: API Keys Setup Instructions
print("1. API Keys Setup Instructions")
print("-" * 35)

print("📋 Required API Keys:")
print("1. GOOGLE_API_KEY - Get from Google AI Studio: https://makersuite.google.com/app/apikey")
print("2. LANGFUSE_SECRET_KEY - Get from Langfuse Cloud or self-hosted instance")
print("3. LANGFUSE_PUBLIC_KEY - Get from Langfuse Cloud or self-hosted instance")
print("4. LANGFUSE_HOST - Default: https://cloud.langfuse.com (or your self-hosted URL)")
print()

# Check API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

print(langfuse_public_key, langfuse_secret_key, langfuse_host)

print("🔍 API Key Status:")
print(f"  GOOGLE_API_KEY: {'✓ Set' if google_api_key else '❌ Missing'}")
print(
    f"  LANGFUSE_SECRET_KEY: {'✓ Set' if langfuse_secret_key else '❌ Missing'}")
print(
    f"  LANGFUSE_PUBLIC_KEY: {'✓ Set' if langfuse_public_key else '❌ Missing'}")
print(f"  LANGFUSE_HOST: {langfuse_host}")
print()

if not google_api_key:
    print("❌ Google API key is required. Please add GOOGLE_API_KEY to your .env file")
    exit(1)

# Section 2: Setup Instructions for Langfuse
print("2. Langfuse Setup Instructions")
print("-" * 32)

if not (langfuse_secret_key and langfuse_public_key):
    print("🚀 To get started with Langfuse:")
    print()
    print("Option 1 - Langfuse Cloud (Recommended for beginners):")
    print("1. Go to https://cloud.langfuse.com")
    print("2. Sign up for a free account")
    print("3. Create a new project")
    print("4. Go to Settings → API Keys")
    print("5. Copy the Public Key and Secret Key")
    print("6. Add to your .env file:")
    print("   LANGFUSE_PUBLIC_KEY=pk-lf-...")
    print("   LANGFUSE_SECRET_KEY=sk-lf-...")
    print("   LANGFUSE_HOST=https://cloud.langfuse.com")
    print()
    print("Option 2 - Self-hosted Langfuse:")
    print("1. Follow the self-hosting guide: https://langfuse.com/docs/deployment/self-host")
    print("2. Set LANGFUSE_HOST to your instance URL")
    print("3. Create API keys in your instance")
    print()

# Section 3: Define Tools for Demonstration
print("3. Define Tools for Demonstration")
print("-" * 36)


@tool
def research_topic(topic: str) -> str:
    """Research a topic and return key information.

    Args:
        topic: Topic to research

    Returns:
        Research findings about the topic
    """
    # Simulate research with predefined information
    research_data = {
        "artificial intelligence": "AI involves machine learning, neural networks, and automation. Key applications include natural language processing, computer vision, and robotics.",
        "climate change": "Global warming caused by greenhouse gas emissions. Major impacts include rising sea levels, extreme weather, and ecosystem disruption.",
        "quantum computing": "Computing using quantum mechanical phenomena like superposition and entanglement. Potential to solve complex problems exponentially faster.",
        "blockchain": "Distributed ledger technology providing transparency and security. Used in cryptocurrencies, smart contracts, and supply chain management.",
        "renewable energy": "Energy from sustainable sources like solar, wind, hydro, and geothermal. Critical for reducing carbon emissions and energy independence."
    }

    topic_lower = topic.lower()
    for key, info in research_data.items():
        if key in topic_lower:
            return f"Research on {topic}: {info}"

    return f"Research on {topic}: Limited information available. This appears to be a specialized topic requiring additional research sources."


@tool
def analyze_data(data_description: str) -> str:
    """Analyze data and provide insights.

    Args:
        data_description: Description of data to analyze

    Returns:
        Analysis insights and recommendations
    """
    # Simulate data analysis
    if "sales" in data_description.lower():
        return "Sales Analysis: Revenue trends show 15% growth Q/Q. Recommend focusing on high-performing products and expanding market reach."
    elif "website" in data_description.lower() or "traffic" in data_description.lower():
        return "Website Analysis: Traffic increased 23% with bounce rate at 45%. Recommend A/B testing landing pages and improving page load speeds."
    elif "customer" in data_description.lower():
        return "Customer Analysis: Satisfaction scores averaged 4.2/5. Top issues: response time (32%) and product quality (18%). Recommend support training."
    else:
        return f"Data Analysis for {data_description}: Patterns identified showing growth opportunities. Recommend implementing data-driven decision making."


@tool
def generate_report(topic: str, analysis_type: str = "summary") -> str:
    """Generate a formatted report.

    Args:
        topic: Report topic
        analysis_type: Type of analysis (summary, detailed, executive)

    Returns:
        Formatted report
    """
    import datetime

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    report = f"""
REPORT: {topic.upper()}
Date: {current_date}
Type: {analysis_type.title()} Report

EXECUTIVE SUMMARY:
This report analyzes {topic} based on current data and research findings.

KEY FINDINGS:
• Positive growth indicators observed
• Market opportunities identified
• Implementation recommendations provided

RECOMMENDATIONS:
1. Continue monitoring key metrics
2. Implement data-driven strategies
3. Regular review and optimization

Report generated by AI Assistant with LangGraph + Langfuse tracking.
"""
    return report.strip()


tools = [research_topic, analyze_data, generate_report]
print(f"✓ Created {len(tools)} demonstration tools")
print()

# Section 4: Initialize LLM and Langfuse
print("4. Initialize LLM and Langfuse")
print("-" * 32)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.2
)
print("✓ Initialized Gemini model")

# Initialize Langfuse callback handler if available
langfuse_handler = LangfuseCallbackHandler(
    public_key=langfuse_public_key,
)
print("✓ Langfuse callback handler initialized")

# Test Langfuse connection
langfuse_client = Langfuse()

print(langfuse_client._project_id)

print("✓ Langfuse client connection verified")

print()

# Section 5: Create ReAct Agent with Observability
print("5. Create ReAct Agent with Observability")
print("-" * 45)

memory = MemorySaver()

system_prompt = """You are an AI research assistant that helps users with comprehensive analysis and reporting.

Your capabilities include:
- Researching topics and gathering information
- Analyzing data and providing insights
- Generating professional reports

Always be thorough in your research and provide actionable insights. Use multiple tools when necessary to provide comprehensive answers."""

# Create ReAct agent
agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
    prompt=system_prompt
)

print("✓ ReAct agent created with research and analysis capabilities")
print()

# Section 6: Demonstrate Agent with Langfuse Tracking
print("6. Demonstrate Agent with Langfuse Tracking")
print("-" * 45)

config = {"configurable": {"thread_id": "langfuse_demo_thread"}}
config["callbacks"] = [langfuse_handler]

print("✓ Langfuse tracking enabled for this session")

print()

# Test query that will use multiple tools
test_query = "Research artificial intelligence, analyze its impact on business, and generate an executive summary report"

print(f"Complex Query: {test_query}")
print("-" * 80)

# Invoke agent with tracking
response = agent.invoke(
    {"messages": [HumanMessage(content=test_query)]},
    config=config
)

# Print the response
final_message = response["messages"][-1]
print(f"Agent Response:\n{final_message.content}")
print()

# Show the reasoning process
print("Agent Reasoning Process:")
print("-" * 30)

tool_calls_made = []
for msg in response["messages"]:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tool_call in msg.tool_calls:
            tool_calls_made.append(tool_call['name'])
            print(f"  🔧 Tool: {tool_call['name']}")
            print(f"     Args: {tool_call['args']}")
    elif hasattr(msg, 'name') and msg.name:  # Tool response
        print(
            f"  📊 Result: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")

print(
    f"\nTotal tools used: {len(tool_calls_made)} - {', '.join(set(tool_calls_made))}")


print("\n" + "=" * 80 + "\n")

# Section 7: Langfuse Dashboard Overview
print("7. Langfuse Dashboard Overview")
print("-" * 33)

print("🎯 What to look for in your Langfuse dashboard:")
print()
print("📊 Traces View:")
print("  • Complete conversation flows")
print("  • Tool usage patterns")
print("  • Response times and latency")
print("  • Token usage and costs")
print()
print("🔧 Spans View:")
print("  • Individual LLM calls")
print("  • Tool executions")
print("  • Processing steps")
print()
print("💬 Generations View:")
print("  • LLM input/output pairs")
print("  • Token counts")
print("  • Model performance metrics")
print()
print("📈 Analytics:")
print("  • Usage patterns over time")
print("  • Cost analysis")
print("  • Performance trends")
print()
print(f"🌐 View your data at: {langfuse_host}")

print()


# TODO: Student Exercise
print("\n" + "=" * 65)
print("TODO: Student Exercise - Full Observability Implementation")
print("=" * 65)
print("""
Set up complete observability for a customer service agent:

1. Set up Langfuse (if not done yet):
   - Sign up at https://cloud.langfuse.com
   - Create a project and get API keys
   - Add keys to your .env file

2. Create a customer service agent with these tools:
   - check_order_status: Look up order information
   - process_refund: Handle refund requests
   - escalate_to_human: Transfer complex issues
   - send_email: Send confirmation emails
   - update_customer_info: Modify customer details

3. Implement advanced Langfuse tracking:
   - Custom trace names for different interaction types
   - User IDs and session tracking
   - Metadata for customer context
   - Tags for categorizing interactions

4. Test scenarios and analyze in Langfuse:
   - "I need to return my recent order"
   - "My package hasn't arrived, what's happening?"
   - "Change my email address to..."
   - "This is too complicated, I need human help"

5. Dashboard analysis tasks:
   - Identify most common customer issues
   - Find longest resolution times
   - Track tool usage effectiveness
   - Monitor customer satisfaction patterns

6. Optimization based on data:
   - Improve prompts based on failure cases
   - Optimize tool selection logic
   - Reduce average resolution time
   - Enhance user experience

Advanced: Set up alerts for high-cost conversations or error rates.

Pro tip: Use Langfuse's score feature to track customer satisfaction!
""")
