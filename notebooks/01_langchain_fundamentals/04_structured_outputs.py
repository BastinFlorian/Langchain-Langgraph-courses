# !pip install -q langchain-google-genai python-dotenv langchain-core pydantic
"""
Structured Outputs with LangChain and Google Gemini
==================================================

In this notebook, we learn:
- Using .with_structured_output() method
- Defining Pydantic models for structured data
- Extracting structured information from text
- Working with different data types and validation

Official documentation:
- LangChain Structured Outputs: https://python.langchain.com/docs/concepts/structured_outputs/
- Pydantic Models: https://docs.pydantic.dev/latest/concepts/models/
"""

from enum import Enum

from dotenv import load_dotenv

# First, let's handle pydantic import with error handling
try:
    from typing import List, Optional

    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
    print("✅ Pydantic imported successfully")
except ImportError:
    print("❌ Pydantic not available. Install with: pip install pydantic")
    print("Exiting...")
    exit(1)

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
# 1. BASIC STRUCTURED OUTPUT
# =============================================================================

print("\n🏗️ 1. Basic Structured Output")

# Define a simple Pydantic model


class PersonInfo(BaseModel):
    """Information about a person extracted from text."""
    name: str = Field(description="The person's full name")
    age: Optional[int] = Field(
        description="The person's age if mentioned", default=None)
    profession: Optional[str] = Field(
        description="The person's job or profession", default=None)
    location: Optional[str] = Field(
        description="Where the person lives or is from", default=None)


# Create structured LLM
structured_llm = llm.with_structured_output(PersonInfo)

# Test with sample text
text_about_person = """
Hi, my name is Sarah Johnson and I'm 28 years old.
I work as a software engineer in San Francisco, California.
I love coding and building web applications.
"""

print(f"Input text: {text_about_person.strip()}")
print("\n" + "-" * 50)

try:
    # Get structured output
    person_data = structured_llm.invoke(
        f"Extract person information from this text: {text_about_person}")

    print(f"✅ Structured output type: {type(person_data)}")
    print(f"Name: {person_data.name}")
    print(f"Age: {person_data.age}")
    print(f"Profession: {person_data.profession}")
    print(f"Location: {person_data.location}")

    # Show it's a proper Python object
    print(f"\nAs Python dict: {person_data.model_dump()}")

except Exception as e:
    print(f"❌ Structured output error: {e}")

# =============================================================================
# 2. COMPLEX STRUCTURED OUTPUT
# =============================================================================

print("\n📊 2. Complex Structured Output with Lists and Validation")


class ProductReview(BaseModel):
    """Product review analysis with multiple data types."""
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Rating from 1 to 5 stars", ge=1, le=5)
    sentiment: str = Field(
        description="Overall sentiment: positive, negative, or neutral")
    pros: List[str] = Field(description="List of positive aspects mentioned")
    cons: List[str] = Field(description="List of negative aspects mentioned")
    would_recommend: bool = Field(
        description="Whether the reviewer would recommend the product")
    confidence: float = Field(
        description="Confidence in the analysis from 0.0 to 1.0", ge=0.0, le=1.0)


# Create review analysis LLM
review_llm = llm.with_structured_output(ProductReview)

review_text = """
I bought this laptop last month and I'm really impressed! The battery life is amazing -
I can work for 8 hours straight without plugging in. The screen is bright and crisp,
perfect for coding and watching videos. The build quality feels solid too.

However, it does get quite hot during intensive tasks like video editing, and the
keyboard feels a bit cheap compared to my old ThinkPad. Also, the price was higher
than I expected for these specs.

Overall, I'd give it 4 stars and would definitely recommend it to others looking
for a good work laptop. It's not perfect, but the pros outweigh the cons.
"""

print(f"Review text: {review_text[:100]}...")
print("\n" + "-" * 50)

try:
    review_analysis = review_llm.invoke(
        f"Analyze this product review: {review_text}")

    print(f"✅ Product: {review_analysis.product_name}")
    print(f"Rating: {review_analysis.rating}/5 stars")
    print(f"Sentiment: {review_analysis.sentiment}")
    print(f"Pros: {', '.join(review_analysis.pros)}")
    print(f"Cons: {', '.join(review_analysis.cons)}")
    print(f"Would recommend: {review_analysis.would_recommend}")
    print(f"Analysis confidence: {review_analysis.confidence:.2f}")

except Exception as e:
    print(f"❌ Review analysis error: {e}")

# =============================================================================
# 3. NESTED STRUCTURED OUTPUT
# =============================================================================

print("\n🔗 3. Nested Structured Output")


class Address(BaseModel):
    """Address information."""
    street: Optional[str] = Field(description="Street address", default=None)
    city: Optional[str] = Field(description="City name", default=None)
    country: Optional[str] = Field(description="Country name", default=None)


class ContactInfo(BaseModel):
    """Contact information."""
    email: Optional[str] = Field(description="Email address", default=None)
    phone: Optional[str] = Field(description="Phone number", default=None)


class BusinessCard(BaseModel):
    """Complete business card information."""
    name: str = Field(description="Person's full name")
    title: Optional[str] = Field(description="Job title", default=None)
    company: Optional[str] = Field(description="Company name", default=None)
    address: Address = Field(description="Address information")
    contact: ContactInfo = Field(description="Contact information")


# Create business card parser
card_llm = llm.with_structured_output(BusinessCard)

business_card_text = """
John Smith
Senior Software Engineer
TechCorp Solutions

123 Innovation Drive
San Francisco, CA 94105
United States

Email: john.smith@techcorp.com
Phone: +1 (555) 123-4567
"""

print(f"Business card text:\n{business_card_text}")
print("-" * 50)

try:
    card_data = card_llm.invoke(
        f"Extract all information from this business card: {business_card_text}")

    print(f"✅ Name: {card_data.name}")
    print(f"Title: {card_data.title}")
    print(f"Company: {card_data.company}")
    print(
        f"Address: {card_data.address.street}, {card_data.address.city}, {card_data.address.country}")
    print(f"Email: {card_data.contact.email}")
    print(f"Phone: {card_data.contact.phone}")

    print(f"\nComplete structured data: {card_data.model_dump()}")

except Exception as e:
    print(f"❌ Business card parsing error: {e}")

# =============================================================================
# 4. MULTIPLE CHOICE STRUCTURED OUTPUT
# =============================================================================

print("\n📋 4. Multiple Choice and Enums")


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    IMPROVEMENT = "improvement"
    DOCUMENTATION = "documentation"


class TicketInfo(BaseModel):
    """Support ticket information."""
    title: str = Field(description="Short title summarizing the issue")
    description: str = Field(description="Detailed description of the issue")
    priority: Priority = Field(description="Priority level of the ticket")
    category: Category = Field(description="Type/category of the ticket")
    estimated_hours: Optional[int] = Field(
        description="Estimated hours to resolve", default=None)
    requires_customer_input: bool = Field(
        description="Whether customer input is needed")


# Create ticket analyzer
ticket_llm = llm.with_structured_output(TicketInfo)

support_request = """
Hi support team,

Our website has been loading very slowly for the past 3 days.
Pages that used to load in 2-3 seconds are now taking 15-20 seconds.
This is affecting our sales and customer experience significantly.

We've tried clearing cache and cookies but the problem persists.
Please help us resolve this as soon as possible as it's impacting our business.

Best regards,
Customer
"""

print(f"Support request: {support_request[:100]}...")
print("\n" + "-" * 50)

try:
    ticket_data = ticket_llm.invoke(
        f"Create a support ticket from this request: {support_request}")

    print(f"✅ Title: {ticket_data.title}")
    print(f"Priority: {ticket_data.priority.value}")
    print(f"Category: {ticket_data.category.value}")
    print(f"Description: {ticket_data.description[:100]}...")
    print(f"Estimated hours: {ticket_data.estimated_hours}")
    print(f"Requires customer input: {ticket_data.requires_customer_input}")

except Exception as e:
    print(f"❌ Ticket creation error: {e}")

# =============================================================================
# 5. ERROR HANDLING AND VALIDATION
# =============================================================================

print("\n🛡️ 5. Error Handling and Validation")


class StrictProduct(BaseModel):
    """Product with strict validation."""
    name: str = Field(description="Product name", min_length=1, max_length=100)
    price: float = Field(description="Product price in USD", gt=0, le=10000)
    in_stock: bool = Field(description="Whether product is in stock")
    category: str = Field(description="Product category",
                          pattern="^[a-zA-Z ]+$")


strict_llm = llm.with_structured_output(StrictProduct)

# Test with edge case
edge_case_text = "This amazing free product costs nothing and belongs to the electronics123 category"

print(f"Edge case text: {edge_case_text}")
print("-" * 50)

try:
    product_data = strict_llm.invoke(
        f"Extract product information: {edge_case_text}")
    print(f"✅ Product extracted: {product_data}")
except Exception as e:
    print(f"⚠️ Validation error (expected): {e}")
    print("Note: This demonstrates Pydantic's validation capabilities")

# =============================================================================
# 6. PRACTICAL EXAMPLE: EMAIL CLASSIFICATION
# =============================================================================

print("\n📧 6. Practical Example: Email Classification")


class EmailClassification(BaseModel):
    """Email classification and analysis."""
    subject: str = Field(description="Email subject line")
    category: str = Field(
        description="Email category: spam, important, newsletter, support, personal")
    urgency: str = Field(description="Urgency level: low, medium, high")
    sender_type: str = Field(
        description="Type of sender: customer, vendor, internal, unknown")
    action_required: bool = Field(
        description="Whether action is required from recipient")
    key_points: List[str] = Field(
        description="Main points or topics in the email")
    suggested_response_time: str = Field(
        description="Suggested response time: immediate, today, this_week, no_response")


email_classifier = llm.with_structured_output(EmailClassification)

sample_email = """
Subject: URGENT: Server outage affecting production environment

Hi DevOps Team,

We're experiencing a complete server outage on our main production environment.
All user-facing services are down and customers cannot access our platform.

The issue started at 2:30 PM EST. We've identified it might be related to
the database server, but need immediate investigation.

Please treat this as highest priority and escalate to on-call engineers immediately.

Error logs are attached. Need status update within 30 minutes.

Best regards,
Site Reliability Team
"""

print(f"Email sample: {sample_email[:100]}...")
print("\n" + "-" * 50)

try:
    email_analysis = email_classifier.invoke(
        f"Classify this email: {sample_email}")

    print(f"✅ Subject: {email_analysis.subject}")
    print(f"Category: {email_analysis.category}")
    print(f"Urgency: {email_analysis.urgency}")
    print(f"Sender type: {email_analysis.sender_type}")
    print(f"Action required: {email_analysis.action_required}")
    print(f"Key points: {', '.join(email_analysis.key_points)}")
    print(f"Suggested response time: {email_analysis.suggested_response_time}")

except Exception as e:
    print(f"❌ Email classification error: {e}")

# =============================================================================
# 7. KEY INFORMATION
# =============================================================================

print("\n📋 Key Information:")
print("✅ .with_structured_output(): Converts LLM responses to structured objects")
print("✅ Pydantic BaseModel: Defines data structure with types and validation")
print("✅ Field(): Adds descriptions and validation constraints")
print("✅ Optional[]: Makes fields optional with default values")
print("✅ List[]: Creates list/array fields")
print("✅ Enums: Restricts values to specific choices")
print("✅ Nested models: Supports complex hierarchical data")
print("✅ Validation: Automatic data validation based on field constraints")
print("✅ .model_dump(): Converts Pydantic object to dictionary")

# =============================================================================
# STUDENT TODO EXERCISE
# =============================================================================

print("\n" + "=" * 60)
print("📝 STUDENT TODO EXERCISE")
print("=" * 60)

"""
TODO: Create a Recipe Analyzer

Requirements:
1. Create a Pydantic model called 'Recipe' with these fields:
   - name: str (recipe name)
   - cuisine: str (type of cuisine: italian, chinese, mexican, etc.)
   - difficulty: str (easy, medium, hard)
   - prep_time: int (preparation time in minutes)
   - cook_time: int (cooking time in minutes)
   - servings: int (number of servings)
   - ingredients: List[str] (list of ingredients)
   - dietary_restrictions: List[str] (vegetarian, vegan, gluten-free, etc.)
   - is_healthy: bool (whether the recipe is considered healthy)

2. Test your model with this recipe text:
   "This delicious Italian Margherita Pizza takes 30 minutes to prep and 15 minutes to cook.
   It serves 4 people and is perfect for beginners. You'll need pizza dough, tomato sauce,
   fresh mozzarella, basil leaves, olive oil, and salt. It's vegetarian-friendly but
   not particularly healthy due to the cheese and refined flour."

3. Print all the extracted information in a readable format

Your code here:
"""


# TODO: Define your Recipe model
# class Recipe(BaseModel):
#     # Add your fields here with proper Field() descriptions
#     pass

# TODO: Create structured LLM for recipe analysis
# recipe_llm = llm.with_structured_output(Recipe)

# TODO: Test with the provided recipe text
# recipe_text = "This delicious Italian Margherita Pizza..."
#
# try:
#     recipe_data = recipe_llm.invoke(f"Extract recipe information: {recipe_text}")
#
#     print("Recipe Analysis:")
#     print(f"Name: {recipe_data.name}")
#     # Add more print statements for all fields
#
# except Exception as e:
#     print(f"Error: {e}")

print("\n💡 Hint: Use Field() with good descriptions to help the LLM extract accurate information!")
