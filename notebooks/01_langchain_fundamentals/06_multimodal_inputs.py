# !pip install -q langchain-google-genai python-dotenv langchain-core pillow requests
"""
Multimodal Inputs with Google Gemini
====================================

In this notebook, we learn:
- Sending images to Gemini models
- Handling different image formats and sources
- Combining text and image prompts
- Best practices for multimodal applications

Official documentation:
- LangChain Multimodal: https://python.langchain.com/docs/concepts/multimodality/
- Google Gemini Vision: https://ai.google.dev/gemini-api/docs/vision
"""

import base64
import io

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize Gemini Pro model (supports multimodal)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Use 2.0 flash for multimodal
    temperature=0.7,
    max_tokens=1000,
)

print("✅ Gemini 2.0 Flash model initialized for multimodal")

# =============================================================================
# 1. IMAGE FROM URL
# =============================================================================

print("\n🖼️ 1. Analyzing Image from URL")


def create_image_message_from_url(image_url, text_prompt):
    """Create a message with image from URL and text prompt"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    )
    return message


# Example with a public image URL (Google Cloud sample image)
image_url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/suitcase.png"

image_message = create_image_message_from_url(
    image_url,
    "Describe what you see in this image. What are colors and can I travel with it?"
)

try:
    response = llm.invoke([image_message])
    print(f"Image description: {response.content}")
except Exception as e:
    print(f"Error processing image from URL: {e}")
    print("Note: Image URL analysis may have limitations")

# =============================================================================
# 2. IMAGE FROM BASE64
# =============================================================================

print("\n📸 2. Processing Base64 Encoded Image")


def encode_image_to_base64(image_path):
    """Encode local image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        return None


def create_image_message_from_base64(base64_image, text_prompt, image_type="jpeg"):
    """Create message with base64 image"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{base64_image}"
                }
            }
        ]
    )
    return message

# Create a simple test image programmatically


def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (200, 100), color='lightblue')

    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Encode to base64
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')


# Create and analyze test image
test_image_b64 = create_test_image()
test_message = create_image_message_from_base64(
    test_image_b64,
    "What color is this image? Describe its characteristics.",
    "png"
)

try:
    response = llm.invoke([test_message])
    print(f"Test image analysis: {response.content}")
except Exception as e:
    print(f"Error processing base64 image: {e}")

# =============================================================================
# 3. MULTIMODAL PROMPT TEMPLATES
# =============================================================================

print("\n📝 3. Multimodal Prompt Templates")

# Create a template for image analysis
multimodal_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert image analyst. Provide detailed, accurate descriptions."),
    ("human", [
        {"type": "text", "text": "Analyze this image for {analysis_type}. Focus on {focus_areas}."},
        {"type": "image_url", "image_url": {"url": "{image_url}"}}
    ])
])

# Test the template (with a safe fallback)
template_input = {
    "analysis_type": "composition and colors",
    "focus_areas": "visual elements, lighting, and overall mood",
    "image_url": "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/suitcase.png"
}

try:
    formatted_messages = multimodal_template.format_messages(**template_input)
    response = llm.invoke(formatted_messages)
    print(f"Template-based analysis: {response.content}")
except Exception as e:
    print(f"Template analysis error: {e}")

# =============================================================================
# 4. COMBINING MULTIPLE IMAGES
# =============================================================================

print("\n🖼️🖼️ 4. Analyzing Multiple Images")


def create_multi_image_message(image_urls, text_prompt):
    """Create message with multiple images"""
    content = [{"type": "text", "text": text_prompt}]

    for i, url in enumerate(image_urls):
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    return HumanMessage(content=content)


# Example with multiple accessible images
simple_images = [
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/suitcase.png",
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/scones.jpg"
]

multi_image_message = create_multi_image_message(
    simple_images,
    "Compare these two images. What are the differences in style, purpose, and visual elements?"
)

try:
    response = llm.invoke([multi_image_message])
    print(f"Multi-image comparison: {response.content}")
except Exception as e:
    print(f"Multi-image analysis error: {e}")

# =============================================================================
# 5. PRACTICAL APPLICATIONS
# =============================================================================

print("\n🛠️ 5. Practical Multimodal Applications")


class ImageAnalyzer:
    """A practical image analysis assistant"""

    def __init__(self, model):
        self.model = model

    def describe_image(self, image_url):
        """Get general description of an image"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Provide a detailed description of this image."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        return self.model.invoke([message]).content

    def extract_text(self, image_url):
        """Extract and read text from an image (OCR)"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract and transcribe any text visible in this image."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        return self.model.invoke([message]).content

    def identify_objects(self, image_url):
        """Identify objects and elements in an image"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "List and identify all objects, people, and elements visible in this image."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        return self.model.invoke([message]).content

    def analyze_mood(self, image_url):
        """Analyze the mood and emotional tone of an image"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze the mood, atmosphere, and emotional tone conveyed by this image."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        return self.model.invoke([message]).content


# Use the image analyzer
analyzer = ImageAnalyzer(llm)

# Test with a Google Cloud sample image
test_image_url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/suitcase.png"

print("\n--- Image Analysis Results ---")

try:
    description = analyzer.describe_image(test_image_url)
    print(f"Description: {description}")
except Exception as e:
    print(f"Description error: {e}")

try:
    objects = analyzer.identify_objects(test_image_url)
    print(f"Objects identified: {objects}")
except Exception as e:
    print(f"Object identification error: {e}")

try:
    mood = analyzer.analyze_mood(test_image_url)
    print(f"Mood analysis: {mood}")
except Exception as e:
    print(f"Mood analysis error: {e}")

# =============================================================================
# 6. BEST PRACTICES AND TIPS
# =============================================================================

print("\n💡 6. Best Practices for Multimodal Applications")


def demonstrate_best_practices():
    """Show best practices for multimodal applications"""

    print("✅ Image Quality Tips:")
    print("  - Use clear, well-lit images")
    print("  - Ensure reasonable resolution (not too small)")
    print("  - Avoid heavily compressed images")

    print("\n✅ Prompt Design Tips:")
    print("  - Be specific about what you want to analyze")
    print("  - Provide context when needed")
    print("  - Ask focused questions rather than general ones")

    print("\n✅ Error Handling:")
    print("  - Always wrap image processing in try-catch blocks")
    print("  - Have fallback strategies for failed image loads")
    print("  - Validate image URLs before processing")

    print("\n✅ Performance Tips:")
    print("  - Resize large images before encoding")
    print("  - Use appropriate image formats (JPEG for photos, PNG for graphics)")
    print("  - Cache frequently analyzed images")


demonstrate_best_practices()

# =============================================================================
# 7. CHAIN WITH MULTIMODAL INPUT
# =============================================================================

print("\n🔗 7. Creating Chains with Multimodal Input")


# Create a multimodal analysis chain
def create_multimodal_chain():
    """Create a chain that processes image and text together"""

    def process_multimodal_input(input_dict):
        """Process multimodal input and create appropriate message"""
        text = input_dict["text"]
        image_url = input_dict["image_url"]

        message = HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
        return [message]

    from langchain_core.runnables import RunnableLambda

    # Create the multimodal chain
    chain = RunnableLambda(process_multimodal_input) | llm | StrOutputParser()
    return chain


# Test the multimodal chain
multimodal_chain = create_multimodal_chain()

chain_input = {
    "text": "What objects and items can you identify in this image?",
    "image_url": "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/suitcase.png"
}

try:
    chain_result = multimodal_chain.invoke(chain_input)
    print(f"Chain result: {chain_result}")
except Exception as e:
    print(f"Chain processing error: {e}")

# =============================================================================
# 8. KEY INFORMATION
# =============================================================================

print("\n📋 Key Information:")
print("✅ Gemini models support text + image inputs")
print("✅ Images can be provided via URL or base64 encoding")
print("✅ Use HumanMessage with content list for multimodal")
print("✅ Support for multiple images in single request")
print("✅ Combine with templates and chains for complex workflows")
print("✅ Always implement proper error handling")
print("✅ Consider image quality and format optimization")

print("\n🎯 Next step: Learn about streaming responses")
print("📚 See: 07_streaming_responses.py")
