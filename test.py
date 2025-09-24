"""API"""

import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print(os.environ["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            "You are a question answering chatbot. You must provide the answer in {language}.",
        ),
        ("human", "The question is: {question}"),
    ]
)

chain = prompt | llm
answer = chain.invoke({
    "language": "francais",
    "question": 'ca va',
}).content


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Écris-moi un haïku sur le code.")
print(response.text)
