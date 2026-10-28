
# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise ValueError("GEMINI_API_KEY is missing from .env")

# client = genai.Client(api_key=api_key)

# for model in client.models.list():
#     if "generateContent" in (model.supported_actions or []):
#         print(model.name)


# python utils/llm_utils.py or 



import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "").strip()

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)
models = client.models.list()

print("\nModels returned by Groq API:\n")

for model in models.data:
    print(model.id)
