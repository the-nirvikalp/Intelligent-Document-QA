
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Sample text
sample_text = """
Artificial Intelligence is a field of computer science.
Machine Learning is a subset of Artificial Intelligence.
Deep Learning uses neural networks.
"""

# Generate embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sample_text,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768
    )
)

# Get embedding vector
embedding = response.embeddings[0].values

print("\nEMBEDDING GENERATED SUCCESSFULLY!")
print("Vector dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
