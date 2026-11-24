
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=20000)
)

MODEL = "gemini-3.5-flash-lite"

print(f"Testing model: {MODEL}", flush=True)

start = time.perf_counter()

try:
    response = client.models.generate_content(
        model=MODEL,
        contents=(
            "Explain Machine Learning in three simple sentences. "
            "Give a meaningful and complete answer."
        ),
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=150
        )
    )

    elapsed = time.perf_counter() - start

    print("\nGenerated Answer:")
    print(response.text)

    print(f"\nResponse Time: {elapsed:.2f} seconds")

except Exception as e:
    elapsed = time.perf_counter() - start

    print("\nError:", str(e))
    print(f"Response Time: {elapsed:.2f} seconds")
