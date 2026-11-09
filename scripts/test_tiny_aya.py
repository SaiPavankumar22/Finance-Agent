"""Run one small Tiny Aya Global inference request with secondary HF credits."""

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(override=False)

token = os.getenv("HF_SECONDARY_TOKEN")
model = os.getenv("HF_MODEL_ID", "CohereLabs/tiny-aya-global")
if not token:
    raise SystemExit("HF_SECONDARY_TOKEN is not set.")

client = InferenceClient(
    provider="cohere",
    model=model,
    api_key=token,
)

response = client.chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a concise financial research assistant. Do not invent facts.",
        },
        {
            "role": "user",
            "content": "What is a stock ticker? Answer in one sentence.",
        },
    ],
    max_tokens=64,
    temperature=0,
)

print(response.choices[0].message.content)
