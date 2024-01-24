import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    model="cognitivecomputations/dolphin-mixtral-8x7b",
    messages=[
        {
            "role": "user",
            "content": "Tell Me About Yourself",
        },
    ],
)
print(completion.choices[0].message.content)
