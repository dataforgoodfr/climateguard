import os

import requests
from dotenv import load_dotenv

load_dotenv()

# OpenAI API standard endpoint
SERVER_URL = "https://localhost/v1/chat/completions"

request_data = {
    "model": "",
    "stream": True,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant on subjects related to climate change.",
        },
        {"role": "user", "content": "What is climate change?"},
    ],
}
headers = {"Authorization": f"Bearer {os.getenv('LLAMA_API_KEY', '12345')}"}

if __name__ == "__main__":
    response = requests.post(SERVER_URL, json=request_data, headers=headers, verify=False)
    response.raise_for_status()
    print(response.json())
