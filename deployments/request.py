import requests


# OpenAI API standard endpoint
SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"

request_data = {
    "model": "",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant on subjects related to climate change."},
        {"role": "user", "content": "What is climate change?"}
    ]
}

if __name__ == "__main__":
    response = requests.post(SERVER_URL, json=request_data)    
    print(response.json())