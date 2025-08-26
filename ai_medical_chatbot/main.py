import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY is not set in the .env file")

def process_image(image_path, query):
    with open(image_path, "rb") as f:
        image_content = f.read()
    encoded_image = base64.b64encode(image_content).decode("utf-8")
    
    # Validate image
    try:
        img = Image.open(io.BytesIO(image_content))
        img.verify()
    except Exception as e:
        logger.error(f"Invalid image format: {e}")
        return {"error": f"Invalid image format: {e}"}
    
    # Combine text + image into a single string
    combined_content = f"{query}\n![image](data:image/jpeg;base64,{encoded_image})"
    messages = [{"role": "user", "content": combined_content}]
    
    def make_api_request(model):
        return requests.post(
            GROQ_API_URL,
            json={"model": model, "messages": messages, "max_tokens": 1000},
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
    
    # Use a supported multimodal model
    resp = make_api_request("meta-llama/llama-4-scout-17b-16e-instruct")
    results = {}
    
    if resp.status_code == 200:
        answer = resp.json()["choices"][0]["message"]["content"]
        logger.info(f"Response: {answer}")
        results["llama-4-scout-17b"] = answer
    else:
        logger.error(f"API Error: {resp.status_code} - {resp.text}")
        results["llama-4-scout-17b"] = f"Error: {resp.status_code}"
    
    return results

if __name__ == "__main__":
    image_path = "test1.png"
    query = "what are the encoders in this picture?"
    result = process_image(image_path, query)
    print(result)