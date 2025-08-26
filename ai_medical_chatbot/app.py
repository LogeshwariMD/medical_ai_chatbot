import os
import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# Load environment variables (API key etc.)
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API details
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # Adjust if needed


# Pydantic model for frontend queries
class QueryRequest(BaseModel):
    query: str


@app.post("/upload_and_query")
async def upload_and_query(request: QueryRequest):
    """Handles user query and sends to LLaMA model"""
    try:
        user_query = request.query.strip()

        if not user_query:
            return {"answer": "⚠️ Please enter a valid query."}

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": user_query},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {"answer": f"⚠️ API Error: {response.status_code}"}

        response_json = response.json()
        model_answer = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not model_answer:
            model_answer = "⚠️ Model did not return a response."

        return {"answer": model_answer}

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return {"answer": f"⚠️ Error: {str(e)}"}


@app.get("/")
async def root():
    return {"message": "✅ AI Medical Chatbot Backend is running!"}






from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h2>AI Medical Chatbot</h2>
            <form action="/upload_and_query" enctype="multipart/form-data" method="post">
                <input type="file" name="file" /><br><br>
                <input type="text" name="query" placeholder="Enter your query" /><br><br>
                <input type="submit" value="Ask" />
            </form>
        </body>
    </html>
    """

# Handle upload + query
@app.post("/upload_and_query")
async def upload_and_query(file: UploadFile, query: str = Form(...)):
    return {"message": f"Your query: '{query}', file: '{file.filename}'"}

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
