import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize Hugging Face InferenceClient
client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2",
    token=os.getenv("HF_TOKEN")
)

# Pydantic model for incoming request
class Exam_Data(BaseModel):
    question_id: str
    type: str
    answer: str
    correct: str
    status: str

# Function to get normalized embedding from Inference API
def get_sentence_embedding(text):
    try:
        embedding = client.feature_extraction(text, normalize=True)
        return np.array(embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {e}")

@app.get("/")
def home():
    return "Hello World!"

@app.post("/score_result")
def return_score(data: Exam_Data):
    json_data = data.dict()
    json_data["answer"] = json_data["answer"].strip("</p>")

    if json_data["type"].lower() == "theory":
        try:
            # Get embeddings
            correct_emb = get_sentence_embedding(json_data['correct'])  
            answer_emb = get_sentence_embedding(json_data['answer'])   

            if correct_emb.shape != (768,) or answer_emb.shape != (768,):
                raise HTTPException(status_code=500, detail="Invalid embedding shape")

            # Compute cosine similarity
            dot_product = np.dot(correct_emb, answer_emb)
            norm_product = np.linalg.norm(correct_emb) * np.linalg.norm(answer_emb)
            similarity = dot_product / norm_product if norm_product != 0 else 0.0

            # Clamp and round
            similarity = max(0.0, min(similarity, 1.0))
            json_data["score"] = round(float(similarity), 3)

            # Change the status from pending to marked
            json_data["status"] = "MARKED"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    return json_data

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)