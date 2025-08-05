import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from huggingface_hub import hf_hub_download, InferenceClient
import joblib
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# Load model
REPO_ID = "Ayonitemi-Feranmi/aes_grader"
FILENAME = "aes_grader.pkl"
model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=os.getenv("HF_TOKEN")))

# Initialize inference client once
client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2",
    token=os.getenv("HF_TOKEN")
)


# Pydantic model
class Exam_Data(BaseModel):
    question_id: str
    type: str
    answer: str
    correct: str
    status: str

# Get embedding from Inference API
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
            correct_emb = get_sentence_embedding(json_data['correct'])  
            answer_emb = get_sentence_embedding(json_data['answer'])    

            # Take absolute difference and reshape for model input
            input_vector = np.abs(correct_emb - answer_emb).reshape(1, 1, -1)

            # Make predictions
            prediction = model.predict(input_vector)[0][0] 

            # Apply Sigmoid 
            score = 1 / (1 + np.exp(-prediction))

            # Set a threshold for the model
            json_data["score"] = float(score)
            json_data["passed"] = bool(score >= 0.5)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    return json_data

# Run the fastapi
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
