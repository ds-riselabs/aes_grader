import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI

# Load environment variables
load_dotenv()

# Instantiate openai's client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# instantiate app for fastapi
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

            # Justify the similarity score of hf sentence transformer
            student_resp, teacher_resp, similarity_score = json_data["answer"], json_data["correct"], json_data["score"]
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
                                    You are an academic subject matter expert.
                                    Task:
                                    1. Evaluate whether the provided similarity score is justified based on semantic and contextual alignment between the teacher's and student's responses.
                                    2. If the similarity score is justified, return exactly: "JUSTIFIED"
                                    3. If the similarity score is NOT justified, grade the student's response strictly between 0 and 1 using domain knowledge, where:
                                        1.0 = fully correct and aligned in meaning,
                                        0.0 = entirely incorrect or irrelevant.
                                        Award partial credit only if the meaning is substantially correct.
                                    4. Ensure consistent grading across all cases.
                                    5. Output must be either:
                                        - "JUSTIFIED" (if similarity score is acceptable), or
                                        - a float value (if grading is needed).
                                    """
                    },
                    {
                        "role": "user",
                        "content": f"""Grade the response 
                                    similarity score: {similarity_score}
                                    student's response: {student_resp},
                                    teacher's resp: {teacher_resp}
                                    """
                    }
                ]
            )
            ai_output = completion.choices[0].message.content.strip()
            # applying the conditional formatting
            if ai_output == "JUSTIFIED":
                json_data["score"] = similarity_score
            else:
                json_data["score"] = float(ai_output)
            
            # change th status of the data:
            json_data["status"] = "MARKED"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
    return json_data

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)