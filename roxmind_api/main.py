from fastapi import FastAPI
from feedback_logic import generate_feedback
from models import HyroxModelRequest

app = FastAPI()

@app.post("/predict_feedback")
def predict_feedback(data: HyroxModelRequest):
    return {"Performance Feedback": generate_feedback(data)}