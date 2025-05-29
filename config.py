import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_STORE_PATH = "vector store/coffee_disease_knowledge"
YOLO_WEIGHTS_PATH = "runs/detect/train8/weights/best.pt"
