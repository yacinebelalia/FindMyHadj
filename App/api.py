from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd 
import os
from App.processing import process_image
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client
import io

app = FastAPI()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


load_dotenv()
#SUPABASE_URL = os.getenv("SUPABASE_URL")
#SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = "https://pafrlirbdezfaloemgxg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBhZnJsaXJiZGV6ZmFsb2VtZ3hnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwMDk4MzIsImV4cCI6MjA2MDU4NTgzMn0.gXO_hRuznX1wFNBCnElcOiWthZ93t-cVWeQGOYZw6j4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def Load_data():
    response = supabase.table("Hadj").select("*").execute()
    print("Raw Supabase response:", response)

    if response.data:
        #print("Data fetched successfully:")
        #print(response.data)
        return response.data
    else:
        print("Error fetching data:", response.error)
        return []

'''
@app.get("/data")
def get_data():
    dataset = Load_data()
    if dataset:
        return {
            "success": True,
            "count": len(dataset),
            "data": dataset
        }
    else:
        return {
            "success": False,
            "message": "No data found or error occurred.",
            "hint": "Check server logs for Supabase errors."
        }
'''

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Process uploaded image
    test_embedding = process_image(image)
    
    if test_embedding is None:
        return {"message": "No face detected"}
    
    # Compare with dataset (to be modified later)
    max_similarity = -1
    best_match = None

    '''
    for _, entry in dataset.iterrows():
        similarity = cosine_similarity(test_embedding, entry['embedding'])
        if similarity > max_similarity:
           max_similarity = similarity
           best_match = entry
'''

    dataset = Load_data()
    for record in dataset:
        embedding = np.array(record['face_embedings'], dtype=np.float32)
        similarity = cosine_similarity(test_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = record
    
    threshold = 0.4  # Adjust as needed
    
    if max_similarity > threshold:
        return {
            "id": best_match["id"],
            "full_name": best_match["full_name"],
            "age": best_match["age"],
            "nationality": best_match["nationality"],
            "blood_type": best_match["blood_type"],
            "gender": best_match["gender"],
            "contact": best_match["contact"],
            "illness": best_match["illness"]
        }
    else:
        return {"message": "No matching identity found"}
        
    
