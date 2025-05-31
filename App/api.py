from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import pandas as pd 
import os
from processing import process_image
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client
import io
import onnxruntime
from typing import Dict, Any, Optional
from insightface.app import FaceAnalysis
app = FastAPI()

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    try:
        # Ensure inputs are numpy arrays
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        
        # Calculate norms
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(a, b) / (norm_a * norm_b)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0
    
# Face embedings extraction 
    
face_app = FaceAnalysis(name="buffalo_l")  # buffalo_l uses ArcFace with a high-quality model
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 -> CPU, det_size controls face detection size

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

def process_image(input_image):
    # Handle if input is a file path (string or Path)
    if isinstance(input_image, (str, os.PathLike)):
        img = Image.open(input_image).convert('RGB')
    # Handle if input is a PIL Image
    elif isinstance(input_image, Image.Image):
        img = input_image.convert('RGB')
    else:
        raise ValueError("Unsupported input type for image processing. Expected file path or PIL.Image.Image")

    img_np = np.array(img)

    # 🔹 Detect face and get embedding
    faces = face_app.get(img_np)

    if len(faces) == 0:
        print("❌ No face detected!")
        return None

    # Take the first detected face
    face = faces[0]

    # Get embedding
    embedding = face.normed_embedding  # Already normalized!

    return embedding.tolist()

# Load environment variables
load_dotenv()

# Use environment variables with fallback (remove hardcoded credentials in production)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://pafrlirbdezfaloemgxg.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBhZnJsaXJiZGV6ZmFsb2VtZ3hnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwMDk4MzIsImV4cCI6MjA2MDU4NTgzMn0.gXO_hRuznX1wFNBCnElcOiWthZ93t-cVWeQGOYZw6j4")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    supabase = None

def load_data():
    """Load data from Supabase with proper error handling"""
    if not supabase:
        print("Supabase client not initialized")
        return []
        
    try:
        response = supabase.table("Hadj").select("*").execute()
        print("Raw Supabase response:", response)

        if response.data:
            print(f"Data fetched successfully: {len(response.data)} records")
            return response.data
        else:
            print("No data found in response")
            return []
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        return []

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Face recognition API is running"}

@app.get("/data")
def get_data():
    """Get all data from database"""
    dataset = load_data()
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
#post Embedings 
@app.post("/get_embedding")
async def get_face_embedding(file: UploadFile = File(...)):
    """Return face embedding from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process image to get embedding
        embedding = process_image(image)

        if embedding is None:
            return {
                "success": False,
                "message": "No face detected in the uploaded image"
            }

        return {
            "success": True,
            "embedding": embedding
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_face_embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
#post Recognition
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize face from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process uploaded image to get embedding
        test_embedding = process_image(image)
        
        if test_embedding is None:
            return {
                "success": False,
                "message": "No face detected in the uploaded image"
            }
        
        # Load dataset
        dataset = load_data()
        if not dataset:
            return {
                "success": False,
                "message": "Database is empty or unavailable"
            }
        
        # Compare with dataset
        max_similarity = -1
        best_match = None

        for record in dataset:
            try:
                # Handle different possible field names for embeddings
                embedding_field = None
                for field in ['face_embedings', 'face_embeddings', 'embedding', 'embeddings']:
                    if field in record and record[field] is not None:
                        embedding_field = field
                        break
                
                if embedding_field is None:
                    print(f"No embedding field found in record {record.get('id', 'unknown')}")
                    continue
                
                stored_embedding = record[embedding_field]
                
                # Convert to numpy array if it's a list
                if isinstance(stored_embedding, list):
                    embedding = np.array(stored_embedding, dtype=np.float32)
                else:
                    embedding = np.array(stored_embedding, dtype=np.float32)
                
                # Calculate similarity
                similarity = cosine_similarity(test_embedding, embedding)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = record
                    
            except Exception as e:
                print(f"Error processing record {record.get('id', 'unknown')}: {e}")
                continue
        
        # Set threshold for matching
        threshold = 0.7  # Adjust as needed
        
        if max_similarity > threshold and best_match is not None:
            return {
                "success": True,
                    "id": best_match.get("id"),
                    "full_name": best_match.get("full_name"),
                    "age": best_match.get("age"),
                    "nationality": best_match.get("nationality"),
                    "blood_type": best_match.get("blood_type"),
                    "gender": best_match.get("gender"),
                    "contact": best_match.get("contact"),
                    "illness": best_match.get("illness")
            }
        else:
            return {
                "success": False,
                "message": "No matching identity found",
                "max_similarity": float(max_similarity) if max_similarity > -1 else 0.0,
                "threshold": threshold
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in recognize_face: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
