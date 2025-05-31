from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd 
import os
import logging
from App.face_processor import process_image, get_face_info
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client
import io
import onnxruntime
from typing import Dict, Any, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition and matching",
    version="1.0.0"
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error", "detail": str(exc)}
    )

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
            logger.warning("Zero norm detected in cosine similarity calculation")
            return 0.0
            
        similarity = np.dot(a, b) / (norm_a * norm_b)
        logger.debug(f"Cosine similarity calculated: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}", exc_info=True)
        return 0.0

# Load environment variables
load_dotenv()

# Get environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing required environment variables: SUPABASE_URL and/or SUPABASE_KEY")
    raise EnvironmentError("Missing required environment variables")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully connected to Supabase")
except Exception as e:
    logger.error(f"Failed to connect to Supabase: {str(e)}", exc_info=True)
    supabase = None

def load_data():
    """Load data from Supabase with proper error handling"""
    if not supabase:
        logger.error("Supabase client not initialized")
        return []
        
    try:
        response = supabase.table("Hadj").select("*").execute()
        logger.debug(f"Raw Supabase response: {response}")

        if response.data:
            logger.info(f"Data fetched successfully: {len(response.data)} records")
            return response.data
        else:
            logger.warning("No data found in Supabase response")
            return []
    except Exception as e:
        logger.error(f"Error fetching data from Supabase: {str(e)}", exc_info=True)
        return []

@app.get("/")
def read_root():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    return {"message": "Face recognition API is running"}

@app.get("/data")
def get_data():
    """Get all data from database"""
    logger.info("Data endpoint called")
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

@app.post("/get_embedding")
async def get_face_embedding(file: UploadFile = File(...)):
    """Return face embedding from uploaded image"""
    logger.info(f"Get embedding endpoint called with file: {file.filename}")
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")

        # Validate file size (max 10MB)
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"File too large: {file_size} bytes")
                raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        # Reset file pointer
        await file.seek(0)
        
        # Read and process image
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        logger.debug(f"Image loaded successfully: {image.size}")

        # Process image to get embedding
        embedding = process_image(image)

        if embedding is None:
            logger.warning("No face detected in the uploaded image")
            return {
                "success": False,
                "message": "No face detected in the uploaded image"
            }

        logger.info(f"Successfully extracted embedding of dimension: {len(embedding)}")
        return {
            "success": True,
            "embedding": embedding
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_face_embedding: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize face from uploaded image"""
    logger.info(f"Recognize endpoint called with file: {file.filename}")
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate file size (max 10MB)
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"File too large: {file_size} bytes")
                raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        # Reset file pointer
        await file.seek(0)
        
        # Read and process image
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        logger.debug(f"Image loaded successfully: {image.size}")
        
        # Process uploaded image to get embedding
        test_embedding = process_image(image)
        
        if test_embedding is None:
            logger.warning("No face detected in the uploaded image")
            return {
                "success": False,
                "message": "No face detected in the uploaded image"
            }
        
        # Load dataset
        dataset = load_data()
        if not dataset:
            logger.warning("Database is empty or unavailable")
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
                    logger.warning(f"No embedding field found in record {record.get('id', 'unknown')}")
                    continue
                
                stored_embedding = record[embedding_field]
                
                # Convert to numpy array if it's a list
                if isinstance(stored_embedding, list):
                    embedding = np.array(stored_embedding, dtype=np.float32)
                else:
                    embedding = np.array(stored_embedding, dtype=np.float32)
                
                # Calculate similarity
                similarity = cosine_similarity(test_embedding, embedding)
                logger.debug(f"Similarity with record {record.get('id')}: {similarity}")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = record
                    
            except Exception as e:
                logger.error(f"Error processing record {record.get('id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        # Set threshold for matching
        threshold = 0.4
        
        if max_similarity > threshold and best_match is not None:
            logger.info(f"Found match with similarity {max_similarity} for record {best_match.get('id')}")
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
            logger.info(f"No match found. Max similarity: {max_similarity}, Threshold: {threshold}")
            return {
                "success": False,
                "message": "No matching identity found",
                "max_similarity": float(max_similarity) if max_similarity > -1 else 0.0,
                "threshold": threshold
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recognize_face: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
