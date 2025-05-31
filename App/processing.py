from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import onnxruntime
import cv2

# Global variable to store the face analysis app
face_app = None

def initialize_face_app():
    """Initialize InsightFace application with error handling"""
    global face_app
    try:
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0)
        print("✅ InsightFace initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing InsightFace: {e}")
        print("Make sure you have installed insightface: pip install insightface")
        return False

def process_image(img):
    """
    Process image to extract face embedding
    
    Args:
        img: PIL Image object or string path to image
        
    Returns:
        list: Face embedding as a list, or None if no face detected
    """
    global face_app
    
    # Initialize face_app if not already done
    if face_app is None:
        if not initialize_face_app():
            return None
    
    try:
        # Handle different input types
        if isinstance(img, str):
            # If input is a file path
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            # If input is numpy array or other format
            img = Image.fromarray(img).convert('RGB')
        
        # Convert PIL Image to numpy array (RGB format)
        img_np = np.array(img)
        
        # Convert RGB to BGR for OpenCV/InsightFace compatibility
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(img_bgr)
        
        if not faces:
            print("❌ No face detected in the image")
            return None
        
        if len(faces) > 1:
            print(f"⚠️  Multiple faces detected ({len(faces)}), using the first one")
        
        # Take the first detected face
        face = faces[0]
        
        # Get the embedding (already normalized by InsightFace)
        embedding = face.normed_embedding
        
        # Ensure it's the right shape and type
        if embedding is None:
            print("❌ Failed to extract face embedding")
            return None
            
        # Convert to list for JSON compatibility
        embedding_list = embedding.astype(np.float32).tolist()
        
        print(f"✅ Face embedding extracted successfully (dimension: {len(embedding_list)})")
        return embedding_list
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def get_face_info(img):
    """
    Get additional face information like age, gender, etc.
    
    Args:
        img: PIL Image object or string path to image
        
    Returns:
        dict: Face information including embedding, age, gender, etc.
    """
    global face_app
    
    # Initialize face_app if not already done
    if face_app is None:
        if not initialize_face_app():
            return None
    
    try:
        # Handle different input types
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert('RGB')
        
        # Convert PIL Image to numpy array and then to BGR
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(img_bgr)
        
        if not faces:
            return None
        
        face = faces[0]
        
        # Extract various face attributes
        face_info = {
            'embedding': face.normed_embedding.astype(np.float32).tolist(),
            'bbox': face.bbox.tolist(),  # Bounding box
            'det_score': float(face.det_score),  # Detection confidence
        }
        
        # Add age and gender if available
        if hasattr(face, 'age'):
            face_info['age'] = int(face.age)
        if hasattr(face, 'gender'):
            face_info['gender'] = int(face.gender)  # 0: female, 1: male
        
        return face_info
        
    except Exception as e:
        print(f"Error getting face info: {e}")
        return None

# Initialize the face app when module is imported
initialize_face_app()
