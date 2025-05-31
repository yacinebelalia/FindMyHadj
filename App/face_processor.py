from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import onnxruntime
import cv2
import logging
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store the face analysis app
face_app = None

def initialize_face_app():
    """Initialize InsightFace application with error handling"""
    global face_app
    try:
        # Set providers for both CUDA and CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logger.info("Using both CUDA and CPU providers")

        face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✅ InsightFace initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing InsightFace: {str(e)}", exc_info=True)
        logger.error("Make sure you have installed insightface: pip install insightface")
        return False

def validate_image(img):
    """Validate image dimensions and format"""
    try:
        if isinstance(img, str):
            if not os.path.exists(img):
                logger.error(f"Image file not found: {img}")
                return False
            img = Image.open(img)
        
        # Check image dimensions
        width, height = img.size
        if width < 20 or height < 20:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        if width > 4096 or height > 4096:
            logger.warning(f"Image too large: {width}x{height}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}", exc_info=True)
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
        logger.info("Face app not initialized, attempting initialization")
        if not initialize_face_app():
            logger.error("Failed to initialize face app")
            return None
    
    try:
        # Validate image
        if not validate_image(img):
            return None

        # Handle different input types
        if isinstance(img, str):
            logger.debug(f"Processing image from path: {img}")
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            logger.debug("Converting input to PIL Image")
            img = Image.fromarray(img).convert('RGB')
        
        # Convert PIL Image to numpy array (RGB format)
        img_np = np.array(img)
        logger.debug(f"Image converted to numpy array with shape: {img_np.shape}")
        
        # Convert RGB to BGR for OpenCV/InsightFace compatibility
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(img_bgr)
        
        if not faces:
            logger.warning("❌ No face detected in the image")
            return None
        
        if len(faces) > 1:
            logger.warning(f"⚠️  Multiple faces detected ({len(faces)}), using the first one")
        
        # Take the first detected face
        face = faces[0]
        
        # Get the embedding (already normalized by InsightFace)
        embedding = face.normed_embedding
        
        # Ensure it's the right shape and type
        if embedding is None:
            logger.error("❌ Failed to extract face embedding")
            return None
            
        # Convert to list for JSON compatibility
        embedding_list = embedding.astype(np.float32).tolist()
        
        logger.info(f"✅ Face embedding extracted successfully (dimension: {len(embedding_list)})")
        return embedding_list
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}", exc_info=True)
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
        logger.info("Face app not initialized, attempting initialization")
        if not initialize_face_app():
            logger.error("Failed to initialize face app")
            return None
    
    try:
        # Validate image
        if not validate_image(img):
            return None

        # Handle different input types
        if isinstance(img, str):
            logger.debug(f"Processing image from path: {img}")
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            logger.debug("Converting input to PIL Image")
            img = Image.fromarray(img).convert('RGB')
        
        # Convert PIL Image to numpy array and then to BGR
        img_np = np.array(img)
        logger.debug(f"Image converted to numpy array with shape: {img_np.shape}")
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_app.get(img_bgr)
        
        if not faces:
            logger.warning("No face detected in the image")
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
            logger.debug(f"Age detected: {face_info['age']}")
        if hasattr(face, 'gender'):
            face_info['gender'] = int(face.gender)  # 0: female, 1: male
            logger.debug(f"Gender detected: {'Male' if face_info['gender'] == 1 else 'Female'}")
        
        logger.info("✅ Face information extracted successfully")
        return face_info
        
    except Exception as e:
        logger.error(f"Error getting face info: {str(e)}", exc_info=True)
        return None

# Initialize the face app when module is imported
logger.info("Initializing face app on module import")
initialize_face_app()
