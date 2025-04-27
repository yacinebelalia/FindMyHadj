from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

# Initialize InsightFace
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Define process_image correctly
def process_image(img):
    # If input is a path, load the image
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')

    img_np = np.array(img)
    faces = face_app.get(img_np)

    if not faces:
        print("❌ No face detected")
        return None

    # Take the first detected face
    face = faces[0]
    embedding = face.normed_embedding  # Already normalized
    return embedding.tolist()  # Convert to list for JSON compatibility
