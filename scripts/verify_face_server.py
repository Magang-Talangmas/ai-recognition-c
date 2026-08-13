import os
import cv2
import numpy as np
import urllib.request
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize InsightFace model globally so it's loaded once at startup
try:
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load InsightFace model: {e}")
    face_app = None

def download_image(url):
    """Downloads an image from a URL and converts it to OpenCV format (BGR)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
    return None

def extract_embedding(img):
    """Extracts the face embedding with the highest detection score."""
    if face_app is None:
        raise RuntimeError("InsightFace model is not initialized.")
    
    faces = face_app.get(img)
    if not faces or len(faces) == 0:
        return None
    
    # Get the face with the highest detection score
    best_face = max(faces, key=lambda f: f.det_score)
    
    # L2 normalize the embedding
    embedding = best_face.embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

@app.route("/verify-face", methods=["POST"])
def verify_face():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON body"}), 400
        
        employee_id = data.get("employeeId")
        photo_url = data.get("photoUrl")
        master_photo_url = data.get("masterPhotoUrl")
        
        if not employee_id or not photo_url or not master_photo_url:
            return jsonify({"success": False, "error": "Missing required fields: employeeId, photoUrl, masterPhotoUrl"}), 400
            
        logger.info(f"Verification requested for Employee: {employee_id}")
        
        # Download images
        liveness_img = download_image(photo_url)
        if liveness_img is None:
            return jsonify({"success": False, "error": "Failed to download photoUrl"}), 400
            
        master_img = download_image(master_photo_url)
        if master_img is None:
            return jsonify({"success": False, "error": "Failed to download masterPhotoUrl"}), 400
            
        # Extract embeddings
        liveness_emb = extract_embedding(liveness_img)
        if liveness_emb is None:
            return jsonify({"success": False, "error": "No face detected in photoUrl"}), 400
            
        master_emb = extract_embedding(master_img)
        if master_emb is None:
            return jsonify({"success": False, "error": "No face detected in masterPhotoUrl"}), 400
            
        # Compute Cosine Similarity (since both are L2 normalized, dot product = cosine similarity)
        cos_sim = float(np.dot(liveness_emb, master_emb))
        
        # Convert to 0-100 scale.
        # InsightFace typical match threshold is ~0.35. We provide the raw percentage here.
        similarity_percentage = min(100.0, max(0.0, cos_sim * 100.0))
        
        logger.info(f"Similarity for {employee_id}: {similarity_percentage:.2f}% (Raw Cosine: {cos_sim:.4f})")
        
        return jsonify({
            "success": True,
            "similarity": similarity_percentage
        })
        
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting Face Verification server on port {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)
