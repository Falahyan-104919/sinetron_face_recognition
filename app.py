import os

import cv2
import flask
import numpy as np
from flask import Flask, jsonify, request
from scipy.spatial.distance import cosine  # For cosine distance

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
KNOWN_EMBEDDINGS_DIR = os.path.join(BASE_DIR, "known_embeddings")

# --- Model Paths (Update filenames if you download newer versions) ---
# Detector (YuNet)
YUNET_MODEL_FILENAME = (
    "face_detection_yunet_2023mar_int8bq.onnx"  # Or your latest YuNet version
)
DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "detector", YUNET_MODEL_FILENAME)

# Recognizer (SFACE)
SFACE_MODEL_FILENAME = "face_recognition_sface_2021dec_int8bq.onnx"
RECOGNIZER_MODEL_PATH = os.path.join(MODELS_DIR, "recognizer", SFACE_MODEL_FILENAME)

# --- Thresholds and Settings ---
# YuNet Detector Parameters
YUNET_SCORE_THRESHOLD = (
    0.95  # Filter out faces with score lower than this. Adjust as needed.
)
YUNET_NMS_THRESHOLD = 0.3  # Non-Maximum Suppression threshold.
YUNET_TOP_K = 5000  # Keep top K results before NMS.

# SFACE Recognizer Parameters
RECOGNITION_THRESHOLD = 0.5  # Cosine distance threshold for matching. SFACE paper suggests 0.363 for 1% FAR. Lower is stricter.

# --- Global Model Instances ---
face_detector_yunet = None
face_recognizer_sface = None


def load_models():
    global face_detector_yunet, face_recognizer_sface
    try:
        print(f"Loading Face Detector (YuNet) from: {DETECTOR_MODEL_PATH}")
        if not os.path.exists(DETECTOR_MODEL_PATH):
            raise FileNotFoundError(
                f"Detector model not found at {DETECTOR_MODEL_PATH}"
            )

        # Input size for YuNet can be dynamic, but often a fixed size or image-derived size is used.
        # Here, we'll set it per image in the detection function.
        # We initialize with a placeholder size; it will be updated by setInputSize later.
        face_detector_yunet = cv2.FaceDetectorYN_create(
            DETECTOR_MODEL_PATH,
            "",  # Config path, usually empty for YuNet ONNX
            (320, 320),  # Initial input size (W, H), will be overridden
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
        print("Face Detector (YuNet) loaded.")

        print(f"Loading Face Recognizer (SFACE) from: {RECOGNIZER_MODEL_PATH}")
        if not os.path.exists(RECOGNIZER_MODEL_PATH):
            raise FileNotFoundError(
                f"Recognizer model not found at {RECOGNIZER_MODEL_PATH}"
            )
        face_recognizer_sface = cv2.FaceRecognizerSF_create(
            RECOGNIZER_MODEL_PATH,
            "",  # Config path, usually empty for SFACE ONNX
        )
        print("Face Recognizer (SFACE) loaded.")

    except cv2.error as e:
        print(f"OpenCV Error loading models: {e}")
        print(
            "Please ensure model files are correctly placed and OpenCV is installed correctly."
        )
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")


# --- Create directories if they don't exist ---
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)
if not os.path.exists(KNOWN_EMBEDDINGS_DIR):
    os.makedirs(KNOWN_EMBEDDINGS_DIR)


# --- Helper Functions ---
def get_aligned_face_and_embedding(image_path):
    """
    Detects faces using YuNet, aligns the best one using SFACE utility,
    and extracts its embedding using SFACE.
    Returns: (aligned_face_chip, embedding_vector) or (None, None)
    """
    if face_detector_yunet is None or face_recognizer_sface is None:
        print("Models not loaded.")
        return None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None

    height, width, _ = img.shape
    face_detector_yunet.setInputSize(
        (width, height)
    )  # Set input size for current image

    status, faces = face_detector_yunet.detect(img)

    if faces is None or len(faces) == 0:
        print("No faces detected by YuNet.")
        return None, None

    # `faces` is a NumPy array where each row is a detected face:
    # [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    # We usually pick the face with the highest score or largest area if needed.
    # For simplicity, let's take the first one (often the most prominent or highest score already by top_k).
    # You might want to add logic to select the 'best' or largest face if multiple are detected.
    best_face_data = faces[0]  # Assuming the first face is the one of interest

    # Align the detected face using SFACE's alignCrop utility
    # alignCrop expects the original image and the full face data (1x15 matrix)
    try:
        aligned_face_chip = face_recognizer_sface.alignCrop(
            img, best_face_data.reshape(1, -1)
        )
        if aligned_face_chip is None or aligned_face_chip.size == 0:
            print("Face alignment failed or resulted in an empty chip.")
            return None, None
    except cv2.error as e:
        print(f"Error during alignCrop: {e}")
        # This can happen if the face landmarks are outside the image due to aggressive detection at edges.
        # A more robust approach might involve padding or careful landmark validation.
        return None, None

    # Extract features (embedding) from the aligned face chip using SFACE
    embedding_vector = face_recognizer_sface.feature(aligned_face_chip)

    # SFACE's feature method already returns a L2-normalized embedding
    return aligned_face_chip, embedding_vector.flatten()


# --- API Endpoints ---
@app.route("/register", methods=["POST"])
def register_face():
    if face_detector_yunet is None or face_recognizer_sface is None:
        return jsonify({"error": "Models not loaded. Check server logs."}), 500

    if "face_image" not in request.files or "nip" not in request.form:
        return jsonify(
            {
                "error": "Missing face_image (form field for image file) or nip (form field for Nomor Induk Pegawai)"
            }
        ), 400

    face_image_file = request.files["face_image"]
    nip = request.form["nip"]

    if not nip:
        return jsonify({"error": "NIP cannot be empty"}), 400

    # Basic filename sanitization for NIP (can be improved)
    safe_nip = "".join(c for c in nip if c.isalnum() or c in ("-", "_")).rstrip()
    if not safe_nip:
        return jsonify({"error": "Invalid NIP format after sanitization."}), 400

    # Save uploaded file temporarily
    filename = f"temp_register_{safe_nip}_{face_image_file.filename}"
    temp_image_path = os.path.join(UPLOADS_DIR, filename)

    try:
        face_image_file.save(temp_image_path)

        _, embedding = get_aligned_face_and_embedding(
            temp_image_path
        )  # We don't need aligned_chip here

        if embedding is None:
            return jsonify(
                {"error": "Could not detect face or generate embedding from the image."}
            ), 400

        embedding_path = os.path.join(KNOWN_EMBEDDINGS_DIR, f"{safe_nip}.npy")
        np.save(embedding_path, embedding)

        return jsonify(
            {"message": f"Face registered successfully for NIP: {safe_nip}"}
        ), 200

    except Exception as e:
        print(f"Error during registration for NIP {safe_nip}: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.route("/match", methods=["POST"])
def match_face():
    if face_detector_yunet is None or face_recognizer_sface is None:
        return jsonify({"error": "Models not loaded. Check server logs."}), 500

    if "face_image" not in request.files or "nip" not in request.form:
        return jsonify(
            {
                "error": "Missing face_image (form field for image file) or nip (form field for Nomor Induk Pegawai)"
            }
        ), 400

    face_image_file = request.files["face_image"]
    nip = request.form["nip"]

    if not nip:
        return jsonify({"error": "NIP cannot be empty"}), 400

    safe_nip = "".join(c for c in nip if c.isalnum() or c in ("-", "_")).rstrip()
    if not safe_nip:
        return jsonify({"error": "Invalid NIP format after sanitization."}), 400

    registered_embedding_path = os.path.join(KNOWN_EMBEDDINGS_DIR, f"{safe_nip}.npy")
    if not os.path.exists(registered_embedding_path):
        return jsonify({"error": f"No registered face found for NIP: {safe_nip}"}), 404

    try:
        registered_embedding = np.load(registered_embedding_path)
    except Exception as e:
        print(f"Error loading registered embedding for NIP {safe_nip}: {e}")
        return jsonify(
            {
                "error": f"Could not load registered embedding for NIP: {safe_nip}. It might be corrupted."
            }
        ), 500

    filename = f"temp_match_{safe_nip}_{face_image_file.filename}"
    temp_image_path = os.path.join(UPLOADS_DIR, filename)

    try:
        face_image_file.save(temp_image_path)
        _, current_embedding = get_aligned_face_and_embedding(temp_image_path)

        if current_embedding is None:
            return jsonify(
                {
                    "error": "Could not detect face or generate embedding from the provided image for matching."
                }
            ), 400

        # Calculate Cosine Distance (lower means more similar)
        # SFACE embeddings are L2 normalized, so cosine similarity = dot product
        # Cosine distance = 1 - cosine_similarity
        # Alternatively, use scipy.spatial.distance.cosine

        # distance = cosine(registered_embedding, current_embedding)
        # Or using match method if available and preferred (match gives score, often similarity)
        # Score from SFACE match is cosine similarity. Higher is better.
        # score = face_recognizer_sface.match(registered_embedding.reshape(1,-1), current_embedding.reshape(1,-1), cv2.FaceRecognizerSF_FR_COSINE)
        # If using score (similarity), thresholding condition would be score >= SIMILARITY_THRESHOLD

        # Let's stick to cosine distance for consistency with the RECOGNITION_THRESHOLD definition (lower is stricter)
        distance = cosine(registered_embedding, current_embedding)

        print(
            f"NIP: {safe_nip}, Calculated Cosine Distance: {distance:.4f}, Threshold: {RECOGNITION_THRESHOLD}"
        )

        if distance <= RECOGNITION_THRESHOLD:
            return jsonify(
                {"match": True, "nip": safe_nip, "distance": float(distance)}
            ), 200
        else:
            return jsonify(
                {"match": False, "nip": safe_nip, "distance": float(distance)}
            ), 200

    except Exception as e:
        print(f"Error during matching for NIP {safe_nip}: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


if __name__ == "__main__":
    load_models()  # Load models when the application starts
    if face_detector_yunet is None or face_recognizer_sface is None:
        print(
            "CRITICAL: Application cannot start due to model loading issues. Please check logs and model paths."
        )
    else:
        print("Flask server starting...")
        # For deployment, use a production WSGI server like Gunicorn:
        # gunicorn -w 4 -b 0.0.0.0:5000 app:app
        app.run(
            host="0.0.0.0", port=20253, debug=True
        )  # debug=True for development, False for production
