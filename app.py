from flask import Flask, request, jsonify
import face_recognition
import os
import numpy as np

app = Flask(__name__)

KNOWN_FACES_DIR = 'known_faces'
UPLOADS_DIR = 'uploads'
TOLERANCE = 0.6  # Lower tolerance means stricter matching

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def get_known_encodings(nip):
    """Loads known face encodings for a given NIP."""
    nip_dir = os.path.join(KNOWN_FACES_DIR, nip)
    if not os.path.exists(nip_dir):
        return []

    encodings = []
    for filename in os.listdir(nip_dir):
        image_path = os.path.join(nip_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            encodings.append(encoding[0])
    return encodings

@app.route('/register', methods=['POST'])
def register_face():
    """Registers a user's face with their NIP."""
    if 'face_image' not in request.files or 'nip' not in request.form:
        return jsonify({'error': 'Missing face image or NIP'}), 400

    face_image = request.files['face_image']
    nip = request.form['nip']

    nip_dir = os.path.join(KNOWN_FACES_DIR, nip)
    if not os.path.exists(nip_dir):
        os.makedirs(nip_dir)

    image_path = os.path.join(nip_dir, f"{nip}.jpg")
    face_image.save(image_path)

    return jsonify({'message': f'Face registered for NIP: {nip}'}), 200

@app.route('/match', methods=['POST'])
def match_face():
    """Matches a user's face with their registered face using NIP."""
    if 'face_image' not in request.files or 'nip' not in request.form:
        return jsonify({'error': 'Missing face image or NIP'}), 400

    face_image = request.files['face_image']
    nip = request.form['nip']

    known_encodings = get_known_encodings(nip)

    if not known_encodings:
        return jsonify({'error': f'No registered face found for NIP: {nip}'}), 404

    image_path = os.path.join(UPLOADS_DIR, f"temp_{nip}.jpg")
    face_image.save(image_path)

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    os.remove(image_path)  # Clean up the uploaded image

    if not unknown_encodings:
        return jsonify({'error': 'No face detected in the provided image'}), 400

    unknown_encoding = unknown_encodings[0]
    results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=TOLERANCE)

    if True in results:
        return jsonify({'match': True}), 200
    else:
        return jsonify({'match': False}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20253, debug=True)
