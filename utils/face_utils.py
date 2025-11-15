# utils/face_utils.py
import cv2
import numpy as np
import os

DB_FILE = "database.npz"

# Load Haar cascade shipped with OpenCV
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


def extract_embedding(bgr_image, size=(64, 64)):
    """
    Input: BGR image (as read by cv2)
    Output: 1D numpy vector (L2-normalized) or None if no face found.
    """
    if bgr_image is None:
        return None

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # detectMultiScale params can be tuned
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Choose the largest detected face (single-face app)
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    face = gray[y:y + h, x:x + w]

    # Resize to fixed size, flatten and normalize
    face_resized = cv2.resize(face, size)
    vec = face_resized.astype(np.float32).flatten() / 255.0

    # L2-normalize
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    vec = vec / norm
    return vec


def load_db():
    """
    Returns (names_list, embeds_list). If no DB, returns ([], []).
    """
    if not os.path.exists(DB_FILE):
        return [], []

    try:
        data = np.load(DB_FILE, allow_pickle=True)
        names = data["names"].tolist()
        embeds = data["embeds"].tolist()
        # ensure each embedding is np.array
        embeds = [np.array(e, dtype=np.float32) for e in embeds]
        return names, embeds
    except Exception:
        return [], []


def save_db(names, embeds):
    """
    names: list of strings
    embeds: list of numpy arrays (vectors)
    """
    # convert to arrays that can be saved
    arr_names = np.array(names, dtype=object)
    arr_embeds = np.array([e.tolist() for e in embeds], dtype=object)
    np.savez(DB_FILE, names=arr_names, embeds=arr_embeds)


def best_match(embedding, embeds, names, threshold=0.70):
    """
    embedding: single normalized vector
    embeds: list of normalized vectors
    names: list of names
    returns (best_name, score) or (None, None) if no match
    """
    if embedding is None or len(embeds) == 0:
        return None, None

    # compute cosine similarity (since vectors are normalized, dot is cosine)
    sims = [float(np.dot(embedding, e)) for e in embeds]
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]
    if best_score >= threshold:
        return names[best_idx], best_score
    return None, best_score
