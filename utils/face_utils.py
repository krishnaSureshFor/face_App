import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

DB_FILE = "database.npz"

def extract_embedding(image):
    result = mp_face.process(image)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0]
    pts = []
    for lm in landmarks.landmark:
        pts.append([lm.x, lm.y, lm.z])
    return np.array(pts).flatten()


def load_db():
    try:
        data = np.load(DB_FILE, allow_pickle=True)
        return data["names"].tolist(), data["embeds"].tolist()
    except:
        return [], []


def save_db(names, embeds):
    np.savez(DB_FILE, names=names, embeds=embeds)
