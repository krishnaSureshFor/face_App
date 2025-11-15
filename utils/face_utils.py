import face_recognition
import pickle
import os

DB_FILE = "database.pkl"

def load_database():
    if os.path.exists(DB_FILE):
        return pickle.load(open(DB_FILE, "rb"))
    return {}

def save_face(name, encoding):
    db = load_database()
    db[name] = encoding
    pickle.dump(db, open(DB_FILE, "wb"))

def encode_face(image):
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None
