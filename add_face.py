import face_recognition
import os
import pickle
import cv2

DB_FILE = "database.pkl"

def add_face(name, image_path):
    # Load image
    img = face_recognition.load_image_file(image_path)
    enc = face_recognition.face_encodings(img)

    if len(enc) == 0:
        print("❌ No face detected!")
        return

    enc = enc[0]

    # Load existing DB
    if os.path.exists(DB_FILE):
        db = pickle.load(open(DB_FILE, "rb"))
    else:
        db = {}

    db[name] = enc

    pickle.dump(db, open(DB_FILE, "wb"))
    print(f"✅ Face added: {name}")

if __name__ == "__main__":
    name = input("Enter name: ")
    image_path = input("Enter image file path: ")
    add_face(name, image_path)
