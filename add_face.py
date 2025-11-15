# add_face.py
import cv2
import sys
from utils.face_utils import extract_embedding, load_db, save_db

def main():
    name = input("Enter name to enroll: ").strip()
    if not name:
        print("Name is required.")
        return

    path = input("Enter path to image file (jpg/png): ").strip()
    img = cv2.imread(path)
    if img is None:
        print("Could not read image at", path)
        return

    emb = extract_embedding(img)
    if emb is None:
        print("No face detected in the image.")
        return

    names, embeds = load_db()
    names.append(name)
    embeds.append(emb)
    save_db(names, embeds)
    print(f"✅ Enrolled '{name}' successfully. DB saved to database.npz")

if __name__ == "__main__":
    main()
