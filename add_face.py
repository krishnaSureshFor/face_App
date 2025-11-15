import cv2
from utils.face_utils import extract_embedding, load_db, save_db

name = input("Enter name: ")
path = input("Enter image path: ")

img = cv2.imread(path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

emb = extract_embedding(img_rgb)

if emb is None:
    print("❌ No face found")
else:
    names, embeds = load_db()
    names.append(name)
    embeds.append(emb)

    save_db(names, embeds)
    print("✅ Face added:", name)
