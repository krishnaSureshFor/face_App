import streamlit as st
import cv2
import numpy as np
from utils.face_utils import extract_embedding, load_db

st.title("👤 Face Recognition Greeting App")

names, embeds = load_db()

if len(names) == 0:
    st.warning("No faces in database. Add using add_face.py")
else:
    st.success("Loaded persons: " + ", ".join(names))

choice = st.radio("Input:", ["Upload Image", "Camera"])

img = None

if choice == "Upload Image":
    file = st.file_uploader("Upload", type=["jpg","jpeg","png"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

elif choice == "Camera":
    cam = st.camera_input("Take picture")
    if cam:
        img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), 1)

if img is not None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    emb = extract_embedding(rgb)

    if emb is None:
        st.error("No face detected")
    else:
        dists = [np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e)) for e in embeds]
        best = np.argmax(dists)

        if dists[best] > 0.80:
            st.success(f"👋 Hello **{names[best]}**!")
        else:
            st.error("Unknown person")

    st.image(rgb)
