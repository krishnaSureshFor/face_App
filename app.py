import streamlit as st
import face_recognition
import numpy as np
from utils.face_utils import load_database, encode_face

st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("👤 Face Recognition Greeting App")

db = load_database()

st.write("### Known People:")
if len(db) == 0:
    st.warning("No faces added yet!")
else:
    st.success(", ".join(db.keys()))

option = st.radio("Choose input source:", ["Upload Image", "Camera"])

image = None

if option == "Upload Image":
    upload = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if upload:
        image = face_recognition.load_image_file(upload)
        st.image(upload, caption="Uploaded Image")

elif option == "Camera":
    cam = st.camera_input("Take a picture")
    if cam:
        image = face_recognition.load_image_file(cam)
        st.image(cam, caption="Camera Capture")

if image is not None:
    st.write("Processing...")

    enc = encode_face(image)

    if enc is None:
        st.error("No face detected!")
    else:
        names = list(db.keys())
        encodings = list(db.values())

        distances = face_recognition.face_distance(encodings, enc)
        best_match = np.argmin(distances)

        if distances[best_match] < 0.45:
            person = names[best_match]
            st.success(f"👋 Hello **{person}**!")
        else:
            st.error("Unknown Person")
