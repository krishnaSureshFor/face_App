# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.face_utils import extract_embedding, load_db, save_db, best_match

st.set_page_config("Face Greet", layout="centered")
st.title("👋 Face Greeting App — Single Face")

st.sidebar.header("Mode")
mode = st.sidebar.radio("Select mode", ["Recognize", "Enroll"])

# Load DB
names, embeds = load_db()

st.sidebar.markdown("**Known people:**")
if len(names) == 0:
    st.sidebar.warning("No enrolled faces. Use 'Enroll' or run add_face.py locally.")
else:
    st.sidebar.info(", ".join(names))

def read_uploaded_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

if mode == "Enroll":
    st.header("Enroll a new person")
    st.write("You can upload an image or use your webcam. For persistence, run `add_face.py` locally and commit `database.npz` to repo.")
    name = st.text_input("Name to enroll")
    col1, col2 = st.columns(2)
    upload = col1.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
    cam_img = None
    with col2:
        cam = st.camera_input("Or take a picture")
        if cam:
            cam_img = cam

    if st.button("Enroll"):
        if not name:
            st.error("Please enter a name")
        else:
            img_bgr = None
            if upload:
                img_bgr = read_uploaded_image(upload)
            elif cam_img:
                img_bgr = read_uploaded_image(cam_img)
            else:
                st.error("Please upload or capture an image.")
            if img_bgr is not None:
                emb = extract_embedding(img_bgr)
                if emb is None:
                    st.error("No face detected. Try a clearer photo or larger face.")
                else:
                    names, embeds = load_db()
                    names.append(name)
                    embeds.append(emb)
                    save_db(names, embeds)
                    st.success(f"✅ Enrolled {name}")
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Enrolled image", use_column_width=True)

elif mode == "Recognize":
    st.header("Recognize and Greet")
    st.write("Upload an image or use your webcam. App will detect a single face and greet the matched name.")
    col1, col2 = st.columns(2)
    upload = col1.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
    cam_img = None
    with col2:
        cam = st.camera_input("Or take a picture")
        if cam:
            cam_img = cam

    if upload or cam_img:
        if upload:
            img_bgr = read_uploaded_image(upload)
        else:
            img_bgr = read_uploaded_image(cam_img)

        emb = extract_embedding(img_bgr)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
        if emb is None:
            st.error("No face detected.")
        else:
            names, embeds = load_db()
            if len(names) == 0:
                st.warning("No enrolled faces. Switch to 'Enroll' or run add_face.py locally.")
            else:
                name, score = best_match(emb, embeds, names, threshold=0.70)
                if name is not None:
                    st.success(f"👋 Hello **{name}**!  (score={score:.2f})")
                else:
                    st.info(f"Unknown person. Best score: {score:.2f}" if score is not None else "No match.")
