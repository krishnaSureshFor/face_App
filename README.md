# Face Greeting — Streamlit (single-face)

How it works:
- Detects single face using OpenCV Haar cascade.
- Embedding: 64x64 grayscale flattened and normalized.
- Recognition: cosine similarity.

Install locally:
pip install -r requirements.txt

To enroll faces that persist across deploys:
1. Run `python add_face.py` locally and add faces with local images.
2. Commit the generated `database.npz` to your repository.
3. Push to GitHub and deploy on Streamlit Cloud.

You can also enroll from the web UI, but files saved by a deployed app may not persist across redeploys or restarts.
