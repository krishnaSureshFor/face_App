import cv2
import face_recognition
import pickle
import numpy as np
import time

DB_FILE = "database.pkl"

# Load database
db = pickle.load(open(DB_FILE, "rb"))
known_names = list(db.keys())
known_encodings = list(db.values())

print("Loaded faces:", known_names)

video = cv2.VideoCapture(0)
last_greet = {}

while True:
    ret, frame = video.read()
    rgb_frame = frame[:, :, ::-1]

    # Detect faces
    faces = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, faces)

    for (top, right, bottom, left), face_enc in zip(faces, encodings):

        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.45)
        distances = face_recognition.face_distance(known_encodings, face_enc)
        best_match = np.argmin(distances)

        name = "Unknown"
        if matches[best_match]:
            name = known_names[best_match]

            # Greet once every 5 seconds
            if name not in last_greet or time.time() - last_greet[name] > 5:
                print(f"👋 Hello {name}!")
                last_greet[name] = time.time()

        # Draw box & name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition App", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
