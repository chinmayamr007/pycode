import os
import sys
import cv2
import numpy as np
import face_recognition
import pickle
import csv
from datetime import datetime

# Fix the model path issue for PyInstaller EXE
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # Running from PyInstaller EXE
else:
    base_path = os.path.dirname(__file__)

# Paths
encodings_file = "D:/final_project/encodings.pickle"
csv_file = "D:/final_project/attendance.csv"

# Load known face encodings and names
if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        known_data = pickle.load(f)
        known_encodings = known_data["encodings"]
        known_names = known_data["names"]
    print(f"✅ Loaded {len(known_names)} known faces from encodings.pickle")
else:
    print("❌ Encodings file not found!")
    known_encodings = []
    known_names = []

# Ensure CSV file exists
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using HOG model for better CPU performance
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"🔍 Found {len(face_encodings)} faces in the frame")

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Use a higher tolerance value
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        print(f"🎭 Detected: {name}")

        # Display name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mark attendance (avoid duplicate consecutive entries)
        try:
            with open(csv_file, "r+", newline="") as f:
                reader = list(csv.reader(f))
                if len(reader) <= 1 or reader[-1][0] != name:
                    writer = csv.writer(f)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([name, now])
                    print(f"📌 Marking attendance: {name} at {now}")

        except PermissionError:
            print(f"❌ ERROR: Could not write to CSV - Permission denied: {csv_file}")

    # Show webcam feed
    cv2.imshow("Face Attendance", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
