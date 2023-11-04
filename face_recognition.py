import cv2
import os
from deepface import DeepFace
import pandas as pd
import time

# Path to the known faces
db_path = 'known_people/'

# Start webcam
cap = cv2.VideoCapture(0)

# Variables to keep track of time and labels
last_recognition_time = time.time()
label = 'detecting'

while True:
    ret, frame = cap.read()  # Read the frame from the webcam

    if not ret:
        print("Failed to grab frame")
        break

    # Calculate time elapsed
    time_elapsed = time.time() - last_recognition_time

    # Convert to grayscale for the face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Haar Cascade from OpenCV to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in the frame.")
    else:

        # Perform face recognition every 5 seconds
        if time_elapsed > 5:
            last_recognition_time = time.time()  # Reset the timer

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_area = frame[y:y+h, x:x+w]
                face_area_rgb = cv2.cvtColor(face_area, cv2.COLOR_BGR2RGB)

                try:
                    # Perform face recognition using DeepFace
                    search_result = DeepFace.find(img_path=face_area_rgb, db_path=db_path, enforce_detection=False, model_name='VGG-Face', distance_metric='cosine')
                    print(search_result)

                    if len(search_result) > 0 and isinstance(search_result[0], pd.DataFrame) and not search_result[0].empty:
                        # Get the most similar face entry
                        best_match = search_result[0]
                        identity = best_match['identity']
                        similarity_score = best_match['VGG-Face_cosine']

                        if isinstance(identity, pd.Series):
                            identity = identity.iloc[0]  # Get the first item if it's a Series

                        person_name = os.path.splitext(os.path.basename(identity))[0]

                        # Convert similarity_score to a single float value if it's a Series
                        if isinstance(similarity_score, pd.Series):
                            similarity_score = similarity_score.values[0]

                        if similarity_score < 0.20:
                            # Extract the person's name from the file path
                            person_name = os.path.splitext(os.path.basename(identity))[0]
                            label = person_name
                            print(f"Best match: {person_name} with similarity score: {similarity_score}")
                        else:
                            label = "Unknown"
                            print("No similar faces found with high confidence.")
                    else:
                        label = "Unknown"
                        print("No faces recognized from DeepFace.find()")

                except Exception as e:
                    print(f"An error occurred during recognition: {e}")

        # Draw rectangles and labels around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Put the label on the image
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows when done
cap.release()
cv2.destroyAllWindows()