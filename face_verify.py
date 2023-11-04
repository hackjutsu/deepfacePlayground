from deepface import DeepFace
import pandas as pd
import os

def find_best_match(input_image_path, db_path):
    try:
        # Search for the face in the database of known faces
        search_result = DeepFace.find(img_path=input_image_path, db_path=db_path, enforce_detection=False, model_name='VGG-Face', distance_metric='cosine')

        if len(search_result) > 0 and isinstance(search_result[0], pd.DataFrame):
            # Get the most similar face entry
            best_match = search_result[0]
            identity = best_match['identity']
            similarity_score = best_match['VGG-Face_cosine']

            print(best_match)

            if isinstance(identity, pd.Series):
                identity = identity.iloc[0]  # Get the first item if it's a Series

            person_name = os.path.splitext(os.path.basename(identity))[0]

            # Convert similarity_score to a single float value if it's a Series
            if isinstance(similarity_score, pd.Series):
                similarity_score = similarity_score.values[0]

            # You might want to adjust the threshold for similarity score as needed
            if similarity_score < 0.40:
                # Extract the person's name from the file path
                person_name = os.path.splitext(os.path.basename(identity))[0]
                print(f"Best match: {person_name} with similarity score: {similarity_score}")
            else:
                print("No similar faces found with high confidence.")
        else:
            print("No matching faces found in the database.")
    except Exception as e:
        print(f"An error occurred: {e}")

def stream_from(db_path):
    DeepFace.stream(db_path)

# Set the paths accordingly
input_image_path = 'test.jpg'
db_path = 'known_people/'

# Call the function to find the best match
find_best_match(input_image_path, db_path)
# result = DeepFace.verify(img1_path = "test.jpg", img2_path = "known_people/image1.jpg")
# print(result)

# stream_from(db_path)