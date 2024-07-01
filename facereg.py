import face_recognition
import pickle
import os

known_face_encodings = []
known_face_names = []

# Loop through each file in the known_faces directory
for filename in os.listdir('known_faces'):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = face_recognition.load_image_file(f'known_faces/{filename}')
        
        # Encode the face
        encodings = face_recognition.face_encodings(image)
        
        # Ensure that the image has exactly one face
        if len(encodings) == 1:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            
            # Use the filename (without extension) as the person's name
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"Image {filename} does not contain exactly one face and will be skipped.")

# Save the encodings and names
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Known faces and encodings saved successfully.")
