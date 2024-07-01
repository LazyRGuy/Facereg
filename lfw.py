import os
import face_recognition
import pickle

known_face_encodings = []
known_face_names = []
i = 0
for person_name in os.listdir("lfw"):
    i +=1
    person_path = os.path.join("lfw", person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 1:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)
    print(person_name,i)
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("LFW face encodings saved successfully.")
