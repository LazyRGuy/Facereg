from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
import pickle
import dlib
import os

app = Flask(__name__)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Load the known faces and encodings
with open('face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

def align_face_dlib(image, rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    landmarks = [(p.x, p.y) for p in shape.parts()]

    left_eye_pts = np.array(landmarks[36:42])
    right_eye_pts = np.array(landmarks[42:48])

    left_eye_center = np.mean(left_eye_pts, axis=0)
    right_eye_center = np.mean(right_eye_pts, axis=0)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    eye_dist = np.sqrt((dx**2) + (dy**2))
    desired_eye_dist = 0.3 * 256
    scale = desired_eye_dist / eye_dist
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    M[0, 2] += 256 * 0.5 - eyes_center[0]
    M[1, 2] += 256 * 0.5 - eyes_center[1]

    aligned_face = cv2.warpAffine(image, M, (256, 256), flags=cv2.INTER_CUBIC)
    return aligned_face

def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        for landmarks, location in zip(face_landmarks_list, face_locations):
            top, right, bottom, left = location
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            rect = dlib.rectangle(left, top, right, bottom)

            aligned_face = align_face_dlib(frame, rect)

            # Save aligned face image
            # cv2.imwrite(f'aligned_faces/face_{np.random.randint(1e6)}.jpg', aligned_face)

            rgb_aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_aligned_face)

            if len(face_encoding) > 0:
                face_encoding = face_encoding[0]
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    name = known_face_names[best_match_index]
                    if distances[best_match_index] < 0.5:
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return render_template('add_face.html', message='Name cannot be empty')
        
        # Create a directory for the new person's face images
        if not os.path.exists(f'known_faces/{name}'):
            os.makedirs(f'known_faces/{name}')
        
        # Add the new face
        capture_new_face(name)
        return redirect(url_for('result', name=name))
    
    return render_template('add_face.html')

def capture_new_face(name):
    cap = cv2.VideoCapture(0)
    count = 0
    num_images = 10

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(f'known_faces/{name}/face_{count}.jpg', face_image)
            count += 1
            if count >= num_images:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Add the new face encodings
    add_new_face(name)

def add_new_face(name):
    image_dir = f'known_faces/{name}'
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                face_encoding = face_encoding[0]  # Assumes one face per image
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
    with open("face_encodings.pkl", 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

@app.route('/result/<name>')
def result(name):
    return render_template('result.html', name=name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
