## Real-Time Face Recognition with Flask
This project is a real-time face recognition web application built using Flask and Python. It captures video from a webcam, performs face recognition, and displays the results on a local web page. Users can also add new faces to the system, which will be used for future recognition. face_recognition is [reported](https://github.com/ageitgey/face_recognition) to have an accuracy of 99.38% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset which I am using here.
The misaligned frames captured are aligned for added accuracy using dlib and for faster performance image resizing is done.
## Libraries Used
* **Flask**: A lightweight WSGI web application framework for Python.
* **OpenCV**: A library for computer vision tasks.
* **face_recognition:** A Python library for face recognition built on top of dlib.
* **pickle**: Python's built-in module for object serialization.

## Usage
To start the Flask web application, execute:
```python
python app.py
```
Open your web browser and navigate to the webpage mentioned in terminal like http://127.0.0.1:5000.



