from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import os

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Face Recognititon Intitalization

path = 'Images' # Folder where known faces are stored
images = []
known_face_names = []
myList = os.listdir(path)

# Create array of known face names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    known_face_names.append(os.path.splitext(cl)[0].upper())

str_known_face_names = ", ".join(known_face_names) # To return all face names to html page

# Create array of known face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

known_face_encodings = findEncodings(images)

# Get a reference to webcam #0 (the default one)
cam = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []


# Object Detection Inititalization

thres = 0.5 # Threshold to detect object

# Setup
camera = cam
camera.set(3,640)
camera.set(4,480)
camera.set(10,70)
 
classNames= []
classFile = 'coco.names' # Microsoft's object detection dataset
with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f]

# Requirements for Common Objects in Context dataset to run
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Setup 
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
objectNames = []


# Face Recognititon

def face_rec():  
    success, frame = cam.read()  # read the cam frame

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # If there is a face match
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name.upper(), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/facenames')
def tempA():
    return render_template('faceRecognition.html')

# Will contain the face recognition image
@app.route('/face_feed')
def face_feed():
    return Response(face_rec(), mimetype='multipart/x-mixed-replace; boundary=frame')

# To return recognized face names
@app.route('/face_recog')
def face_rec_names():  
    str_names = "NONE"
    success, frame = cam.read()  # Read the cam frame
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name.upper())
        str_names = ", ".join(face_names) # List to String to return all the matched face names

    return render_template('faceRecognition.html', value=str_names, known = str_known_face_names)

# Object Detection

def obj_det():  
    success, frame = camera.read()  # read the camera frame
    classIds, confs, bbox = net.detect(frame,confThreshold=thres)

    # If objects are detected
    if len(classIds) != 0:
        for classId, confidence ,box in zip(classIds.flatten(),confs.flatten(),bbox):
            # Draw a box around the object 
            cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
            # Draw a label with a name and confidence of the object
            cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
    
    # If there are no objects detected return just the captured image
    else:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()        

    yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/objnames')
def tempB():
    return render_template('objectDetection.html')

# Will contain the object detection image
@app.route('/obj_feed')
def obj_feed():
    return Response(obj_det(), mimetype='multipart/x-mixed-replace; boundary=frame')

# To return recognized object names
@app.route('/obj_detect')
def obj_det_names():  
    str_obj_names = "NONE"
    success, frame = camera.read()  # read the camera frame
    classIds, confs, bbox = net.detect(frame,confThreshold=thres)
    objectNames = []

    # If objects are detected
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            # Put all the objects detected names in a list
            objectNames.append(classNames[classId-1].upper())
        str_obj_names = ", ".join(objectNames) # List to String to return all the recognised object names
    return render_template('objectDetection.html', value=str_obj_names)

if __name__=='__main__':
    app.run(debug=True)