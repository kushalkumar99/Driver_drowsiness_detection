from flask import Flask, render_template, Response
#Flask: Web framework for Python to create web applications.
#render_template: Renders an HTML template.
#Response: Constructs a response object for the web application.
from scipy.spatial import distance as dist
#distance as dist: Imports distance functions.
from imutils import face_utils
#face_utils: Utility functions for facial landmarks detection.
from threading import Thread
#Thread: Allows running code in parallel using threading.
import imutils # A library to simplify OpenCV functionality.
import time # Standard Python library for time-related functions.
import dlib # A toolkit containing machine learning algorithms and tools, including facial landmark detection.
import cv2 # OpenCV library for computer vision tasks.
import winsound # Library for sound playing functionality on Windows.

app = Flask(__name__) # Initializes the Flask application.

def sound_alarm():
    # Function to play a beep sound at 2500 Hz for 2000 ms.
    winsound.Beep(2500, 2000)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])#vertical distance
    B = dist.euclidean(eye[2], eye[4])#vertical distance
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])#horizontal distance
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

detector = dlib.get_frontal_face_detector() # Initializes the frontal face detector.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Loads the facial landmark predictor.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #Indices for left and right eye landmarks.

def generate_frames():
    global COUNTER, ALARM_ON #Declares these variables as global, so the function can modify the shared state of COUNTER and ALARM_ON.
    vs = cv2.VideoCapture(0) #Opens the default webcam (index 0) for video capture.
    time.sleep(1.0) #Pauses for 1 second to allow the camera to warm up.

    while True: #Starts an infinite loop to continuously capture frames from the webcam.
        ret, frame = vs.read() #Reads a frame from the webcam. ret is a boolean indicating if the frame was successfully read, and frame contains the image.
        if not ret:
            break
        frame = imutils.resize(frame, width=450)# Resizes the frame to a width of 450 pixels while maintaining aspect ratio.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Converts the frame to grayscale, which is necessary for the dlib face detector.
        rects = detector(gray, 0)#Detects faces in the grayscale frame. rects is a list of rectangles where faces were detected.

        if rects:
            rect = rects[0]  # Only process the first detected face
            shape = predictor(gray, rect) #Predicts facial landmarks for the detected face.
            shape = face_utils.shape_to_np(shape)# Converts the facial landmarks to  a NumPy array.

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            #Extracts the coordinates of the left and right eyes using the indices.
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            #Calculates the eye aspect ratio for both eyes.
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            #Computes the convex hull of the eye landmarks, which is the smallest polygon that can contain all the points.

            # Calculate the bounding box of the eyes region
            (x1, y1, w1, h1) = cv2.boundingRect(leftEyeHull)
            (x2, y2, w2, h2) = cv2.boundingRect(rightEyeHull)

            # Determine the bounding box that covers both eyes
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

            # Add some margin around the eyes
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)

            # Crop the frame to the eyes region
            eye_frame = frame[y:y+h, x:x+w].copy()

            # Draw contours around the eyes in the cropped frame
            cv2.drawContours(eye_frame, [leftEyeHull - [x, y]], -1, (0, 255, 0), 1)
            cv2.drawContours(eye_frame, [rightEyeHull - [x, y]], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH: #Checks if the EAR is below the threshold, indicating closed eyes.
                COUNTER += 1 #Increments the counter if EAR is below the threshold.
                if COUNTER >= EYE_AR_CONSEC_FRAMES: #Checks if the counter has reached the consecutive frame threshold.
                    if not ALARM_ON:
                        ALARM_ON = True #If the alarm is not already on, it starts the alarm in a separate thread.
                        t = Thread(target=sound_alarm)
                        t.daemon = True
                        t.start()

            else:
                COUNTER = 0 #Resets the counter and turns off the alarm if EAR is above the threshold.
                ALARM_ON = False

        else:
            eye_frame = frame #If no face is detected, the entire frame is used.

        ret, buffer = cv2.imencode('.jpg', eye_frame)#Encodes the frame as a JPEG image.
        eye_frame = buffer.tobytes() #Converts the encoded image to bytes.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + eye_frame + b'\r\n') #Streams the frame as a part of an HTTP response, with appropriate MIME type and boundary for multipart data.

    vs.release()#Releases the webcam resource.

@app.route('/')#This is a decorator in Flask that associates the URL / (the root URL) with the index function. 
def index(): #This defines the index function that will be executed when the root URL is accessed.
    return render_template('index.html')
    #This renders and returns the index.html template. Flask looks for the index.html file in the templates directory of the project.
@app.route('/video_feed')#This decorator associates the URL /video_feed with the video_feed function.
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #This constructs a response object with the content generated by the generate_frames function.
if __name__ == "__main__":
    app.run(debug=True)
