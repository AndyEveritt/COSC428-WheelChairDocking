# face_detection.py

import numpy as np
import cv2

# Load the pre-trained classifiers. These can be found in opencv/data/haarcascades
# but are also inside the lab directory for your convenience.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)  # Open the webcam device.

while True:
    ret, img = cap.read()  # Read an frame from the webcam.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection.

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # Draw a blue box around each face.
        # Define a "Region of Interest" for the eye detector, since eyes are typically found on a face.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # For each face, detect eyes within the region of interest.
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  # Draw a green box around each eye.
#        smile = smile_cascade.detectMultiScale(roi_gray)
#        for (ex,ey,ew,eh) in smile:
#            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    cv2.imshow('img',img)

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
