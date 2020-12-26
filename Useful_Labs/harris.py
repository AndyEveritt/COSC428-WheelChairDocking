# harris.py

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Open the first camera connected to the computer.

while True:
    ret, frame = cap.read()
    # The Harris corner detector operates on a grayscale image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(gray,2,3,0.04)
    # Dialate the detected corners to make them clearer in the output image.
    corners = cv2.dilate(corners,None)

    # Perform thresholding on the corners to throw away some false positives.
    frame[corners > 0.1 * corners.max()] = [0,0,255]

    cv2.imshow("Harris", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):  # Close the script when q is pressed.
        break

cap.release()
cv2.destroyAllWindows()
