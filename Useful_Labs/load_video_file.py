# load_video_file.py

import cv2

# Load the video file.
cap = cv2.VideoCapture('../output.avi')

while cap.isOpened():
    ret, frame = cap.read()  # Read an frame from the video file.

    # If we cannot read any more frames from the video file, then exit.
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale.
    cv2.imshow('frame', frame)  # Display the grayscale frame on the screen.

    # Close the script if q is pressed.
    # Note that the delay in cv2.waitKey affects how quickly the video will play on screen.
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# Release the video file, and close the GUI.
cap.release()
cv2.destroyAllWindows()
