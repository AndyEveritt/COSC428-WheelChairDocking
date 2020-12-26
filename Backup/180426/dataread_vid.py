# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 2018

@author: Andy Everitt

Reads data from color and depth directories

"""

data_path = "testdata" #read path
name_path = "test06"

import cv2

# Load the video file.
cap_color = cv2.VideoCapture('./{}/{}/color.avi'.format(data_path, name_path))
cap_depth = cv2.VideoCapture('./{}/{}/depth.avi'.format(data_path, name_path))

while True:  # Loop over the images until we decide to quit.
    ret_color, c = cap_color.read()  # Read an frame from the color video file.
    ret_depth, d = cap_depth.read()  # Read an frame from the depth video file.

    # If we cannot read any more frames from the video file, then reload video.
    if not (ret_color or ret_depth):
        cap_color.release()
        cap_depth.release()
        # Load the video file.
        cap_color = cv2.VideoCapture('./{}/{}/color.avi'.format(data_path, name_path))
        cap_depth = cv2.VideoCapture('./{}/{}/depth.avi'.format(data_path, name_path))

    if (ret_color and ret_depth):
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale.
        cv2.imshow('Colour', c)
        cv2.imshow('Depth', d * 18)  # Scale the depth data for visibility
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_color.release()
cap_depth.release()
cv2.destroyAllWindows()