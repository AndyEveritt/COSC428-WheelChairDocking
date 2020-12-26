# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 2018

@author: Nathan Ferguson

Reads data from color and depth directories

"""

fps = 15 #playback framerate
data_path = "testdata/images" #read path
name_path = "test00"



import cv2
import os


color_path = "./{}/{}/color".format(data_path, name_path)
depth_path = "./{}/{}/depth".format(data_path, name_path)

color_images = os.listdir(color_path)
color_images.sort()
depth_images = os.listdir(depth_path)
depth_images.sort()


delay = 1000 // fps #delay between frames in ms
img_index = 0

while True:  # Loop over the images until we decide to quit.
    
    color_filename = color_images[img_index]
    depth_filename = depth_images[img_index]
    if not color_filename.endswith(".png") or not depth_filename.endswith(".png"):
        continue
    
    # Note that if the cv2.IMREAD_ANYDEPTH flag is missing, OpenCV will load the 16-bit depth data as an 8-bit image.
    c = cv2.imread("{}/{}".format(color_path, color_filename))
    d = cv2.imread("{}/{}".format(depth_path, depth_filename), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    cv2.imshow('Colour', c)
    cv2.imshow('Depth', d * 18)  # Scale the depth data for visibility
    
    
    img_index = (img_index + 1) % len(color_images)
    
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()