# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 2018

@author: Nathan Ferguson

Write data (colour and depth) to file

For Intel Realsense D435

Resolutions available:
Color: 424x240 640x360 640x480 848x480 960x540 1280x720 1980x1080
Depth: 424x240 480x270 640x360 640x400 640x480 848x480 1280x720 1280x800

Framerates available:
Color: 6 15 30 60
Depth: 6 15 25 30 60 90
"""

fps = 15 #framerate
res_col = (640, 480) #color resolution
res_dep = res_col #depth resolution

write = True #enable write
data_path = "testdata/images" #write path
name_path = "test08"



import numpy as np
import cv2
import pyrealsense2 as pyrs
import pathlib


color_path = "./{}/{}/color".format(data_path, name_path)
depth_path = "./{}/{}/depth".format(data_path, name_path)

#Create directories
if write:
    pathlib.Path(color_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(depth_path).mkdir(parents=True, exist_ok=True)


#Set up camera
cam = pyrs.pipeline()

config = pyrs.config()
config.enable_stream(pyrs.stream.color, res_col[0], res_col[1], pyrs.format.bgr8, fps)
config.enable_stream(pyrs.stream.depth, res_dep[0], res_dep[1], pyrs.format.z16, fps)

cam.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = pyrs.stream.color
align = pyrs.align(align_to)

index = 0

while True:
    
    #Get frame
    frames = cam.wait_for_frames()
    
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    
    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue
    
    d = np.asanyarray(aligned_depth_frame.get_data())
    c = np.asanyarray(color_frame.get_data())
    
    #Write frame
    if write:
        cv2.imwrite("{}/{:04d}.png".format(color_path, index), c)
        cv2.imwrite("{}/{:04d}.png".format(depth_path, index), d)
        
        index += 1
    
    #Show images
    cv2.imshow('Colour', c)
    cv2.imshow('Depth', d * 18)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cam.stop()

