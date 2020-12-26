# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 2018

@author: Andy Everitt

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
#data_path = "testdata" #write path
data_path = "tmp" #write path
name_path = "test00"



import numpy as np
import cv2
import pyrealsense2 as pyrs
import pathlib


file_path = "./{}/{}".format(data_path, name_path)

#Create directories
if write:
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
color_out = cv2.VideoWriter('./{}/{}/color.avi'.format(data_path, name_path), fourcc, 15.0, (640, 480))
depth_out = cv2.VideoWriter('./{}/{}/depth.avi'.format(data_path, name_path), fourcc, 15.0, (640, 480))

#Set up camera
cam = pyrs.pipeline()

config = pyrs.config()
config.enable_stream(pyrs.stream.color, res_col[0], res_col[1], pyrs.format.bgr8, fps)
config.enable_stream(pyrs.stream.depth, res_dep[0], res_dep[1], pyrs.format.z16, fps)

cam.start(config)


index = 0

while True:
    
    #Get frame
    frames = cam.wait_for_frames()
    
    c = frames.get_color_frame()
    d = frames.get_depth_frame()
    if not c or not d:
        continue
    
    c = np.asanyarray(c.get_data()) 
    d = np.asanyarray(d.get_data())
    d_vid = np.dstack((d,d,d))
    d_vid = (d_vid/256).astype('uint8')
    
    
    #Write frame
    if write:
        color_out.write(c)
        depth_out.write(d_vid)
        
        index += 1
    
    #Show images
    cv2.imshow('Colour', c)
    cv2.imshow('Depth', d * 18)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cam.stop()

