# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:50:25 2018

@author: Andy Everitt
"""

import numpy as np
import cv2
import pyrealsense2 as pyrs

###############################################################################
#   Global variables
###############################################################################

"""
Video source
"""
live = False # enables live video feed
data_path = "testdata" #read path
name_path = "stairs.bag"

fps = 15 #framerate

resolution = [(1280, 720),(848, 480),(640, 480)]
res_col = resolution[2] #color resolution
res_dep = resolution[2] #depth resolution


"""
Set up camera
"""
cam = pyrs.pipeline()
config = pyrs.config()

# Set stream path. If live == True then use camera, else use .bag file
if live:
    config.enable_stream(pyrs.stream.color, res_col[0], res_col[1], pyrs.format.bgr8, fps)
    config.enable_stream(pyrs.stream.depth, res_dep[0], res_dep[1], pyrs.format.z16, fps)
    

else:
    path_to_bag = "./{}/{}".format(data_path, name_path)
    config.enable_device_from_file(path_to_bag)
    
    
cam.start(config)

"""
Create an align object
rs.align allows us to perform alignment of depth frames to others frames
The "align_to" is the stream type to which we plan to align depth frames.
"""
align_to = pyrs.stream.color
align = pyrs.align(align_to)


while True:  # Loop over the images until we decide to quit.
    
###############################################################################  
#   Obtaining frame and image data
###############################################################################    
    
    """
    Get frameset of color and depth
    """
    frames = cam.wait_for_frames()
        
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    
    # Validate that both frames are valid
    if not depth_frame or not color_frame:
        continue
    
    """
    Unnecessary for this demo but useful if you want to do analysis on the image
    """
    d = np.asanyarray(depth_frame.get_data())
    c = np.asanyarray(color_frame.get_data())
        
    d_scaled = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    
    
    """
    Calculate 3D world distances
    """
    pixel1 = [10,10]    # the pixel of interest
    dpt_frame = depth_frame.as_depth_frame()    # gets suitable depth information from frame
    depth1 = dpt_frame.get_distance(pixel1[0], pixel1[1])   # returns depth in meters of pixel in image
    dept_intrin = depth_frame.profile.as_video_stream_profile().intrinsics   # depth camera intrinsics
    coordinate1 = pyrs.rs2_deproject_pixel_to_point(dept_intrin, [pixel1[0], pixel1[1]], depth1) # returns (x,y,z) relative to the camera in meters
    
    print (coordinate1)
    
    
    """
    Render images
    """
    cv2.imshow('Colour', c)
    cv2.imshow('Depth', d_scaled *10)  # Scale the depth data for visibility
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
"""
Release camera and close windows
"""  
cam.stop()
cv2.destroyAllWindows()