# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:01:36 2018

@author: Andy Everitt

Detects desk using contours from depth image

"""

import numpy as np
import cv2
import os
import pyrealsense2 as pyrs


data_path = "testdata/images" #read path
name_path = "test01"

fps = 15 #framerate
res_col = (640, 480) #color resolution
res_dep = res_col #depth resolution

live = True # enables live video feed

buffer_index = 0
buffer = 30
d_avg_min = np.linspace(0,0,buffer)


def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


if live:
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
else:
    color_path = "./{}/{}/color".format(data_path, name_path)
    depth_path = "./{}/{}/depth".format(data_path, name_path)
    
    color_images = os.listdir(color_path)
    color_images.sort()
    depth_images = os.listdir(depth_path)
    depth_images.sort()
    
    delay = 1000 // fps #delay between frames in ms
    img_index = 0


# Create Trackbars
cv2.namedWindow('Threshold Range')
cv2.createTrackbar('Auto Range', 'Threshold Range', 1, 1, nothing)
cv2.createTrackbar('Min Range', 'Threshold Range', 1, 66000, nothing)
cv2.createTrackbar('Max Range', 'Threshold Range', 3, 66000, nothing)

cv2.namedWindow('Morphological Transform')
cv2.createTrackbar('Opening Iterations', 'Morphological Transform', 2, 10, nothing)
cv2.createTrackbar('Closing Iterations', 'Morphological Transform', 10, 50, nothing)


while True:  # Loop over the images until we decide to quit.
    if live:
        #Get frame
        frames = cam.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
#        c = frames.get_color_frame()
#        d = frames.get_depth_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
    else:
        color_filename = color_images[img_index]
        depth_filename = depth_images[img_index]
        if not color_filename.endswith(".png") or not depth_filename.endswith(".png"):
            continue
    
        # Note that if the cv2.IMREAD_ANYDEPTH flag is missing, OpenCV will load the 16-bit depth data as an 8-bit image.
        c = cv2.imread("{}/{}".format(color_path, color_filename))
        d = cv2.imread("{}/{}".format(depth_path, depth_filename), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        
        img_index = (img_index + 1) % len(color_images)
        
    
    d_scaled = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    
    
    # Calculate horizontal depth average
    d_avg_h = np.average(d_scaled[:,:,0], 1)
    d_avg_h_img = d_avg_h * np.ones(res_dep)
    d_avg_h_img = np.transpose(d_avg_h_img)
    d_avg_h_img = d_avg_h_img.astype('uint16')
    d_avg_h_img = cv2.cvtColor(d_avg_h_img, cv2.COLOR_GRAY2BGR)
    d_avg_h_img = cv2.blur(d_avg_h_img, (19, 19))  # Gaussian blur to reduce noise in the image.

    # Calculate closest average depth and location using circular buffer
    d_avg_min[buffer_index] = min(d_avg_h[10:res_dep[1]]) # ignores bottom and top 10 rows as they have the most inaccurate depth data
    print (np.average(d_avg_min))
    buffer_index += 1
    if buffer_index == buffer:
        buffer_index = 0
    
    d_avg_index = np.argmin(d_avg_h[10:res_dep[1]-1]) + 10
    roi = [(10, d_avg_index-1), (629, 479)]
    cv2.rectangle(c,roi[0],roi[1],(65535,0,0),2)
    cv2.rectangle(d_scaled,roi[0],roi[1],(65535,0,0),2)
    cv2.rectangle(d_avg_h_img,(10,d_avg_index),(630,d_avg_index),(65535,0,0),2)
    roi_depth = d_scaled[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    roi_color = c[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    
    autoRange = cv2.getTrackbarPos('Auto Range', 'Threshold Range')
    if autoRange:
        minRange = int(np.average(d_avg_min) * 0.1)
#        minRange = 10
        maxRange = int((np.average(d_avg_min)) * 2.5)
        if maxRange < 500:
            maxRange = 500
    else:
        # Get trackbar information
        minRange = cv2.getTrackbarPos('Min Range', 'Threshold Range')
        maxRange = cv2.getTrackbarPos('Max Range', 'Threshold Range')
    
    

    # Do a simple "in range" threshold to find all objects between 2 and 5 units of distance away.
    # Note that each increment is equal to approximately 256mm. This is because the inRange function
    # only accepts 8-bit integers, so we must scale it down.
    thresh = cv2.inRange(roi_depth[:,:,0], minRange, maxRange)

    # Perform some morphological operations to help distinguish the features in the image.
    openIter = cv2.getTrackbarPos('Opening Iterations', 'Morphological Transform')
    closeIter = cv2.getTrackbarPos('Closing Iterations', 'Morphological Transform')
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=openIter)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=closeIter)
    
    # Edge detection
#    edges_depth = cv2.Canny(closing, 0, 0)
#    edges_colour = cv2.Canny(c, 0, 100)
#    edges_combined = edges_depth + edges_colour

    # Detect the external contour in the thresholded image.
    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    i = 0
    try:
        contour_size = 0
        for contour in contours:
            if (np.size(contour) > contour_size):
                contour_size = np.size(contour)
                contour_desk = contour
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(roi_depth,(x,y),(x+w,y+h),(0,65535,65535),2)
            cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,65535,65535),2)
#        contour_exter = cv2.convexHull(contours[0]) # Convex hull of first external contour
        # Detect & draw bounding rectangle
        x,y,w,h = cv2.boundingRect(contour_desk)
#        cv2.rectangle(c,(x,y),(x+w,y+h),(0,65535,0),2)
#        cv2.rectangle(roi_depth,(x,y),(x+w,y+h),(0,65535,0),2)
#        cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,65535,0),2)
    except IndexError:
        i += 1
        if (i == 10):
            print ('No contours')
    

    
    # Draw the first external contour around the detected object.        
    cv2.drawContours(roi_color, contours, -1, (0,0,65535), 1)
    cv2.drawContours(roi_depth, contours, -1, (0,0,65535), 1)
#
#    # Use the moments of the contour to draw a dot at the centroid of the object.
#    for contour in contours:
#        moments = cv2.moments(contour)
#        cX = int(moments["m10"] / moments["m00"])
#        cY = int(moments["m01"] / moments["m00"])
#        # Draw the centroid on the colour and depth images.
#        cv2.circle(c, (cX, cY), 7, (0, 65535, 0), -1)
#        cv2.circle(d_scaled, (cX, cY), 7, (0, 65535, 0), -1)
        
        
    # Render images
    cv2.namedWindow('Colour',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Colour', 640,480)
    
    cv2.namedWindow('Depth',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth', 640,480)
    
    cv2.namedWindow('Depth Average',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth Average', 320,240)
    
    cv2.namedWindow('Morphology',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Morphology', 480,120)
    
    
#    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow('Colour', c)
    cv2.imshow('Depth', d_scaled * 18)  # Scale the depth data for visibility
    cv2.imshow('Depth Average', d_avg_h_img * 18)  # Scale the depth data for visibility
    cv2.imshow('Depth ROI', roi_depth * 18)
    
    morphology_images = np.hstack((thresh, opening, closing))
#    morphology_images = cv2.resize(morphology_images, (0, 0), None, .25, .25)
    cv2.imshow('Morphology', morphology_images)
#    cv2.imshow('edges_depth', edges_depth)
#    cv2.imshow('edges_colour', edges_colour)
#    cv2.imshow('edges_combined', edges_combined)
    
    if 0xFF == ord('p'):
        os.system("pause")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
if live:    
    cam.stop()
cv2.destroyAllWindows()