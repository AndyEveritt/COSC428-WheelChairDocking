# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:01:36 2018

@author: Andy Everitt

Detects desk using contours from depth image

"""

import numpy as np
import cv2
import pyrealsense2 as pyrs


data_path = "testdata/video" #read path
name_path = "test11"

fps = 15 #framerate
res_col = (640, 480) #color resolution
res_dep = res_col #depth resolution

live = False # enables live video feed

index = 0
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
else:
    # Load the video file.
    cap_color = cv2.VideoCapture('./{}/{}/color.avi'.format(data_path, name_path))
    cap_depth = cv2.VideoCapture('./{}/{}/depth.avi'.format(data_path, name_path))


# Create Trackbars
cv2.namedWindow('Threshold Range')
cv2.createTrackbar('Auto Range', 'Threshold Range', 1, 1, nothing)
cv2.createTrackbar('Min Range', 'Threshold Range', 1, 66000, nothing)
cv2.createTrackbar('Max Range', 'Threshold Range', 3, 66000, nothing)

cv2.namedWindow('Morphological Transform')
cv2.createTrackbar('Opening Iterations', 'Morphological Transform', 0, 10, nothing)
cv2.createTrackbar('Closing Iterations', 'Morphological Transform', 0, 50, nothing)


while True:  # Loop over the images until we decide to quit.
    if live:
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
    else:
        ret_color, c = cap_color.read()  # Read an frame from the color video file.
        ret_depth, d = cap_depth.read()  # Read an frame from the depth video file.
    
        # If we cannot read any more frames from the video file, then reload video.
        if not ret_color or not ret_depth:
            cap_color.release()
            cap_depth.release()
            # Load the video file.
            cap_color = cv2.VideoCapture('./{}/{}/color.avi'.format(data_path, name_path))
            cap_depth = cv2.VideoCapture('./{}/{}/depth.avi'.format(data_path, name_path))
            continue
        d = d[:,:,0]
    
    
    d_scaled = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    
    
    # Calculate horizontal depth average
    d_avg_h = np.average(d_scaled[:,:,0], 1)
    d_avg_h_img = d_avg_h * np.ones((640, 480))
    d_avg_h_img = np.transpose(d_avg_h_img)
    d_avg_h_img = d_avg_h_img.astype('uint16')
    d_avg_h_img = cv2.cvtColor(d_avg_h_img, cv2.COLOR_GRAY2BGR)

    # Calculate closest average depth and location using circular buffer
    d_avg_min[index] = min(d_avg_h[10:res_dep[1]]) # ignores bottom and top 10 rows as they have the most inaccurate depth data
    print (np.average(d_avg_min))
    index += 1
    if index == buffer:
        index = 0
    
    d_avg_index = np.argmin(d_avg_h[10:res_dep[1]]) + 10
    roi = [(10, d_avg_index-1), (629, 479)]
    cv2.rectangle(c,roi[0],roi[1],(65535,0,0),2)
    cv2.rectangle(d_scaled,roi[0],roi[1],(65535,0,0),2)
    cv2.rectangle(d_avg_h_img,(10,d_avg_index),(630,d_avg_index),(65535,0,0),2)
    roi_depth = d_scaled[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    
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
#        contour_exter = cv2.convexHull(contours[0]) # Convex hull of first external contour
        # Detect & draw bounding rectangle
        x,y,w,h = cv2.boundingRect(contour_desk)
#        cv2.rectangle(c,(x,y),(x+w,y+h),(0,65535,0),2)
        cv2.rectangle(roi_depth,(x,y),(x+w,y+h),(0,65535,0),2)
    except IndexError:
        i += 1
        if (i == 10):
            print ('No contours')
    

    
    # Draw the first external contour around the detected object.        
    cv2.drawContours(c, contours, -1, (0,0,65535), 10)
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
        
        
    cv2.imshow('Colour', c)
    cv2.imshow('Depth', d_scaled * 18)  # Scale the depth data for visibility
    cv2.imshow('Depth Average', d_avg_h_img * 18)  # Scale the depth data for visibility
    cv2.imshow('Depth ROI', roi_depth * 18)
    
    cv2.imshow('thresh', thresh)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
#    cv2.imshow('edges_depth', edges_depth)
#    cv2.imshow('edges_colour', edges_colour)
#    cv2.imshow('edges_combined', edges_combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
if live:    
    cam.stop()
else:
    cap_color.release()
    cap_depth.release()
cv2.destroyAllWindows()