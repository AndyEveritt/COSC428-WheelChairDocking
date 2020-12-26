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



"""
test00 - Bedroom desk, unobstructed -> knee. (unaligned color & depth)
test01 - Bedroom desk, unobstructed -> 2 knees -> desk at angle. (aligned color & depth)
test02 - Kitchen table, 2 chairs side on
test03 - Bedroom desk, unobstructed, straight on
test04 - Bedroom desk, chair in middle, straight on
test05 - Bedroom desk, knee & wire, straight on, getting closer
test06 - Lab desk, computer & bag, varying distance
test07 - Lab desk, chair, varying distance
test08 - No desk -> desk -> window
"""

data_path = "testdata/images" #read path
name_path = "test01"


fps = 30 #framerate
res_col = (640, 480) #color resolution
res_dep = res_col #depth resolution

live = False # enables live video feed

obstacles = False # enables obstacle detection
buffer_index = 0
buffer = 50
d_avg_min = np.linspace(0,0,buffer)
maxRange_multi = 2.2
minRange_multi = 0.1


def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


if live:
    """
    Set up camera
    """
    cam = pyrs.pipeline()
    
    config = pyrs.config()
    config.enable_stream(pyrs.stream.color, res_col[0], res_col[1], pyrs.format.bgr8, fps)
    config.enable_stream(pyrs.stream.depth, res_dep[0], res_dep[1], pyrs.format.z16, fps)
    
    cam.start(config)
    
    """
    Create an align object
    rs.align allows us to perform alignment of depth frames to others frames
    The "align_to" is the stream type to which we plan to align depth frames.
    """
    align_to = pyrs.stream.color
    align = pyrs.align(align_to)
else:
    color_path = "./{}/{}/color".format(data_path, name_path)
    depth_path = "./{}/{}/depth".format(data_path, name_path)
    
    color_images = os.listdir(color_path)
    color_images.sort()
    depth_images = os.listdir(depth_path)
    depth_images.sort()
    
    delay = int (1000 // fps) #delay between frames in ms
    img_index = 0


"""
Create Trackbars
"""
cv2.namedWindow('Threshold Range')
cv2.createTrackbar('Auto Range', 'Threshold Range', 1, 1, nothing)
cv2.createTrackbar('Min Range', 'Threshold Range', 1, 66000, nothing)
cv2.createTrackbar('Max Range', 'Threshold Range', 3, 66000, nothing)


cv2.namedWindow('Morphological Transform')
cv2.createTrackbar('Opening Iterations', 'Morphological Transform', 2, 10, nothing)
cv2.createTrackbar('Closing Iterations', 'Morphological Transform', 10, 50, nothing)
cv2.createTrackbar("HoughThreshold", 'Morphological Transform', 0, 200, nothing)

while True:  # Loop over the images until we decide to quit.
        
    """
    Pause program on pressing 'p'
    """
#    while cv2.waitKey(delay) & 0xFF == ord('p'):
#        continue
        
    """
    Load frames from camera if live=True or memory if live=False
    """
    if live:
        # Get frame
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
    
    
    """
    Calculate horizontal depth average
    """
    d_avg_h = np.average(d_scaled[:,:,0], 1)
    d_avg_h_img = d_avg_h * np.ones(res_dep)
    d_avg_h_img = np.transpose(d_avg_h_img)
    d_avg_h_img = d_avg_h_img.astype('uint16')
    d_avg_h_img = cv2.cvtColor(d_avg_h_img, cv2.COLOR_GRAY2BGR)
    d_avg_h_img = cv2.blur(d_avg_h_img, (1, 20))  # Gaussian blur to reduce noise in the image.


    """
    Calculate closest average depth and location using circular buffer
    """
    d_avg_min[buffer_index] = min(d_avg_h[10:res_dep[1]]) # ignores bottom and top 10 rows as they have the most inaccurate depth data
#    print (np.average(d_avg_min))
    buffer_index += 1
    if buffer_index == buffer:
        buffer_index = 0
    
    d_avg_index = np.argmin(d_avg_h[10:res_dep[1]-1]) + 10
    
    
    """
    Calculate threshold depth range
    """
    autoRange = cv2.getTrackbarPos('Auto Range', 'Threshold Range')
    if autoRange:
        minRange = int(np.average(d_avg_min) * minRange_multi)
#        minRange = 10
        maxRange = int((np.average(d_avg_min)) * maxRange_multi)
#        if maxRange < 500:
#            maxRange = 500
    else:
        # Get trackbar information
        minRange = cv2.getTrackbarPos('Min Range', 'Threshold Range')
        maxRange = cv2.getTrackbarPos('Max Range', 'Threshold Range')
        
        
    """
    Obstacle detection using thresholding and morphological operations
    """
    if obstacles:
        """
        Identify region of interest (ROI) for beneath the desk
        """
        roi = [(10, d_avg_index-1), (629, 479)]
        cv2.rectangle(c,roi[0],roi[1],(65535,0,0),2)
        cv2.rectangle(d_scaled,roi[0],roi[1],(65535,0,0),2)
        roi_depth = d_scaled[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        roi_color = c[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    
        """
        Do "in range" threshold to find all objects between minRange and maxRange units of distance away in ROI.
        """
        thresh = cv2.inRange(roi_depth[:,:,0], minRange, maxRange)
    
        """
        Perform some morphological operations to help distinguish the features in the image.
        """
        openIter = cv2.getTrackbarPos('Opening Iterations', 'Morphological Transform')
        closeIter = cv2.getTrackbarPos('Closing Iterations', 'Morphological Transform')
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=openIter)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=closeIter)
    
        """
        Detect & draw the external contours in the thresholded ROI.
        """
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        i = 0
        try:
            contour_size = 0
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(roi_depth,(x,y),(x+w,y+h),(0,65535,65535),2)  # yellow bounding rectangles
                cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,65535,65535),2)  # yellow bounding rectangles
    #            contour_exter = cv2.convexHull(contour) # Convex hull of first external contour
    #            cv2.drawContours(roi_color, [contour_exter], -1, (65535,0,65535), 4)   # red contours
        except IndexError:  # if no contours exist
            i += 1
            if (i == 10):
                print ('No contours')
        
        """
        Draw the all external contours around the detected object.        
        """
        cv2.drawContours(roi_color, contours, -1, (0,0,65535), 1)   # red contours
        cv2.drawContours(roi_depth, contours, -1, (0,0,65535), 1)   # red contours
    
        """
        Use the moments of the contour to draw a dot at the centroid of the object.
        """
#        for contour in contours:
#            moments = cv2.moments(contour)
#            cX = int(moments["m10"] / moments["m00"])
#            cY = int(moments["m01"] / moments["m00"])
#            # Draw the centroid on the colour and depth images.
#            cv2.circle(roi_color, (cX, cY), 7, (0, 65535, 0), -1)
#            cv2.circle(roi_depth, (cX, cY), 7, (0, 65535, 0), -1)
        
        
    """
    Desk edge ROI
    """
    roi_edge = [(10, d_avg_index-10), (629, d_avg_index+10)]
    cv2.rectangle(d_avg_h_img,(10,d_avg_index),(630,d_avg_index),(65535,65535,0),2)
#    cv2.rectangle(c,roi_edge[0],roi_edge[1],(65535,65535,0),2)
    cv2.rectangle(d_scaled,roi_edge[0],roi_edge[1],(65535,65535,0),2)
    roi_edge_depth = d_scaled[roi_edge[0][1]:roi_edge[1][1],roi_edge[0][0]:roi_edge[1][0]]
    roi_edge_color = c[roi_edge[0][1]:roi_edge[1][1],roi_edge[0][0]:roi_edge[1][0]]
    
    """
    Edge detection
    """
    houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
#    edges_depth = cv2.Canny(roi_edge_depth, 0, 100)
    edges_color = cv2.Canny(roi_edge_color, 0, 100)
#    edges_combined = edges_depth + edges_colour
    lines = cv2.HoughLines(edges_color, 1, np.pi/180, houghThreshold)

    # For each line that was detected, draw it on the img.
#    if lines is not None:
#	     for line in lines:
#		      for rho,theta in line:
#		          a = np.cos(theta)
#		          b = np.sin(theta)
#		          x0 = a*rho
#		          y0 = b*rho
#		          x1 = int(x0 + 1000*(-b))
#		          y1 = int(y0 + 1000*(a))
#		          x2 = int(x0 - 1000*(-b))
#		          y2 = int(y0 - 1000*(a))
#
#		          cv2.line(edges_color,(x1,y1),(x2,y2),(0,0,65535),1)
    
    
    """
    Render images
    """
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
    
    if obstacles:
        cv2.imshow('Depth ROI', roi_depth * 18)
        morphology_images = np.hstack((thresh, opening, closing))
        cv2.imshow('Morphology', morphology_images)

    cv2.imshow('Color ROI', roi_edge_color)
#    cv2.imshow('edges_depth', edges_depth)
    cv2.imshow('edges_color', edges_color)
#    cv2.imshow('edges_combined', edges_combined)
    

    
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

"""
Release camera and close windows
"""
if live:    
    cam.stop()
cv2.destroyAllWindows()