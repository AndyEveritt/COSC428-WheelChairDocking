# r200_segmentation.py

import numpy as np
import cv2
import os

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

color_path = "./colour/"
depth_path = "./depth/"
index = -1

# Load the list of image filenames.
# We are assuming that the depth files have the exact same filename.
color_images = os.listdir(color_path)
color_images.sort()  # Make sure we get all our images in order.

cv2.namedWindow('Hough Line Transform')
cv2.createTrackbar('Canny Threshold 1', 'Hough Line Transform', 0, 1200, nothing)
cv2.createTrackbar('Canny Threshold 2', 'Hough Line Transform', 0, 1200, nothing)
cv2.createTrackbar("Min Line Length", 'Hough Line Transform', 0, 100, nothing)
cv2.createTrackbar("Max Line Gap", 'Hough Line Transform', 0, 1000, nothing)

while True:  # Loop over the images until we decide to quit.
    index = (index + 1) % len(color_images)
    filename = color_images[index]
    
    # Get track bar positions for hough line and canny edge
    minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Line Transform')
    maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Line Transform')
    cannyThreshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Hough Line Transform')
    cannyThreshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Hough Line Transform')

    # Note that if the cv2.IMREAD_ANYDEPTH flag is missing,
    # OpenCV will load the 16-bit depth data as an 8-bit image.
    c = cv2.imread("{}{}".format(color_path, filename))
    d = cv2.imread("{}{}".format(depth_path, filename), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    # Create a new 8-bit colour version of the depth data for drawing on later.
    # This is scaled for visibility (65536 / 3500 ~= 18)
    d_scaled = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    d_scaled = ((d_scaled*18)/(256)).astype('uint8')

    # Do a simple "in range" threshold to find all objects between 2 and 5 units of distance away.
    # Note that each increment is equal to approximately 256mm. This is because the inRange function
    # only accepts 8-bit integers, so we must scale it down.
    thresh = cv2.inRange((d / 256).astype(np.uint8), 2, 5)

    # Perform some morphological operations to help distinguish the features in the image.
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    # Edge detection
    edges_depth = cv2.Canny(closing, 0, 0)
    edges_colour = cv2.Canny(c, cannyThreshold1, cannyThreshold2)
    edges_combined = edges_depth + edges_colour

    # Detect the contour in the image.
    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_exter = cv2.convexHull(contours[0])
    
    # Draw the contour around the detected object.        
    cv2.drawContours(c, contour_exter, -1, (0,0,255), 10)
    cv2.drawContours(d_scaled, contours, -1, (0,0,255), 1)
    
    x,y,w,h = cv2.boundingRect(contours[0])
    cv2.rectangle(c,(x,y),(x+w,y+h),(0,255,0),2)

    # Use the moments of the contour to draw a dot at the centroid of the object.
    for contour in contours:
        moments = cv2.moments(contour)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        # Draw the centroid on the colour and depth images.
        cv2.circle(c, (cX, cY), 7, (0, 255, 0), -1)
        cv2.circle(d_scaled, (cX, cY), 7, (0, 255, 0), -1)
    
#    # Detect the contour in the image.
#    _, contours_edge, hierarchy = cv2.findContours(edges_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    
#    
#    
#    # Draw the contour around the detected object.
#    cv2.drawContours(c, contours_edge[0], -1, (255,0,0), 1)
#    cv2.drawContours(d_scaled, contours_edge, -1, (255,0,0), 1)
    
#    lines = cv2.HoughLinesP(edges_depth, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
#
#    # For each line that was detected, draw it on the img.
#    if lines is not None:
#        for line in lines:
#            for x1,y1,x2,y2 in line:
#                cv2.line(c,(x1,y1),(x2,y2),(0,255,0),2)
                
#    corners = cv2.cornerHarris(closing,2,3,0.04)
#    # Dialate the detected corners to make them clearer in the output image.
#    corners = cv2.dilate(corners,None)
#
#    # Perform thresholding on the corners to throw away some false positives.
#    frame[corners > 0.1 * corners.max()] = [0,0,255]
#
#    cv2.imshow("Harris", closing)
#    
    cv2.imshow('c', c)
    cv2.imshow('d', d_scaled)
    cv2.imshow('thresh', thresh)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    cv2.imshow('edges_depth', edges_depth)
    cv2.imshow('edges_colour', edges_colour)
    cv2.imshow('edges_combined', edges_combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
