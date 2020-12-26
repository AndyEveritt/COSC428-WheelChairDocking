# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:01:36 2018

@author: Andy Everitt

Detects the front of a desk using an Intel D435 depth camera.
The distance and angle of the camera relative to the desk are calculated.
Any obstructions under the desk are detected and the largest unobstructed
docking area is highlighted and measured to identify whether a wheelchair
will fit.

"""



###############################################################################
""" Future Improvements """
#
#    Convert distance and angle into wheelchair speeds
#    
#    Calculate best path to largest docking location
#
#    Detect when a desk is NOT present
#
#    Use a kalman filter to track desk between frames
#
###############################################################################

import numpy as np
import cv2
import pyrealsense2 as pyrs
import pathlib


###############################################################################
#   Test data
###############################################################################

#   Resolution = 1280x720p
"""
test00 - close, 0 degrees
test01 - 
test02 - bedroom, right & straight
test03 - bedroom, left & right rotation
test04 - no desk -> desk
test05 - kitchen, far -> close
test06 - dinning room, mid -> close
test07 - dinning room, rotation
"""

#   Resolution = 640x480p
"""
test01 - plain desk, 2m -> 0.4m
test02 - angle test, +/- 10 degrees, 0.5m
test03 - really cluttered desk 1.2m -> 0.4m
test04 - no movement, docking location detection
test05 - double high desk & leg
test06 - round table
test07 - round table with chair in foreground
"""


###############################################################################
#   Global variables
###############################################################################

"""
Video source
"""
live = False # enables live video feed
data_path = "testdata/bag/480p" #read path
name_path = "test03.bag"

fps = 15 #framerate

# Resolutions that are compatible with both colour and depth cameras
resolution = [(1280, 720),(848, 480),(640, 480)]
res_col = resolution[2] #color resolution
res_dep = resolution[2] #depth resolution


obstacles = True # enables obstacle detection
show_all = True # shows all steps of image processing in individual windows if True, else only shows colour image

"""
Output video
"""
save_vid = False
vid_name = 'output05.avi'


"""
Error frames
"""
save_error = False
error_path = "./error_images/01" #write path
error_index = 0

"""
Variables
"""
buffer_index = 0
buffer = 1
d_avg_min = np.linspace(0,0,buffer)
maxRange_multi = 1.2
minRange_multi = 0.8

chair_width = 0.6

total_frames = 0
successful_frames = 0


###############################################################################
#   Functions
###############################################################################

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

"""
Saves the colour, depth, depth average and depth average threshold images if
an error occurs
"""
def save_error_image():
    if save_error:
        global error_index
        cv2.imwrite("{}/{:04d}c.png".format(error_path, error_index), c)
        cv2.imwrite("{}/{:04d}d.png".format(error_path, error_index), d_scaled*10)
        cv2.imwrite("{}/{:04d}d_avg.png".format(error_path, error_index), d_avg_h_img*10)
        cv2.imwrite("{}/{:04d}d_t.png".format(error_path, error_index), desk_thresh_avg)
        error_index += 1

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


"""
Save images that cause an error
"""
if save_error:
    pathlib.Path(error_path).mkdir(parents=True, exist_ok=True)

"""
Save video format
"""
if save_vid:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(vid_name, fourcc, 20.0, (640, 480))


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
#cv2.createTrackbar("HoughThreshold", 'Morphological Transform', 0, 200, nothing)

while True:  # Loop over the images until we decide to quit.
    
    total_frames += 1

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
    
    d = np.asanyarray(depth_frame.get_data())
    c = np.asanyarray(color_frame.get_data())
        
    d_scaled = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    

    
    
###############################################################################  
#   Finding the front of desk and calulating the range of the desk using the
#   depth image
############################################################################### 
    
    """
    Calculate horizontal depth average
    """
    d_avg_h = d_scaled.astype(np.float64)   # convert to float
    d_avg_h[d_avg_h == 0] = np.nan    # replace missing depth values with "nan"
    d_avg_h = np.uint16(np.nanmean(d_avg_h[:,:,0],1))   # average each row of depth values
    d_avg_h = cv2.blur(d_avg_h, (1, 5))  # Gaussian blur to reduce noise in the image.
    
    
    # Manipulate d_avg_h so it can be displayed using "imshow"
    d_avg_h_img = d_avg_h[:,0] * np.ones(res_dep)
    d_avg_h_img = np.transpose(d_avg_h_img)
    d_avg_h_img = d_avg_h_img.astype('uint16')
    d_avg_h_img = cv2.cvtColor(d_avg_h_img, cv2.COLOR_GRAY2BGR)
    d_avg_h_img = cv2.GaussianBlur(d_avg_h_img, (1, 5),0)  # Gaussian blur to reduce noise in the image.


    """
    Calculate closest average depth and location using circular buffer
    """
    d_avg_min[buffer_index] = min(d_avg_h[10:res_dep[1]-10]) # ignores bottom and top 10 rows as they have the most inaccurate depth data
#    print (np.average(d_avg_min))
    buffer_index += 1
    if buffer_index == buffer:
        buffer_index = 0
    
    d_avg_index = np.argmin(d_avg_h[10:res_dep[1]-10]) + 10
    
    
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
        
        
    desk_thresh_avg = cv2.inRange(d_avg_h_img[:,:,0], minRange, maxRange)


    """
    Finds contours in the average horizontal depth values and uses the largest 
    to find the height of the desk in the image
    """
    _, contours, hierarchy = cv2.findContours(desk_thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(d_avg_h_img, contours, -1, (0,0,65535), 1)   # red contours
    
    try:    
        contour = max(contours, key = cv2.contourArea)
    except ValueError:
        print ("ERROR: No desk contour")
        continue
    
    x,y,w,h = cv2.boundingRect(contour)
    
    desk_edge = [x,y,w,h]   # Desk edge vertical bounds, has not detected the horizontal edge boundary
    
    """
    Desk edge ROI to find the desk width
    """
    roi_edge_depth = d_scaled[desk_edge[1]:desk_edge[1]+desk_edge[3], desk_edge[0]:desk_edge[0]+desk_edge[2]]
    
    desk_thresh = cv2.inRange(roi_edge_depth[:,:,0], minRange, maxRange)
    
    """
    Perform some morphological operations to help distinguish the front of the desk in the image.
    """
    kernel = np.ones((3,3), np.uint8)
    opening_desk = cv2.morphologyEx(desk_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing_desk = cv2.morphologyEx(opening_desk, cv2.MORPH_CLOSE, kernel, iterations=10)
    
    _, contours, hierarchy = cv2.findContours(closing_desk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    
    ###################
    #   Finding desk width Option 1
    ###################
#    min_x, min_y = x+w, y+h
#    max_x = max_y = 0
#    
##     computes the bounding box for the contour, and draws it on the frame,
#    for contour in contours:
#        if (cv2.contourArea(contour) > 100):
#            (x,y,w,h) = cv2.boundingRect(contour)
#            min_x, max_x = min(x, min_x), max(x+w, max_x)
#            min_y, max_y = min(y, min_y), max(y+h, max_y)
#    
#    """
#    Highlight front of desk on images with blue rectangle
#    """
#    if max_x - min_x > 0 and max_y - min_y > 0:
#        cv2.rectangle(roi_edge_depth, (min_x, min_y), (max_x, max_y), (65535, 0, 0), 2)
#        cv2.rectangle(d_avg_h_img, (min_x, min_y + desk_edge[1]), (max_x, max_y + desk_edge[1]), (65535, 0, 0), 2)
#        cv2.rectangle(c, (min_x, min_y + desk_edge[1]), (max_x, max_y + desk_edge[1]), (65535, 0, 0), 2)
#    
#    
#    desk_edge = [min_x, min_y + desk_edge[1], max_x - min_x, max_y - min_y]
    
    
    ###############
    #   Finding desk width Option 2
    ###############
    try:    
        contour = max(contours, key = cv2.contourArea)
    except ValueError:
        print ("ERROR: No desk contour")
        continue
    
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(c, (x, y + desk_edge[1]), (x+w, y+h + desk_edge[1]), (65535, 0, 0), 2)
    cv2.rectangle(d_avg_h_img, (x, y + desk_edge[1]), (x+w, y+h + desk_edge[1]), (65535, 0, 0), 2)
    cv2.rectangle(roi_edge_depth, (x, y), (x+w, y+h), (65535, 0, 0), 2)
    
    desk_edge = [x,y+desk_edge[1],x+w,y+h]
    
    ###############
    
    desk_mid = (int(desk_edge[0] + desk_edge[2]/2), int(desk_edge[1] + desk_edge[3]/2))

    



###############################################################################
#   Finds the distance (meters) and the angle (degrees) of the desk relative to
#   the camera so that it can be implemented as wheelchair movements
###############################################################################   

    """
    These are the pixel coordinates I want to find the 3D world coordinates of
    """
    multiplier = 0.2
    desk_pixel1 = (int(desk_mid[0] - desk_edge[2]*multiplier), desk_mid[1])
    desk_pixel2 = (int(desk_mid[0] + desk_edge[1]*multiplier), desk_mid[1])
    

    """
    Filtering dodgy data
    """
    kernel = (5,5)
    dpt_frame = depth_frame.as_depth_frame()
    depth1 = dpt_frame.get_distance(desk_pixel1[0], desk_pixel1[1])
    depth2 = dpt_frame.get_distance(desk_pixel2[0], desk_pixel2[1])
    
    while depth1 == 0:
        multiplier += 0.05
        desk_pixel1 = (int(desk_mid[0]-desk_edge[2]*multiplier), desk_mid[1])
        depth1 = dpt_frame.get_distance(desk_pixel2[0], desk_pixel2[1])
#            cv2.rectangle(d_scaled,(desk_pixel1[0]-kernel[0], desk_pixel1[1]-kernel[1]),(desk_pixel1[0]+kernel[0], desk_pixel1[1]+kernel[1]),(0,65000,65000),1)
        if multiplier > 0.4:
            break
        
    multiplier = 0.2
    while depth2 == 0:
        multiplier += 0.01
        desk_pixel2 = (int(desk_mid[0]+desk_edge[2]*multiplier), desk_mid[1])
        depth2 = dpt_frame.get_distance(desk_pixel2[0], desk_pixel2[1])
#            cv2.rectangle(d_scaled,(desk_pixel2[0]-kernel[0], desk_pixel2[1]-kernel[1]),(desk_pixel2[0]+kernel[0], desk_pixel2[1]+kernel[1]),(65000,65000,0),1)
        if multiplier > 0.4:
            break

#    for a in range(-kernel[0], kernel[0]):
#        for b in range(-kernel[1], kernel[1]):
#            dist = dpt_frame.get_distance(desk_pixel1[0]+a,desk_pixel1[1]+b)
#            if dist > depth1:
#                depth1 = dist
#            dist = dpt_frame.get_distance(desk_pixel2[0]+a,desk_pixel2[1]+b)
#            if dist > depth2:
#                depth2 = dist

#    print ("Depth1 {:.3f} \t Depth2 {:.3f}".format(depth1, depth2))
    
    cv2.rectangle(d_scaled,(desk_pixel1[0]-kernel[0], desk_pixel1[1]-kernel[1]),(desk_pixel1[0]+kernel[0], desk_pixel1[1]+kernel[1]),(0,65000,0),1)
    cv2.rectangle(d_scaled,(desk_pixel2[0]-kernel[0], desk_pixel2[1]-kernel[1]),(desk_pixel2[0]+kernel[0], desk_pixel2[1]+kernel[1]),(0,65000,0),1)
#    print (pixel_distance_in_meters)
    
    
    if (depth1 != 0 and depth2 != 0):
        
        
        """
        Calculate 3D world space coordinates in meters relative to camera
        """
        dept_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        desk_coordinate1 = pyrs.rs2_deproject_pixel_to_point(dept_intrin, [desk_pixel1[0], desk_pixel1[1]], depth1)
        desk_coordinate2 = pyrs.rs2_deproject_pixel_to_point(dept_intrin, [desk_pixel2[0], desk_pixel2[1]], depth2)
        
#        print ("Coor1 {} \nCoor2 {}".format(desk_coordinate1, desk_coordinate2))
        
        
        (x1, x2) = (desk_coordinate1[0], desk_coordinate2[0])    # Calculated 3D horizontal coordinates
        (y1, y2) = (desk_coordinate1[1], desk_coordinate2[1])
        (depth1, depth2) = (desk_coordinate1[2], desk_coordinate2[2])   # Calculated depth (in meters) at each coordinate x1 & x2 along front of desk (y = d_avg_index)
        
        
        """
        Calculate mean desk distance and angle relative to camera
        """
        try:
            desk_angle = np.arctan((depth2 - depth1)/(x2 - x1))
            desk_angle = desk_angle * 180 / np.pi
        except ZeroDivisionError:
            print ("ERROR: ZeroDivisionError")
            save_error_image()
            pass
        
        desk_dist = (depth1 + depth2)/2
        
        
        cv2.rectangle(c, (0,0), (700,80), (0,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(c,"Angle {:.3f}".format(desk_angle),(10,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(c,"Distance {:.3f} m".format(desk_dist),(200,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    
#        print ("Angle {:.3f} \t Distance {:.3f}".format(desk_angle, desk_dist))
    
    
    
    
###############################################################################  
#   Obstacle detection using thresholding and morphological operations
############################################################################### 
    

    if obstacles:
        """
        Identify region of interest (ROI) for beneath the desk
        """
#            roi = [(10, d_avg_index-1), (res_dep[0]-10, res_dep[1]-1)]
        roi = [(desk_edge[0], desk_edge[1] + desk_edge[3] + 20), (desk_edge[0] + desk_edge[2]-1, res_dep[1]-1)]
        cv2.rectangle(c,roi[0],roi[1],(0,65535,0),2)
        cv2.rectangle(d_scaled,roi[0],roi[1],(0,65535,0),2)
        roi_depth = d_scaled[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        roi_color = c[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
    
        """
        Do "in range" threshold to find all objects between minRange and maxRange units of distance away in ROI.
        """
        thresh = cv2.inRange(roi_depth[:,:,0], 10, maxRange)
    
        """
        Perform some morphological operations to help distinguish the features in the image.
        """
        try:
            openIter = cv2.getTrackbarPos('Opening Iterations', 'Morphological Transform')
            closeIter = cv2.getTrackbarPos('Closing Iterations', 'Morphological Transform')
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=openIter)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=closeIter)
        except:
            print ("ERROR: Can't find underdesk")
            save_error_image()
            continue
    
        """
        Detect & draw the external contours in the thresholded ROI.
        """
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_list = []

        contour_size = 0
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            obstacle_list.append((x,y,w,h))
            cv2.rectangle(roi_depth,(x,y),(x+w,y+h),(0,65535,65535),2)  # yellow bounding rectangles
            cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,65535,65535),2)  # yellow bounding rectangles

        
        """
        Draw the all external contours around the detected object.        
        """
        cv2.drawContours(roi_color, contours, -1, (0,0,65535), 1)   # red contours
        cv2.drawContours(roi_depth, contours, -1, (0,0,65535), 1)   # red contours
        

        """
        Find all unobstructed points beneath desk
        """
        clear_list = []
        clear_index = []
        
        alpha = 0.6
        roi_overlay = roi_depth.copy()
        
        
        for pixel_x in range(np.shape(closing)[1]):
            clear = True
            for pixel_y in range(np.shape(closing)[0]):
                if closing[pixel_y, pixel_x] != 0:
                    clear = False
                    break
            clear_list.append(clear)
            if clear == True:
                cv2.rectangle(roi_overlay,(pixel_x,0),(pixel_x,np.shape(closing)[0]),(0,0,0),-1)

        # apply the overlay
        cv2.addWeighted(roi_overlay, alpha, roi_depth, 1 - alpha,0, roi_depth)
        
        
        """
        Find size of each docking location
        """
        index1 = 0
        index2 = 0
        index_diff = 0            
        for i in range(1, len(clear_list)):

            if clear_list[i-1] == False and clear_list[i] == True:
                index_diff = 0
                index1 = i
            elif clear_list[i-1] == True and clear_list[i] == True:
                index_diff += 1
            elif clear_list[i-1] == True and clear_list[i] == False:
                index2 = i
                clear_index.append((index1,index2,index_diff))
        
            if (i == len(clear_list)-1 and clear_list[i] == True):
                index2 = i
                clear_index.append((index1,index2,index_diff))
        
        
        """
        Find largest docking location
        """        
        docking_width = 0
        for i in range(len(clear_index)):
            if clear_index[i][2] > docking_width:
                docking_width = clear_index[i][2]
        
                docking_location = (clear_index[i][0],clear_index[i][1])
        try:
            docking_location
        except NameError:
            continue
        
        """
        Calculate size of docking location
        """
        docking_coordinate1 = pyrs.rs2_deproject_pixel_to_point(dept_intrin, [docking_location[0], desk_mid[1]], depth1)
        docking_coordinate2 = pyrs.rs2_deproject_pixel_to_point(dept_intrin, [docking_location[1], desk_mid[1]], depth2)
        
        (docking_x1, docking_x2) = (docking_coordinate1[0], docking_coordinate2[0])
        docking_width = docking_x2 - docking_x1
        
#        print ("Docking1 {} \nDocking2 {}".format(docking_coordinate1, docking_coordinate2))
#        print ("Docking Width {:.3f} m".format(docking_width))
        
        if docking_width > chair_width:    
            cv2.rectangle(roi_depth,(docking_location[0], 0),(docking_location[1], np.shape(closing)[0]),(65535,65535,65535),5)
            cv2.rectangle(c,(docking_location[0] + roi[0][0], roi[0][1]),(docking_location[1] + roi[0][0], roi[1][1]),(65535,65535,65535),5)
            cv2.putText(c,"Docking Width {:.3f} m".format(docking_width),(400,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.rectangle(roi_depth,(docking_location[0], 0),(docking_location[1], np.shape(closing)[0]),(0,0,65535),5)
            cv2.rectangle(c,(docking_location[0] + roi[0][0], roi[0][1]),(docking_location[1] + roi[0][0], roi[1][1]),(0,0,65535),5)
            cv2.putText(c,"Docking Width {:.3f} m".format(docking_width),(400,50), font, 0.5,(0,0,255),1,cv2.LINE_AA)
        
        
    

###############################################################################  
#   Attempt at detecting angle of desk relative to camera -------- Unsuccessful
#    
#   Might be useful in future to detect if desk is rotated (It was not)
############################################################################### 
        
    """
    Edge detection
    """
    
#        roi_edge = [(10, d_avg_index-10), (res_dep[0]-10, d_avg_index+10)]
#        cv2.rectangle(d_avg_h_img,(10,d_avg_index),(res_dep[0]-10,d_avg_index),(65535,65535,0),2)
#        cv2.rectangle(c,roi_edge[0],roi_edge[1],(65535,65535,0),2)
#        cv2.rectangle(d_scaled,roi_edge[0],roi_edge[1],(65535,65535,0),2)
#        roi_edge_depth = d_scaled[roi_edge[0][1]:roi_edge[1][1],roi_edge[0][0]:roi_edge[1][0]]
#        roi_edge_color = c[roi_edge[0][1]:roi_edge[1][1],roi_edge[0][0]:roi_edge[1][0]]
    
#        houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
#        edges_depth = cv2.Canny(roi_edge_depth, 0, 100)
#        edges_color = cv2.Canny(roi_edge_color, 0, 100)
#        edges_combined = edges_depth + edges_colour
#        lines = cv2.HoughLines(edges_color, 1, np.pi/180, houghThreshold)
    



###############################################################################  
#   Graph stuff
############################################################################### 

    
    """
    Render images
    """
    cv2.namedWindow('Colour',cv2.WINDOW_NORMAL)
    
    if (show_all):
        cv2.resizeWindow('Colour', 640,480)
    
        cv2.namedWindow('Depth',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth', 640,480)
        cv2.imshow('Depth', d_scaled *10)  # Scale the depth data for visibility
        
        cv2.namedWindow('Depth Average',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth Average', 320,240)
        cv2.imshow('Depth Average', d_avg_h_img * 10)  # Scale the depth data for visibility
        
        cv2.namedWindow('Depth Average Threshold',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Depth Average Threshold', 320,240)
        cv2.imshow('Depth Average Threshold', desk_thresh_avg * 20)  # Scale the depth data for visibility
        
        cv2.imshow('Depth Threshold', closing_desk * 20)  # Scale the depth data for visibility
        
        
#        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(d_scaled, alpha=0.03), cv2.COLORMAP_JET)
#        cv2.imshow('Depth_Colour', depth_colormap)
    
        
        if obstacles:
            cv2.namedWindow('Morphology',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Morphology', 640,120)
            cv2.imshow('Depth ROI', roi_depth * 20)
            morphology_images = np.hstack((thresh, opening, closing))
            cv2.imshow('Morphology', morphology_images)
    
        cv2.imshow('Desk ROI', roi_edge_depth * 20)
        
    cv2.imshow('Colour', c)
    
    successful_frames += 1
#    cv2.putText(c,"Frames - Total: {:.3f}  Successful: {:.3f} ".format(total_frames, successful_frames),(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)

#    print ("\n")
    
    if save_vid:
        out.write(c)  # Write the frame to the output file.
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

"""
Release camera and close windows
"""  
if (live==False):
    print ("File {}".format(name_path))
print ("\nTotal Frames {}\nSuccessful Frames {}\nRatio {:.1f}%".format(total_frames, successful_frames, (successful_frames/total_frames)*100))
cam.stop()
cv2.destroyAllWindows()