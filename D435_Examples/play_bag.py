# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

path_to_bag = "../testdata/bag/stairs.bag"

config = rs.config()
config.enable_device_from_file(path_to_bag)

# Create a pipeline
pipeline = rs.pipeline()

# Start streaming
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)


while True:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()


    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    cv2.imshow('Colour ', color_image)
    cv2.imshow('Depth', depth_image)
    cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
