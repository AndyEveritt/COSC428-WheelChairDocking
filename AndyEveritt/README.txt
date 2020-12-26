My code is trying to find a docking location based on finding the front edge of the top of a table or desk. It also identifies any obstructions beneath the desk, and the distance and angle of the camera relative to the desk.

GETTING STARTED:
To run this code your have to set up the appropriate libraries. It uses opencv 3.4.1,  pyrealsense2, numpy, and pathlib libraries. The realsense libraries come with the realsense 2.0 SDK. The project was coded using python 3.6 in the Spyder IDE that comes packaged with Anaconda.


To run my program an D435 camera needs to be plugged in, or a ".bag" file produced by a D435 camera needs to be provided. The intel realsense viewer can be used to produce a .bag file for testing.

The main file to run the code is "Desk_detection.py"
	Each section of the code is separated by a header briefly explaining what it does. Individual subsections and lines of code are also commented to give a more in-depth explaination.

To stop the code after it has begun running, press "Q".




HOW TO USE:
-Open Python IDE
-Set the global variables (near the top of the code) so the video source is correct.
	-"Live" controls whether a stream from the connected D435 camera is used or if the .bag file found at the "./data_path/name_path" is used.
	-"fps" only affects the live stream
	-"resolution" contains all the resolutions that can be used for both the colour and depth sensors on the D435
		-To maintain a decent fps it is suggested to use 480x640p
	-"obstacles" controls whether obstacles are detected beneath the desk.
		-Disabling slightly improves fps
	-"show_all" will display all the major steps of the code using imshow windows.
		-Requires a decent graphics card to get good fps due to the number of windows.
	-"save_vid" will save the colour stream as an ".avi" file if True.
	-All the "Error frames" variables control whether specific frames that cause a part of the code to fail are saved and where they are saved to.
		-"error_index" should always be 0.
	-"buffer" sets the number of frames used in the circular buffer to have some rudamentary tracking.
		-"Buffer_index" should always be 0.
	-"maxRange_multi"/"minRange_multi" are multipliers for the average desk distance when thresholding occurs to ensure the desk is detected.
	-"chair_width" is the width of the wheelchair in meters.
-Run "Desk_detection.py"
-Press "Q" to stop.
	

LEGEND:
Various rectangles are overlaid onto the colour and depth frames. The colour idicates what object has been identified.
-Blue: Front edge of the desk
-Green: Area beneath the desk
-Yellow: Bounding rectangle of obstacles beneath the desk
-Red (thin): Contours of obstacles beneath the desk
-White (bold): Largest docking location that is wide enough for the wheelchair
-Red (bold): Largest docking location that is not wide enough for the wheelchair


THE CODE WILL:
Get a color frame and a depth frame. The colour frame is only used to display the output on so it is more user friendly.

Find the front of the desk using the closest averaged horizontal depth value in the frame.
	-This assumption works most of the time assuming the desk is the primary object in view.

Thresholding, contours and morpholgical operations are then used in various combinations to find obstacles and improve the recognition of the edge of the desk.

3D pixel to point deprojection allows the conversion of pixel coordinates into real world coordinates in meters relative to the camera. This depends on the camera intrinsics which are easily found using the pyrealsense2 library.
