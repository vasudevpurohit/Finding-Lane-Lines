### FINDING LANE LINES ON THE ROAD

The aim of this project is to find the lane lines on the road using the canny edge detector and Hough Line Transforms.

# Following are the files & folders contained in this repository:
---------------------------------------------------------------

1. The main code is contained in 'Main_code.py'
2. All the test videos on which the code has been run are contained in 'test_videos'
3. The output obtained is stored in 'test_videos_output'
4. 'Project_Writeup.docx' contains the pipeline for the code, shortcomings, and the ways to improve the pipeline.


# Running the code:
-------------------

To run the code on different images, you would have to change the name of the input video on which the code needs
to be run, and the name of the output video to which the appended video will be stored.

Hence, you have to change the following lines of code accordingly,

1. To run the code on different input videos, ref line 24:
		cap = cv2.VideoCapture('test_videos/<filename>')

2. To store the corresponding appended videos, ref line 27:
		out = cv2.VideoWriter('test_videos_output/<filename>', fourcc, 25, (960,  540))
