import numpy as np
import cv2

#creating a function to convert the RGB image to Grayscale

def RGBtoGray(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

#creating a function for Smoothing the Grayscale image
def smoothing(image,kernel_size):
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

#this creates a canny edge detector image
def detected_edges(image,low_threshold,high_threshold):
    return cv2.Canny(image,low_threshold,high_threshold)

#creating a mask to just run the Canny Edge detector in the region of the image where we have the lanes
def mask_image(image,vertices,cvtToColor):
    a = np.zeros_like(image)*0
    b = cv2.fillPoly(a,vertices,cvtToColor)
    return cv2.bitwise_and(image,b)


cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    ##Main Code
    image1_gray = RGBtoGray(frame)
    image1_gray_smoothed = smoothing(image1_gray,7)

    edges_image1 = detected_edges(image1_gray_smoothed,20,75)

## all of this to find lines on the right side of the image
    vertices = np.array([[[(480,539),(900,539),(480,285),(490,285)]],[[(500,539),(80,539),(490,285),(500,285)]]])
    edges_r = mask_image(edges_image1,(vertices[0]),255)
    edges_l = mask_image(edges_image1,(vertices[1]),255)

    d=1
    theta = np.pi/180
    max_length = 50
    min_gap = 10
    pixel_no = 10      #no of pixels that will be used to plot the lines
    alpha = 1.5
    beta = 0.8
    gamma = 0

    HL_r = cv2.HoughLinesP(edges_r,d,theta,max_length,min_gap) #right lane
    HL_l = cv2.HoughLinesP(edges_l,d,theta,max_length,min_gap) #left lane

    s_r = HL_r.shape[0]
    s_l = HL_l.shape[0]
    b = np.zeros_like(frame)*0

##to detect the right-lanes

    x1 = np.ones((s_r),dtype='int32')
    x2 = np.ones((s_r),dtype='int32')
    y1 = np.ones((s_r),dtype='int32')
    y2 = np.ones((s_r),dtype='int32')

    for i in range(s_r):
        j=0
        x1[i] = (HL_r[i][0][j])
        x2[i]= (HL_r[i][0][j+2])
        y1[i] = (HL_r[i][0][j+1])
        y2[i] = (HL_r[i][0][j+3])


    X1 = np.hstack((x1,x2))
    Y1 = np.hstack((y1,y2))
    lane_right = np.polyfit(X1,Y1,1)
    x1_new = np.linspace(500,900,500)
    y1_new = np.polyval(lane_right,x1_new)
    y1_min = int(np.min(y1_new))
    y1_max = int(np.max(y1_new))
    point1_1 = (500,y1_min)
    point2_1 = (900,y1_max)
    lines_right = cv2.line(b,point1_1,point2_1,(0,0,255),pixel_no)
    final_right = cv2.addWeighted(lines_right,alpha,frame,beta,gamma)

##to detect the left-lanes

    x3 = np.ones((s_l),dtype='int32')
    x4 = np.ones((s_l),dtype='int32')
    y3 = np.ones((s_l),dtype='int32')
    y4 = np.ones((s_l),dtype='int32')

    for i in range(s_l):
        j=0
        x3[i] = (HL_l[i][0][j])
        x4[i]= (HL_l[i][0][j+2])
        y3[i] = (HL_l[i][0][j+1])
        y4[i] = (HL_l[i][0][j+3])


    X2 = np.hstack((x3,x4))
    Y2 = np.hstack((y3,y4))
    lane_left = np.polyfit(X2,Y2,1)
    x2_new = np.linspace(80,450,500)
    y2_new = np.polyval(lane_left,x2_new)
    y2_min = int(np.min(y2_new))
    y2_max = int(np.max(y2_new))
    point1_2 = (80,y2_max)
    point2_2 = (450,y2_min)
    lines_left = cv2.line(b,point1_2,point2_2,(0,0,255),pixel_no)
    frame = cv2.addWeighted(lines_left,alpha,frame,beta,gamma)


    cv2.imshow('frame',frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





