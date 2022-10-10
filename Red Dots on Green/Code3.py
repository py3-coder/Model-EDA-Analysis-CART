import cv2
import numpy as np
import os

cap =cv2.VideoCapture(os.path.join('Video 3.mp4'))   #Requesting the input from the video file

# Properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video Writer 
video_writer = cv2.VideoWriter(os.path.join('Output1.MP4'), cv2.VideoWriter_fourcc('P','I','M','1'), fps, (width, height))



# Loop through each frame
for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    _, img=cap.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                     

    l_b = np.array([40, 100, 100])           #setting the upper and lower bound for green color                                                                                                   
    u_b = np.array([80, 255, 255])                                              

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(img, img, mask=mask)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT,1,10,
                            param1=50,param2=12,minRadius=0,maxRadius=40)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),3,(0,0,255),3)


    cv2.imshow("video", img)   #Displaying the output to user
        
    # Write out frame 
    video_writer.write(img)

    key=cv2.waitKey(1) 
    if key & 0xFF==ord('q'):   # here we are specifying the key which will stop the loop and stop all the processes going
        break
        
cap.release()
cv2.destroyAllWindows()

# Release video writer
video_writer.release()