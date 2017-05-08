import cv2
import numpy as np

DILATION_SIZE = 10 #This is a constant variable. Never write to it. 
                   #In C++ you would just declare it to be a const, but Python does not support that. 


cap = cv2.VideoCapture("green_test.mp4"); #open the default camera

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.medianBlur(frame, 3)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red_hue_range = hsv_image
    upper_red_hue_range = hsv_image
    green_hue_range = hsv_image # You may wonder why I am setting them equal to frame. That is just so
                            # that I know that they are all the same datatype. I was initially 
                            # using blank numpy arrays and they were failing. 

    lower_red_hue_range = cv2.inRange(hsv_image, np.array([0, 80, 100]) , np.array([10, 255, 255]))
    upper_red_hue_range = cv2.inRange(hsv_image, np.array([160, 80, 100]) , np.array([180, 255, 255]))
    green_hue_range = cv2.inRange(hsv_image, np.array([55, 50, 50]) , np.array([70, 255, 255]))

    all_roombas =  cv2.addWeighted(lower_red_hue_range, 1, upper_red_hue_range, 1, 0)

    all_roombas = cv2.addWeighted(all_roombas, 1, green_hue_range, 1, 0)

    cv2.imshow('all', all_roombas)

    kernel = np.ones((5,5),np.uint8)
    # Apparently the python version of OpenCV has some different options for dilation. Ended 
    # up getting a little distracted messing around with them
    all_roombas = cv2.dilate(all_roombas,kernel,iterations = 1)
    opening = cv2.morphologyEx(all_roombas, cv2.MORPH_OPEN, kernel)

    all_roombas = cv2.Canny(all_roombas, 50, 200)

    contours, hierarchy = cv2.findContours(all_roombas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_x = 0;
    max_y = 0;
    min_x = 2500;
    min_y = 2500;

    for contour in contours:

        x,y,w,h = cv2.boundingRect(contour)

        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y)
        max_y = max(max_y, y + h)


    cv2.rectangle(all_roombas ,(min_x, min_y),(max_x, max_y),(100,0,100),3)

    cv2.imshow('output', all_roombas)
    #cv2.imshow('output2', opening)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
