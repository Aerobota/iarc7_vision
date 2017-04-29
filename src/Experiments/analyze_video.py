import cv2
import numpy as np

DILATION_SIZE = 10 #This is a constant variable. Never write to it. 
                   #In C++ you would just declare it to be a const, but Python does not support that. 


cap = cv2.VideoCapture("red_test.mp4"); #open the default camera

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

    #So here are the alternatives 1. I include the above 3 lines, and lower_red_hue just ends up being equal to the hsv_image
    #2. I don't and Python complains that "lower_red_hue_range is not defined".

    cv2.inRange(hsv_image, np.array([0, 100, 100]) , np.array([10, 200, 255]) , lower_red_hue_range)
    cv2.inRange(hsv_image, np.array([160, 100, 100]) , np.array([179, 200, 255]), upper_red_hue_range)

    lower_red_hue_range =  cv2.addWeighted(lower_red_hue_range, 1, upper_red_hue_range, 1, 0)

    cv2.imshow('output', lower_red_hue_range)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
