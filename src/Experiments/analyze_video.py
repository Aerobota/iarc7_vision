import cv2
import numpy as np
#test commit

DILATION_SIZE = 10 #This is a constant variable. Never write to it. 
                   #In C++ you would just declare it to be a const, but Python does not support that. 

kernel_size = 3
scale = 1
delta = 0
track_window = None
ret = None
roomba_found = False
# ddepth = CV_16S;
cap = cv2.VideoCapture("green_test.mp4"); #open the default camera

while(True):

    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    if roomba_found is False:

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.medianBlur(frame, 3)    
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h,s,v = cv2.split(hsv_image)

        cv2.imshow('original', frame)

        cv2.imshow('saturation', s)

        cv2.imshow('value', v)

        s = cv2.inRange(s, 190, 210)

        #s = cv2.Canny(s, 10, 50)

        #cv2.imshow('Canny on saturation', s)

        # lower_red_hue_range = hsv_image
        # upper_red_hue_range = hsv_image
        # green_hue_range = hsv_image # You may wonder why I am setting them equal to frame. That is just so
        #                         # that I know that they are all the same datatype. I was initially 
        #                         # using blank numpy arrays and they were failing. 

        # lower_red_hue_range = cv2.inRange(hsv_image, np.array([0, 80, 100]) , np.array([10, 255, 255]))
        # upper_red_hue_range = cv2.inRange(hsv_image, np.array([160, 80, 100]) , np.array([180, 255, 255]))
        # green_hue_range = cv2.inRange(hsv_image, np.array([55, 50, 30]) , np.array([70, 255, 255]))

        # all_roombas =  cv2.addWeighted(lower_red_hue_range, 1, upper_red_hue_range, 1, 0)

        # all_roombas = cv2.addWeighted(all_roombas, 1, green_hue_range, 1, 0)

        # all_roombas = cv2.addWeighted(all_roombas, 1, s, 1, 0)

        # # kernel = np.ones((5,5),np.uint8)

        # all_roombas = cv2.erode(all_roombas, kernel, iterations=2)
        # # Apparently the python version of OpenCV has some different options for dilation. 
        # # Messed around with them a little and I think I got pretty close to the C++ version
        # all_roombas = cv2.dilate(all_roombas,kernel,iterations = 3)

        # cv2.imshow('inRange + dilate', all_roombas)

        # all_roombas = cv2.Canny(all_roombas, 50, 200)

        #  # detect circles in the image
        # circles = cv2.HoughCircles(all_roombas, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
         
        # # ensure at least some circles were found
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")
         
        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(all_roombas, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(all_roombas, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
         
        # # cv2.imshow('all', all_roombas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
