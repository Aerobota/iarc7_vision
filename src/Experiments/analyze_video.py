import cv2
import numpy as np

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

        s = cv2.inRange(s, 160, 180)

        cv2.imshow('saturation filtered', s)

        s = cv2.Canny(s, 50, 200)

        cv2.imshow('Canny on saturation', s)

        lower_red_hue_range = hsv_image
        upper_red_hue_range = hsv_image
        green_hue_range = hsv_image # You may wonder why I am setting them equal to frame. That is just so
                                # that I know that they are all the same datatype. I was initially 
                                # using blank numpy arrays and they were failing. 

        lower_red_hue_range = cv2.inRange(hsv_image, np.array([0, 80, 100]) , np.array([10, 255, 255]))
        upper_red_hue_range = cv2.inRange(hsv_image, np.array([160, 80, 100]) , np.array([180, 255, 255]))
        green_hue_range = cv2.inRange(hsv_image, np.array([55, 50, 30]) , np.array([70, 255, 255]))

        all_roombas =  cv2.addWeighted(lower_red_hue_range, 1, upper_red_hue_range, 1, 0)

        all_roombas = cv2.addWeighted(all_roombas, 1, green_hue_range, 1, 0)

        cv2.imshow('all', all_roombas)

        kernel = np.ones((5,5),np.uint8)

        # Apparently the python version of OpenCV has some different options for dilation. 
        # Messed around with them a little and I think I got pretty close to the C++ version
        all_roombas = cv2.dilate(all_roombas,kernel,iterations = 1)
        opening = cv2.morphologyEx(all_roombas, cv2.MORPH_OPEN, kernel)

        all_roombas = cv2.Canny(all_roombas, 50, 200)

        # cv2.Laplacian(all_roombas, 165, all_roombas, kernel_size, scale, delta)

        # cv2.imshow('laplace', all_roombas)

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

        contour_list = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 250:

                contour_list.append(contour)


        cv2.drawContours(all_roombas, contour_list,  -1, (255,0,0), 2)
        print len(contour_list)

        cv2.imshow('output', all_roombas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if len(contour_list) == 2:
            
        #     x,y,w,h = cv2.boundingRect(contour_list[0])

        #     track_window = (x,y,x+w,y+h)
        #     ret, track_window = cv2.CamShift(all_roombas,track_window, term_crit)
        #     cv2.ellipse(all_roombas, track_window, np.array([0, 0, 100]))
        #     roomba_found = True

        # cv2.imshow('output', all_roombas)


    # elif roomba_found is True:
    #     ret, frame = cap.read()

    #     ret, track_window = cv2.CamShift(frame,track_window, term_crit)

    #     cv2.ellipse(all_roombas, track_window, np.array([0, 0, 100])
    #     cv2.imshow('output', all_roombas)


        #cv2.imshow('output2', opening)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
