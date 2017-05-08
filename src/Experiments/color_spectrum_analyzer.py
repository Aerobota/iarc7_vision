import cv2
import numpy as np

color_spect = cv2.imread("spectrum_chart.jpg", 1)

while(1):
    hsv_image = cv2.cvtColor(color_spect, cv2.COLOR_BGR2HSV)

    lower_red_hue_range = hsv_image
    upper_red_hue_range = hsv_image
    green_hue_range = hsv_image # You may wonder why I am setting them equal to frame. That is just so
                            # that I know that they are all the same datatype. I was initially 
                            # using blank numpy arrays and they were failing. 

    lower_red_hue_range = cv2.inRange(hsv_image, np.array([0, 80, 100]) , np.array([10, 255, 255]))
    upper_red_hue_range = cv2.inRange(hsv_image, np.array([160, 80, 100]) , np.array([179, 255, 255]))
    green_hue_range = cv2.inRange(hsv_image, np.array([55, 50, 50]) , np.array([70, 200, 255]))

    all_roombas =  cv2.addWeighted(lower_red_hue_range, 1, upper_red_hue_range, 1, 0)

    all_roombas = cv2.addWeighted(all_roombas, 1, green_hue_range, 1, 0)

    cv2.imshow('blobs',  green_hue_range)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
