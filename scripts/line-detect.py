import cv2
import numpy as np
import time

cap = cv2.VideoCapture("../Default Project.mp4")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = frame

    #image = cv2.imread("../grid_sample_2.png", cv2.CV_LOAD_IMAGE_COLOR)

    # Just downsampling so I can see the image on my screen. If removed morphology kernel sizes will need to go up.
    image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #image = cv2.GaussianBlur(image, (7,7), 5)

    cv2.imshow("raw", image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7,7), 5)

    # define range of blue color in HSV

    # Wide HSV slice
    low = np.array([0,0,200])
    up = np.array([180,80,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, low, up)

    cv2.imshow("mask", mask)

    # Morphology
    # Spec removal

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #erosion = cv2.erode(mask,kernel,iterations = 10)
    #cv2.imshow("erosion", erosion)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #dilation = cv2.dilate(erosion,kernel,iterations = 10)
    #cv2.imshow("dilation", dilation)
    # Use the mask on the original image
    
    #masked_image = cv2.bitwise_and(image, image, mask=erosion)
    #cv2.imshow("masked_image", masked_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("gradient", gradient)

    #edges = cv2.Canny(mask,10,100,apertureSize = 3)
    #cv2.imshow("edges", edges)

    # Now run line detector

    lines = cv2.HoughLines(gradient,1,np.pi/180,200)

    wait = False
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    else:
        wait = False


    cv2.imshow("with lines", image)

    if wait == True:
        for i in range(0, 30):
            key = cv2.waitKey(0)
    key = cv2.waitKey(0)

    # idk its the number that key equals too and I didn't want to figure it out
    if key == 1048689:
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
