#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

const int DILATION_SIZE = 10;


int main(int argc, char** argv)
{

VideoCapture cap("red_test.mp4"); // open the default camera

if(!cap.isOpened()) // check if we succeeded
{
    return -1;
}

while(true)
{

    cv::Mat frame;
    //Take from the capture video buffer and feed it into the Mat frame
    cap >> frame;
    if(frame.empty())
    {
      return -1;
    }
    //This medianBlur is just to reduce noise. Because the internet told me to. 
    cv::medianBlur(frame,frame,3);
    imshow("source", frame);
    cv::Mat hsv_image;
    //Turning the original image into an hsv image because again, the internet told me to
    //(But for real, this makes it possible to threshold out the roombas by the red and green switches on top of them)
    cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

    // Threshold the HSV image, keep only the red pixels
    cv::Mat lower_red_hue_range, upper_red_hue_range;
    cv::Mat red_hue_image;
    cv::Mat green_hue_range;
    cv::inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 200, 255), lower_red_hue_range);
    cv::inRange(hsv_image, cv::Scalar(160, 100, 100), cv::Scalar(179, 200, 255), upper_red_hue_range);

    //Now keep only the green pixels from the source image
    cv::inRange(hsv_image, cv::Scalar(55, 50, 20), cv::Scalar(70, 70, 50), green_hue_range); //(98,51,23) (147,54,40) (80,54,31) (104,45,23)

    //Do another blur to reduce noise (and again, the internet told me to and it seems to work)
    cv::GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);

    cv::Mat all_roombas;
    //Add the two red ranges
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

    //Add the green range to the result of the above statement
    cv::addWeighted(red_hue_image, 1.0, green_hue_range, 1.0, 0.0, all_roombas);

    Mat dst, cdst;
    //Getting the "element" allows us to dilate the image
    Mat element = getStructuringElement( MORPH_RECT, Size( 2*DILATION_SIZE + 1, 2*DILATION_SIZE+1 ), Point( DILATION_SIZE, DILATION_SIZE ) );
    //This makes lines thicker so that we get better at finding bigger contours (the end result I was going for here was fewer contours, and it worked)
    dilate(all_roombas, all_roombas, element);
    vim -r /home/ritesh/Documents/iarc/src/iarc7_vision/.git/COMMIT_EDITMSG

    //This takes the thresholded image and finds the outline of the objects within the image
    //The outline allows us to use findContours to get lines surrounding the blobs
    Canny(all_roombas, dst, 50, 200, 3);  
    vector<vector<Point>> contours;
    //hierarchy is just here in case I need it later (might be useful for separating roombas that are close to each other)
    vector<Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // Use the Hough transform to detect circles in the combined threshold image
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(all_roombas, circles, CV_HOUGH_GRADIENT, 1, all_roombas.rows/8, 100, 20, 0, 0);
  
/*    if(circles.size() == 0) 
        {
            cout << "Killed with fire";
            std::exit(-1);
        }
    for(size_t current_circle = 0; current_circle < circles.size(); ++current_circle) {
        cv::Point center(std::round(circles[current_circle][0]), std::round(circles[current_circle][1]));
        int radius = std::round(circles[current_circle][2]);

        cv::circle(dst, center, radius, cv::Scalar(0, 255, 0), 5);
        cout << "Test\n";
    }*/

    
    int max_x = 0;
    int max_y = 0;
    int min_x = 2500;
    int min_y = 2500;
    int j = 0;
    vector<Rect> Objects;

    for(int i = 0; i < contours.size(); i++)
    {

        Rect Contourbound = boundingRect(contours[i]);

        //cv::rectangle(dst, Contourbound , Scalar( 100, 100, 100 ), 3 , CV_AA);

        //Contourbound x,y makes up the top left corner of the rectangle 
        min_x = std::min(Contourbound.x, min_x); 
        max_x = std::max(Contourbound.x + Contourbound.width, max_x); 
        max_y = std::max(Contourbound.y + Contourbound.height, max_y);
        min_y = std::min(Contourbound.y, min_y);

        cout << Contourbound.x << "," << Contourbound.y << "," << Contourbound.x + Contourbound.width << "," << Contourbound.y + Contourbound.height << "\n";

        //This is an attempt to separate contours by color (however, it fails horribly)
        if(i % 3 == 1)
        {
            drawContours(dst, contours, i, Scalar(120,200,50), 2);
        }

        if(i % 3 == 2)
        {
            drawContours(dst, contours, i, Scalar(120,200,100), 2);
        }

        if(i % 3 == 0)
        {
            drawContours(dst, contours, i, Scalar(120,200,150), 2);
        }

    }



    cv::rectangle(dst, Point2i(min_x, max_y), Point2i(max_x, min_y) , Scalar( 80, 180, 180 ), 6 , CV_AA);

    if(dst.empty())
    {
        cout << "This ain't working :( \n";
        //This really means that the image was empty and I completely failed :( )
        //This is a holdover from when I tried using HoughCircles on the output of the canny image.
        //Though I am no longer trying the HoughCircles method (but now I am rethinking it as I wrtie this comment)
        //This conditional statement is still useful in some way. 
    }
    else{
    imshow("rectangles", dst);
    }

    if(waitKey(30) >= 0) break;

}


return 0;
}
