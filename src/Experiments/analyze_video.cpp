#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iomanip>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

const int DILATION_SIZE = 10;


int main(int argc, char** argv)
{

cout << "test" << std::endl;

VideoCapture cap("green_test.mp4"); // open the default camera

cout << "test" << std::endl;

Ptr< GeneralizedHough > generalHough =  cv::GeneralizedHough::create(GHT_POSITION | GHT_SCALE | GHT_ROTATION);

cv::Mat roomba_template = imread("finalcrop.png", 0);

cout << roomba_template.type()  << std::endl;

roomba_template.convertTo(roomba_template, CV_8UC1);

cout << roomba_template.type()  << std::endl;

generalHough->setTemplate(roomba_template, 100);

if(!cap.isOpened()) // check if we succeeded
{
    return -1;
}

    while(true)
    {

        cv::Mat frame, hsv_image;
        //Take from the capture video buffer and feed it into the Mat frame
        cap >> frame;
        if(frame.empty())
        {
          return -1;
        }
        //This medianBlur is just to reduce noise. Because the internet told me to. 
        cv::medianBlur(frame,frame,3);

        imshow("source", frame);
        
        cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);
        cv::Mat hsv[3];
        cv::split(hsv_image, hsv);

        //Turning the original image into an hsv image because again, the internet told me to
        //(But for real, this makes it possible to threshold out the roombas by the red and green switches on top of them)

        cv::Mat saturation = hsv[2];

        cv::inRange(saturation, 160, 180, saturation);

        std::vector<cv::Vec2i> roomba_positions;
        std::vector<cv::Vec2i> roomba_votes;

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        cout << "before detect" << std::endl;

        generalHough->detect(saturation, roomba_positions, roomba_votes);

        
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "GHT took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << "us." << std::endl;

        if(roomba_positions.size() > 0)
        {
            cv::rectangle(saturation, Point2i(roomba_positions[0][0], roomba_positions[0][1]), Point2i(roomba_positions[0][0] + 50, roomba_positions[0][1] + 50) , Scalar( 80, 180, 180 ), 6 , CV_AA);
        }

        cout << "after detect" << std::endl;

        imshow("saturation", saturation);

    }


return 0;
}
