#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "pic1.png";

    cv::Mat src = imread(filename, IMREAD_GRAYSCALE);
    Mat roomba_canny, hsv_image;
    
    cout << "test" << std::endl;

    // Basing my usage off of the example here (https://github.com/opencv/opencv/blob/master/samples/gpu/generalized_hough.cpp)
    // And what I could glean from reading the generalHough code on the 2.4 branch in the opencv repository.

    int method = GHT_POSITION + GHT_ROTATION + GHT_SCALE;

    Ptr< GeneralizedHough > generalHough =  generalHough->create(method);
    generalHough->set("angleThresh", 10000);
    generalHough->set("scaleThresh", 1000);
    generalHough->set("posThresh", 100);
    generalHough->set("maxSize", 1000);
    generalHough->set("levels", 360);
    generalHough->set("minAngle", 0);
    generalHough->set("maxAngle", 360);
    generalHough->set("angleStep", 1);
    generalHough->set("minScale", 0.5);
    generalHough->set("maxScale", 2);
    generalHough->set("dp", 2);
    
    // generalHough->set("levels", 100);

    cv::Mat roomba_template = imread("templ.png", IMREAD_GRAYSCALE);

    cout << roomba_template.type()  << std::endl;

    generalHough->setTemplate(roomba_template);

    vector<Vec4f> roomba_positions;
    vector<Vec3i> roomba_votes;

    generalHough->detect(src, roomba_positions);

    if(roomba_positions.size() == 0)
    {
        cout << "nooooo" << std::endl;
    }

    for (int i = 0; i < roomba_positions.size(); i++)
    {
        cv::rectangle(src, Point2i(roomba_positions[i][0]-50, roomba_positions[i][1]-50), Point2i(roomba_positions[i][0] + 50, roomba_positions[i][1] + 50) , Scalar( 100, 180, 180 ), 6 , CV_AA);
    }

    imshow("Output", src);
    waitKey();

    return 0;

}
