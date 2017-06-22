#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{

    const char* filename = argc >= 2 ? argv[1] : "testGeneralHough.png";

    cv::Mat src = imread(filename, IMREAD_GRAYSCALE);
    Mat roomba_canny, hsv_image;
    
    cout << "test" << std::endl;

    // Basing my usage off of the example here (https://github.com/opencv/opencv/blob/master/samples/gpu/generalized_hough.cpp)
    // And what I could glean from reading the generalHough code on the 2.4 branch in the opencv repository.

    Ptr< GeneralizedHough > generalHough =  cv::GeneralizedHough::create(GHT_POSITION | GHT_ROTATION | GHT_SCALE);

    cv::Mat roomba_template = imread("blackOutline.png", IMREAD_GRAYSCALE);

    cout << roomba_template.type()  << std::endl;

    roomba_template.convertTo(roomba_template, CV_8UC1);

    cout << roomba_template.type() << std::endl;

    generalHough->setTemplate(roomba_template, 100);

    vector<Vec4f> roomba_positions;
    vector<Vec3i> roomba_votes;

    // Canny(src, roomba_canny, 100, 50);

    // imshow("canny output", roomba_canny);

    generalHough->detect(src, roomba_positions, roomba_votes, 100);

    if(roomba_positions.size() == 0)
    {
        cout << "nooooo" << std::endl;
    }

    for (int i = 0; i < roomba_positions.size(); i++)
    {
        cv::rectangle(src, Point2i(roomba_positions[i][0], roomba_positions[i][1]), Point2i(roomba_positions[i][0] + 50, roomba_positions[i][1] + 50) , Scalar( 80, 180, 180 ), 6 , CV_AA);
    }

    imshow("Output", src);
    waitKey();

    return 0;

}
