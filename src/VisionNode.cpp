// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop

#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

#include <sensor_msgs/image_encodings.h>

#include "iarc7_vision/GridLineEstimator.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision");

    ros::NodeHandle nh;

    iarc7_vision::GridLineEstimator gridline_estimator;

    std::function<void(const sensor_msgs::Image::ConstPtr&)> handler =
        [&](const sensor_msgs::Image::ConstPtr& message) {
            ROS_WARN("Updating");
            gridline_estimator.update(cv_bridge::toCvShare(message)->image);
        };

    image_transport::ImageTransport image_transporter{nh};
    ros::Subscriber sub = nh.subscribe("/bottom_image_raw/image", 100, &std::function<void(const sensor_msgs::Image::ConstPtr&)>::operator(), &handler);

    ros::Rate rate (100);
    while (ros::ok() && ros::Time::now() == ros::Time(0)) {
        // wait
        ros::spinOnce();
    }

//    cv::VideoCapture cap ("/home/aaron/Videos/Default Project.mp4");
//    cv::Mat image;

    while (ros::ok())
    {
      //  cap >> image;
      //  image = cv::imread("/home/aaron/Pictures/grid_sample_2.png");
      //  gridline_estimator.update(image);

        ros::spinOnce();
        rate.sleep();
    }

    // All is good.
    return 0;
}
