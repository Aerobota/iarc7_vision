#ifndef IARC7_VISION_GRIDLINE_ESTIMATOR_HPP_
#define IARC7_VISION_GRIDLINE_ESTIMATOR_HPP_

#include <opencv2/core/core.hpp>
#include <ros/ros.h>

namespace iarc7_vision {

class GridLineEstimator {
  public:
    GridLineEstimator();
    void update(const cv::Mat& image);

  private:
    void getLines(std::vector<cv::Vec2f>& lines, const cv::Mat& image);

    ros::Publisher image_pub_;
};

} // namespace iarc7_vision

#endif // include guard
