// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop

#include <cmath>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include "iarc7_vision/GridLineEstimator.hpp"

static void drawLines(const std::vector<cv::Vec2f>& lines, cv::Mat image) {
	for (auto& line : lines)
	{
		float rho = line[0], theta = line[1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 10000*(-b));
		pt1.y = cvRound(y0 + 10000*(a));
		pt2.x = cvRound(x0 - 10000*(-b));
		pt2.y = cvRound(y0 - 10000*(a));
		cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 3, CV_AA);
	}
}

namespace iarc7_vision {

	GridLineEstimator::GridLineEstimator() {
		ros::NodeHandle local_nh ("grid_line_estimator");
		image_pub_ = local_nh.advertise<sensor_msgs::Image>("lines_image", 10);
	}

	void GridLineEstimator::update(const cv::Mat& image) {
		//////////////////////////////////////////////////
		// extract lines
		//////////////////////////////////////////////////
		std::vector<cv::Vec2f> lines;
		getLines(lines, image);
		ROS_WARN("%lu", lines.size());
		drawLines(std::vector<cv::Vec2f>(
					lines.begin(),
					lines.begin() + std::min(80ul, lines.size())),
				image);



		//////////////////////////////////////////////////
		// compute tranformation from lines to gridlines
		//////////////////////////////////////////////////
		/*
		   f = transform (1 0 0) from camera frame to level_quad
		 */

		//////////////////////////////////////////////////
		// transform lines into gridlines
		//////////////////////////////////////////////////
		/*
		   for each line:
		   v = (0 sin(theta) cos(theta))
		   global_v = transform v from camera frame to level_quad
		   offset = (0 -r*cos(theta) r*sin(theta))
		   global_offset = transform offset from camera_frame to level_quad
		   ray1 = f + global_offset
		   ray2 = f + global_offset + line_time * global_v

		// think about what happens when these don't intersect
		p1 = intersection of ray1 with the plane z=0
		p2 = intersection of ray2 with the plane z=0
		result = linebetween p1 and p2
		 */

		//////////////////////////////////////////////////
		// cluster gridlines
		//////////////////////////////////////////////////
		/*
		   sort gridlines by angle
		   run k-means with k=2? or run sliding window over list and look for high counts?
		   maybe repeatedly pair each line with the nearest line or nearest cluster?

		   make a group for each direction and throw out outliers (lines more than a constant number of standard deviations away)
		 */

		//////////////////////////////////////////////////
		// return estimate
		//////////////////////////////////////////////////
	}

	void GridLineEstimator::getLines(std::vector<cv::Vec2f>& lines, const cv::Mat& image) {
		cv::Mat image_hsv,
			image_gray,
			image_grad,
			image_blurred,
			image_edges,
			color_mask;
		cv::Mat image_hsv_channels[3];

		double hough_rho_resolution = 0.2;
		double hough_theta_resolution = M_PI / 180;
		int hough_threshold = 30;
		cv::Mat gradient_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

		if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
			ROS_WARN_ONCE("Doing OpenCV operations on CPU");
			cv::Mat	image_sized,
				image_hsv,
				image_blurred,
				mask_saturation,
				mask_value,
				mask,
				grad,
				morphology_buf_1,
				morphology_buf_2,
				lines;
			cv::Mat image_hsv_channels[3];


			cv::Mat image_gray;
			cv::resize(image, image_sized, cv::Size(240, 135));
			cv::cvtColor(image_sized, image_gray, CV_BGRA2GRAY);
			cv::Canny(image_gray, grad, 20000, 25000, 7, true);
			cv::HoughLines(grad,
					lines,
					hough_rho_resolution,
					hough_theta_resolution,
					hough_threshold);

			image_edges = grad;
			//cv::cvtColor(image, image_hsv, CV_BGR2HSV);
			//cv::split(image_hsv, image_hsv_channels);
			//cv::threshold(image_hsv_channels[1],
			//              color_mask,

			//              saturation_threshold,
			//              255,
			//              cv::THRESH_BINARY_INV);
			//cv::erode(color_mask, color_mask, erode_kernel);

			//cv::cvtColor(image, image_gray, CV_BGR2GRAY);
			//cv::GaussianBlur(image_gray, image_blurred, blur_size, blur_sigma);
			//cv::Canny(image_blurred,
			//          image_edges,
			//          canny_low_threshold,
			//          canny_high_threshold,
			//          canny_sobel_size);
			//cv::bitwise_and(color_mask, image_edges, image_edges);
			//cv::HoughLines(image_edges,
			//               lines,
			//               hough_rho_resolution,
			//               hough_theta_resolution,
			//               hough_threshold);
		} else {
			cv::gpu::HoughLinesBuf hough_buf;
			cv::gpu::GpuMat gpu_image,
				gpu_image_sized,
				gpu_image_hsv,
				gpu_image_blurred,
				gpu_mask_saturation,
				gpu_mask_value,
				gpu_mask,
				gpu_grad,
				morphology_buf_1,
				morphology_buf_2,
				gpu_lines;
			cv::gpu::GpuMat gpu_image_hsv_channels[3];

			gpu_image.upload(image);

#if 0
			cv::Size blur_size (7, 7);
			double blur_sigma = 5;
			double saturation_threshold = 80;
			double value_threshold = 200;
			cv::gpu::resize(gpu_image, gpu_image_sized, cv::Size(960, 540));
			cv::gpu::cvtColor(gpu_image_sized, gpu_image_hsv, CV_BGR2HSV);
			cv::gpu::GaussianBlur(gpu_image_hsv,
					gpu_image_blurred,
					blur_size,
					blur_sigma);
			cv::gpu::split(gpu_image_blurred, gpu_image_hsv_channels);
			cv::gpu::threshold(gpu_image_hsv_channels[1],
					gpu_mask_saturation,
					saturation_threshold,
					255,
					cv::THRESH_BINARY_INV);
			cv::gpu::threshold(gpu_image_hsv_channels[2],
					gpu_mask_value,
					value_threshold,
					255,
					cv::THRESH_BINARY);
			cv::gpu::bitwise_and(gpu_mask_saturation, gpu_mask_value, gpu_mask);
			cv::gpu::morphologyEx(gpu_mask,
					gpu_grad,
					cv::MORPH_GRADIENT,
					gradient_kernel,
					morphology_buf_1,
					morphology_buf_2);
#else
			cv::gpu::GpuMat gpu_image_gray;
			cv::gpu::resize(gpu_image, gpu_image_sized, cv::Size(240, 135));
			cv::gpu::cvtColor(gpu_image_sized, gpu_image_gray, CV_BGRA2GRAY);
			cv::gpu::Canny(gpu_image_gray, gpu_grad, 20000, 25000, 7, true);
#endif
			cv::gpu::HoughLines(gpu_grad,
					gpu_lines,
					hough_buf,
					hough_rho_resolution,
					hough_theta_resolution,
					hough_threshold);

			gpu_grad.download(image_edges);
			cv::gpu::HoughLinesDownload(gpu_lines, lines);
		}

		cv_bridge::CvImage cv_image { std_msgs::Header(), sensor_msgs::image_encodings::MONO8, image_edges };
		//    image_pub_.publish(cv_image.toImageMsg());
		cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display window", image_edges );                   // Show our image inside it.

		cv::waitKey(0);
	}

} // namespace iarc7_vision
