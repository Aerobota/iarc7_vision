#include "iarc7_vision/GridLineEstimator.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ctime>
std::string type2str(int type) {
    std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


void drawLines(std::vector<cv::Vec2f> lines, cv::Mat image) {
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
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

void GridLineEstimator::update() {
    //////////////////////////////////////////////////
    // extract lines
    //////////////////////////////////////////////////
    cv::Mat image,
            image_hsv,
            image_gray,
            image_blurred,
            image_edges,
            color_mask;
    cv::Mat image_hsv_channels[3];
    std::vector<cv::Vec2f> lines;

clock_t t;

    cv::Size blur_size (9, 9);
    double blur_sigma = 5;
    double saturation_threshold = 40;
    double canny_low_threshold = 1000;
    double canny_high_threshold = 1000;
    int canny_sobel_size = 5;
    double hough_rho_resolution = 1;
    double hough_theta_resolution = 0.01;
    int hough_threshold = 200;
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    // cv::VideoCapture cap ("/home/aaron/Videos/Default Project.mp4");

    //while (true) {
      //  cap >> image;
        image = cv::imread("/home/ubuntu/grid_sample_2.png");

        if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
            cv::cvtColor(image, image_hsv, CV_BGR2HSV);
            cv::split(image_hsv, image_hsv_channels);
            cv::threshold(image_hsv_channels[1],
                          color_mask,
                          saturation_threshold,
                          255,
                          cv::THRESH_BINARY_INV);
            cv::erode(color_mask, color_mask, erode_kernel);

            cv::cvtColor(image, image_gray, CV_BGR2GRAY);
            cv::GaussianBlur(image_gray, image_blurred, blur_size, blur_sigma);
            cv::Canny(image_blurred,
                      image_edges,
                      canny_low_threshold,
                      canny_high_threshold,
                      canny_sobel_size);
            cv::bitwise_and(color_mask, image_edges, image_edges);
            cv::HoughLines(image_edges,
                           lines,
                           hough_rho_resolution,
                           hough_theta_resolution,
                           hough_threshold);
        } else {
            cv::gpu::CannyBuf canny_buf;
            cv::gpu::HoughLinesBuf hough_buf;
            cv::gpu::GpuMat gpu_image_hsv,
                            gpu_color_mask,
                            gpu_color_mask_2,
                            gpu_mask_buf,
                            gpu_image_gray,
                            gpu_image_blurred,
                            gpu_image_grad_x,
                            gpu_image_grad_y,
                            gpu_image_grad_x_U,
                            gpu_image_grad_y_U,
                            gpu_image_edges,
                            gpu_lines;
            cv::gpu::GpuMat gpu_image_hsv_channels[3];
cv::gpu::GpuMat gpu_image;

            for (int i = 0; i < 2; i++) {
t = clock();
cv::gpu::CudaMem image_cudamem(image, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
gpu_image = image_cudamem.createGpuMatHeader();
//                gpu_image.upload(image);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;

t = clock();
                cv::gpu::cvtColor(gpu_image, gpu_image_hsv, CV_BGR2HSV);
                cv::gpu::split(gpu_image_hsv, gpu_image_hsv_channels);
                cv::gpu::threshold(gpu_image_hsv_channels[1],
                                   gpu_color_mask_2,
                                   saturation_threshold,
                                   255,
                                   cv::THRESH_BINARY_INV);
                cv::gpu::erode(gpu_color_mask_2, gpu_color_mask, erode_kernel, gpu_mask_buf);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;

t = clock();
                cv::gpu::cvtColor(gpu_image, gpu_image_gray, CV_BGR2GRAY);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;
t = clock();
                cv::gpu::GaussianBlur(gpu_image_gray, gpu_image_blurred, blur_size, blur_sigma);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;
t = clock();
                // cv::gpu::Canny(gpu_image_blurred,
                //                canny_buf,
                //                gpu_image_edges,
                //                canny_low_threshold,
                //                canny_high_threshold,
                //                canny_sobel_size);
                cv::gpu::Sobel(gpu_image_blurred,
                               gpu_image_grad_x,
                               CV_16S,
                               1,
                               0,
                               canny_sobel_size);
                cv::gpu::abs(gpu_image_grad_x, gpu_image_grad_x);
                gpu_image_grad_x.convertTo(gpu_image_grad_x_U, CV_16U);
                cv::gpu::Sobel(gpu_image_blurred,
                               gpu_image_grad_y,
                               CV_16S,
                               0,
                               1,
                               canny_sobel_size);
                cv::gpu::abs(gpu_image_grad_y, gpu_image_grad_y);
                gpu_image_grad_y.convertTo(gpu_image_grad_y_U, CV_16U);
                cv::gpu::addWeighted(gpu_image_grad_x_U, 0.5, gpu_image_grad_y_U, 0.5, 0, gpu_image_edges);
                std::cout << type2str(gpu_image_edges.type()) << std::endl;
                std::cout << type2str(gpu_image_grad_x_U.type()) << std::endl;
                std::cout << type2str(gpu_image_grad_y_U.type()) << std::endl;
                std::cout << type2str(gpu_image_grad_x.type()) << std::endl;
                std::cout << type2str(gpu_image_grad_y.type()) << std::endl;
                cv::gpu::threshold(gpu_image_edges, gpu_image_edges, 700, 255, cv::THRESH_BINARY);
                gpu_image_edges.convertTo(gpu_image_edges, CV_8U);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;
t = clock();
                cv::gpu::bitwise_and(gpu_color_mask, gpu_image_edges, gpu_image_edges);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;
t = clock();
                cv::gpu::HoughLines(gpu_image_edges,
                                    gpu_lines,
                                    hough_buf,
                                    hough_rho_resolution,
                                    hough_theta_resolution,
                                    hough_threshold);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;

t = clock();
                gpu_image_edges.download(image_edges);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;
t = clock();
                cv::gpu::HoughLinesDownload(gpu_lines, lines);
std::cout << (clock() - t) / double(CLOCKS_PER_SEC) << std::endl;
            }
        }

        drawLines(lines, image);
        cv::imshow("edges", image_edges);
        cv::imshow("image", image);
        cv::waitKey(0);
//        if (cv::waitKey(30) >= 0) break;
    //}

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

} // namespace iarc7_vision
