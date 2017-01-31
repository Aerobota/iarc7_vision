#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <ctime>

void printtime(clock_t begin, std::string texto)
{
    clock_t end = clock();
    double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
    std::cout << "Time elapsed " << texto << " : " << timeSec << std::endl;
}

int main()
{
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    int blurwindow = 5; // blur kernel size
    int threshlimit = 55; // threshold limit

    clock_t begin, end, begin_partial,begin_1frame ;
    double timeSec;
    begin = clock();

    cv::Mat orig_image, dst_host, grey, blurred, thresh;

    // Stream and string to generate filenames
    std::stringstream ss;
    std::string filename;

    char key = 0;
    begin = clock();
    //    cv::Mat image;
    unsigned int rowBytes;

    // GPU definitions
    cv::gpu::GpuMat gpu_src, gpu_grey, gpu_thresh, gpu_blurred, gpu_edges, gpu_lines;
    //      http://stackoverflow.com/a/19454917
    //   The first call of any gpu function is slow due to CUDA context initialization. All next calls wil be faster. Call some gpu function before time measurement:
    cv::gpu::GpuMat test;
    test.create(1, 1, CV_8U); // Just to initialize context

    std::vector<int> compression_parms;
    compression_parms.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_parms.push_back(1);

    begin = clock();

    while (key != 'q')
    {
        begin_1frame = clock();

        orig_image = cv::imread("/home/aaron/Pictures/grid_sample.png");

        // CPU operations
        begin_partial = clock();
        cv::cvtColor(orig_image, grey, CV_BGR2GRAY);
        cv::blur(grey, blurred, cv::Size(blurwindow, blurwindow));
        cv::threshold(blurred, thresh, threshlimit, 255.0, CV_THRESH_BINARY_INV);
        printtime(begin_partial, "cpu_complete");

        // GPU operations
        begin_partial = clock();
        gpu_src.upload(orig_image);
        cv::gpu::cvtColor(gpu_src, gpu_grey, CV_BGR2GRAY); // not good to optimize with
        cv::gpu::blur(gpu_grey, gpu_blurred, cv::Size(blurwindow, blurwindow));
        cv::gpu::threshold(gpu_grey, gpu_thresh, threshlimit, 255.0, CV_THRESH_BINARY_INV);
        gpu_thresh.download(thresh);
        printtime(begin_partial, "gpu_complete");

        // STEP BY STEP CPU
        // STEP BY STEP GPU
        begin_partial = clock();
        gpu_src.upload(orig_image);
        printtime(begin_partial, "gpu_upload");

        begin_partial = clock();
        cv::cvtColor(orig_image, grey, CV_BGR2GRAY);
        printtime(begin_partial, "cpu_cvtColor");

        begin_partial = clock();
        cv::gpu::cvtColor(gpu_src, gpu_grey, CV_BGR2GRAY); // not good to optimize with gpu
        printtime(begin_partial, "gpu_cvtColor");

        begin_partial = clock();
        cv::blur(grey, blurred, cv::Size(blurwindow, blurwindow));
        printtime(begin_partial, "cpu_blur");

        begin_partial = clock();
        cv::gpu::blur(gpu_grey, gpu_blurred, cv::Size(blurwindow, blurwindow));
        printtime(begin_partial, "gpu_blur");

        begin_partial = clock();
        cv::threshold(blurred, thresh, threshlimit, 255.0, CV_THRESH_BINARY_INV);
        printtime(begin_partial, "cpu_threshold");

        begin_partial = clock();
        cv::gpu::threshold(gpu_grey, gpu_thresh, threshlimit, 255.0, CV_THRESH_BINARY_INV);
        printtime(begin_partial, "gpu_threshold");


        begin_partial = clock();
        gpu_thresh.download(thresh);
        printtime(begin_partial, "gpu_download");

        begin_partial = clock();
        cv::gpu::Canny(gpu_blurred, gpu_edges, 50, 150);
        printtime(begin_partial, "gpu canny");

        begin_partial = clock();
        cv::gpu::HoughLines(gpu_edges, gpu_lines, 1, 0.01, 500, false, 20);
        printtime(begin_partial, "gpu hough");

        cv::Mat cdst, edges;
        gpu_edges.download(edges);
        cv::cvtColor(grey, cdst, CV_GRAY2BGR);
        std::vector<cv::Vec2f> lines;
        cv::gpu::HoughLinesDownload(gpu_lines, lines);
        std::cout << lines.size() << std::endl;
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
           cv::line( cdst, pt1, pt2, cv::Scalar(0,0,255), 3, CV_AA);
        }
        cv::imshow("source", edges);
        cv::imshow("lines", cdst);
        cv::waitKey(0);
    }

    return 0;
}
