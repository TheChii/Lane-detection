#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void applyContrastStretching(Mat& img, double minPercentile, double maxPercentile)
{
    // Compute the percentiles
    double minVal, maxVal;
    cvtColor(img, img, COLOR_BGR2GRAY);
    minMaxLoc(img, &minVal, &maxVal);

    // Calculate the new intensity range based on percentiles
    double range = maxVal - minVal;
    double newMin = minVal + (minPercentile / 100.0) * range;
    double newMax = minVal + (maxPercentile / 100.0) * range;

    // Apply contrast stretching
    img = (img - newMin) * (255.0 / (newMax - newMin));

    // Clip values to the valid intensity range
    img.setTo(0, img < 0);
    img.setTo(255, img > 255);
}

void resizeImage(Mat& img, double scale)
{
    int newWidth = static_cast<int>(img.cols * scale);
    int newHeight = static_cast<int>(img.rows * scale);

    cv::resize(img, img, Size(newWidth, newHeight));
}

void applyCannyEdgeDetection(Mat& img, double lowThreshold, double highThreshold)
{
    if (img.channels() > 1)
        cvtColor(img, img, COLOR_BGR2GRAY);

    Canny(img, img, lowThreshold, highThreshold);
}

Rect roi(491, 436, 625, 233);
void processFrame(Mat& frame) {
    // Extract the ROI from the frame
    Mat roiFrame = frame(roi);

    // Apply contrast stretching
    applyContrastStretching(roiFrame, 40.0, 175.0);

    // Apply Canny edge detection with specified thresholds
    double lowThreshold = 160.0;
    double highThreshold = 255.0;
    applyCannyEdgeDetection(roiFrame, lowThreshold, highThreshold);

    // Find contours from the Canny edges
    vector<vector<Point>> contours;
    findContours(roiFrame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create a mask to fill the contours
    Mat mask = Mat::zeros(roiFrame.size(), CV_8U);
    fillPoly(mask, contours, Scalar(255));

    // Apply the mask to fill the edges in the original frame
    frame(roi).setTo(Scalar(0, 0, 255), mask);

    // Display or save the processed frame
    resizeImage(frame, 0.5);
    imshow("Processed Frame", frame);
}

int main() {
    VideoCapture cap("C:/Users/theo/Desktop/projects/lane detection/testimg/video2.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open the video file!" << std::endl;
        return -1;
    }

    Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Error: Empty frame received!" << std::endl;
            break;
        }

        // Process the frame
        processFrame(frame);

        // Display or save the processed frame
        imshow("Processed Frame", frame);

        // Check for key press or break condition
        char key = waitKey(30);
        if (key == 27)  // 'Esc' key
            break;
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
