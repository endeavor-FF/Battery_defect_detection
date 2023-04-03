#include"utils.h"

int main()
{
    vector<string> filenames;
    string imagePath = "C:\\Users\\Administrator\\Desktop\\sample";
    getSubdirs(imagePath, filenames);
    std::ofstream dataFile;
    vector<NewData> data;
    for (int i = 0; i < filenames.size(); i++) {
        Mat image = imread(filenames[i] + "\\1.png", IMREAD_UNCHANGED);
        int left_size = 4;//4mm
        float resolution = 0.016;//解析值
        int offest = 10;//偏移值
        int pixelSize = left_size / resolution + offest;//像素尺度
        int leftSize = pixelSize + 90;
        cv::Mat binary_image;
        cv::threshold(image, binary_image, 0, 65535, cv::THRESH_BINARY);
        Mat img_8uc1;
        binary_image.convertTo(img_8uc1, CV_8UC1);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(img_8uc1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        //寻找最大轮廓索引位置
        int largestContourIndex = returnMaxIndex(contours);
        Rect rect = boundingRect(contours[largestContourIndex]);//最小外接矩形
        Mat roi = image(rect);
        Rect left_rect(0, 0, leftSize, rect.height);
        Mat roi_left = roi(left_rect);//bug
        Rect center_rect(pixelSize, 0, rect.width - 2 * pixelSize, rect.height);
        Mat roi_center = roi(center_rect);
        Rect right_rect(rect.width - pixelSize, 0, pixelSize, rect.height);
        Mat roi_right = roi(right_rect);
        leftEdgeDetect(roi, data, filenames[i]);
        cout << "stop";
    }
    dataFile.open("C:\\Users\\Administrator\\Desktop\\sampleInfo\\右侧缺陷样本.csv", ios::out | ios::trunc);
    dataFile << "path" << "," << "x" << "," << "y" << "," << "Area" << "," << "rate" << "," << "Mean" << "," << "Length" << "," << "flag";
    dataFile << endl;
    if (!data.empty()) {
        for (int j = 0; j < data.size(); j++) {
            dataFile << data[j].path << "," << data[j].x << "," << data[j].y 
                << "," << data[j].Area << "," << data[j].Rate << "," 
                << data[j].grayscaleValue << "," << data[j].Length << "," << data[j].flag;
            dataFile << endl;
        }
    }
    else {
        dataFile << "nothing";
    }
    dataFile.close();
    return 0;
}