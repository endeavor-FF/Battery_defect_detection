// Battery_defect_detection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include<atomic>
#include<mutex>
#include <numeric>
#include <string>
#include <iostream>
#include<math.h>
#include <io.h>
#include<chrono>
#include<fstream>
#include <algorithm>
#include<sstream>
#include"utils.h"
using namespace cv;
using namespace std;
int fitCurve(std::vector<double> x, std::vector<double> y);
struct element {
    int XAxis;
    int YAxis;
    ushort pixelValue;
    double coefofVar;
    bool isDetect;
    double stddev;
    int width;
    int hight;
    vector<ushort> colsMean;
    double stdRange;
    double Max;
    double Min;
    double Mean;
};

struct FlawPoint {
    int XAxis;
    int YAxis;
    double Area;
};
//void getSubdirs(std::string path, std::vector<std::string>& files);
mutex foundThresholdMutex;//互斥锁
condition_variable foundThresholdCond;//条件变量
atomic<bool> foundThreshold(false);//原子操作
bool isBatteryDefect(Mat& grayImagePath);
int returnMaxIndex(vector<vector<Point>> contours);
bool comparePixel(const element& f1, const element& f2);
//void EdgeDetect(Mat roi, int relativeX, int relativeY, FlawPoint& flawPoint, int edgeFlag, bool swithValue, std::string path);
bool isDefect(Mat& image, std::string path);

//int main()
//{
//    vector<string> filenames;
//    string imagePath = "E:\\report\\batteryrecognition\\BatteryRecognition\\Result\\OK";
//    getSubdirs(imagePath, filenames);
//    for (int i = 0; i < filenames.size(); i++) {
//        Mat image = imread(filenames[i] + "\\11左.png", IMREAD_UNCHANGED);
//        bool defect = isDefect(image, filenames[i]);
//        if (defect == true) {
//            std::cout << "该样本含有缺陷" << endl;
//        }
//        else {
//            std::cout << "该样本不含有缺陷" << endl;
//        }
//    }
//    //Mat image = imread(imagePath + "\\u16c1_src.png", IMREAD_UNCHANGED);
//    //bool defect = isDefect(image, imagePath);
//    
//    
//    
//
//    //Mat img = imread("C:\\Users\\Administrator\\Desktop\\sampleAn\\2\\r\\rowFine1.png", IMREAD_UNCHANGED);
//    //Mat columnMean;
//    //reduce(img, columnMean, 0, REDUCE_AVG, CV_64FC1);//列平均值
//    return 0;
//}
bool compareArea(const NewData& f1, const NewData& f2) {
    return f1.Area > f2.Area;
}
cv::Mat diffEveryTwoCols(cv::Mat input);
bool isDefect(Mat& image ,std::string path) {
    if (image.empty())
    {
        throw runtime_error("读取图片失败");
    }
    //image.convertTo(grayImage, CV_16UC1);
    auto start = std::chrono::high_resolution_clock::now();//开始计算算法用时
    //---------------区域分割------------------
    int left_size = 4;//4mm
    float resolution = 0.016;//解析值
    int offest = 10;//偏移值
    int pixelSize = left_size / resolution + offest;//像素尺度
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
    Rect left_rect(0, 0, pixelSize, rect.height);
    Mat roi_left = roi(left_rect);//bug
    Rect center_rect(pixelSize, 0, rect.width - 2 * pixelSize, rect.height);
    Mat roi_center = roi(center_rect);
    Rect right_rect(rect.width - pixelSize, 0, pixelSize, rect.height);
    Mat roi_right = roi(right_rect);

    FlawPoint flawPoint;
    std::vector<thread> threads;

    //threads.emplace_back(EdgeDetect, roi_right, rect.x + right_rect.x, rect.y + right_rect.y, flawPoint, true);//右边缘检测
    //threads.emplace_back(EdgeDetect, roi_left, rect.x + left_rect.x, rect.y + left_rect.y, flawPoint, true);//左边缘检测
    
    //unique_lock<mutex> lock(foundThresholdMutex);//加锁
    //foundThresholdCond.wait(lock, [] { return foundThreshold.load(); });//等待条件变量被修改为TRUE
    //lock.unlock();
    //for (auto& thread : threads) {
    //    thread.join();
    //}
 
   
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    float fDuration = duration.count() / 1000000.0;
    std::cout << fDuration << endl;

    //Scharr(diffImage2, resultX, CV_16U, 3, 0, 7, 1000);
    //Mat resultY;
    //Sobel(diffImage2, resultY, CV_16U, 0, 1, 11);
    /*Mat kernel = getStructuringElement(2,Size(3,3));
    Mat open, close, gradient, tophat, blackhat, hitmiss;
    morphologyEx(diffImage2, open, MORPH_OPEN, kernel);
    morphologyEx(diffImage2, close, MORPH_CLOSE, kernel);
    morphologyEx(diffImage2, gradient, MORPH_GRADIENT, kernel);
    morphologyEx(diffImage2, tophat, MORPH_TOPHAT, kernel);
    morphologyEx(diffImage2, blackhat, MORPH_BLACKHAT, kernel);
    morphologyEx(diffImage2, hitmiss, MORPH_HITMISS, kernel);*/

    //Mat diffMat = diffEveryTwoCols(roi_right);

}


void EdgeDetect(Mat roi ,std::string path){

        bool isFlaw = false;
        //按列下采样
        cv::Mat diffImageCols = cv::Mat::zeros(roi.rows, roi.cols/5, CV_16U);
        for (int i = 0; i < roi.cols - 5; i+=5) {
			Mat colMean;
			Mat colPerFive(roi, Range::all(), Range(i, i + 5));
			reduce(colPerFive, colMean, 1, REDUCE_AVG, CV_32F);
            colMean.copyTo(diffImageCols.col(i/5));
        }
        cv::Mat diffImage = cv::Mat::zeros(diffImageCols.rows, diffImageCols.cols, CV_64F);
        for (int i = 0; i < diffImageCols.rows - 1; i++) {
            cv::Mat row1 = diffImageCols.row(i);
            cv::Mat row2 = diffImageCols.row(i + 1);
            cv::Mat diffCol = row2 - row1;
            diffCol.copyTo(diffImage.row(i));
        }

        cv::Mat diffImage2 = cv::Mat::zeros(diffImageCols.rows, diffImageCols.cols, CV_32F);
        for (int i = 0; i < diffImageCols.cols - 1; i++) {
            cv::Mat col1 = diffImage.col(i);
            cv::Mat col2 = diffImage.col(i + 1);
            cv::Mat diffCol = col2 - col1;
            diffCol.copyTo(diffImage2.col(i));
        }
        //Mat downSampleMean;
        //reduce(diffImageCols, downSampleMean, 0, REDUCE_AVG, CV_32F);
        //Mat _32f;
        //diffImageCols.convertTo(_32f,CV_32F);
        //cv::Mat result1 = cv::Mat::zeros(diffImageCols.rows, diffImageCols.cols, CV_32F);
        //for (int i = 0; i < diffImageCols.rows -1; i ++) {
        //    Mat colMean;
        //    cv::Mat row = _32f.row(i);
        //    cv::Mat diffCol = row - downSampleMean;
        //    diffCol.copyTo(result1.row(i));
        //}
        //Mat rowMean;
        //reduce(roi, rowMean, 0, REDUCE_AVG, CV_32F);
        //Mat downRowMean = cv::Mat::zeros(rowMean.rows, rowMean.cols/20 -1, CV_16U);
        //vector<double> x;
        //vector<double> y;
        //for (int i = 20; i < rowMean.cols; i += 20) {
        //    float colValve = rowMean.at<float>(0, i);
        //    x.push_back(i);
        //    y.push_back(colValve);
        //    //point.push_back(Point2f(i, colValve));
        //    //cv::Mat col = rowMean.col(i);
        //    //col.copyTo(downRowMean.col(i / 20 - 1));
        //}
        //fitCurve(x,y);

        // vector<vector<element>> maxsVectors;
		//int block_size_22 = 30;
  //      Mat diffLocalImg = diffImageCols.clone();
  //      Mat _32fimg;
  //      diffLocalImg.convertTo(_32fimg, CV_32F);
  //      int rows = diffLocalImg.rows;
  //      int cols = diffLocalImg.cols;
  //      vector<uchar> stepSize = {15,30};
		//for (int j = 0; j < stepSize.size(); j++) {
		//	for (int k = 0; k < rows; k += stepSize[j])
		//	{
		//		int block_height_2 = (k + stepSize[j] > rows) ? (rows - k) : stepSize[j];
		//		Rect rect_2(0, k, cols, block_height_2);
		//		Mat block_2 = _32fimg(rect_2);
		//		Mat rowMean, rowMean16,colMean;
		//		reduce(block_2, rowMean, 0, REDUCE_AVG, CV_32F);
  //              //reduce(block_2, colMean, 1, REDUCE_AVG, CV_32F);
		//		//rowMean.convertTo(rowMean16, CV_16U);
		//		Mat colPerFive(_32fimg, Range(k, k + block_height_2), Range::all());
		//		cv::Mat diffImage(colPerFive, Range::all(), Range::all());//= cv::Mat::zeros(colPerFive.rows, colPerFive.cols, CV_32F);
		//		for (int i = 0; i < colPerFive.rows; i++) {
		//			cv::Mat row1 = colPerFive.row(i);
		//			cv::Mat diffRow = row1 - rowMean;
		//			diffRow.copyTo(diffImage.row(i));
		//		}
  //              //for (int i = 0; i < colPerFive.cols; i++) {
  //              //    cv::Mat col = colPerFive.col(i);
  //              //    cv::Mat diffCol = col - colMean;
  //              //    diffCol.copyTo(diffImage.col(i));
  //              //}
		//	}
		//}
		//
  //      Mat _8uresult;
  //      _32fimg.convertTo(_8uresult, CV_8U);

  //      Mat colMean;
  //      reduce(_8uresult, colMean, 1, REDUCE_AVG, CV_32F);
  //      colMean.convertTo(colMean, CV_8U);
  //      cv::Mat diffImage3 = cv::Mat::zeros(_32fimg.rows, _32fimg.cols, CV_8U);
  //      for (int i = 0; i < _8uresult.cols; i++) {
  //          cv::Mat col1 = _8uresult.col(i);
  //         
  //          cv::Mat diffCol = col1- colMean;
  //          //diffCol.setTo(255, diffCol > 0);
  //          diffCol.setTo(0, diffCol < 0);
  //          diffCol.copyTo(diffImage3.col(i));
  //      }
  //      diffImage3.convertTo(_8uresult, CV_8U);
        /*cv::Mat diffImage = cv::Mat::zeros(roi.rows, roi.cols, CV_64F);
        for (int i = 0; i < roi.rows - 1; i++) {
            cv::Mat row1 = roi.row(i);
            cv::Mat row2 = roi.row(i + 1);
            cv::Mat diffCol = row2 - row1;
            diffCol.copyTo(diffImage.row(i));
        }
       
        cv::Mat diffImage2 = cv::Mat::zeros(roi.rows, roi.cols, CV_32F);
        for (int i = 0; i < roi.cols - 1; i++) {
            cv::Mat col1 = diffImage.col(i);
            cv::Mat col2 = diffImage.col(i + 1);
            cv::Mat diffCol = col2 - col1;
            diffCol.copyTo(diffImage2.col(i));
        }*/
        Mat diffBinaryImg;
        //diffImage2.setTo(0, diffImage2 < 0);
        //diffImage2.setTo(255, diffImage2 > 0);
        //diffImage2.convertTo(diffBinaryImg, CV_8U);
        //Mat diffBinaryImg = cv::Mat::zeros(diffImage2.rows, diffImage2.cols,CV_8U);
        //threshold(diffImage2, diffBinaryImg, 0, 255, THRESH_BINARY);

        //掩膜
        Mat srcBinary, srcBinary8U,srcDilate;
        threshold(roi, srcBinary, 0, 32676, THRESH_BINARY);
        srcBinary.convertTo(srcBinary8U, CV_8U);//原图二值化
        Mat dilatekernel = getStructuringElement(1, Size(20, 20));
        dilate(~srcBinary8U, srcDilate, dilatekernel);//取反膨胀
        //dilate(srcDilate, srcDilate, dilatekernel);


        //拉普拉斯算子轮廓检测
        Mat laplaceResult;
        cv::Laplacian(diffImage2, laplaceResult, CV_16S, 5, 1, 1000);
        Mat _8uimg;
        Mat binary_16S;
        threshold(laplaceResult, binary_16S, 0, 32676, THRESH_BINARY);
        binary_16S.convertTo(_8uimg,CV_8U);
        Mat blackHatImg;
        Mat kernel = getStructuringElement(0, Size(5, 5));
        morphologyEx(_8uimg, blackHatImg, MORPH_BLACKHAT, kernel);
        Mat result_8uc1;
        Mat OPENkernel = getStructuringElement(1, Size(2, 2));
        dilate(blackHatImg, result_8uc1, OPENkernel);


        //相减去边界
        Mat result;
        subtract(result_8uc1, srcDilate, result);
        result.setTo(0, result < 0);

        //检测
        std::vector<std::vector<cv::Point>> _contours;
        cv::findContours(result, _contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        imwrite(path+"\\roi_right_new.png", result);

   //     ////////////
   //     int colsCsv = _contours.size();//小
   //     //int rowsCsv = rowsMeans[0].size();//大




        /*std::ofstream dataFile;
        dataFile.open(path + "\\sample.csv", ios::out | ios::trunc);
        dataFile << "contoursSize" << "," << "X" << "," << "Y";
        dataFile << endl;*/
   //     
   //     for (int i = 0; i < colsCsv-1; i++) {

			//dataFile << _contours[i].size() << ","
   //             << _contours[i][0].x<<","
   //             << _contours[i][0].y;
			//dataFile << endl;
   //     }
   //     dataFile.close();
   //     //////////
        //cout << "debug";
        ////drawContours(result_8uc1, _contours, _contours.size() - 1, Scalar(255, 255, 255), -1, LINE_8);
        //int flawX;
        //int flawY;
        //double flawArea;
        //if (edgeFlag == 1) {
        //    for (int i = 0; i < _contours.size() - 1; i++) {
        //        if (_contours[i].size() >= 70 && _contours[i].size() <= 500) {
        //            RotatedRect contoursRect = minAreaRect(_contours[i]);
        //            if (contoursRect.center.x <= 245 && contoursRect.center.y >= 30) {
        //                isFlaw = true;
        //                flawX = contoursRect.center.x + relativeX;
        //                flawY = contoursRect.center.y + relativeY;
        //                flawArea = contourArea(_contours[i]);
        //                break;
        //            }
        //        }
        //    }
        //}
        //if (edgeFlag == 0) {
        //    for (int i = 0; i < _contours.size() - 1; i++) {
        //        if (_contours[i].size() >= 80 && _contours[i].size() <= 500) {
        //            RotatedRect contoursRect = minAreaRect(_contours[i]);
        //            if (contoursRect.center.x >= 65 && contoursRect.center.y >= 50 && contoursRect.center.y <= roi.rows-50) {
        //                isFlaw = true;
        //                flawX = contoursRect.center.x + relativeX;
        //                flawY = contoursRect.center.y + relativeY;
        //                flawArea = contourArea(_contours[i]);
        //                /*dataFile << _contours[i].size() << ","
        //                    << _contours[i][0].x << ","
        //                    << _contours[i][0].y;
        //                dataFile << endl;*/
        //                //break;
        //            }
        //        }
        //    }
        //}
        //dataFile.close();
        //if (isFlaw == true) {
        //    unique_lock<mutex> lock(foundThresholdMutex);
        //    if (!foundThreshold.load()) {
        //        flawPoint.XAxis = flawX;
        //        flawPoint.YAxis = flawY;
        //        flawPoint.Area = flawArea;
        //        foundThreshold.store(true);
        //        foundThresholdCond.notify_all();
        //    }
        //    lock.unlock();
        //}
    
}
cv::Mat diffEveryTwoCols(cv::Mat input) {
    int cols = input.cols;

    //int out_cols = (cols + 1) / 2;
    int out_cols = cols ;

    cv::Mat output = cv::Mat::zeros(input.rows, out_cols, CV_32F);

    //for (int col = 0; col < cols - 2; col += 2) {
    //    cv::Mat diff = input.col(col) - input.col(col + 2);

    //    cv::Mat output_col = output.col((col + 2) / 2);
    //    diff.copyTo(output_col);
    //}

    for (int col = 0; col < cols - 1; col += 1) {
        cv::Mat diff = input.col(col) - input.col(col + 1);

        cv::Mat output_col = output.col(col);
        diff.copyTo(output_col);
    }
    return output;
}


bool isBatteryDefect(Mat& image) {

    
    if (image.empty())
    {
        throw runtime_error("读取图片失败");
    }
    //image.convertTo(grayImage, CV_16UC1);
    auto start = std::chrono::high_resolution_clock::now();//开始计算算法用时
    //---------------区域分割------------------
    int left_size = 4;//4mm
    float resolution = 0.016;//解析值
    int offest = 10;//偏移值
    int pixelSize = left_size / resolution + offest;//像素尺度
    cv::Mat binary_image;
    cv::threshold(image, binary_image, 0, 65535, cv::THRESH_BINARY);
    Mat img_8uc1;
    binary_image.convertTo(img_8uc1, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img_8uc1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //寻找最大轮廓索引位置
    int largestContourIndex = returnMaxIndex(contours);
    Rect rect = boundingRect(contours[largestContourIndex]);//最小外接矩形
    

    /// <summary>
    Mat comImg = image(rect);
    Mat columnMean;
   
    reduce(comImg, columnMean, 0, REDUCE_AVG,CV_64FC1);//列平均值
    int x1 = 500;
    int x2 = 1000;
    double y1 = columnMean.at<double>(0, x1);
    double y2 = columnMean.at<double>(0, x2);
    double k = (y1 - y2) / (x1 - x2);

    Mat K = Mat::zeros(comImg.cols, 1, CV_32FC1);//列
    for (int i = 0; i < K.rows; i++)
    {
        K.at<float>(i, 0) = i;
    }
    Mat oneM = Mat::ones(1, comImg.rows, CV_32FC1);//行
    Mat compensateM;
    gemm(K, oneM,1,Mat(),0.0, compensateM,0);
    Mat trans = (compensateM * k).t();
    Mat xBalanceSrc(comImg.rows, comImg.cols,CV_16UC1);
    add(comImg, -trans, xBalanceSrc,cv::noArray(),CV_16UC1);
    /// </summary>
    /// <param name="grayImagePath"></param>
    /// <returns></returns>
    int stepSize = 5;
    vector<vector<string>> rowsMeans;
    vector<string> rowMeans;
    
    Mat roi = xBalanceSrc.clone();
    //Mat dst;
    //cv::normalize(image, dst, 0, 65535, NORM_MINMAX, CV_16U);
    //for (int i = 0; i <= roi.cols; i = i + stepSize) {
    //    rowMeans.clear();
    //    if (i + 5 >= roi.cols)
    //        break;
    //    Rect rowRoiRect(i, 0, stepSize, roi.rows);
    //    Mat rowRoi = roi(rowRoiRect);
    //    for (int j = 0; j < rowRoi.rows; j++) {
    //        if (j == 0) {
    //            string index = to_string(i) + "-" + to_string(i + stepSize);
    //            rowMeans.push_back(index);
    //        }
    //        double rowMean = mean(rowRoi.row(j))[0];
    //        rowMeans.push_back(to_string(rowMean));
    //    }
    //    rowsMeans.push_back(rowMeans);

    //    rowMeans.clear();
    //    if (i + stepSize >= roi.cols) {
    //        break;
    //        Rect rowRoiRect(i, 0, roi.cols - i, roi.rows);
    //        Mat rowRoi = roi(rowRoiRect);
    //        for (int j = 0; j < rowRoi.rows; j++) {
    //            double rowMean = mean(rowRoi.row(j))[0];
    //            rowMeans.push_back(to_string(rowMean));
    //        }
    //        rowsMeans.push_back(rowMeans);
    //    }
    //}
    //int colsCsv = rowsMeans.size();//小
    //int rowsCsv = rowsMeans[0].size();//大
    //std::ofstream dataFile;
    //dataFile.open("C:\\Users\\Administrator\\Desktop\\sample.csv",ios::out|ios::trunc);
    //for (int i = 0; i < rowsCsv; i++) {
    //    for (int j = 0; j < colsCsv; j++) {
    //        dataFile << rowsMeans[j][i] << ",";
    //    }
    //    dataFile << endl;
    //}
    //dataFile.close();
    //imwrite("C:\\Users\\Administrator\\Desktop\\modifiyPng.png", roi);
    //三区域分离
    Rect left_rect(0, 0, pixelSize, rect.height);
    Mat roi_left = roi(left_rect);//bug
    Rect center_rect(pixelSize, 0, rect.width - 2 * pixelSize, rect.height);
    Mat roi_center = roi(center_rect);
    Rect right_rect(rect.width - pixelSize, 0, pixelSize, rect.height);
    Mat roi_right = roi(right_rect);

    //---------------区域分割------------------
    Mat paintImg = roi_center.clone();
    int rows = roi_center.rows;
    int cols = roi_center.cols;
    int block_size = 300;
    int block_size_2 = 500;

    int block_size_22 = 80; 
   // vector<vector<element>> maxsVectors;
   // for (int k = 0; k < rows; k += block_size_22)
   // {
   //     vector<element> rowmax;
   //     for (int l = 0; l < cols; l += block_size_22)
   //     {
   //         int block_width_2 = (l + block_size_22 > cols) ? (cols - l) : block_size_22;
   //         int block_height_2 = (k + block_size_22 > rows) ? (rows - k) : block_size_22;

   //         Rect rect_2(l, k, block_width_2, block_height_2);
   //         Mat block_2 = roi_center(rect_2);
   //         double max_val_2;
   //         minMaxLoc(block_2, nullptr, &max_val_2);
   //         element rowElement;
   //         rowElement.pixelValue = max_val_2;
   //         rowElement.XAxis = rect_2.x;
   //         rowElement.YAxis = rect_2.y;
   //         rowmax.push_back(rowElement);
   //     }
   //     maxsVectors.push_back(rowmax);
   // }
   // vector<element> filterVectors;
   // for (int i = 0; i < maxsVectors.size(); i++) {//读取每行的容器
   //     for (int j = 0; j < maxsVectors[i].size(); j++) {
			//if (i >= 1 && i <= maxsVectors.size() - 2) {//去掉上下两个侧

			//	if (j == 0) {//左侧一列
			//		ushort max = 0;
			//		ushort selfValue = maxsVectors[i][j].pixelValue;
			//		for (int indexi = i - 1; indexi <= i + 1; i++) {
			//			for (int indexj = j; indexj <= j + 2; j++) {
			//				if (indexi == i && indexj == j)
			//					continue;
			//				if (selfValue <= maxsVectors[indexi][indexj].pixelValue)
			//				{
			//					break;
			//				}
			//				else {
			//					max = maxsVectors[indexi][indexj].pixelValue;
			//				}
			//			}
			//		}
			//		if (max < selfValue) {
			//			filterVectors.push_back(maxsVectors[i][j]);
			//		}
			//	}

			//	if (j == maxsVectors[i].size() - 1) {//右侧一列
			//		ushort max = 0;
			//		ushort selfValue = maxsVectors[i][j].pixelValue;
			//		for (int indexi = i - 1; indexi <= i + 1; i++) {
			//			for (int indexj = j - 2; indexj <= j; j++) {
			//				if (indexi == i && indexj == j)
			//					continue;
			//				if (selfValue <= maxsVectors[indexi][indexj].pixelValue)
			//				{
			//					break;
			//				}
			//				else {
			//					max = maxsVectors[indexi][indexj].pixelValue;
			//				}
			//			}
			//		}
			//		if (max < selfValue) {
			//			filterVectors.push_back(maxsVectors[i][j]);
			//		}
			//	}
			//	//中心位置
			//	ushort max = 0;
			//	ushort selfValue = maxsVectors[i][j].pixelValue;
			//	for (int indexi = i - 1; indexi <= i + 1; i++) {
			//		for (int indexj = j - 1; indexj <= j + 1; j++) {
			//			if (indexi == i && indexj == j)
			//				continue;
			//			if (selfValue <= maxsVectors[indexi][indexj].pixelValue)
			//			{
			//				break;
			//			}
			//			else {
			//				max = maxsVectors[indexi][indexj].pixelValue;
			//			}
			//		}
			//	}
			//	if (max < selfValue) {
			//		filterVectors.push_back(maxsVectors[i][j]);
			//	}

			//}
   //     }
   // }
   // 
    bool isFlaw = false;
    int flawx = 300;
    int flawy = 1821;
    int rowStep = 30;
    int colStep = 300;
    vector<element> varVector;
    for (int i = 0; i < rows; i += rowStep)
    {
        for (int j = 0; j < cols; j += colStep)
        {
            int block_height = (i + rowStep > rows) ? (rows - i) : rowStep;
            int block_width = (j + colStep > cols) ? (cols - j) : colStep;
            Rect rect1(j, i, block_width, block_height);
            Mat block = roi_center(rect1);
            /*if (i <= 2370 && i + block_height >= 2370) {
                if (j <= 700 && j + block_width >= 700) {
                    imwrite("C:\\Users\\Administrator\\Desktop\\rowFlaw.png", roi_center(rect));
                }
            }
            if (i <= 1168 && i + block_height >= 1168) {
                if (j <= 287 && j + block_width >= 287) {
                    imwrite("C:\\Users\\Administrator\\Desktop\\rowFine1.png", roi_center(rect));
                }
            }
            if (i <= 2132 && i + block_height >= 2132) {
                if (j <= 716 && j + block_width >= 716) {
                    imwrite("C:\\Users\\Administrator\\Desktop\\rowFine.png", roi_center(rect));
                }
            }*/
            vector<int> colsMean;
			Mat columnMean;
			reduce(block, columnMean, 0, REDUCE_AVG, CV_64FC1);//列平均值
            
			//const ushort* pSrcData = columnMean.ptr<ushort>(0);
			//for (int indexv = 0; indexv <= columnMean.cols; indexv++) {
   //             int clum = columnMean.at<ushort>(0,0);
			//	//colsMean.push_back(*pSrcData++);
			//}
                
                
            
            Mat mat_mean, mat_stddev;
            meanStdDev(columnMean, mat_mean, mat_stddev);
            double mean = mat_mean.at<double>(0, 0);
            double stddev = mat_stddev.at<double>(0, 0);
            double coefofVar = stddev / mean;
            element e;
            e.isDetect = false;
            //if (j <= flawx && j + block_width >= flawx) {
            //    if (i <= flawy && i + block_height >= flawy) {
            //        e.isDetect = true;
            //    }
            //}
            double max_val_2;
            double max_val_3;
            minMaxLoc(block, &max_val_3, &max_val_2);
            //if (rect1.y >= 3000 || rect1.y <= 360 ) {
            //    continue;
            //}
            //if (stddev >= 100) {
            //    continue;
            //}
            e.coefofVar = coefofVar;
            e.XAxis = rect1.x;
            e.YAxis = rect1.y;
            e.stddev = stddev;
            e.hight = block_height;
            e.width = block_width;
            //e.colsMean = colsMean;
            e.stdRange = (max_val_2 - max_val_3) / mean;
            e.Mean = mean;
            e.Max = max_val_2;
            e.Min = max_val_3;
            varVector.push_back(e);
        }  
    }
    sort(varVector.begin(), varVector.end(), comparePixel);
    Mat paint = image(rect)(center_rect);
    if (varVector[0].coefofVar >= 0.0019 && varVector[0].coefofVar <= 0.0032) {
        isFlaw = true;
        
        Rect rect1(varVector[0].XAxis, varVector[0].YAxis, varVector[0].width, varVector[0].hight);
        rectangle(paint, rect1, Scalar(0, 0, 0), 8);
    }
   
	std::ofstream dataFile;
    dataFile.open("C:\\Users\\Administrator\\Desktop\\new_sample.csv", ios::in);
    if (!dataFile) {
        ofstream fout("C:\\Users\\Administrator\\Desktop\\new_sample.csv");
        fout.close();
    }
	dataFile.open("C:\\Users\\Administrator\\Desktop\\new_sample.csv", ios::out | ios::trunc);
    dataFile << "X" << "," << "Y" << "," << "Var" << "," << "Dev" << "," << "isDetect" << "," << "StdRange" << "," << "MAX" << "," << "MIN" << "," << "Mean" << "," << "Range";
    //for (int j = 1; j <= 300; j++) {
    //    dataFile << "," << to_string(j);
    //}
	int rowsCsv = varVector.size();
	for (int i = 0; i < rowsCsv; i++) {
        dataFile << endl;
        dataFile<< to_string(varVector[i].XAxis)<< "," << to_string(varVector[i].YAxis) <<
            "," << to_string(varVector[i].coefofVar) << "," << to_string(varVector[i].stddev) << 
            "," << varVector[i].isDetect <<"," << varVector[i].stdRange << "," << varVector[i].Max << "," << varVector[i].Min << "," << varVector[i].Mean << "," << varVector[i].Max- varVector[i].Min;
        /*for (int j = 0; j <= varVector[i].colsMean.size()-1; j++) {
            dataFile << "," << to_string(varVector[i].colsMean[j]);
        }*/
	}
	dataFile.close();
    return isFlaw;
     //循环切割图像
    //for (int i = 0; i < rows; i += block_size)
    //{
    //    for (int j = 0; j < cols; j += block_size)
    //    {
    //        // 判断是否越界
    //        int block_width = (j + block_size > cols) ? (cols - j) : block_size;
    //        int block_height = (i + block_size > rows) ? (rows - i) : block_size;

    //        // 切割矩形块
    //        Rect rect(j, i, block_width, block_height);
    //        //rectangle(paintImg, rect,Scalar(0,0,0),8);
    //        Mat block = roi_center(rect);
    //        double max_val, mean_val;
    //        minMaxLoc(block, nullptr, &max_val);
    //        //mean_val = mean(block)[0];
    //        

    //        //中位数
    //        /*vector<ushort> pixels;
    //        pixels.assign(block.data, block.data + block.total());
    //        sort(pixels.begin(), pixels.end());
    //        double median;
    //        size_t pixelCount = pixels.size();
    //        if (pixelCount % 2 == 0) {
    //            median = (double)(pixels[pixelCount / 2 - 1] + pixels[pixelCount / 2]) / 2.0;
    //        }
    //        else {
    //            median = (double)pixels[pixelCount / 2];
    //        }*/
    //        double non_zero_pixels = countNonZero(block);
    //        double sum = cv::sum(block)[0];
    //        double mean = sum / non_zero_pixels;
    //        int threshold;
    //        if (max_val - mean > 200 && max_val - mean < 500) {
    //            std::cout << "极差";
    //            threshold = max_val - 50;
    //            for (int col = rect.x; col < rect.x + block_width; ++col) {
    //                for (int row = rect.y; row < rect.y + block_height; ++row) {

    //                    // 读取像素值
    //                    ushort pixel = paintImg.at<ushort>(row, col);
    //                    // 修改像素值
    //                    if (threshold > pixel) {
    //                        pixel = 0;
    //                    }
    //                    else {
    //                        pixel = 65535;
    //                    }
    //                    //pixel = 65535;
    //                // 写入像素值
    //                    paintImg.at<ushort>(row, col) = pixel;
    //                }
    //            }
    //        }
    //        else {
    //            for (int col = rect.x; col < rect.x + block_width; ++col) {
    //                for (int row = rect.y; row < rect.y + block_height; ++row) {

    //                    ushort pixel = paintImg.at<ushort>(row, col);
    //                    pixel = 0;
    //                    paintImg.at<ushort>(row, col) = pixel;
    //                }
    //            }
    //        }
    //        if (j <= 660 && block_width + j >= 660) {
    //            if (i <= 2367 && i + block_height >= 2367) {
    //                imwrite("C:\\Users\\Administrator\\Desktop\\1flaw.png", roi_center(rect));
    //            }
    //        }
    //        if (j <= 742 && block_width + j >= 742) {
    //            if (i <= 2366 && i + block_height >= 2366) {
    //                imwrite("C:\\Users\\Administrator\\Desktop\\2flaw.png", roi_center(rect));
    //            }
    //        }
    //        if (j <= 409 && block_width + j >= 409) {
    //            if (i <= 905 && i + block_height >= 905) {
    //                imwrite("C:\\Users\\Administrator\\Desktop\\1noise.png", roi_center(rect));
    //            }
    //        }
    //        if (j <= 755 && block_width + j >= 755) {
    //            if (i <= 1203 && i + block_height >= 1203) {
    //                imwrite("C:\\Users\\Administrator\\Desktop\\2noise.png", roi_center(rect));
    //            }
    //        }
    //        if (j <= 146 && block_width + j >= 146) {
    //            if (i <= 1663 && i + block_height >= 1663) {
    //                imwrite("C:\\Users\\Administrator\\Desktop\\1fine.png", roi_center(rect));
    //            }
    //        }
    //    }
    //}
    //    }
    //}

    //bool isBatteryDefect = false;
    ////二值化得到掩膜
    //Mat binaryImageMask;
    //threshold(grayImage, binaryImageMask, 0, 255, IMREAD_GRAYSCALE);

    ////手动调参
    ////// 二值化
    ////Mat binaryImage;
    ////threshold(image, binaryImage,250,255,THRESH_BINARY);
    //Mat dst;
    //cv::Canny(grayImage, dst, 2, 6, 3);


    //// 在掩膜图像中寻找轮廓
    //vector<vector<Point>> contours;
    //findContours(binaryImageMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ////寻找最大轮廓索引位置
    //int largestContourIndex = -1;
    //double largestContourArea = 0;
    //for (int i = 0; i < contours.size(); i++) {
    //    double area = contourArea(contours[i]);
    //    if (area > largestContourArea) {
    //        largestContourArea = area;
    //        largestContourIndex = i;
    //    }
    //}

    ////将图像在轮廓内部分割成若干正方形，求取正方形内部不为零像素部分均值
    //int square_side = 300;
    //vector<double> mean_vals;
    //Rect rect = boundingRect(contours[largestContourIndex]);//最小外接矩形
    //for (int y = rect.y + square_side / 2; y + square_side / 2 < rect.y + rect.height; y += square_side) {
    //    for (int x = rect.x + square_side / 2; x + square_side / 2 < rect.x + rect.width; x += square_side) {
    //        Rect square(x - square_side / 2, y - square_side / 2, square_side, square_side);
    //        //rectangle(image, square, Scalar(0, 0, 0), 2);//绘制300×300区域
    //        Mat square_roi = grayImage(square);
    //        double non_zero_pixels = countNonZero(square_roi);
    //        double sum = cv::sum(square_roi)[0];
    //        double mean = sum / non_zero_pixels;
    //        mean_vals.push_back(mean);
    //    }
    //}

    ////求取vector中的均值
    //double mean = accumulate(mean_vals.begin(), mean_vals.end(), 0.0) / mean_vals.size();

    ////保护措施2：除去表面高度差较小的样本
    //double imageAltitude = 255 - mean;
    //cout << "均值与纯白的差" << imageAltitude << endl;//如果遍历全部的像素的话极差相似，分辨不出
    //if (imageAltitude < 15) {
    //    return isBatteryDefect;
    //}

    ////求取vector中的最大值
    //double max_val = *std::max_element(mean_vals.begin(), mean_vals.end());//返回一个迭代器指向最大值

    //

    //////求取vector中的最小值
    ////double min_val = *std::min_element(mean_vals.begin(), mean_vals.end());

    ////求取vector中的中位数
    ////std::sort(mean_vals.begin(), mean_vals.end());
    ////double median = mean_vals[mean_vals.size() / 2];
    ////if (mean_vals.size() % 2 == 0) {
    ////    median = (mean_vals[mean_vals.size() / 2 - 1] + mean_vals[mean_vals.size() / 2]) / 2;
    ////}

    //vector<double> firstAreas;
    //vector<double> Areas;
    //Mat binaryImage;
    //threshold(grayImage, binaryImage, max_val, 255, THRESH_BINARY);

    //contours.clear();
    //findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //for (int i = 0; i < contours.size(); i++) {
    //    double area = contourArea(contours[i]);
    //    if (area > 10000) {
    //        firstAreas.push_back(area);
    //    }
    //}
    //Areas = firstAreas;
    //while (Areas.size() <= firstAreas.size()) {
    //    max_val--;
    //    Areas.clear();
    //    vector<vector<Point>> _contours;
    //    Mat _binaryImage;
    //    threshold(grayImage, _binaryImage, max_val, 255, THRESH_BINARY);
    //    findContours(_binaryImage, _contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //    for (int i = 0; i < _contours.size(); i++) {
    //        double area = contourArea(_contours[i]);
    //        if (area > 10000) {
    //            Areas.push_back(area);
    //        }
    //    }
    //}
    //max_val++;
    //Mat binaryImageFin;
    //threshold(grayImage, binaryImageFin, max_val, 255, THRESH_BINARY);
    //contours.clear();
    //findContours(binaryImageFin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ////保护措施1：
    //while (contours.empty()) {//一个轮廓也过滤不出——max值太大
    //    max_val -= 2;
    //    threshold(grayImage, binaryImage, max_val, 255, THRESH_BINARY);
    //}

    //

    //Mat imageOutput = grayImage.clone();
    //int j = 0;
    //for (int i = 0; i < contours.size(); i++) {
    //    double area = contourArea(contours[i]);
    //    if (area > 10000) {
    //        RotatedRect rect = minAreaRect(contours[i]);
    //        Point2f vertices[4];
    //        rect.points(vertices);
    //        for (int j = 0; j < 4; j++)
    //        {
    //            line(imageOutput, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 0), 2);
    //        }
    //        j = j + 1;
    //        cout << "第" << to_string(j) << "个缺陷的面积为：" << area << endl;
    //        isBatteryDefect = true;
    //    }
    //    
    //}

    //if (isBatteryDefect == true) {
    //    string outputPath = grayImagePath + "\\Detection result.png";
    //    imwrite(outputPath, imageOutput);
    //}

}
int returnMaxIndex(vector<vector<Point>> contours) {
    //寻找最大轮廓索引位置
    int largestContourIndex = -1;
    double largestContourArea = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > largestContourArea) {
            largestContourArea = area;
            largestContourIndex = i;
        }
    }

    if (largestContourIndex == -1) {
        largestContourIndex = 0;
    }
    return largestContourIndex;
}

bool comparePixel(const element& f1, const element& f2) {
    return f1.stddev > f2.stddev;
}

//bool FlawDetect(Mat img) {
//    auto start = std::chrono::high_resolution_clock::now();//开始计算算法用时
//
//    //---------------区域分割------------------
//    int left_size = 4;//4mm
//    float resolution = 0.016;//解析值
//    int offest = 10;//偏移值
//    int pixelSize = left_size / resolution + offest;//像素尺度
//    cv::Mat binary_image;
//    cv::threshold(img, binary_image, 0, 65535, cv::THRESH_BINARY);
//    Mat img_8uc1;
//    binary_image.convertTo(img_8uc1, CV_8UC1);
//    std::vector<std::vector<cv::Point>> contours;
//    cv::findContours(img_8uc1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//    //寻找最大轮廓索引位置
//    int largestContourIndex = returnMaxIndex(contours);
//    Rect rect = boundingRect(contours[largestContourIndex]);//最小外接矩形
//
//    //---------------图像校正------------------
//    Mat comImg = img(rect);
//    Mat columnMean;
//    reduce(comImg, columnMean, 0, REDUCE_AVG, CV_64FC1);//列平均值
//    int x1 = 500;
//    int x2 = 1000;
//    double y1 = columnMean.at<double>(0, x1);
//    double y2 = columnMean.at<double>(0, x2);
//    double k = (y1 - y2) / (x1 - x2);
//    Mat K = Mat::zeros(comImg.cols, 1, CV_32FC1);//列
//    for (int i = 0; i < K.rows; i++)
//    {
//        K.at<float>(i, 0) = i;
//    }
//    Mat oneM = Mat::ones(1, comImg.rows, CV_32FC1);//行
//    Mat compensateM;
//    gemm(K, oneM, 1, Mat(), 0.0, compensateM, 0);
//    Mat trans = (compensateM * k).t();
//    Mat xBalanceSrc(comImg.rows, comImg.cols, CV_16UC1);
//    add(comImg, -trans, xBalanceSrc, cv::noArray(), CV_16UC1);
//    Mat roi = xBalanceSrc.clone();
//    //---------------图像校正------------------
//    Rect left_rect(0, 0, pixelSize, rect.height);
//    Mat roi_left = roi(left_rect);//bug
//    Rect center_rect(pixelSize, 0, rect.width - 2 * pixelSize, rect.height);
//    Mat roi_center = roi(center_rect);
//    Rect right_rect(rect.width - pixelSize, 0, pixelSize, rect.height);
//    Mat roi_right = roi(right_rect);
//    //---------------区域分割------------------
//
//    //---------------局部检测------------------
//    bool isFlaw = false;
//    int flawx = 300;
//    int flawy = 1821;
//    int rowStep = 30;
//    int colStep = 300;
//    vector<element> varVector;
//    int rows = roi_center.rows;
//    int cols = roi_center.cols;
//    for (int i = 0; i < rows; i += rowStep)
//    {
//        for (int j = 0; j < cols; j += colStep)
//        {
//            int block_height = (i + rowStep > rows) ? (rows - i) : rowStep;
//            int block_width = (j + colStep > cols) ? (cols - j) : colStep;
//            Rect rect1(j, i, block_width, block_height);
//            Mat block = roi_center(rect1);
//            vector<int> colsMean;
//            Mat columnMean;
//            reduce(block, columnMean, 0, REDUCE_AVG, CV_64FC1);//列平均值
//            Mat mat_mean, mat_stddev;
//            meanStdDev(columnMean, mat_mean, mat_stddev);
//            double mean = mat_mean.at<double>(0, 0);
//            double stddev = mat_stddev.at<double>(0, 0);
//            double coefofVar = stddev / mean;
//            element e;
//            e.isDetect = false;
//            double max;
//            double min;
//            minMaxLoc(block, &min, &max);
//            e.coefofVar = coefofVar;
//            e.XAxis = rect1.x;
//            e.YAxis = rect1.y;
//            e.stddev = stddev;
//            e.hight = block_height;
//            e.width = block_width;
//            e.stdRange = (max - min) / mean;
//            e.Mean = mean;
//            e.Max = max;
//            e.Min = min;
//            varVector.push_back(e);
//        }
//    }
//    //---------------局部检测------------------
//    // 
//    //---------------缺陷标注------------------
//    element flawE;
//    if (flawE.isDetect = true) {//是缺陷
//        int imgx = rect.x + flawE.XAxis;
//        int imgy = rect.y + flawE.YAxis;
//        Rect flawRect(imgx, imgy, flawE.width, flawE.hight);
//        Mat flawRoi = img(flawRect);
//        double rectMax;
//        minMaxLoc(flawRoi,nullptr, &rectMax);
//        Mat mat_mean, mat_std;
//        meanStdDev(flawRoi, mat_mean, mat_std);
//        double mean = mat_mean.at<double>(0, 0);
//        double threshold = (rectMax + mean)/2;
//
//        for (int indexc = 0; indexc <= flawRoi.rows; indexc++) {
//            ushort* pSrcData = flawRoi.ptr<ushort>(indexc);
//            for (int indexv = 0; indexv <= flawRoi.cols; indexv++) {
//                if ((*pSrcData++) >= threshold) {
//                    (*pSrcData++) = 65536;
//                }
//                else {
//                    *pSrcData++ = 0;
//                }
//            }
//        }
//		
//        Mat img_8uc1;
//        flawRoi.convertTo(img_8uc1, CV_8UC1);
//        vector<vector<Point>> _contours;
//        findContours(img_8uc1, _contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//        int maxIndex = returnMaxIndex(_contours);
//        double maxArea = contourArea(_contours[maxIndex]);
//        RotatedRect flawRect = minAreaRect(_contours[maxIndex]);
//        double area = flawRect.size().area();
//
//        Mat centerImg_16S;
//        img.convertTo(centerImg_16S, CV_16S, 1.0, -32768.0);
//
//        //short v = centerImg_16S.at<short>(flawRect.center.y + square.y, flawRect.center.x + square.x);
//		
//    }
//
//}
void getSubdirs(std::string path, std::vector<std::string>& files)
{
    long long hFile = 0;
    struct _finddata_t fileinfo;
    std::string p;
    if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))//是目录
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//
                {
                    getSubdirs(p.assign(path).append("\\").append(fileinfo.name), files);
                    //files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                }
            }
            else//是文件不是目录
            {
                //files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                if (std::find(files.begin(), files.end(), p.assign(path)) == files.end())
                    files.push_back(p.assign(path));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}
int fitCurve(std::vector<double> x, std::vector<double> y)
{
    //columns is 3, rows is x.size()
    cv::Mat A = cv::Mat::zeros(cv::Size(2, x.size()), CV_64FC1);
    for (int i = 0; i < x.size(); i++)
    {
        A.at<double>(i, 0) = 1;
        A.at<double>(i, 1) = sqrt(x[i]);
        //A.at<double>(i, 2) ;
        
    }

    cv::Mat b = cv::Mat::zeros(cv::Size(1, y.size()), CV_64FC1);
    for (int i = 0; i < y.size(); i++)
    {
        b.at<double>(i, 0) = y[i];
    }

    cv::Mat c;
    c = A.t() * A;
    cv::Mat d;
    d = A.t() * b;

    cv::Mat result = cv::Mat::zeros(cv::Size(1, 2), CV_64FC1);
    cv::solve(c, d, result);
    //std::cout << "A = " << A << std::endl;
    //std::cout << "b = " << b << std::endl;
    //std::cout << "result = " << result << std::endl;
    double a0 = result.at<double>(0, 0);
    double a1 = result.at<double>(1, 0);
    //double a2 = result.at<double>(2, 0);
    //std::cout << "对称轴：" << -a1 / a2 / 2 << std::endl;
    std::cout << "拟合方程：" << a0 << " + (" << a1 << "x) ";//+ (" << a2 << "x^2)"
    return 0;
}
//主函数输入(主函数在最下面)：不需要矫正的整张ROI图



Mat Feature3(Mat rowsFeature) {
    int block = 50;
    int border = 20;//取块避开上下border行
    Mat feature3 = Mat::zeros(rowsFeature.rows, rowsFeature.cols, CV_16UC1);
    Mat base = Mat::zeros(rowsFeature.rows / block, rowsFeature.cols, CV_16UC1);
    for (int j = border; j < rowsFeature.rows - block - border; j += block)
    {
        for (int i = 0; i < rowsFeature.cols; i++)
        {
            Rect core = Rect(i, j, 1, block);
            Mat Slide = rowsFeature(core).clone();
            Mat SlideSort = Slide.clone();
            cv::sort(Slide, SlideSort, SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
            int baseAvg = SlideSort.at<ushort>(block / 2, 0);
            base.at<ushort>(j / block, i) = baseAvg;
        }
    }

    for (int j = 0; j < rowsFeature.rows; j++)
    {
        ushort* p = rowsFeature.ptr<ushort>(j);
        int index = 0;
        if (j > border) {
            index = (j - border) / block;
        }
        if (index >= base.rows) {
            index = base.rows - 1;
        }
        int borderFlag = 0;
        for (int i = 1; i < rowsFeature.cols; i++)
        {
            int data = p[i];
            int Dev = 100 * abs(data - base.at<ushort>(index, i));
            if (Dev > 60000) {
                Dev = 60000;
            }

            int threshold = 10000;//真正的左边界灰度值阈值
            if ((data > threshold) && (borderFlag == 0)) {
                feature3.at<ushort>(j, 0) = i;
                borderFlag = 1;
            }
            else if (data < threshold) {
                borderFlag = 0;
                feature3.at<ushort>(j, 0) = 0;
            }
            feature3.at<ushort>(j, i) = Dev;
        }
    }
    return feature3;
}


void RowsFilter(Mat rowDataMat, int leftBegin) {
    int threshold = 40000;
    int threshold1 = 5;//从左开始大于threshold的值的个数阈值
    int threshold2 = 5000;//波动阈值
    int threshold3 = 2;//完成一次波动需要的点数阈值
    int fluctuateNum = 0;//连续波动点
    int min = 100000;
    int max = 0;
    bool flag = false;
    bool filterFlag = false;
    int extraNum = 0;
    bool extraFlag = false;
    for (int i = leftBegin; i < rowDataMat.cols; i++) {
        int data = rowDataMat.at<ushort>(0, i);
        if (data > threshold) {
            extraNum++;
            extraFlag = true;
        }
        else {
            extraNum = 0;
            extraFlag = false;
        }
        if (extraNum > threshold1) {
            filterFlag = true;
            break;
        }
        if ((data < min) && (data > 100)) {
            min = data;
        }
        else if (data - min > threshold2) {
            flag = true;
        }
        if (flag) {
            fluctuateNum++;
            if (data > max) {
                max = data;
            }
            else if (max - data > threshold2) {
                if (fluctuateNum > threshold3) {
                    filterFlag = true;
                    break;
                }
                else {
                    flag = false;
                    min = 100000;//单点波动太大，更新最小值
                    max = 0;
                    extraNum = 0;
                    fluctuateNum = 0;
                }
            }
        }
        else if (!extraFlag) {
            rowDataMat.at<ushort>(0, i) = 0;
        }
    }
    if (!filterFlag) {
        rowDataMat = 0;
    }
}
void RowsFilter_inverted(Mat rowDataMat) {
    int threshold2 = 5000;//波动阈值
    int threshold3 = 2;//完成一次波动需要的点数阈值
    int fluctuateNum = 0;//连续波动点
    int min = 100000;
    bool flag = false;
    for (int i = rowDataMat.cols - 1; i > 0; i--) {
        int data = rowDataMat.at<ushort>(0, i);
        if ((data < min) && (data > 100)) {
            min = data;
        }
        else if (data - min > threshold2) {
            flag = true;
        }
        if (!flag) {
            rowDataMat.at<ushort>(0, i) = 0;
        }
        else {
            break;
        }
    }
}


Mat Feature5(Mat rowsFeature) {
    Mat feature = rowsFeature.clone();
    for (int j = 0; j < feature.rows; j++) {
        int index = rowsFeature.at<ushort>(j, 0);
        Mat rowDataMat(1, feature.cols, feature.type(), feature.ptr(j));
        RowsFilter(rowDataMat, index);//按行进行滤波(从左往右)
        RowsFilter_inverted(rowDataMat);//按行进行滤波(从右往左)
    }
    return feature;
}

void Enhance(Mat rowsFeature, Mat Feature, vector<Point> Index) {
    Mat feature1 = Feature.clone();
    Mat feature2 = Feature.clone();
    int cols = Feature.cols;
    for (int k = 0; k < Index.size(); k++) {
        int beginRow = Index[k].x;
        int RowsNum = Index[k].y - Index[k].x + 1;
        int endRow = beginRow + RowsNum;
        int expand = RowsNum / 2;
        bool border = false;
        if (beginRow<30 || endRow>Feature.rows - 30) {
            border = true;
        }
        else {
            border = false;
        }

        if (border) {
            expand = 0;
        }
        else if (expand > 10) {
            expand = 10;
        }
        beginRow = beginRow - expand;
        RowsNum = RowsNum + 2 * expand;
        int begin_up = beginRow - RowsNum;
        int begin_down = beginRow + RowsNum;
        int end_down = begin_down + RowsNum;
        if (begin_up < 20) {
            begin_up = 20;
        }
        if (end_down > Feature.rows - 20) {
            begin_down = Feature.rows - 20 - RowsNum;
        }
        Rect defect = Rect(0, beginRow, cols, RowsNum);
        Mat result = Feature(defect);
        if (border) {
            result = 0;
        }
        else {
            Mat result_up = feature1(defect);//缺陷与上基底相减结果
            Mat result_down = feature2(defect);//缺陷与下基底相减结果
            Mat src_defect = rowsFeature(defect);//实际缺陷
            Rect background_up = Rect(0, begin_up, cols, RowsNum);
            Rect background_down = Rect(0, begin_down, cols, RowsNum);
            Mat src_up = rowsFeature(background_up);
            Mat src_down = rowsFeature(background_down);
            result_up = (src_defect - src_up) * 100;
            result_down = (src_defect - src_down) * 100;
            int threshold = 5000;
            for (int j = 0; j < result_up.rows; j++)
            {
                for (int i = 0; i < result_up.cols; i++)
                {
                    int data1 = result_up.at<ushort>(j, i);
                    int data2 = result_down.at<ushort>(j, i);
                    int data = 0;
                    if ((data1 > threshold) && (data2 > threshold)) {
                        //data = (data1 + data2) / 2;
                        data = data1 * data2 / (data1 + data2);
                    }
                    result.at<ushort>(j, i) = data;
                }
            }
        }
    }
}

Mat ObviousHandle(Mat rowsFeature, Mat Feature) {
    int begin = 0;
    int end = Feature.cols - 1;
    int cols = end - begin;
    Mat feature = Feature.clone();
    vector<Point> Index;//存储增强行的起始位置
    bool leftFlag = false;
    bool enhanceFlag = false;
    Point index;
    Mat firstRow(1, feature.cols, feature.type(), feature.ptr(0));
    Mat endRow(1, feature.cols, feature.type(), feature.ptr(feature.rows - 1));
    firstRow = 0;
    endRow = 0;
    for (int j = 1; j < feature.rows; j++) {
        ushort* p = feature.ptr<ushort>(j);
        int data = p[0];
        if ((data != 0) && (leftFlag == false)) {
            index.x = j;
            leftFlag = true;
        }
        else if ((data == 0) && (leftFlag == true)) {
            index.y = j - 1;
            leftFlag = false;
            enhanceFlag = true;
        }
        if (enhanceFlag) {
            enhanceFlag = false;
            Index.push_back(index);
        }
    }
    Enhance(rowsFeature, feature, Index);
    return feature;
}


uint RangeFeature(Mat rowsFeature, vector<NewData>& data, string path) {
    uint result = 0;
    Mat img;
    int type = 2;
    int length = 70;
    if (type == 1) {//左边界
        Rect core = Rect(0, 0, length, rowsFeature.rows);
        img = rowsFeature(core).clone();
    }
    else if (type == 2) {//右边界
        flip(rowsFeature, rowsFeature, 1); // 1 表示左右翻转
        Rect core1 = Rect(0, 0, length, rowsFeature.rows);
        img = rowsFeature(core1).clone();
    }
    Mat feature3 = Feature3(img);//块的列均值作为基底求落差特征图
    Mat feature5 = Feature5(feature3);//按行滤波(不满足要求将其滤为0)
    Mat feature = ObviousHandle(rowsFeature, feature5);//原图、待增强图(第一列是增强索引)
    feature.col(0) = 0;//从第一列开始、将索引列归零
    Mat Feature_result = Feature5(feature);

    bool resultFlaw = newLeftEdgeDetect(Feature_result, path, data);
    return result;
}





void leftEdgeDetect(Mat roi_image,vector<NewData>& data,string path) {
    uint result;
    int step1 = 5;
    Mat rowsFeature = Mat::zeros(roi_image.rows, roi_image.cols / step1, CV_16UC1);
    for (int j = 0; j < roi_image.rows; j++)
    {
        ushort* p = roi_image.ptr<ushort>(j);
        for (int i = 0; i < roi_image.cols - step1; i += step1)
        {
            int sum = 0;
            int num = 0;
            for (int k = 0; k < step1; k++) {
                if (p[i + k] > 10000) {
                    sum = sum + p[i + k];
                    num++;
                }
            }
            int avg = 0;
            if (num == 5) {
                avg = sum / step1;
            }
            else if (i > 9) {
                avg = rowsFeature.at<ushort>(j, (i / step1) - 2);
            }
            rowsFeature.at<ushort>(j, i / step1) = avg;
        }

    }
    result = RangeFeature(rowsFeature, data, path);
    
}
bool newLeftEdgeDetect(Mat image,string path, vector<NewData>& data)
{
    Mat _8uImg, dilateImg;
    Mat strElement = getStructuringElement(1, Size(3, 3));
    dilate(image, dilateImg, strElement);
    dilateImg.convertTo(_8uImg, CV_8U);
    vector<vector<Point>> contours;
    findContours(_8uImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    vector<NewData> newDataLocal;
    if (!contours.empty()) {
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 20) {
                continue;
            }
            RotatedRect minRect = minAreaRect(contour);
            Rect boundingRect = minRect.boundingRect();//容易越界
            limitBounding(boundingRect, dilateImg);
            double areaLength = 0;
            Mat roi = dilateImg(boundingRect);
            int x = 0;
            int y = 0;
            int areaMean = 0;
            Mat secondBinary, secondBinary8U;
            int thresholdMean = getHexGrayMean(roi);
            threshold(roi, secondBinary, thresholdMean, 65535, THRESH_BINARY);
            secondBinary.convertTo(secondBinary8U, CV_8U);
            vector<vector<Point>> secondContours;

            findContours(secondBinary8U, secondContours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            int secondArea = 0;
            if (!secondContours.empty()) {
                sort(secondContours.begin(), secondContours.end(), compareSize);
                RotatedRect secondMinRect = minAreaRect(secondContours[0]);
                Rect secondBounding = secondMinRect.boundingRect();
                limitBounding(secondBounding, roi);
                Mat secondRoi = roi(secondBounding);
                areaMean = getHexGrayMean(secondRoi);
                y = secondMinRect.center.y + boundingRect.y;
                x = secondMinRect.center.x + boundingRect.x;
                areaLength = secondMinRect.size.width > secondMinRect.size.height ?
                    secondMinRect.size.width : secondMinRect.size.height;
                secondArea = contourArea(secondContours[0]);
                int len = arcLength(secondContours[0], true);
                double circularity = (4 * secondArea * CV_PI) / (len * len);
                NewData newDataElement;
                newDataElement.Area = secondArea;
                newDataElement.grayscaleValue = areaMean;
                newDataElement.Rate = circularity;
                newDataElement.Length = areaLength;
                newDataElement.x = x;
                newDataElement.y = y;
                newDataElement.path = path;
                //太小                  太长              太暗	             不够圆
                //if (secondArea > 60 && areaLength < 45  ) {
                //    if(areaMean >= 7000 && circularity > 0.6)
                //        newDataLocal.push_back(newDataElement);
                //    else if(circularity <= 0.6 && circularity >= 0.3 && areaMean >= 20000)
                //        newDataLocal.push_back(newDataElement);
                //}
                newDataLocal.push_back(newDataElement);
            }
        }
    }
    for (const auto& element : newDataLocal) {
        data.push_back(element);
    }
    return true;
}
void limitBounding(Rect& rect, Mat srcImage) {
    rect.x = std::max(0, std::min(rect.x, srcImage.cols - 1));
    rect.y = std::max(0, std::min(rect.y, srcImage.rows - 1));
    rect.width = std::max(0, std::min(rect.width, srcImage.cols - rect.x));
    rect.height = std::max(0, std::min(rect.height, srcImage.rows - rect.y));
}
int getHexGrayMean(Mat roi) {
    double sum = 0;
    int count = 0;
    for (int row = 0; row < roi.rows; row++) {
        ushort* ptr = roi.ptr<ushort>(row);
        for (int col = 0; col < roi.cols; col++) {
            ushort v = *ptr++;
            if (v != 0) {
                sum = sum + v;
                count++;
            }
        }
    }
    return sum / count;
}
bool compareSize(const vector<Point>& f1, const vector<Point>& f2) {
    return f1.size() > f2.size();
}