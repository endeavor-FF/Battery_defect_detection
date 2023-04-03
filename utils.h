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
using namespace cv;
using namespace std; 
//struct Data {
//	//int x;
//	//int y;
//	//int mean;
//	//float std;
//	//int range;
//	int centerX;
//	int centerY;
//	int Area;
//	float rate;
//	int length;
//	string path;
//	int flag = 0;
//};

struct NewData {
	int x;
	int y;
	int Area;
	float Rate;
	int Length;
	int grayscaleValue;
	string path;
	int flag;
};
void limitBounding(Rect& rect, Mat srcImage);
int getHexGrayMean(Mat roi);
bool compareSize(const vector<Point>& f1, const vector<Point>& f2);
void EdgeDetect(Mat roi, std::string path);
bool newLeftEdgeDetect(Mat image, string path, vector<NewData>& data);
void getSubdirs(std::string path, std::vector<std::string>& files);
int returnMaxIndex(vector<vector<Point>> contours);
void leftEdgeDetect(Mat roi_image, vector<NewData>& data,string path);
uint RangeFeature(Mat rowsFeature, vector<NewData>& data, string path);
Mat ObviousHandle(Mat rowsFeature, Mat Feature, int begin, int end);
void Enhance(Mat rowsFeature, Mat Feature, vector<Point> Index);
Mat Feature5(Mat rowsFeature);
void RowsFilter(Mat rowDataMat, int leftBegin);
Mat Feature3(Mat rowsFeature);
bool compareArea(const NewData& f1, const NewData& f2);
#pragma once
