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
struct Data {
	//int x;
	//int y;
	//int mean;
	//float std;
	//int range;
	int centerX;
	int centerY;
	int Area;
	float rate;
	int length;
	string path;
	int flag = 0;
};

void EdgeDetect(Mat roi, std::string path);
void getSubdirs(std::string path, std::vector<std::string>& files);
int returnMaxIndex(vector<vector<Point>> contours);
void leftEdgeDetect(Mat roi_image, vector<Data>& data,string path);
uint RangeFeature(Mat rowsFeature, vector<Data>& data, string path);
Mat ObviousHandle(Mat rowsFeature, Mat Feature, int begin, int end, vector<Data>& data, string path);
void Enhance(Mat rowsFeature, Mat Feature, vector<Point> Index, vector<Data>& data, string path);
Mat Feature5(Mat rowsFeature);
void RowsFilter(Mat rowDataMat, int leftBegin);
Mat Feature3(Mat rowsFeature);
bool compareArea(const Data& f1, const Data& f2);
#pragma once
