#pragma once

#include <iostream>
#include <fstream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>

using namespace cv;
using namespace std;

void onMouseL(int event, int x, int y, int flags, void *param);
void onMouseL_ROI(int event, int x, int y, int flags, void *param);
void onMouseR(int event, int x, int y, int flags, void *param);
void onMouseR_ROI(int event, int x, int y, int flags, void *param);
void getROI(Mat& img, Point center, Size roiSize, Rect& roi, Mat& roiImg);
string num2str(int num);

void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics, Mat &stdDevExtrinsics, vector<double> &perViewErrors);
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics, Mat &stdDevExtrinsics, vector<double> &perViewErrors, ofstream &fout);
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics);
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError);

