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

enum FeatureType { SIFT, GMS };
enum FitType { Poly, Exp2 };

/**
* 数字转字符串
* @param num     输入数字
* @return outStr 返回对应的字符串
*/
string num2str(int num);

/**
* getROI 根据选取的点在输入图像上生成ROI区域，并返回ROI图像
* @param img     [input]输入图像
* @param center  [input]输入的ROI中心点
* @param roiSize [input]ROI的大小（半径）
* @param roi     [output]ROI区域位置
* @param roiImg  [output]ROI图像
*/
void getROI(Mat& img, Point center, Size roiSize, Rect& roi, Mat& roiImg);

/**
* 输出标定结果到控制台。
* 输出内参数矩阵、畸变向量、总重投影误差；
* 输出焦距及误差、主点坐标及误差、畸变向量及误差；
* 输出畸变向量其他值的误差、外参数误差、每幅图像的重投影误差。
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
* @param stdDevIntrinsics
* @param stdDevExtrinsics
* @param perViewErrors
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics, Mat &stdDevExtrinsics, vector<double> &perViewErrors);

/**
* 输出标定结果到文件。
* 输出内参数矩阵、畸变向量、总重投影误差；
* 输出焦距及误差、主点坐标及误差、畸变向量及误差；
* 输出畸变向量其他值的误差、外参数误差、每幅图像的重投影误差。
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
* @param stdDevIntrinsics
* @param stdDevExtrinsics
* @param perViewErrors
* @param fout
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics, Mat &stdDevExtrinsics, vector<double> &perViewErrors, ofstream &fout);

/**
* 输出标定结果到控制台。
* 输出内参数矩阵、畸变向量、总重投影误差；
* 输出焦距及误差、主点坐标及误差、畸变向量及误差；
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
* @param stdDevIntrinsics
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics);

/**
* 输出标定结果到文件。
* 输出内参数矩阵、畸变向量、总重投影误差；
* 输出焦距、主点坐标、畸变向量；
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError);

/**
* saveCorrspondingPoints 保存匹配点坐标到文件
* @param fileName 保存到的文件路径
* @param K1 第1组图像对应相机的内参数矩阵
* @param K2 第2组图像对应相机的内参数矩阵
* @param R 相机2相对于相机1的旋转矩阵
* @param t 相机2相对于相机1的平移矩阵
* @param points1 第1组匹配点坐标
* @param points2 第2组匹配点坐标
* @param groundTruth 真实测距值
*/
void saveCorrspondingPoints(string fileName, Mat K1, Mat K2, Mat R, Mat t, vector<vector<Point2f>> points1, vector<vector<Point2f>> points2, int corrPointsCount, vector<float> groundTruth);

/**
* saveTrainResults 保存训练参数到文件
* @param fileName 保存到的文件路径
* @param K1 第1组图像对应相机的内参数矩阵
* @param K2 第2组图像对应相机的内参数矩阵
* @param R 相机2相对于相机1的旋转矩阵
* @param t 相机2相对于相机1的平移矩阵
* @param trainVal 用于训练的测距值
* @param groundTruth 真实测距值
* @param fit 拟合模型类别
* @param coef 拟合出的参数
*/
void saveTrainResults(string fileName, Mat K1, Mat K2, Mat R, Mat t, vector<float> trainVal, vector<float> groundTruth, FitType fit, vector<float> coef);