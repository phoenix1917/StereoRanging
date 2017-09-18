#pragma once

#include <iostream>
#include <fstream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
//#include "createFit.h"

using namespace cv;
using namespace std;

enum FeatureType { SIFT, GMS };

// 显示角点提取结果
bool showCornerExt = false;
// 进行单目标定（true通过单目标定确定内参，false输入内参）
bool doSingleCalib = true;
// 进行双目标定（true通过双目标定确定外参，false输入外参）
bool doStereoCalib = true;
// 增强图像
bool doEnhance = true;
// 保存增强的图像（标定图像）
bool doSaveEnhancedImg = false;
// 进行畸变矫正
bool doDistortionCorrect = false;
// 是否训练补偿模型
bool doTrain = true;
// 对匹配点的距离进行处理，得到唯一值（true返回唯一值，false返回所有匹配点距离）
bool processRange = true;
// 对测距值进行补偿
bool doCompensate = true;
// 特征提取方式（用于manualPoints = false）
FeatureType type = GMS;
// ROI大小（横向半径，纵向半径）
Size roiSize = Size(45, 75);

// 读入的测距图像序列
vector<Mat> trainSetL, trainSetR;
vector<Mat> testSetL, testSetR;
// 执行增强时的滑动条数据
int clipL = 20, gridXL = 2, gridYL = 2;
int clipR = 20, gridXR = 2, gridYR = 2;
// 临时保存的增强图像
Mat tempEnhanceL, tempEnhanceR;
// 用于记录测距图像中的同名点
Point2f targetL, targetR;
// 用于记录测距图像中的ROI
Rect roiL, roiR;
// 用于特征匹配的ROI图像
Mat roiImgL, roiImgR;
// 使用的标定板棋盘格内角点的个数
Size boardSize;
// 标定板上每个方格的大小
float squareSize;

// 使用的数据集
String dataset = "20170909";
// 使用的标定数据组别
String calibset = "calib2";
// 使用的训练数据组别
String trainset = "train2-2";
// 使用的测试数据组别
String testset = "test2-all";
// 用于保存测距值文件的后缀
String testlabel = "compensate";

// 加载标定所用图像文件的路径
ifstream finL("data/" + dataset + "/" + calibset + "_L.txt");
ifstream finR("data/" + dataset + "/" + calibset + "_R.txt");
// 保存增强图像（标定图像）的路径
String pathEnhanced = "data/" + dataset + "/" + calibset + "_enhanced/";
// 加载训练所用图像文件的路径
ifstream fTrainL("data/" + dataset + "/" + trainset + "_L.txt");
ifstream fTrainR("data/" + dataset + "/" + trainset + "_R.txt");
// 加载测距所用的图像文件路径
ifstream fTestL("data/" + dataset + "/" + testset + "_L.txt");
ifstream fTestR("data/" + dataset + "/" + testset + "_R.txt");
ofstream foutTest("data/" + dataset + "/result_" + testset + "_" + calibset + "_" + testlabel + ".txt");
// 保存标定结果的文件
ofstream foutL("data/" + dataset + "/result_" + calibset + "_L.txt");
ofstream foutR("data/" + dataset + "/result_" + calibset + "_R.txt");
ofstream foutStereo("data/" + dataset + "/result_" + calibset + "_stereo.txt");

#define MAX_CLIP_LIMIT 200
#define MAX_GRID_SIZE_X 100
#define MAX_GRID_SIZE_Y 100

/**
* 同名点选取鼠标回调事件（左）
* @param event 鼠标操作事件类型
*          enum cv::MouseEventTypes
*          EVENT_MOUSEMOVE     滑动
*          EVENT_LBUTTONDOWN   左键按下
*          EVENT_RBUTTONDOWN   右键按下
*          EVENT_MBUTTONDOWN   中键按下
*          EVENT_LBUTTONUP     左键释放
*          EVENT_RBUTTONUP     右键释放
*          EVENT_MBUTTONUP     中键释放
*          EVENT_LBUTTONDBLCLK 左键双击
*          EVENT_RBUTTONDBLCLK 右键双击
*          EVENT_MBUTTONDBLCLK 中键双击
*          EVENT_MOUSEWHEEL    滚轮上下滑动
*          EVENT_MOUSEHWHEEL   滚轮左右滑动
* @param x     鼠标位于窗口的x坐标位置（窗口左上角默认为原点，向右为x轴，向下为y轴）
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
*          enum cv::MouseEventFlags
*          EVENT_FLAG_LBUTTON  左键拖拽
*          EVENT_FLAG_RBUTTON  右键拖拽
*          EVENT_FLAG_MBUTTON  中键拖拽
*          EVENT_FLAG_CTRLKEY  Ctrl键按下
*          EVENT_FLAG_SHIFTKEY Shift键按下
*          EVENT_FLAG_ALTKEY   Alt键按下
* @param param 自定义数据
*/
void onMouseL(int event, int x, int y, int flags, void *param);
void onMouseL_Train(int event, int x, int y, int flags, void *param);

/**
* onMouseL 局部特征提取ROI选取事件（左）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onMouseL_ROI(int event, int x, int y, int flags, void *param);
void onMouseL_ROI_Train(int event, int x, int y, int flags, void *param);

/**
* 同名点选取鼠标回调事件（右）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onMouseR(int event, int x, int y, int flags, void *param);
void onMouseR_Train(int event, int x, int y, int flags, void *param);

/**
* onMouseL 局部特征提取ROI选取事件（右）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onMouseR_ROI(int event, int x, int y, int flags, void *param);
void onMouseR_ROI_Train(int event, int x, int y, int flags, void *param);

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
* onEnhanceTrackbarL 图像增强参数调整回调（左目）
* @param pos TrackBar当前数值
* @param userdata 用户数据（optional）
*/
void onEnhanceTrackbarL(int pos, void *userdata);
void onEnhanceTrackbarL_Train(int pos, void *userdata);

/**
* onEnhanceTrackbarR 图像增强参数调整回调（右目）
* @param pos TrackBar当前数值
* @param userdata 用户数据（optional）
*/
void onEnhanceTrackbarR(int pos, void *userdata);
void onEnhanceTrackbarR_Train(int pos, void *userdata);

/**
* onEnhanceMouseL 增强窗口鼠标回调事件（左）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onEnhanceMouseL(int event, int x, int y, int flags, void* param);
void onEnhanceMouseL_Train(int event, int x, int y, int flags, void* param);

/**
* onEnhanceMouseR 增强窗口鼠标回调事件（右）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onEnhanceMouseR(int event, int x, int y, int flags, void* param);
void onEnhanceMouseR_Train(int event, int x, int y, int flags, void* param);

/**
* 数字转字符串
* @param num     输入数字
* @return outStr 返回对应的字符串
*/
string num2str(int num);

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
