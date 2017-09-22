#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>

#include "utils.hpp"

using namespace cv;
using namespace std;

// 增强图像
extern bool doEnhance;
// 保存增强的图像（标定图像）
extern bool doSaveEnhancedImg;
// 进行畸变矫正
extern bool doDistortionCorrect;
// 对匹配点的距离进行处理，得到唯一值（true返回唯一值，false返回所有匹配点距离）
extern bool processRange;
// 手动选取测距点
extern bool manualPoints;
// 补偿模型选择
extern FitType fit;
// 特征提取方式
extern FeatureType feature;
// ROI大小（横向半径，纵向半径）
extern Size roiSize;

// 使用的数据集
extern String dataset;
// 使用的标定数据组别
extern String calibset;
// 使用的训练数据组别
extern String trainset;
// 使用的测试数据组别
extern String testset;
// 用于保存测距值文件的后缀
extern String testlabel;

// 读入的测距图像序列
extern vector<Mat> testSetL, testSetR;
// 执行增强时的滑动条数据
extern int clipL, gridXL, gridYL;
extern int clipR, gridXR, gridYR;
// 临时保存的增强图像
extern Mat tempEnhanceL, tempEnhanceR;
// 用于记录测距图像中的同名点
extern Point2f targetL, targetR;
// 用于记录测距图像中的ROI
extern Rect roiL, roiR;
// 用于特征匹配的ROI图像
extern Mat roiImgL, roiImgR;
// 使用的标定板棋盘格内角点的个数
extern Size boardSize;
// 标定板上每个方格的大小
extern float squareSize;

// 加载训练所用图像文件的路径
extern String trainResult;
// 加载测距所用的图像文件路径
extern ifstream fTestL;
extern ifstream fTestR;
extern ifstream fTestGT;
extern ofstream foutTest;

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

/**
* onMouseL 局部特征提取ROI选取事件（左）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onMouseL_ROI(int event, int x, int y, int flags, void *param);

/**
* 同名点选取鼠标回调事件（右）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onMouseR(int event, int x, int y, int flags, void *param);

/**
* onMouseL 局部特征提取ROI选取事件（右）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onMouseR_ROI(int event, int x, int y, int flags, void *param);

/**
* onEnhanceTrackbarL 图像增强参数调整回调（左目）
* @param pos TrackBar当前数值
* @param userdata 用户数据（optional）
*/
void onEnhanceTrackbarL(int pos, void *userdata);

/**
* onEnhanceTrackbarR 图像增强参数调整回调（右目）
* @param pos TrackBar当前数值
* @param userdata 用户数据（optional）
*/
void onEnhanceTrackbarR(int pos, void *userdata);

/**
* onEnhanceMouseL 增强窗口鼠标回调事件（左）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onEnhanceMouseL(int event, int x, int y, int flags, void* param);

/**
* onEnhanceMouseR 增强窗口鼠标回调事件（右）
* @param event 鼠标操作事件类型
* @param x     鼠标位于窗口的x坐标位置
* @param y     鼠标位于窗口的y坐标位置
* @param flags 鼠标拖拽及键鼠联合事件标志位
* @param param 自定义数据
*/
void onEnhanceMouseR(int event, int x, int y, int flags, void* param);
