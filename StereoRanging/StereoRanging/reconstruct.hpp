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

/**
* usingBoard：定义所使用的标定板
* @param boardNum   [Input]标定板编号
*                      标定板1：9x7，35mm，白色边缘
*                      标定板2：12x9，30mm，黑色边缘
*                      标定板3：10x9，90mm，白色边缘
*                      标定板4：13x9，30mm，白色边缘
* @param boardSize  [Output]返回标定板棋盘格内角点数目
* @param squareSize [Output]返回标定板方格边长
*/
void usingBoard(int boardNum, Size& boardSize, float& squareSize);

/**
* 计算标定板角点位置
* @param boardSize  [input]标定板内角点size
* @param squareSize [input]标定板方格边长
* @param corners    [output]返回的角点标称位置
*/
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);

/**
* 计算投影误差
* @param objectPoints
* @param imagePoints
* @param rvecs
* @param tvecs
* @param cameraMatrix
* @param distCoeffs
* @param perViewErrors
*/
double computeReprojectionErrors(vector<vector<Point3f> >& objectPoints, vector<vector<Point2f> >& imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs, Mat& cameraMatrix, Mat& distCoeffs, vector<float>& perViewErrors);

/**
* 固定主点坐标到指定值
* @param K     [InputOutputArray] 内参数矩阵(CV_64FC1)
* @param point [InputArray] 要固定的主点坐标位置
* @return      是否成功操作
*/
bool fixPrinciplePoint(Mat& K, Point2f point);

/**
* 求本征矩阵，并分解出相机双目外参（位姿关系）（内参数使用相同矩阵）
* @param K    [InputArray] 内参数矩阵
* @param R    [OutputArray] camera2 对 camera1 的旋转矩阵
* @param T    [OutputArray] camera2 对 camera1 的平移向量
* @param p1   [InputArray] 同名点对在 camera1 图像上的点坐标
* @param p2   [InputArray] 同名点对在 camera2 图像上的点坐标
* @param mask [InputOutputArray] 匹配点对flag，接受的匹配点（inliers）返回正值
*/
bool findTransform(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask);

/**
* 求本征矩阵，并分解出相机双目外参（位姿关系）（内参数使用不同矩阵）
* @param K1   [InputArray] camera1 内参数矩阵
* @param K2   [InputArray] camera2 内参数矩阵
* @param R    [OutputArray] camera2 对 camera1 的旋转矩阵
* @param T    [OutputArray] camera2 对 camera1 的平移向量
* @param p1   [InputArray] 同名点对在 camera1 图像上的点坐标
* @param p2   [InputArray] 同名点对在 camera2 图像上的点坐标
* @param mask [InputOutputArray] 匹配点对flag，接受的匹配点（inliers）返回正值
*/
bool findTransform(Mat& K1, Mat& K2, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask);

/**
* 去除不匹配点对（位置）
* @param p1   [InputOutputArray]
* @param mask [InputArray]
*/
void maskoutPoints(vector<Point2f>& p1, Mat& mask);

/**
* 去除不匹配点对（像素值）
* @param p1   [InputOutputArray]
* @param mask [InputArray]
*/
void maskoutColors(vector<Vec3b>& p1, Mat& mask);

/**
* 三角化重建空间点（内参数使用相同矩阵）
* @param K  [InputArray] 内参数矩阵
* @param R  [InputArray] camera2 对 camera1 的旋转矩阵
* @param T  [InputArray] camera2 对 camera1 的平移向量
* @param p1 [InputArray] 同名点对在 camera1 图像上的点坐标
* @param p2 [InputArray] 同名点对在 camera2 图像上的点坐标
* @param structure [OutputArray] 重建出的空间点（齐次坐标）
*/
void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);

/**
* 三角化重建空间点（内参数使用不同矩阵）
* @param K1 [InputArray] camera1 内参数矩阵
* @param K2 [InputArray] camera2 内参数矩阵
* @param R  [InputArray] camera2 对 camera1 的旋转矩阵
* @param T  [InputArray] camera2 对 camera1 的平移向量
* @param p1 [InputArray] 同名点对在 camera1 图像上的点坐标
* @param p2 [InputArray] 同名点对在 camera2 图像上的点坐标
* @param structure [OutputArray] 重建出的空间点（齐次坐标）
*/
void reconstruct(Mat& K1, Mat& K2, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);

/**
* 三角化重建空间点（内参数使用相同矩阵）
* @param K  [InputArray] 内参数矩阵
* @param R1 [InputArray] camera1 对世界坐标系的旋转矩阵
* @param T1 [InputArray] camera1 对世界坐标系的平移向量
* @param R2 [InputArray] camera2 对世界坐标系的旋转矩阵
* @param T2 [InputArray] camera2 对世界坐标系的平移向量
* @param p1 [InputArray] 同名点对在 camera1 图像上的点坐标
* @param p2 [InputArray] 同名点对在 camera2 图像上的点坐标
* @param structure [OutputArray] 重建出的空间点（空间坐标）
*/
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);

/**
* toPoints3D 将齐次坐标转换为空间坐标。数据为float型（CV_32FC1）。
* @param points4D [InputArray] 齐次坐标点，4xN
* @param points3D [OutputArray] 空间坐标点
*/
void toPoints3D(Mat& points4D, Mat& points3D);

/**
* saveStructure 保存相机位姿和点云坐标到文件
* @param fileName
* @param rotations
* @param motions
* @param structure
* @param colors
*/
void saveStructure(string fileName, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors);

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
* ranging 通过R，t计算中位线距离
* @param structure 点在世界坐标系下的坐标（单位：mm）
* @param R 旋转矩阵
* @param t 平移向量
* @return 计算出的距离（单位：m）
*/
vector<double> ranging(Mat structure, Mat R, Mat t);
