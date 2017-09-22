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
* gmsMatch GMS特征匹配
* @param img1 待匹配图像1
* @param img2 待匹配图像2
*/
void gmsMatch(Mat &img1, Mat &img2);

/**
* gmsMatch GMS特征匹配
* @param img1 图像1
* @param img2 图像2
* @param matchL 图像1的匹配点
* @param matchR 图像2的匹配点
*/
void gmsMatch(Mat &img1, Mat &img2, vector<Point2f> &matchL, vector<Point2f> &matchR);

/**
* extractSIFTFeatures 对图像列表提取SIFT特征
* @param imageNames     [input]图像名称列表
* @param keyPoints4All  [output]所有图像特征点位置列表
* @param descriptor4All [output]所有图像特征描述子列表
* @param colors4All     [output]所有图像特征点像素列表
*/
void extractSIFTFeatures(vector<string>& imageNames, vector<vector<KeyPoint>>& keyPoints4All, vector<Mat>& descriptor4All, vector<vector<Vec3b>>& colors4All);

/**
* extractSIFTFeatures 对图像列表提取SIFT特征
* @param images         [input]图像列表
* @param keyPoints4All  [output]所有图像特征点位置列表
* @param descriptor4All [output]所有图像特征描述子列表
* @param colors4All     [output]所有图像特征点像素列表
*/
void extractSIFTFeatures(vector<Mat>& images, vector<vector<KeyPoint>>& keyPoints4All, vector<Mat>& descriptor4All, vector<vector<Vec3b>>& colors4All);

/**
* matchSIFTFeatures SIFT特征匹配
* @param query   图像1特征点的特征描述
* @param train   图像2特征点的特征描述
* @param matches 匹配点对
*/
void matchSIFTFeatures(Mat& query, Mat& train, vector<DMatch>& matches);

/**
* getMatchedPoints 获取匹配点对的位置
* @param p1 图像1上的特征点
* @param p2 图像2上的特征点
* @param matches 匹配信息
* @param out_p1 匹配后的图像1特征点
* @param out_p2 匹配后的图像2特征点
*/
void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2);
void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<KeyPoint>& out_p1, vector<KeyPoint>& out_p2);

/**
* getMatchedColors 获取匹配点对的像素值
* @param c1
* @param c2
* @param matches
* @param out_c1
* @param out_c2
*/
void getMatchedColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches, vector<Vec3b>& out_c1, vector<Vec3b>& out_c2);

/**
* procCLAHE CLAHE增强
* @param src 原图像
* @param dst 增强的图像
* @param clipLimit 裁剪限幅
* @param tileGridSize 区域分割
*/
void procCLAHE(Mat &src, Mat &dst, double clipLimit, Size tileGridSize);

/**
 * DrawInlier 绘制匹配对
 * @param src1 图像1
 * @param src2 图像2
 * @param kpt1 图像1上的特征点
 * @param kpt2 图像2上的特征点
 * @param inlier 匹配对
 * @param type 图像显示类型
 * @return 绘制了匹配对的图像
 */
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, int type);

/**
 * imresize 按比例放缩图像
 * @param src 图像
 * @param height 要放缩到的高度
 */
inline void imresize(Mat &src, int height);

/**
 * compensateExp2 计算二阶指数补偿
 * @param coef 补偿函数的系数
 * @param x 函数输入
 * @return 补偿后的值
 */
double compensateExp2(vector<float> coef, double x);

/**
* compensatePoly 计算多项式补偿
* @param coef 补偿函数的系数
* @param x 函数输入
* @return 补偿后的值
*/
double compensatePoly(vector<float> coef, double x);

/**
 * polyfit2 拟合多项式补偿函数
 * @param train 训练值
 * @param truth 真值
 * @param n 要拟合的多项式次数
 * @return 补偿函数的系数
 */
vector<float> polyfit2(vector<float> &train, vector<float> &truth, int n);

/**
* exp2fit 拟合二阶指数补偿函数（GD）
* @param train 训练值
* @param truth 真值
* @param learningRate 学习率
* @return 补偿函数的系数
*/
// vector<float> exp2fit(vector<float> &trainVal, vector<float> &groundTruth);
vector<float> exp2fit(vector<float> &trainVal, vector<float> &groundTruth, double learningRate);