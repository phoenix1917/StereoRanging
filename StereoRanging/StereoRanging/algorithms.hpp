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
void extractSIFTFeatures(vector<string>& image_names, vector<vector<KeyPoint>>& key_points_for_all, vector<Mat>& descriptor_for_all, vector<vector<Vec3b>>& colors_for_all);

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
* @param p1
* @param p2
* @param matches
* @param out_p1
* @param out_p2
*/
void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2);

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

inline Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

inline void imresize(Mat &src, int height);
