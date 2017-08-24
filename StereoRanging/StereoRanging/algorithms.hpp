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
* gmsMatch GMS����ƥ��
* @param img1 ��ƥ��ͼ��1
* @param img2 ��ƥ��ͼ��2
*/
void gmsMatch(Mat &img1, Mat &img2);

/**
* gmsMatch GMS����ƥ��
* @param img1 ͼ��1
* @param img2 ͼ��2
* @param matchL ͼ��1��ƥ���
* @param matchR ͼ��2��ƥ���
*/
void gmsMatch(Mat &img1, Mat &img2, vector<Point2f> &matchL, vector<Point2f> &matchR);

/**
* extractSIFTFeatures ��ͼ���б���ȡSIFT����
* @param imageNames     [input]ͼ�������б�
* @param keyPoints4All  [output]����ͼ��������λ���б�
* @param descriptor4All [output]����ͼ�������������б�
* @param colors4All     [output]����ͼ�������������б�
*/
void extractSIFTFeatures(vector<string>& image_names, vector<vector<KeyPoint>>& key_points_for_all, vector<Mat>& descriptor_for_all, vector<vector<Vec3b>>& colors_for_all);

/**
* extractSIFTFeatures ��ͼ���б���ȡSIFT����
* @param images         [input]ͼ���б�
* @param keyPoints4All  [output]����ͼ��������λ���б�
* @param descriptor4All [output]����ͼ�������������б�
* @param colors4All     [output]����ͼ�������������б�
*/
void extractSIFTFeatures(vector<Mat>& images, vector<vector<KeyPoint>>& keyPoints4All, vector<Mat>& descriptor4All, vector<vector<Vec3b>>& colors4All);

/**
* matchSIFTFeatures SIFT����ƥ��
* @param query   ͼ��1���������������
* @param train   ͼ��2���������������
* @param matches ƥ����
*/
void matchSIFTFeatures(Mat& query, Mat& train, vector<DMatch>& matches);

/**
* getMatchedPoints ��ȡƥ���Ե�λ��
* @param p1
* @param p2
* @param matches
* @param out_p1
* @param out_p2
*/
void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2);

/**
* getMatchedColors ��ȡƥ���Ե�����ֵ
* @param c1
* @param c2
* @param matches
* @param out_c1
* @param out_c2
*/
void getMatchedColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches, vector<Vec3b>& out_c1, vector<Vec3b>& out_c2);

/**
* procCLAHE CLAHE��ǿ
* @param src ԭͼ��
* @param dst ��ǿ��ͼ��
* @param clipLimit �ü��޷�
* @param tileGridSize ����ָ�
*/
void procCLAHE(Mat &src, Mat &dst, double clipLimit, Size tileGridSize);

inline Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

inline void imresize(Mat &src, int height);
