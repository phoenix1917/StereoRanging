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
void gmsMatch(Mat &img1, Mat &img2);
void gmsMatch(Mat &img1, Mat &img2, vector<Point2f> &matchL, vector<Point2f> &matchR);
void extractSIFTFeatures(vector<string>& image_names, vector<vector<KeyPoint>>& key_points_for_all, vector<Mat>& descriptor_for_all, vector<vector<Vec3b>>& colors_for_all);
void extractSIFTFeatures(vector<Mat>& images, vector<vector<KeyPoint>>& keyPoints4All, vector<Mat>& descriptor4All, vector<vector<Vec3b>>& colors4All);
void matchSIFTFeatures(Mat& query, Mat& train, vector<DMatch>& matches);
void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2, vector<DMatch> matches, vector<Point2f>& out_p1, vector<Point2f>& out_p2);
void getMatchedColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches, vector<Vec3b>& out_c1, vector<Vec3b>& out_c2);
void procCLAHE(Mat &src, Mat &dst, double clipLimit, Size tileGridSize);


