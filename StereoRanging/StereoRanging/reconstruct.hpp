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

void usingBoard(int boardNum, Size& boardSize, float& squareSize);
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);
double computeReprojectionErrors(vector<vector<Point3f> >& objectPoints, vector<vector<Point2f> >& imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs, Mat& cameraMatrix, Mat& distCoeffs, vector<float>& perViewErrors);
bool fixPrinciplePoint(Mat& K, Point2f point);
bool findTransform(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask);
bool findTransform(Mat& K1, Mat& K2, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask);
void maskoutPoints(vector<Point2f>& p1, Mat& mask);
void maskoutColors(vector<Vec3b>& p1, Mat& mask);
void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);
void reconstruct(Mat& K1, Mat& K2, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);
void toPoints3D(Mat& points4D, Mat& points3D);
void saveStructure(string fileName, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors);
vector<double> ranging(Mat structure, Mat R, Mat t);
