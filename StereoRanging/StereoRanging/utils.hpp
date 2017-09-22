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
* ����ת�ַ���
* @param num     ��������
* @return outStr ���ض�Ӧ���ַ���
*/
string num2str(int num);

/**
* getROI ����ѡȡ�ĵ�������ͼ��������ROI���򣬲�����ROIͼ��
* @param img     [input]����ͼ��
* @param center  [input]�����ROI���ĵ�
* @param roiSize [input]ROI�Ĵ�С���뾶��
* @param roi     [output]ROI����λ��
* @param roiImg  [output]ROIͼ��
*/
void getROI(Mat& img, Point center, Size roiSize, Rect& roi, Mat& roiImg);

/**
* ����궨���������̨��
* ����ڲ������󡢻�������������ͶӰ��
* ������༰���������꼰��������������
* ���������������ֵ�����������ÿ��ͼ�����ͶӰ��
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
* @param stdDevIntrinsics
* @param stdDevExtrinsics
* @param perViewErrors
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics, Mat &stdDevExtrinsics, vector<double> &perViewErrors);

/**
* ����궨������ļ���
* ����ڲ������󡢻�������������ͶӰ��
* ������༰���������꼰��������������
* ���������������ֵ�����������ÿ��ͼ�����ͶӰ��
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
* ����궨���������̨��
* ����ڲ������󡢻�������������ͶӰ��
* ������༰���������꼰��������������
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
* @param stdDevIntrinsics
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError, Mat &stdDevIntrinsics);

/**
* ����궨������ļ���
* ����ڲ������󡢻�������������ͶӰ��
* ������ࡢ�������ꡢ����������
* @param cameraMatrix
* @param distCoeffs
* @param reprojectionError
*/
void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError);

/**
* saveCorrspondingPoints ����ƥ������굽�ļ�
* @param fileName ���浽���ļ�·��
* @param K1 ��1��ͼ���Ӧ������ڲ�������
* @param K2 ��2��ͼ���Ӧ������ڲ�������
* @param R ���2��������1����ת����
* @param t ���2��������1��ƽ�ƾ���
* @param points1 ��1��ƥ�������
* @param points2 ��2��ƥ�������
* @param groundTruth ��ʵ���ֵ
*/
void saveCorrspondingPoints(string fileName, Mat K1, Mat K2, Mat R, Mat t, vector<vector<Point2f>> points1, vector<vector<Point2f>> points2, int corrPointsCount, vector<float> groundTruth);

/**
* saveTrainResults ����ѵ���������ļ�
* @param fileName ���浽���ļ�·��
* @param K1 ��1��ͼ���Ӧ������ڲ�������
* @param K2 ��2��ͼ���Ӧ������ڲ�������
* @param R ���2��������1����ת����
* @param t ���2��������1��ƽ�ƾ���
* @param trainVal ����ѵ���Ĳ��ֵ
* @param groundTruth ��ʵ���ֵ
* @param fit ���ģ�����
* @param coef ��ϳ��Ĳ���
*/
void saveTrainResults(string fileName, Mat K1, Mat K2, Mat R, Mat t, vector<float> trainVal, vector<float> groundTruth, FitType fit, vector<float> coef);