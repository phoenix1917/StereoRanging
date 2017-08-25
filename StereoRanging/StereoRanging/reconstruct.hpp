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
* usingBoard��������ʹ�õı궨��
* @param boardNum   [Input]�궨����
*                      �궨��1��9x7��35mm����ɫ��Ե
*                      �궨��2��12x9��30mm����ɫ��Ե
*                      �궨��3��10x9��90mm����ɫ��Ե
*                      �궨��4��13x9��30mm����ɫ��Ե
* @param boardSize  [Output]���ر궨�����̸��ڽǵ���Ŀ
* @param squareSize [Output]���ر궨�巽��߳�
*/
void usingBoard(int boardNum, Size& boardSize, float& squareSize);

/**
* ����궨��ǵ�λ��
* @param boardSize  [input]�궨���ڽǵ�size
* @param squareSize [input]�궨�巽��߳�
* @param corners    [output]���صĽǵ���λ��
*/
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);

/**
* ����ͶӰ���
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
* �̶��������굽ָ��ֵ
* @param K     [InputOutputArray] �ڲ�������(CV_64FC1)
* @param point [InputArray] Ҫ�̶�����������λ��
* @return      �Ƿ�ɹ�����
*/
bool fixPrinciplePoint(Mat& K, Point2f point);

/**
* �������󣬲��ֽ�����˫Ŀ��Σ�λ�˹�ϵ�����ڲ���ʹ����ͬ����
* @param K    [InputArray] �ڲ�������
* @param R    [OutputArray] camera2 �� camera1 ����ת����
* @param T    [OutputArray] camera2 �� camera1 ��ƽ������
* @param p1   [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
* @param p2   [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
* @param mask [InputOutputArray] ƥ����flag�����ܵ�ƥ��㣨inliers��������ֵ
*/
bool findTransform(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask);

/**
* �������󣬲��ֽ�����˫Ŀ��Σ�λ�˹�ϵ�����ڲ���ʹ�ò�ͬ����
* @param K1   [InputArray] camera1 �ڲ�������
* @param K2   [InputArray] camera2 �ڲ�������
* @param R    [OutputArray] camera2 �� camera1 ����ת����
* @param T    [OutputArray] camera2 �� camera1 ��ƽ������
* @param p1   [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
* @param p2   [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
* @param mask [InputOutputArray] ƥ����flag�����ܵ�ƥ��㣨inliers��������ֵ
*/
bool findTransform(Mat& K1, Mat& K2, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask);

/**
* ȥ����ƥ���ԣ�λ�ã�
* @param p1   [InputOutputArray]
* @param mask [InputArray]
*/
void maskoutPoints(vector<Point2f>& p1, Mat& mask);

/**
* ȥ����ƥ���ԣ�����ֵ��
* @param p1   [InputOutputArray]
* @param mask [InputArray]
*/
void maskoutColors(vector<Vec3b>& p1, Mat& mask);

/**
* ���ǻ��ؽ��ռ�㣨�ڲ���ʹ����ͬ����
* @param K  [InputArray] �ڲ�������
* @param R  [InputArray] camera2 �� camera1 ����ת����
* @param T  [InputArray] camera2 �� camera1 ��ƽ������
* @param p1 [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
* @param p2 [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
* @param structure [OutputArray] �ؽ����Ŀռ�㣨������꣩
*/
void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);

/**
* ���ǻ��ؽ��ռ�㣨�ڲ���ʹ�ò�ͬ����
* @param K1 [InputArray] camera1 �ڲ�������
* @param K2 [InputArray] camera2 �ڲ�������
* @param R  [InputArray] camera2 �� camera1 ����ת����
* @param T  [InputArray] camera2 �� camera1 ��ƽ������
* @param p1 [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
* @param p2 [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
* @param structure [OutputArray] �ؽ����Ŀռ�㣨������꣩
*/
void reconstruct(Mat& K1, Mat& K2, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);

/**
* ���ǻ��ؽ��ռ�㣨�ڲ���ʹ����ͬ����
* @param K  [InputArray] �ڲ�������
* @param R1 [InputArray] camera1 ����������ϵ����ת����
* @param T1 [InputArray] camera1 ����������ϵ��ƽ������
* @param R2 [InputArray] camera2 ����������ϵ����ת����
* @param T2 [InputArray] camera2 ����������ϵ��ƽ������
* @param p1 [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
* @param p2 [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
* @param structure [OutputArray] �ؽ����Ŀռ�㣨�ռ����꣩
*/
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure);

/**
* toPoints3D ���������ת��Ϊ�ռ����ꡣ����Ϊfloat�ͣ�CV_32FC1����
* @param points4D [InputArray] �������㣬4xN
* @param points3D [OutputArray] �ռ������
*/
void toPoints3D(Mat& points4D, Mat& points3D);

/**
* saveStructure �������λ�˺͵������굽�ļ�
* @param fileName
* @param rotations
* @param motions
* @param structure
* @param colors
*/
void saveStructure(string fileName, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors);

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
* ranging ͨ��R��t������λ�߾���
* @param structure ������������ϵ�µ����꣨��λ��mm��
* @param R ��ת����
* @param t ƽ������
* @return ������ľ��루��λ��m��
*/
vector<double> ranging(Mat structure, Mat R, Mat t);
