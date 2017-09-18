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

// ��ʾ�ǵ���ȡ���
bool showCornerExt = false;
// ���е�Ŀ�궨��trueͨ����Ŀ�궨ȷ���ڲΣ�false�����ڲΣ�
bool doSingleCalib = true;
// ����˫Ŀ�궨��trueͨ��˫Ŀ�궨ȷ����Σ�false������Σ�
bool doStereoCalib = true;
// ��ǿͼ��
bool doEnhance = true;
// ������ǿ��ͼ�񣨱궨ͼ��
bool doSaveEnhancedImg = false;
// ���л������
bool doDistortionCorrect = false;
// �Ƿ�ѵ������ģ��
bool doTrain = true;
// ��ƥ���ľ�����д����õ�Ψһֵ��true����Ψһֵ��false��������ƥ�����룩
bool processRange = true;
// �Բ��ֵ���в���
bool doCompensate = true;
// ������ȡ��ʽ������manualPoints = false��
FeatureType type = GMS;
// ROI��С������뾶������뾶��
Size roiSize = Size(45, 75);

// ����Ĳ��ͼ������
vector<Mat> trainSetL, trainSetR;
vector<Mat> testSetL, testSetR;
// ִ����ǿʱ�Ļ���������
int clipL = 20, gridXL = 2, gridYL = 2;
int clipR = 20, gridXR = 2, gridYR = 2;
// ��ʱ�������ǿͼ��
Mat tempEnhanceL, tempEnhanceR;
// ���ڼ�¼���ͼ���е�ͬ����
Point2f targetL, targetR;
// ���ڼ�¼���ͼ���е�ROI
Rect roiL, roiR;
// ��������ƥ���ROIͼ��
Mat roiImgL, roiImgR;
// ʹ�õı궨�����̸��ڽǵ�ĸ���
Size boardSize;
// �궨����ÿ������Ĵ�С
float squareSize;

// ʹ�õ����ݼ�
String dataset = "20170909";
// ʹ�õı궨�������
String calibset = "calib2";
// ʹ�õ�ѵ���������
String trainset = "train2-2";
// ʹ�õĲ����������
String testset = "test2-all";
// ���ڱ�����ֵ�ļ��ĺ�׺
String testlabel = "compensate";

// ���ر궨����ͼ���ļ���·��
ifstream finL("data/" + dataset + "/" + calibset + "_L.txt");
ifstream finR("data/" + dataset + "/" + calibset + "_R.txt");
// ������ǿͼ�񣨱궨ͼ�񣩵�·��
String pathEnhanced = "data/" + dataset + "/" + calibset + "_enhanced/";
// ����ѵ������ͼ���ļ���·��
ifstream fTrainL("data/" + dataset + "/" + trainset + "_L.txt");
ifstream fTrainR("data/" + dataset + "/" + trainset + "_R.txt");
// ���ز�����õ�ͼ���ļ�·��
ifstream fTestL("data/" + dataset + "/" + testset + "_L.txt");
ifstream fTestR("data/" + dataset + "/" + testset + "_R.txt");
ofstream foutTest("data/" + dataset + "/result_" + testset + "_" + calibset + "_" + testlabel + ".txt");
// ����궨������ļ�
ofstream foutL("data/" + dataset + "/result_" + calibset + "_L.txt");
ofstream foutR("data/" + dataset + "/result_" + calibset + "_R.txt");
ofstream foutStereo("data/" + dataset + "/result_" + calibset + "_stereo.txt");

#define MAX_CLIP_LIMIT 200
#define MAX_GRID_SIZE_X 100
#define MAX_GRID_SIZE_Y 100

/**
* ͬ����ѡȡ���ص��¼�����
* @param event �������¼�����
*          enum cv::MouseEventTypes
*          EVENT_MOUSEMOVE     ����
*          EVENT_LBUTTONDOWN   �������
*          EVENT_RBUTTONDOWN   �Ҽ�����
*          EVENT_MBUTTONDOWN   �м�����
*          EVENT_LBUTTONUP     ����ͷ�
*          EVENT_RBUTTONUP     �Ҽ��ͷ�
*          EVENT_MBUTTONUP     �м��ͷ�
*          EVENT_LBUTTONDBLCLK ���˫��
*          EVENT_RBUTTONDBLCLK �Ҽ�˫��
*          EVENT_MBUTTONDBLCLK �м�˫��
*          EVENT_MOUSEWHEEL    �������»���
*          EVENT_MOUSEHWHEEL   �������һ���
* @param x     ���λ�ڴ��ڵ�x����λ�ã��������Ͻ�Ĭ��Ϊԭ�㣬����Ϊx�ᣬ����Ϊy�ᣩ
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
*          enum cv::MouseEventFlags
*          EVENT_FLAG_LBUTTON  �����ק
*          EVENT_FLAG_RBUTTON  �Ҽ���ק
*          EVENT_FLAG_MBUTTON  �м���ק
*          EVENT_FLAG_CTRLKEY  Ctrl������
*          EVENT_FLAG_SHIFTKEY Shift������
*          EVENT_FLAG_ALTKEY   Alt������
* @param param �Զ�������
*/
void onMouseL(int event, int x, int y, int flags, void *param);
void onMouseL_Train(int event, int x, int y, int flags, void *param);

/**
* onMouseL �ֲ�������ȡROIѡȡ�¼�����
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onMouseL_ROI(int event, int x, int y, int flags, void *param);
void onMouseL_ROI_Train(int event, int x, int y, int flags, void *param);

/**
* ͬ����ѡȡ���ص��¼����ң�
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onMouseR(int event, int x, int y, int flags, void *param);
void onMouseR_Train(int event, int x, int y, int flags, void *param);

/**
* onMouseL �ֲ�������ȡROIѡȡ�¼����ң�
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onMouseR_ROI(int event, int x, int y, int flags, void *param);
void onMouseR_ROI_Train(int event, int x, int y, int flags, void *param);

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
* onEnhanceTrackbarL ͼ����ǿ���������ص�����Ŀ��
* @param pos TrackBar��ǰ��ֵ
* @param userdata �û����ݣ�optional��
*/
void onEnhanceTrackbarL(int pos, void *userdata);
void onEnhanceTrackbarL_Train(int pos, void *userdata);

/**
* onEnhanceTrackbarR ͼ����ǿ���������ص�����Ŀ��
* @param pos TrackBar��ǰ��ֵ
* @param userdata �û����ݣ�optional��
*/
void onEnhanceTrackbarR(int pos, void *userdata);
void onEnhanceTrackbarR_Train(int pos, void *userdata);

/**
* onEnhanceMouseL ��ǿ�������ص��¼�����
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onEnhanceMouseL(int event, int x, int y, int flags, void* param);
void onEnhanceMouseL_Train(int event, int x, int y, int flags, void* param);

/**
* onEnhanceMouseR ��ǿ�������ص��¼����ң�
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onEnhanceMouseR(int event, int x, int y, int flags, void* param);
void onEnhanceMouseR_Train(int event, int x, int y, int flags, void* param);

/**
* ����ת�ַ���
* @param num     ��������
* @return outStr ���ض�Ӧ���ַ���
*/
string num2str(int num);

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
