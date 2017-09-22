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
// �ֶ�ѡȡ����
bool manualPoints = true;
// ����ģ��ѡ��
FitType fit = Poly;
// ������ȡ��ʽ
FeatureType feature = GMS;
// ROI��С������뾶������뾶��
Size roiSize = Size(45, 75);

// ʹ�õ����ݼ�
String dataset = "20170909";
// ʹ�õı궨�������
String calibset = "calib2";
// ʹ�õ�ѵ���������
String trainset = "train2-1";
// ʹ�õĲ����������
String testset = "test2-all";
// ���ڱ�����ֵ�ļ��ĺ�׺
String testlabel = "compensate";

// ����Ĳ��ͼ������
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

// ����ѵ������ͼ���ļ���·��
String trainResult = "../data/" + dataset + "/result_" + trainset + "_" + calibset + ".yaml";
// ���ز�����õ�ͼ���ļ�·��
ifstream fTestL("../data/" + dataset + "/" + testset + "_L.txt");
ifstream fTestR("../data/" + dataset + "/" + testset + "_R.txt");
ifstream fTestGT("../data/" + dataset + "/" + testset + "_groundtruth(2).txt");
ofstream foutTest("../data/" + dataset + "/result_" + testset + "_" + calibset + "_" + testlabel + ".txt");

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

/**
* onMouseL �ֲ�������ȡROIѡȡ�¼�����
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onMouseL_ROI(int event, int x, int y, int flags, void *param);

/**
* ͬ����ѡȡ���ص��¼����ң�
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onMouseR(int event, int x, int y, int flags, void *param);

/**
* onMouseL �ֲ�������ȡROIѡȡ�¼����ң�
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onMouseR_ROI(int event, int x, int y, int flags, void *param);

/**
* onEnhanceTrackbarL ͼ����ǿ���������ص�����Ŀ��
* @param pos TrackBar��ǰ��ֵ
* @param userdata �û����ݣ�optional��
*/
void onEnhanceTrackbarL(int pos, void *userdata);

/**
* onEnhanceTrackbarR ͼ����ǿ���������ص�����Ŀ��
* @param pos TrackBar��ǰ��ֵ
* @param userdata �û����ݣ�optional��
*/
void onEnhanceTrackbarR(int pos, void *userdata);

/**
* onEnhanceMouseL ��ǿ�������ص��¼�����
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onEnhanceMouseL(int event, int x, int y, int flags, void* param);

/**
* onEnhanceMouseR ��ǿ�������ص��¼����ң�
* @param event �������¼�����
* @param x     ���λ�ڴ��ڵ�x����λ��
* @param y     ���λ�ڴ��ڵ�y����λ��
* @param flags �����ק�����������¼���־λ
* @param param �Զ�������
*/
void onEnhanceMouseR(int event, int x, int y, int flags, void* param);
