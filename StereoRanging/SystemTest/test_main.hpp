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
extern bool doEnhance;
// ������ǿ��ͼ�񣨱궨ͼ��
extern bool doSaveEnhancedImg;
// ���л������
extern bool doDistortionCorrect;
// ��ƥ���ľ�����д����õ�Ψһֵ��true����Ψһֵ��false��������ƥ�����룩
extern bool processRange;
// �ֶ�ѡȡ����
extern bool manualPoints;
// ����ģ��ѡ��
extern FitType fit;
// ������ȡ��ʽ
extern FeatureType feature;
// ROI��С������뾶������뾶��
extern Size roiSize;

// ʹ�õ����ݼ�
extern String dataset;
// ʹ�õı궨�������
extern String calibset;
// ʹ�õ�ѵ���������
extern String trainset;
// ʹ�õĲ����������
extern String testset;
// ���ڱ�����ֵ�ļ��ĺ�׺
extern String testlabel;

// ����Ĳ��ͼ������
extern vector<Mat> testSetL, testSetR;
// ִ����ǿʱ�Ļ���������
extern int clipL, gridXL, gridYL;
extern int clipR, gridXR, gridYR;
// ��ʱ�������ǿͼ��
extern Mat tempEnhanceL, tempEnhanceR;
// ���ڼ�¼���ͼ���е�ͬ����
extern Point2f targetL, targetR;
// ���ڼ�¼���ͼ���е�ROI
extern Rect roiL, roiR;
// ��������ƥ���ROIͼ��
extern Mat roiImgL, roiImgR;
// ʹ�õı궨�����̸��ڽǵ�ĸ���
extern Size boardSize;
// �궨����ÿ������Ĵ�С
extern float squareSize;

// ����ѵ������ͼ���ļ���·��
extern String trainResult;
// ���ز�����õ�ͼ���ļ�·��
extern ifstream fTestL;
extern ifstream fTestR;
extern ifstream fTestGT;
extern ofstream foutTest;

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
