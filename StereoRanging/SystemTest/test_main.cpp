#include <iostream>
#include <fstream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>

#include "reconstruct.hpp"
#include "algorithms.hpp"
#include "utils.hpp"
#include "test_main.hpp"

using namespace cv;
using namespace std;

//#define UNIT_TEST

#ifdef UNIT_TEST
int main() {

}
#else
int main() {
    // ������ʹ�õı궨��
    usingBoard(3, boardSize, squareSize);

    // ÿ�ж����ͼ��·��
    string fileName;
    // �ڲ�������ͻ���ϵ��
    Mat cameraMatrixL = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsL = Mat::zeros(5, 1, CV_64F);
    Mat cameraMatrixR = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsR = Mat::zeros(5, 1, CV_64F);
    // ˫Ŀ���
    Mat R, T;
    // ��������
    vector<float> coef;

    // ���ڲ���ͼ��Ĵ�С
    Size testImgSize;
    // ���ڲ���ͼ����
    int testImgCount;
    // ���ͼ���ϵ�������λ��
    vector<Point2f> objectPointsL, objectPointsR;
    // ���ͼ���ϵ�����������ֵ
    vector<Vec3b> objectColorsL, objectColorsR;
    // ���ǻ��ؽ�������������ϵ����
    Mat structure, structure3D;
    // ������λ�߾���
    vector<double> dist;
    
    // ��ȡ��������Ͳ���ģ��
    loadTrainResults(trainResult, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,
                     R, T, fit, coef);
    cout << "�����������" << endl;
    cout << "------------------------" << endl << endl;

    // ����
    // ��ȡ��Ŀ���ͼ��
    while(getline(fTestL, fileName)) {
        Mat img = imread(fileName);
        testSetL.push_back(img);
    }
    testImgSize = testSetL[0].size();
    testImgCount = testSetL.size();
    // ��ȡ��Ŀ���ͼ��
    while(getline(fTestR, fileName)) {
        Mat img = imread(fileName);
        testSetR.push_back(img);
    }

    // ����ͼ���е�Ŀ����ʵ����
    vector<float> groundTruth;

    // ��ȡ��ֵ�����ڱȽϣ�
    while(getline(fTestGT, fileName)) {
        groundTruth.push_back(atof(fileName.c_str()));
    }

    cout << "ͼ��������" << endl;
    cout << "------------------------" << endl << endl;

    for(int i = 0; i < testImgCount; i++) {
        // ˫���˲�
        bilateralFilter(testSetL[i], tempEnhanceL, 5, 10, 3);
        bilateralFilter(testSetR[i], tempEnhanceR, 5, 10, 3);
        //medianBlur(testSetL[i], tempEnhanceL, 3);
        //medianBlur(testSetR[i], tempEnhanceR, 3);
        testSetL[i] = tempEnhanceL.clone();
        testSetR[i] = tempEnhanceR.clone();

        // �������
        if(doDistortionCorrect) {
            undistort(testSetL[i].clone(), testSetL[i], cameraMatrixL, distCoeffsL);
            undistort(testSetR[i].clone(), testSetR[i], cameraMatrixR, distCoeffsR);
        }

        if(doEnhance) {
            // CLAHE��ǿ
            procCLAHE(testSetL[i], tempEnhanceL, clipL / 10.0, Size(gridXL, gridYL));
            procCLAHE(testSetR[i], tempEnhanceR, clipR / 10.0, Size(gridXR, gridYR));
            namedWindow("enhance_leftcam");
            createTrackbar("Clip", "enhance_leftcam", &clipL, MAX_CLIP_LIMIT, onEnhanceTrackbarL, (void *)&i);
            createTrackbar("Grid X", "enhance_leftcam", &gridXL, MAX_GRID_SIZE_X, onEnhanceTrackbarL, (void *)&i);
            createTrackbar("Grid Y", "enhance_leftcam", &gridYL, MAX_GRID_SIZE_Y, onEnhanceTrackbarL, (void *)&i);
            setMouseCallback("enhance_leftcam", onEnhanceMouseL, (void *)&i);
            imshow("enhance_leftcam", tempEnhanceL);

            namedWindow("enhance_rightcam");
            createTrackbar("Clip", "enhance_rightcam", &clipR, MAX_CLIP_LIMIT, onEnhanceTrackbarR, (void *)&i);
            createTrackbar("Grid X", "enhance_rightcam", &gridXR, MAX_GRID_SIZE_X, onEnhanceTrackbarR, (void *)&i);
            createTrackbar("Grid Y", "enhance_rightcam", &gridYR, MAX_GRID_SIZE_Y, onEnhanceTrackbarR, (void *)&i);
            setMouseCallback("enhance_rightcam", onEnhanceMouseR, (void *)&i);
            imshow("enhance_rightcam", tempEnhanceR);

            cout << "��" << i + 1 << "��ͼ��" << endl;
            cout << "��ǿͼ��" << endl;
            waitKey();
        }

        imshow("Ranging_leftcam", testSetL[i]);
        imshow("Ranging_rightcam", testSetR[i]);

        if(manualPoints) {
            // �ֶ�ѡ��һ���
            targetL = Point(0, 0);
            targetR = Point(0, 0);
            setMouseCallback("Ranging_leftcam", onMouseL, (void *)&i);
            setMouseCallback("Ranging_rightcam", onMouseR, (void *)&i);
            cout << "��ͼ���и�ѡ��һ��ƥ��㣺" << endl;
            waitKey();
            cout << "Ŀ��㣺L(" << targetL.x << ", " << targetL.y << ")   ";
            cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
            cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;
            objectPointsL.clear();
            objectPointsR.clear();
            objectPointsL.push_back(targetL);
            objectPointsR.push_back(targetR);
        } else {
            // ��ͼ��ѡȡROI
            setMouseCallback("Ranging_leftcam", onMouseL_ROI, (void *)&i);
            setMouseCallback("Ranging_rightcam", onMouseR_ROI, (void *)&i);
            cout << "��ͼ���и�ѡ��һ��ROI��������ƥ�䣺" << endl;
            waitKey();
            cout << "ROI���ģ�L(" << targetL.x << ", " << targetL.y << ")   ";
            cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
            cout << "ROI��С��L(" << roiL.width << ", " << roiL.height << ")   ";
            cout << "R(" << roiR.width << ", " << roiR.height << ")" << endl;
            cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;

            vector<vector<KeyPoint>> keyPoints4All;
            vector<Mat> descriptor4All;
            vector<vector<Vec3b>> colors4All;
            vector<DMatch> matches;
            vector<Mat> rois = { roiImgL, roiImgR };
            vector<KeyPoint> kp1, kp2;
            Mat show;

            switch(feature) {
            case SIFT:
                // �ֲ�����ƥ��
                extractSIFTFeatures(rois, keyPoints4All, descriptor4All, colors4All);
                matchSIFTFeatures(descriptor4All[0], descriptor4All[1], matches);
                getMatchedPoints(keyPoints4All[0], keyPoints4All[1], matches, objectPointsL, objectPointsR);
                getMatchedColors(colors4All[0], colors4All[1], matches, objectColorsL, objectColorsR);
                // ��ʾƥ����
                getMatchedPoints(keyPoints4All[0], keyPoints4All[1], matches, kp1, kp2);
                show = DrawInlier(roiImgL, roiImgR, kp1, kp2, 1);
                imshow("matches", show);
                break;
            case GMS:
                gmsMatch(roiImgL, roiImgR, objectPointsL, objectPointsR);
                break;
            default:
                break;
            }

            if(objectPointsL.size() == 0) {
                cout << "û���ҵ�ƥ��㣬�ֶ�ѡ��һ��" << endl;
                imshow("Ranging_leftcam", testSetL[i]);
                imshow("Ranging_rightcam", testSetR[i]);
                // �ֶ�ѡ��һ���
                targetL = Point(0, 0);
                targetR = Point(0, 0);
                setMouseCallback("Ranging_leftcam", onMouseL, (void *)&i);
                setMouseCallback("Ranging_rightcam", onMouseR, (void *)&i);
                cout << "�ڵ�" << i + 1 << "��ͼ���и�ѡ��һ��ƥ��㣺" << endl;
                waitKey();
                cout << "Ŀ��㣺L(" << targetL.x << ", " << targetL.y << ")   ";
                cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
                cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;
                objectPointsL.clear();
                objectPointsR.clear();
                objectPointsL.push_back(targetL);
                objectPointsR.push_back(targetR);
            } else {
                // ��ROI����ָ���ԭͼ����
                for(int i = 0; i < objectPointsL.size(); i++) {
                    objectPointsL[i] += Point2f(roiL.x, roiL.y);
                    objectPointsR[i] += Point2f(roiR.x, roiR.y);
                }
            }
        }

        // �ؽ�����
        reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                    objectPointsL, objectPointsR, structure);
        toPoints3D(structure, structure3D);
        // ��λ�߾���
        dist = ranging(structure3D, R, T);

        // �������
        if(processRange) {
            double range;
            if(dist.size() > 1) {
                // ������ľ���ȡ��λ��
                int len = dist.size();
                sort(dist.begin(), dist.end());
                if(len % 2) {
                    range = (dist[len / 2 - 1] + dist[len / 2]) / 2;
                } else {
                    range = dist[len / 2];
                }
            } else {
                range = dist[0];
            }
            //���о��벹��
            switch(fit) {
            case Poly:
                // ����ʽ
                range = compensatePoly(coef, range);
                break;
            case Exp2:
                // ����ָ��
                range = compensateExp2(coef, range);
                break;
            default:
                break;
            }
            cout << "����������Ŀ����룺" << range << " m" << endl;
            cout << "����ǲ�������Ŀ����룺" << groundTruth[i] << " m" << endl;
            cout << "��� " << range - groundTruth[i] << " m (" << abs(range - groundTruth[i]) * 100 / groundTruth[i] << "%)" << endl << endl;
            foutTest << range << endl;
        } else {
            // TEST��ֱ���������ԭʼֵ
            for(auto iter = dist.cbegin(); iter < dist.cend(); ++iter) {
                cout << "Ŀ�����(δ����) " << *iter << " m" << endl;
            }
            cout << endl;
        }

        waitKey();
        destroyAllWindows();
    }

    system("pause");
    return 0;
}
#endif


void onMouseL(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = testSetL[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetL = Point(x, y);
        circle(frame, targetL, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetL, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("Ranging_leftcam", frame);
    }
}

void onMouseL_ROI(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = testSetL[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetL = Point(x, y);
        getROI(frame, targetL, roiSize, roiL, roiImgL);
        rectangle(frame, roiL, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("Ranging_leftcam", frame);
    }
}

void onMouseR(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = testSetR[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetR = Point(x, y);
        circle(frame, targetR, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetR, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("Ranging_rightcam", frame);
    }
}
void onMouseR_ROI(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = testSetR[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetR = Point(x, y);
        getROI(frame, targetR, roiSize, roiR, roiImgR);
        rectangle(frame, roiR, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("Ranging_rightcam", frame);
    }
}

void onEnhanceTrackbarL(int pos, void *userdata) {
    int *i = (int *)userdata;
    if(clipL && gridXL && gridYL) {
        double clipVal = double(clipL) / 10;
        procCLAHE(testSetL[*i], tempEnhanceL, clipVal, Size(gridXL, gridYL));
        imshow("enhance_leftcam", tempEnhanceL);
    }
}

void onEnhanceTrackbarR(int pos, void *userdata) {
    int *i = (int *)userdata;
    if(clipR && gridXR && gridYR) {
        double clipVal = double(clipR) / 10;
        procCLAHE(testSetR[*i], tempEnhanceR, clipVal, Size(gridXR, gridYR));
        imshow("enhance_rightcam", tempEnhanceR);
    }
}

void onEnhanceMouseL(int event, int x, int y, int flags, void* param) {
    int *i = (int *)param;

    switch(event) {
    case EVENT_LBUTTONDOWN:
        imshow("enhance_leftcam", testSetL[*i]);
        break;
    case EVENT_LBUTTONUP:
        imshow("enhance_leftcam", tempEnhanceL);
        break;
    case EVENT_RBUTTONUP:
        tempEnhanceL.copyTo(testSetL[*i]);
        destroyWindow("enhance_leftcam");
        break;
    default:
        break;
    }
}

void onEnhanceMouseR(int event, int x, int y, int flags, void* param) {
    int *i = (int *)param;

    switch(event) {
    case EVENT_LBUTTONDOWN:
        imshow("enhance_rightcam", testSetR[*i]);
        break;
    case EVENT_LBUTTONUP:
        imshow("enhance_rightcam", tempEnhanceR);
        break;
    case EVENT_RBUTTONUP:
        tempEnhanceR.copyTo(testSetR[*i]);
        destroyWindow("enhance_rightcam");
        break;
    }
}

