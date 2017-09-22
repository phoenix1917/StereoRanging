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
    // 定义所使用的标定板
    usingBoard(3, boardSize, squareSize);

    // 每行读入的图像路径
    string fileName;
    // 内参数矩阵和畸变系数
    Mat cameraMatrixL = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsL = Mat::zeros(5, 1, CV_64F);
    Mat cameraMatrixR = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsR = Mat::zeros(5, 1, CV_64F);
    // 双目外参
    Mat R, T;
    // 补偿参数
    vector<float> coef;

    // 用于测距的图像的大小
    Size testImgSize;
    // 用于测距的图像数
    int testImgCount;
    // 测距图像上的特征点位置
    vector<Point2f> objectPointsL, objectPointsR;
    // 测距图像上的特征点像素值
    vector<Vec3b> objectColorsL, objectColorsR;
    // 三角化重建出的世界坐标系坐标
    Mat structure, structure3D;
    // 测距点中位线距离
    vector<double> dist;
    
    // 获取相机参数和补偿模型
    loadTrainResults(trainResult, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,
                     R, T, fit, coef);
    cout << "参数加载完成" << endl;
    cout << "------------------------" << endl << endl;

    // 测试
    // 读取左目测距图像
    while(getline(fTestL, fileName)) {
        Mat img = imread(fileName);
        testSetL.push_back(img);
    }
    testImgSize = testSetL[0].size();
    testImgCount = testSetL.size();
    // 读取右目测距图像
    while(getline(fTestR, fileName)) {
        Mat img = imread(fileName);
        testSetR.push_back(img);
    }

    // 测试图像中的目标真实距离
    vector<float> groundTruth;

    // 读取真值（用于比较）
    while(getline(fTestGT, fileName)) {
        groundTruth.push_back(atof(fileName.c_str()));
    }

    cout << "图像加载完成" << endl;
    cout << "------------------------" << endl << endl;

    for(int i = 0; i < testImgCount; i++) {
        // 双边滤波
        bilateralFilter(testSetL[i], tempEnhanceL, 5, 10, 3);
        bilateralFilter(testSetR[i], tempEnhanceR, 5, 10, 3);
        //medianBlur(testSetL[i], tempEnhanceL, 3);
        //medianBlur(testSetR[i], tempEnhanceR, 3);
        testSetL[i] = tempEnhanceL.clone();
        testSetR[i] = tempEnhanceR.clone();

        // 畸变矫正
        if(doDistortionCorrect) {
            undistort(testSetL[i].clone(), testSetL[i], cameraMatrixL, distCoeffsL);
            undistort(testSetR[i].clone(), testSetR[i], cameraMatrixR, distCoeffsR);
        }

        if(doEnhance) {
            // CLAHE增强
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

            cout << "第" << i + 1 << "组图像：" << endl;
            cout << "增强图像" << endl;
            waitKey();
        }

        imshow("Ranging_leftcam", testSetL[i]);
        imshow("Ranging_rightcam", testSetR[i]);

        if(manualPoints) {
            // 手动选择一组点
            targetL = Point(0, 0);
            targetR = Point(0, 0);
            setMouseCallback("Ranging_leftcam", onMouseL, (void *)&i);
            setMouseCallback("Ranging_rightcam", onMouseR, (void *)&i);
            cout << "在图像中各选择一个匹配点：" << endl;
            waitKey();
            cout << "目标点：L(" << targetL.x << ", " << targetL.y << ")   ";
            cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
            cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;
            objectPointsL.clear();
            objectPointsR.clear();
            objectPointsL.push_back(targetL);
            objectPointsR.push_back(targetR);
        } else {
            // 逐图像选取ROI
            setMouseCallback("Ranging_leftcam", onMouseL_ROI, (void *)&i);
            setMouseCallback("Ranging_rightcam", onMouseR_ROI, (void *)&i);
            cout << "在图像中各选择一个ROI进行特征匹配：" << endl;
            waitKey();
            cout << "ROI中心：L(" << targetL.x << ", " << targetL.y << ")   ";
            cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
            cout << "ROI大小：L(" << roiL.width << ", " << roiL.height << ")   ";
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
                // 局部特征匹配
                extractSIFTFeatures(rois, keyPoints4All, descriptor4All, colors4All);
                matchSIFTFeatures(descriptor4All[0], descriptor4All[1], matches);
                getMatchedPoints(keyPoints4All[0], keyPoints4All[1], matches, objectPointsL, objectPointsR);
                getMatchedColors(colors4All[0], colors4All[1], matches, objectColorsL, objectColorsR);
                // 显示匹配结果
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
                cout << "没有找到匹配点，手动选择一组" << endl;
                imshow("Ranging_leftcam", testSetL[i]);
                imshow("Ranging_rightcam", testSetR[i]);
                // 手动选择一组点
                targetL = Point(0, 0);
                targetR = Point(0, 0);
                setMouseCallback("Ranging_leftcam", onMouseL, (void *)&i);
                setMouseCallback("Ranging_rightcam", onMouseR, (void *)&i);
                cout << "在第" << i + 1 << "组图像中各选择一个匹配点：" << endl;
                waitKey();
                cout << "目标点：L(" << targetL.x << ", " << targetL.y << ")   ";
                cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
                cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;
                objectPointsL.clear();
                objectPointsR.clear();
                objectPointsL.push_back(targetL);
                objectPointsR.push_back(targetR);
            } else {
                // 将ROI坐标恢复到原图像中
                for(int i = 0; i < objectPointsL.size(); i++) {
                    objectPointsL[i] += Point2f(roiL.x, roiL.y);
                    objectPointsR[i] += Point2f(roiR.x, roiR.y);
                }
            }
        }

        // 重建坐标
        reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                    objectPointsL, objectPointsR, structure);
        toPoints3D(structure, structure3D);
        // 中位线距离
        dist = ranging(structure3D, R, T);

        // 输出距离
        if(processRange) {
            double range;
            if(dist.size() > 1) {
                // 对求出的距离取中位数
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
            //进行距离补偿
            switch(fit) {
            case Poly:
                // 多项式
                range = compensatePoly(coef, range);
                break;
            case Exp2:
                // 二阶指数
                range = compensateExp2(coef, range);
                break;
            default:
                break;
            }
            cout << "经过修正的目标距离：" << range << " m" << endl;
            cout << "测距仪测量出的目标距离：" << groundTruth[i] << " m" << endl;
            cout << "误差 " << range - groundTruth[i] << " m (" << abs(range - groundTruth[i]) * 100 / groundTruth[i] << "%)" << endl << endl;
            foutTest << range << endl;
        } else {
            // TEST：直接输出所有原始值
            for(auto iter = dist.cbegin(); iter < dist.cend(); ++iter) {
                cout << "目标距离(未补偿) " << *iter << " m" << endl;
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
        // 记录当前位置的坐标，画一个点
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
        // 记录当前位置的坐标，画一个点
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
        // 记录当前位置的坐标，画一个点
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
        // 记录当前位置的坐标，画一个点
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

