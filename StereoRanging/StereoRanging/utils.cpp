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

string num2str(int num) {
    ostringstream s1;
    s1 << num;
    string outStr = s1.str();
    return(outStr);
}

void getROI(Mat& img, Point center, Size roiSize, Rect& roi, Mat& roiImg) {
    Point startPoint, endPoint;
    Size imgSize = img.size();

    // x
    if(roiSize.width * 2 + 1 > imgSize.width) {
        startPoint.x = 0;
        endPoint.x = imgSize.width;
    } else if(center.x < roiSize.width) {
        startPoint.x = 0;
        endPoint.x = center.x + roiSize.width;
    } else if(center.x > imgSize.width - roiSize.width) {
        startPoint.x = imgSize.width;
        endPoint.x = center.x - roiSize.width;
    } else {
        startPoint.x = center.x - roiSize.width;
        endPoint.x = center.x + roiSize.width;
    }

    // y
    if(roiSize.height * 2 + 1 > imgSize.height) {
        startPoint.y = 0;
        endPoint.y = imgSize.height;
    } else if(center.y < roiSize.height) {
        startPoint.y = 0;
        endPoint.y = center.y + roiSize.height;
    } else if(center.y > imgSize.height - roiSize.height) {
        startPoint.y = imgSize.height;
        endPoint.y = center.y - roiSize.height;
    } else {
        startPoint.y = center.y - roiSize.height;
        endPoint.y = center.y + roiSize.height;
    }

    // 返回ROI及图像
    roi = Rect(startPoint, endPoint);
    roiImg = img(roi).clone();
}

void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError,
                       Mat &stdDevIntrinsics, Mat &stdDevExtrinsics, vector<double> &perViewErrors) {
    cout << "标定结果：" << endl;
    cout << "cameraMatrix = " << endl << cameraMatrix << endl << endl;
    cout << "distCoeffs = " << endl << distCoeffs << endl << endl;
    cout << "reprojectionError = " << endl << reprojectionError << endl << endl;

    cout << "内参数：" << endl;
    cout << "[fx, fy] = ";
    cout << "[" << cameraMatrix.at<double>(0, 0) << ", " << cameraMatrix.at<double>(1, 1) << "]";
    cout << " +/- [" << stdDevIntrinsics.at<double>(0) << ", " << stdDevIntrinsics.at<double>(1) << "]" << endl;
    cout << "[Cx, Cy] = ";
    cout << "[" << cameraMatrix.at<double>(0, 2) << ", " << cameraMatrix.at<double>(1, 2) << "]";
    cout << " +/- [" << stdDevIntrinsics.at<double>(2) << ", " << stdDevIntrinsics.at<double>(3) << "]" << endl;
    cout << "[k1, k2, p1, p2, k3] = ";
    cout << "[" << distCoeffs.at<double>(0) << ", " << distCoeffs.at<double>(1) << ", ";
    cout << distCoeffs.at<double>(2) << ", " << distCoeffs.at<double>(3) << ", ";
    cout << distCoeffs.at<double>(4) << "]";
    cout << " +/- [" << stdDevIntrinsics.at<double>(4) << ", " << stdDevIntrinsics.at<double>(5) << ", ";
    cout << stdDevIntrinsics.at<double>(6) << ", " << stdDevIntrinsics.at<double>(7) << ", ";
    cout << stdDevIntrinsics.at<double>(8) << "]" << endl << endl;

    cout << "误差：" << endl;
    cout << "[k4, k5, k6] = ";
    cout << "[" << stdDevIntrinsics.at<double>(9) << ", " << stdDevIntrinsics.at<double>(10) << ", ";
    cout << stdDevIntrinsics.at<double>(11) << "]" << endl;
    cout << "[s1, s2, s3, s4] = ";
    cout << "[" << stdDevIntrinsics.at<double>(12) << ", " << stdDevIntrinsics.at<double>(13) << ", ";
    cout << stdDevIntrinsics.at<double>(14) << ", " << stdDevIntrinsics.at<double>(15) << "]" << endl;
    cout << "[tauX, tauY] = ";
    cout << "[" << stdDevIntrinsics.at<double>(16) << ", " << stdDevIntrinsics.at<double>(17) << "]" << endl << endl;

    for(int i = 0; i < stdDevExtrinsics.rows; i += 6) {
        cout << "R" << i / 6 + 1 << " = ";
        cout << "[" << stdDevExtrinsics.at<double>(i) << ", " << stdDevExtrinsics.at<double>(i + 1) << ", " << stdDevExtrinsics.at<double>(i + 2) << "]" << endl;
        cout << "t" << i / 6 + 1 << " = ";
        cout << "[" << stdDevExtrinsics.at<double>(i + 3) << ", " << stdDevExtrinsics.at<double>(i + 4) << ", " << stdDevExtrinsics.at<double>(i + 5) << "]" << endl << endl;
    }
    cout << endl;

    int size = perViewErrors.size();
    for(int i = 0; i < size; i++) {
        cout << "reprojectionError[" << i + 1 << "] = " << perViewErrors[i] << endl;
    }
    cout << endl << endl;
}

void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError,
                       Mat &stdDevIntrinsics, Mat &stdDevExtrinsics,
                       vector<double> &perViewErrors, ofstream &fout) {
    fout << "标定结果：" << endl;
    fout << "cameraMatrix = " << endl << cameraMatrix << endl << endl;
    fout << "distCoeffs = " << endl << distCoeffs << endl << endl;
    fout << "reprojectionError = " << endl << reprojectionError << endl << endl;

    fout << "内参数：" << endl;
    fout << "[fx, fy] = ";
    fout << "[" << cameraMatrix.at<double>(0, 0) << ", " << cameraMatrix.at<double>(1, 1) << "]";
    fout << " +/- [" << stdDevIntrinsics.at<double>(0) << ", " << stdDevIntrinsics.at<double>(1) << "]" << endl;
    fout << "[Cx, Cy] = ";
    fout << "[" << cameraMatrix.at<double>(0, 2) << ", " << cameraMatrix.at<double>(1, 2) << "]";
    fout << " +/- [" << stdDevIntrinsics.at<double>(2) << ", " << stdDevIntrinsics.at<double>(3) << "]" << endl;
    fout << "[k1, k2, p1, p2, k3] = ";
    fout << "[" << distCoeffs.at<double>(0) << ", " << distCoeffs.at<double>(1) << ", ";
    fout << distCoeffs.at<double>(2) << ", " << distCoeffs.at<double>(3) << ", ";
    fout << distCoeffs.at<double>(4) << "]";
    fout << " +/- [" << stdDevIntrinsics.at<double>(4) << ", " << stdDevIntrinsics.at<double>(5) << ", ";
    fout << stdDevIntrinsics.at<double>(6) << ", " << stdDevIntrinsics.at<double>(7) << ", ";
    fout << stdDevIntrinsics.at<double>(8) << "]" << endl << endl;

    fout << "误差：" << endl;
    fout << "[k4, k5, k6] = ";
    fout << "[" << stdDevIntrinsics.at<double>(9) << ", " << stdDevIntrinsics.at<double>(10) << ", ";
    fout << stdDevIntrinsics.at<double>(11) << "]" << endl;
    fout << "[s1, s2, s3, s4] = ";
    fout << "[" << stdDevIntrinsics.at<double>(12) << ", " << stdDevIntrinsics.at<double>(13) << ", ";
    fout << stdDevIntrinsics.at<double>(14) << ", " << stdDevIntrinsics.at<double>(15) << "]" << endl;
    fout << "[tauX, tauY] = ";
    fout << "[" << stdDevIntrinsics.at<double>(16) << ", " << stdDevIntrinsics.at<double>(17) << "]" << endl << endl;

    for(int i = 0; i < stdDevExtrinsics.rows; i += 6) {
        fout << "R" << i / 6 + 1 << " = ";
        fout << "[" << stdDevExtrinsics.at<double>(i) << ", " << stdDevExtrinsics.at<double>(i + 1) << ", " << stdDevExtrinsics.at<double>(i + 2) << "]" << endl;
        fout << "t" << i / 6 + 1 << " = ";
        fout << "[" << stdDevExtrinsics.at<double>(i + 3) << ", " << stdDevExtrinsics.at<double>(i + 4) << ", " << stdDevExtrinsics.at<double>(i + 5) << "]" << endl << endl;
    }
    fout << endl;

    int size = perViewErrors.size();
    for(int i = 0; i < size; i++) {
        fout << "reprojectionError[" << i + 1 << "] = " << perViewErrors[i] << endl;
    }
    fout << endl << endl;
}

void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs,
                       double reprojectionError, Mat &stdDevIntrinsics) {
    cout << "标定结果：" << endl;
    cout << "cameraMatrix = " << endl << cameraMatrix << endl << endl;
    cout << "distCoeffs = " << endl << distCoeffs << endl << endl;
    cout << "reprojectionError = " << endl << reprojectionError << endl << endl;

    cout << "内参数：" << endl;
    cout << "[fx, fy] = ";
    cout << "[" << cameraMatrix.at<double>(0, 0) << ", " << cameraMatrix.at<double>(1, 1) << "]";
    cout << " +/- [" << stdDevIntrinsics.at<double>(0) << ", " << stdDevIntrinsics.at<double>(1) << "]" << endl;
    cout << "[Cx, Cy] = ";
    cout << "[" << cameraMatrix.at<double>(0, 2) << ", " << cameraMatrix.at<double>(1, 2) << "]";
    cout << " +/- [" << stdDevIntrinsics.at<double>(2) << ", " << stdDevIntrinsics.at<double>(3) << "]" << endl;
    cout << "[k1, k2, p1, p2, k3] = ";
    cout << "[" << distCoeffs.at<double>(0) << ", " << distCoeffs.at<double>(1) << ", ";
    cout << distCoeffs.at<double>(2) << ", " << distCoeffs.at<double>(3) << ", ";
    cout << distCoeffs.at<double>(4) << "]";
    cout << " +/- [" << stdDevIntrinsics.at<double>(4) << ", " << stdDevIntrinsics.at<double>(5) << ", ";
    cout << stdDevIntrinsics.at<double>(6) << ", " << stdDevIntrinsics.at<double>(7) << ", ";
    cout << stdDevIntrinsics.at<double>(8) << "]" << endl << endl << endl;
}

void printCalibResults(Mat &cameraMatrix, Mat &distCoeffs, double reprojectionError) {
    cout << endl << "标定结果：" << endl;
    cout << "cameraMatrix = " << endl << cameraMatrix << endl << endl;
    cout << "distCoeffs = " << endl << distCoeffs << endl << endl;
    cout << "reprojectionError = " << endl << reprojectionError << endl << endl;

    cout << "内参数：" << endl;
    cout << "[fx, fy] = ";
    cout << "[" << cameraMatrix.at<double>(0, 0) << ", " << cameraMatrix.at<double>(1, 1) << "]" << endl;
    cout << "[Cx, Cy] = ";
    cout << "[" << cameraMatrix.at<double>(0, 2) << ", " << cameraMatrix.at<double>(1, 2) << "]" << endl;
    cout << "[k1, k2, p1, p2, k3] = ";
    cout << "[" << distCoeffs.at<double>(0) << ", " << distCoeffs.at<double>(1) << ", ";
    cout << distCoeffs.at<double>(2) << ", " << distCoeffs.at<double>(3) << ", ";
    cout << distCoeffs.at<double>(4) << "]" << endl << endl;
}

void saveCorrspondingPoints(string fileName, Mat K1, Mat K2, Mat R, Mat t,
                            vector<vector<Point2f>> points1, vector<vector<Point2f>> points2,
                            int corrPointsCount, vector<float> groundTruth) {
    FileStorage fs(fileName, FileStorage::WRITE);
    fs << "K1" << K1;
    fs << "K2" << K2;
    fs << "R" << R;
    fs << "t" << t;
    fs << "ImageCount" << (int)points1.size();
    fs << "CorrPointsCount" << corrPointsCount;
    for(size_t i = 0; i < points1.size(); ++i) {
        ostringstream s;
        s << i + 1;
        string imgNum = s.str();
        fs << "L" + imgNum << "[:";
        for(auto iter = points1[i].begin(); iter < points1[i].end(); ++iter) {
            fs << *iter;
        }
        fs << "]";
        fs << "R" + imgNum << "[:";
        for(auto iter = points2[i].begin(); iter < points2[i].end(); ++iter) {
            fs << *iter;
        }
        fs << "]";
        fs << "GroundTruth" + imgNum << groundTruth[i];
    }
    fs.release();
}

void saveTrainResults(string fileName, Mat K1, Mat distCoefs1, Mat K2, Mat distCoefs2, 
                      Mat R, Mat t, vector<float> trainVal, vector<float> groundTruth,
                      FitType fit, vector<float> coef) {
    FileStorage fs(fileName, FileStorage::WRITE);
    fs << "K1" << K1;
    fs << "K2" << K2;
    fs << "distCoefs1" << distCoefs1;
    fs << "distCoefs2" << distCoefs2;
    fs << "R" << R;
    fs << "t" << t;
    fs << "trainVal" << "[";
    for(auto iter = trainVal.begin(); iter < trainVal.end(); ++iter) {
        fs << *iter;
    }
    fs << "]";
    fs << "groundTruth" << "[";
    for(auto iter = groundTruth.begin(); iter < groundTruth.end(); ++iter) {
        fs << *iter;
    }
    fs << "]";
    fs << "fitType" << fit;
    fs << "coef" << "[";
    for(auto iter = coef.begin(); iter < coef.end(); ++iter) {
        fs << *iter;
    }
    fs << "]";
    fs.release();
}

void loadTrainResults(string fileName, Mat& K1, Mat& distCoefs1, Mat& K2, Mat& distCoefs2,
                      Mat& R, Mat& t, FitType& fit, vector<float>& coef) {
    FileStorage fs(fileName, FileStorage::READ);
    fs["K1"] >> K1;
    fs["K2"] >> K2;
    fs["distCoefs1"] >> distCoefs1;
    fs["distCoefs2"] >> distCoefs2;
    fs["R"] >> R;
    fs["t"] >> t;
    int fitTypeNum;
    fs["fitType"] >> fitTypeNum;
    fit = (FitType)fitTypeNum;
    FileNode node = fs["coef"];
    node >> coef;
    fs.release();
}

void printCoef(vector<float> coef, FitType fit) {
    switch(fit) {
    case Poly:
        // 多项式拟合
        cout << "补偿模型：" << endl;
        cout << "f(x) = (" << coef[0] << ") + (" << coef[1] << ")x + (" << coef[2] << ")x^2 + (" << coef[3] << ")x^3" << endl;
        break;
    case Exp2:
        // 二阶指数拟合
        cout << "补偿模型：" << endl;
        cout << "f(x) = (" << coef[0] << ")*exp(" << coef[1] << "*x) + (" << coef[2] << ")*exp(" << coef[3] << "*x)" << endl;
        break;
    default:
        break;
    }
}