#include <iostream>
#include <fstream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>

#include "reconstruct.hpp"

using namespace cv;
using namespace std;


void usingBoard(int boardNum, Size& boardSize, float& squareSize) {
    switch(boardNum) {
    case 1:
        boardSize = Size(8, 6);
        squareSize = 35.0;
        break;
    case 2:
        boardSize = Size(11, 8);
        squareSize = 30.0;
        break;
    case 3:
        boardSize = Size(9, 8);
        squareSize = 90.0;
        break;
    case 4:
        boardSize = Size(12, 8);
        squareSize = 30.0;
        break;
    default:
        cout << "无对应标定板" << endl;
        break;
    }
}

void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners) {
    for(int i = 0; i < boardSize.height; ++i) {
        for(int j = 0; j < boardSize.width; ++j) {
            corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
        }
    }
}

double computeReprojectionErrors(vector<vector<Point3f> >& objectPoints,
                                 vector<vector<Point2f> >& imagePoints,
                                 vector<Mat>& rvecs, vector<Mat>& tvecs,
                                 Mat& cameraMatrix, Mat& distCoeffs,
                                 vector<float>& perViewErrors) {
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    //perViewErrors.resize(objectPoints.size());

    for(i = 0; i < (int)objectPoints.size(); ++i) {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

        int n = (int)objectPoints[i].size();
        //perViewErrors[i] = (float)sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }
    return sqrt(totalErr / totalPoints);
}

bool fixPrinciplePoint(Mat& K, Point2f point) {
    if(K.size() == Size(3, 3)) {
        K.at<double>(0, 2) = point.x;
        K.at<double>(1, 2) = point.y;
        return true;
    } else {
        cout << "内参数矩阵大小不正确，请检查" << endl;
        return false;
    }
}

bool findTransform(Mat& K, Mat& R, Mat& T,
                   vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask) {
    //根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
    double focalLength = 0.5 * (K.at<double>(0) + K.at<double>(4));
    Point2d principlePoint(K.at<double>(2), K.at<double>(5));

    //根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
    Mat E = findEssentialMat(p1, p2, focalLength, principlePoint, RANSAC, 0.999, 1.0, mask);
    if(E.empty()) {
        return false;
    }

    double feasibleCount = countNonZero(mask);
    cout << (int)feasibleCount << " -in- " << p1.size() << endl;
    //对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
    if(feasibleCount <= 15 || (feasibleCount / p1.size()) < 0.6) {
        return false;
    }

    //分解本征矩阵，获取相对变换。返回inliers数目
    int passCount = recoverPose(E, p1, p2, R, T, focalLength, principlePoint, mask);

    //同时位于两个相机前方的点的数量要足够大
    if(((double)passCount) / feasibleCount < 0.7) {
        return false;
    }

    return true;
}

bool findTransform(Mat& K1, Mat& K2, Mat& R, Mat& T,
                   vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask) {
    //根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
    double focalLength = 0.5 * (K1.at<double>(0) + K1.at<double>(4) +
                                K2.at<double>(0) + K2.at<double>(4));
    Point2d principlePoint((K1.at<double>(2) + K2.at<double>(2)) / 2,
                           (K1.at<double>(5) + K2.at<double>(5)) / 2);

    //根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
    Mat E = findEssentialMat(p1, p2, focalLength, principlePoint, RANSAC, 0.999, 1.0, mask);
    if(E.empty()) {
        return false;
    }

    double feasibleCount = countNonZero(mask);
    cout << (int)feasibleCount << " -in- " << p1.size() << endl;
    //对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
    if(feasibleCount <= 15 || (feasibleCount / p1.size()) < 0.6) {
        return false;
    }

    //分解本征矩阵，获取相对变换。返回inliers数目
    int passCount = recoverPose(E, p1, p2, R, T, focalLength, principlePoint, mask);

    //同时位于两个相机前方的点的数量要足够大
    if(((double)passCount) / feasibleCount < 0.7) {
        return false;
    }

    return true;
}

void maskoutPoints(vector<Point2f>& p1, Mat& mask) {
    vector<Point2f> p1_copy = p1;
    p1.clear();

    for(int i = 0; i < mask.rows; ++i) {
        if(mask.at<uchar>(i) > 0) {
            p1.push_back(p1_copy[i]);
        }
    }
}

void maskoutColors(vector<Vec3b>& p1, Mat& mask) {
    vector<Vec3b> p1_copy = p1;
    p1.clear();
    
    for(int i = 0; i < mask.rows; ++i) {
        if(mask.at<uchar>(i) > 0) {
            p1.push_back(p1_copy[i]);
        }
    }
}

void reconstruct(Mat& K, Mat& R, Mat& T,
                 vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure) {
    // 两个相机的投影矩阵（单应性矩阵）K[R T]，triangulatePoints只支持float型（CV_32FC1）
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    // 初始化camera1的投影矩阵
    // 世界坐标系建立在camera1的摄像机坐标系上
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    // 初始化camera2的投影矩阵
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);

    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK * proj1;
    proj2 = fK * proj2;

    // 三角重建
    triangulatePoints(proj1, proj2, p1, p2, structure);
}

void reconstruct(Mat& K1, Mat& K2, Mat& R, Mat& T,
                 vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure) {
    // 两个相机的投影矩阵（单应性矩阵）K[R T]，triangulatePoints只支持float型（CV_32FC1）
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    // 初始化camera1的投影矩阵
    // 世界坐标系建立在camera1的摄像机坐标系上
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    // 初始化camera2的投影矩阵
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);

    Mat fK1, fK2;
    K1.convertTo(fK1, CV_32FC1);
    K2.convertTo(fK2, CV_32FC1);
    proj1 = fK1 * proj1;
    proj2 = fK2 * proj2;

    // 三角重建
    triangulatePoints(proj1, proj2, p1, p2, structure);
}

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2,
                 vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure) {
    //两个相机的投影矩阵（单应性矩阵）K[R T]，triangulatePoints只支持float型
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
    T1.convertTo(proj1.col(3), CV_32FC1);

    R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T2.convertTo(proj2.col(3), CV_32FC1);

    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK * proj1;
    proj2 = fK * proj2;

    //三角重建
    Mat structure4D;
    triangulatePoints(proj1, proj2, p1, p2, structure4D);

    structure.clear();
    structure.reserve(structure4D.cols);
    for(int i = 0; i < structure4D.cols; ++i) {
        //齐次坐标，除以最后一个元素转为空间坐标
        Mat_<float> col = structure4D.col(i);
        col /= col(3);
        structure.push_back(Point3f(col(0), col(1), col(2)));
    }
}

void toPoints3D(Mat& points4D, Mat& points3D) {
    points3D = Mat::zeros(3, points4D.size().width, CV_32FC1);
    for(int i = 0; i < points4D.size().width; i++) {
        points3D.at<float>(0, i) = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        points3D.at<float>(1, i) = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        points3D.at<float>(2, i) = points4D.at<float>(2, i) / points4D.at<float>(3, i);
    }
}

void saveStructure(string fileName, vector<Mat>& rotations,
                   vector<Mat>& motions, vector<Point3f>& structure,
                   vector<Vec3b>& colors) {
    int n = (int)rotations.size();

    FileStorage fs(fileName, FileStorage::WRITE);
    fs << "Camera Count" << n;
    fs << "Point Count" << (int)structure.size();

    fs << "Rotations" << "[";
    for(size_t i = 0; i < n; ++i) {
        fs << rotations[i];
    }
    fs << "]";

    fs << "Motions" << "[";
    for(size_t i = 0; i < n; ++i) {
        fs << motions[i];
    }
    fs << "]";

    fs << "Points" << "[";
    for(size_t i = 0; i < structure.size(); ++i) {
        fs << structure[i];
    }
    fs << "]";

    fs << "Colors" << "[";
    for(size_t i = 0; i < colors.size(); ++i) {
        fs << colors[i];
    }
    fs << "]";

    fs.release();
}

void saveCorrspondingPoints(string fileName, Mat K1, Mat K2, Mat R, Mat t, 
                            vector<vector<Point2f>> points1, vector<vector<Point2f>> points2, vector<float> groundTruth) {
    //FileStorage fs(fileName, FileStorage::WRITE);
    //fs << "K1" << K1;
    //fs << "K2" << K2;
    //fs << "R" << R;
    //fs << "t" << t;
    //fs << "Image Count" << (int)points1.size();
    //fs << "Points" << "[";
    //for(size_t i = 0; i < points1.size(); ++i) {
    //    fs << "Image" << i << "[";
    //    fs << "L" << "[";
    //    for(auto iter = points1[i].begin(); iter < points1[i].end(); ++iter) {
    //        fs << *iter;
    //    }
    //    fs << "]";
    //    fs << "R" << "[";
    //    for(auto iter = points2[i].begin(); iter < points2[i].end(); ++iter) {
    //        fs << *iter;
    //    }
    //    fs << "]";
    //    fs << "Ground Truth" << groundTruth[i];
    //    fs << "]";
    //}
    //fs << "]";
}

vector<double> ranging(Mat structure, Mat R, Mat t) {
    vector<double> range;
    // 输入大小有误，返回空
    if(structure.size().height != 3 || R.size() != Size(3, 3) || t.size() != Size(1, 3)) {
        cout << "Ranging error: wrong input sizes." << endl;
        return range;
    }
    for(int i = 0; i < structure.size().width; i++) {
        Mat pointL, pointR;
        structure.colRange(i, i + 1).clone().convertTo(pointL, CV_64FC1);
        pointR = R * pointL + t;
        double distL, distR, distT, median;
        distL = sqrt(pointL.at<double>(0) * pointL.at<double>(0) +
                     pointL.at<double>(1) * pointL.at<double>(1) +
                     pointL.at<double>(2) * pointL.at<double>(2));
        distR = sqrt(pointR.at<double>(0) * pointR.at<double>(0) +
                     pointR.at<double>(1) * pointR.at<double>(1) +
                     pointR.at<double>(2) * pointR.at<double>(2));
        distT = sqrt(t.at<double>(0) * t.at<double>(0) +
                     t.at<double>(1) * t.at<double>(1) +
                     t.at<double>(2) * t.at<double>(2));
        median = 0.5 * sqrt(2 * distL * distL + 2 * distR * distR - distT * distT) / 1000;
        range.push_back(median);
    }
    return range;
}
    
