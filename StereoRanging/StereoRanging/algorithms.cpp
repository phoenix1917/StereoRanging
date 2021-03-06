#include <iostream>
#include <fstream>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include "gms_matcher.hpp"
#include "algorithms.hpp"
// #include "createFit.h"

using namespace cv;
using namespace std;

void gmsMatch(Mat &img1, Mat &img2) {
    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;
    vector<DMatch> matches_all, matches_gms;

    Ptr<ORB> orb = ORB::create(10000);
    orb->setFastThreshold(0);
    orb->detectAndCompute(img1, Mat(), kp1, d1);
    orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
    GpuMat gd1(d1), gd2(d2);
    Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
    matcher->match(gd1, gd2, matches_all);
#else
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(d1, d2, matches_all);
#endif

    // GMS filter
    int num_inliers = 0;
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    num_inliers = gms.GetInlierMask(vbInliers, false, false);

    cout << "Get total " << num_inliers << " matches." << endl;

    // draw matches
    for(size_t i = 0; i < vbInliers.size(); ++i) {
        if(vbInliers[i] == true) {
            matches_gms.push_back(matches_all[i]);
        }
    }

    Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    imshow("matches", show);
    waitKey();
}

void gmsMatch(Mat &img1, Mat &img2, vector<Point2f> &matchL, vector<Point2f> &matchR) {
    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;
    vector<DMatch> matches_all, matches_gms;

    Ptr<ORB> orb = ORB::create(10000);
    orb->setFastThreshold(0);
    orb->detectAndCompute(img1, Mat(), kp1, d1);
    orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
    GpuMat gd1(d1), gd2(d2);
    Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
    matcher->match(gd1, gd2, matches_all);
#else
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(d1, d2, matches_all);
#endif

    // GMS filter
    int num_inliers = 0;
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    num_inliers = gms.GetInlierMask(vbInliers, false, false);

    cout << "Get total " << num_inliers << " matches." << endl;

    // draw matches
    for(size_t i = 0; i < vbInliers.size(); ++i) {
        if(vbInliers[i] == true) {
            matches_gms.push_back(matches_all[i]);
        }
    }

    // output matched points location
    getMatchedPoints(kp1, kp2, matches_gms, matchL, matchR);

    Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    imshow("matches", show);
}

void extractSIFTFeatures(vector<string>& imageNames,
                         vector<vector<KeyPoint>>& keyPoints4All,
                         vector<Mat>& descriptor4All,
                         vector<vector<Vec3b>>& colors4All) {
    keyPoints4All.clear();
    descriptor4All.clear();
    colors4All.clear();
    Mat image;

    // 读取图像，获取图像特征点，并保存
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

    for(auto it = imageNames.begin(); it != imageNames.end(); ++it) {
        vector<KeyPoint> keyPoints;
        Mat descriptor;

        image = imread(*it);
        if(image.empty()) {
            continue;
        }

        // 偶尔出现内存分配失败的错误
        sift->detectAndCompute(image, noArray(), keyPoints, descriptor);
        //// 特征点过少，则排除该图像
        //if(keyPoints.size() <= 10) {
        //    continue;
        //}

        // 根据特征点位置获取像素值
        vector<Vec3b> colors(keyPoints.size());
        for(int i = 0; i < keyPoints.size(); ++i) {
            Point2f& p = keyPoints[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }

        keyPoints4All.push_back(keyPoints);
        descriptor4All.push_back(descriptor);
        colors4All.push_back(colors);
    }

    //// 若所有图像都没有提出足够多的特征点，则将返回的列表赋为图像列表的长度
    //if(keyPoints4All.size() == 0) {
    //    int size = imageNames.size();
    //    keyPoints4All.resize(size);
    //    descriptor4All.resize(size);
    //    colors4All.resize(size);
    //}
}

void extractSIFTFeatures(vector<Mat>& images,
                         vector<vector<KeyPoint>>& keyPoints4All,
                         vector<Mat>& descriptor4All,
                         vector<vector<Vec3b>>& colors4All) {
    keyPoints4All.clear();
    descriptor4All.clear();
    colors4All.clear();
    Mat image;

    // 读取图像，获取图像特征点，并保存
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

    for(auto it = images.begin(); it != images.end(); ++it) {
        vector<KeyPoint> keyPoints;
        Mat descriptor;

        image = *it;
        // 偶尔出现内存分配失败的错误
        sift->detectAndCompute(image, noArray(), keyPoints, descriptor);
        //// 特征点过少，则排除该图像
        //if(keyPoints.size() <= 10) {
        //    continue;
        //}

        // 根据特征点位置获取像素值
        vector<Vec3b> colors(keyPoints.size());
        for(int i = 0; i < keyPoints.size(); ++i) {
            Point2f& p = keyPoints[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }

        keyPoints4All.push_back(keyPoints);
        descriptor4All.push_back(descriptor);
        colors4All.push_back(colors);
    }

    //// 若所有图像都没有提出足够多的特征点，则将返回的列表赋为图像列表的长度
    //if(keyPoints4All.size() == 0) {
    //    int size = imageNames.size();
    //    keyPoints4All.resize(size);
    //    descriptor4All.resize(size);
    //    colors4All.resize(size);
    //}
}

void matchSIFTFeatures(Mat& query, Mat& train, vector<DMatch>& matches) {
    vector<vector<DMatch> > knnMatches;
    BFMatcher matcher(NORM_L2);
    matcher.knnMatch(query, train, knnMatches, 2);

    //获取满足Ratio Test的最小匹配的距离
    float minDist = FLT_MAX;
    for(int r = 0; r < knnMatches.size(); ++r) {
        //Ratio Test
        if(knnMatches[r][0].distance > 0.6 * knnMatches[r][1].distance) {
            continue;
        }
        float dist = knnMatches[r][0].distance;
        if(dist < minDist) {
            minDist = dist;
        }
    }

    matches.clear();
    for(size_t r = 0; r < knnMatches.size(); ++r) {
        //排除不满足Ratio Test的点和匹配距离过大的点
        if(knnMatches[r][0].distance > 0.6 * knnMatches[r][1].distance ||
           knnMatches[r][0].distance > 5 * max(minDist, 10.0f)) {
            continue;
        }
        //保存匹配点
        matches.push_back(knnMatches[r][0]);
    }
}

void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2,
                      vector<DMatch> matches,
                      vector<Point2f>& out_p1, vector<Point2f>& out_p2) {
    out_p1.clear();
    out_p2.clear();
    for(int i = 0; i < matches.size(); ++i) {
        out_p1.push_back(p1[matches[i].queryIdx].pt);
        out_p2.push_back(p2[matches[i].trainIdx].pt);
    }
}

void getMatchedPoints(vector<KeyPoint>& p1, vector<KeyPoint>& p2,
                      vector<DMatch> matches,
                      vector<KeyPoint>& out_p1, vector<KeyPoint>& out_p2) {
    out_p1.clear();
    out_p2.clear();
    for(int i = 0; i < matches.size(); ++i) {
        out_p1.push_back(p1[matches[i].queryIdx]);
        out_p2.push_back(p2[matches[i].trainIdx]);
    }
}

void getMatchedColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches,
                      vector<Vec3b>& out_c1, vector<Vec3b>& out_c2) {
    out_c1.clear();
    out_c2.clear();
    for(int i = 0; i < matches.size(); ++i) {
        out_c1.push_back(c1[matches[i].queryIdx]);
        out_c2.push_back(c2[matches[i].trainIdx]);
    }
}

void procCLAHE(Mat &src, Mat &dst, double clipLimit = 40.0, Size tileGridSize = Size(8, 8)) {
    Mat tempOneChannel, tempBGR;
    cvtColor(src, tempOneChannel, CV_BGR2GRAY);

    Ptr<CLAHE> clahe = createCLAHE();
    clahe -> setClipLimit(clipLimit);
    clahe -> setTilesGridSize(tileGridSize);
    clahe -> apply(tempOneChannel, tempBGR);

    cvtColor(tempBGR, dst, CV_GRAY2BGR);
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
    const int height = max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
    src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

    if(type == 1) {
        for(size_t i = 0; i < inlier.size(); i++) {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
            line(output, left, right, Scalar(0, 255, 255));
        }
    } else if(type == 2) {
        for(size_t i = 0; i < inlier.size(); i++) {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
            line(output, left, right, Scalar(255, 0, 0));
        }

        for(size_t i = 0; i < inlier.size(); i++) {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
            circle(output, left, 1, Scalar(0, 255, 255), 2);
            circle(output, right, 1, Scalar(0, 255, 0), 2);
        }
    }

    return output;
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, int type) {
    const int height = max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
    src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

    if(type == 1) {
        for(size_t i = 0; i < kpt1.size(); i++) {
            Point2f left = kpt1[i].pt;
            Point2f right = (kpt2[i].pt + Point2f((float)src1.cols, 0.f));
            line(output, left, right, Scalar(0, 255, 255));
        }
    } else if(type == 2) {
        for(size_t i = 0; i < kpt1.size(); i++) {
            Point2f left = kpt1[i].pt;
            Point2f right = (kpt2[i].pt + Point2f((float)src1.cols, 0.f));
            line(output, left, right, Scalar(255, 0, 0));
        }

        for(size_t i = 0; i < kpt1.size(); i++) {
            Point2f left = kpt1[i].pt;
            Point2f right = (kpt2[i].pt + Point2f((float)src1.cols, 0.f));
            circle(output, left, 1, Scalar(0, 255, 255), 2);
            circle(output, right, 1, Scalar(0, 255, 0), 2);
        }
    }

    return output;
}

inline void imresize(Mat &src, int height) {
    double ratio = src.rows * 1.0 / height;
    int width = static_cast<int>(src.cols * 1.0 / ratio);
    resize(src, src, Size(width, height));
}

double compensateExp2(vector<float> coef, double x) {
     double k = coef[0] * exp(coef[1] * x) + coef[2] * exp(coef[3] * x);
     return k;
}

double compensatePoly(vector<float> coef, double x) {
    double k = coef[0];
    for(int i = 1; i < coef.size(); i++) {
        k += coef[i] * pow(x, i);
    }
    return k;
}

vector<float> polyfit2(vector<float> &train, vector<float> &truth, int n) {
    Mat y(train.size(), 1, CV_32F, Scalar::all(0));
    /* ********【预声明phy超定矩阵】************************/
    /* 多项式拟合的函数为多项幂函数
    * f(x)=a0+a1*x+a2*x^2+a3*x^3+......+an*x^n
    * a0、a1、a2......an是幂系数，也是拟合所求的未知量。设有m个抽样点，则：
    * 超定矩阵phy=1 x1 x1^2 ... ...  x1^n
    *            1 x2 x2^2 ... ...  x2^n
    *            1 x3 x3^2 ... ...  x3^n
    *              ... ... ... ...
    *              ... ... ... ...
    *            1 xm xm^2 ... ...  xm^n
    *
    * *************************************************/
    Mat phy(train.size(), n, CV_32F, Scalar::all(0));
    for(int i = 0; i < phy.rows; i++) {
        float* pr = phy.ptr<float>(i);
        for(int j = 0; j < phy.cols; j++) {
            pr[j] = pow(train[i], j);
        }
        y.at<float>(i) = truth[i];
    }
    Mat phy_t = phy.t();
    Mat phyMULphy_t = phy.t() * phy;
    Mat phyMphyInv = phyMULphy_t.inv();
    Mat a = phyMphyInv * phy_t;
    a = a * y;

    vector<float> coef;
    for(int i = 0; i < a.rows; i++) {
        coef.push_back(a.at<float>(i));
    }
    return coef;
}

//vector<float> exp2fit(vector<float> &trainVal, vector<float> &groundTruth) {
//    vector<float> coef(9);
//    // MEX code
//    if(!createFitInitialize()) {
//        cout << "拟合初始化失败" << endl;
//        return coef;
//    }
//    // mwArray
//    mwArray mwFitResult(4, 1, mxDOUBLE_CLASS);
//    mwArray mwGoF(5, 1, mxDOUBLE_CLASS);
//    mwArray mwTrain(trainVal.size(), 1, mxDOUBLE_CLASS);
//    mwArray mwTest(trainVal.size(), 1, mxDOUBLE_CLASS);
//    mwArray mwFlag(1, 1, mxLOGICAL_CLASS);
//    mxLogical mxFlag = false;
//    // Set data
//    mwTrain.SetData(&trainVal[0], trainVal.size());
//    mwTest.SetData(&groundTruth[0], trainVal.size());
//    mwFlag.SetLogicalData(&mxFlag, 1);
//    // Fit
//    createFit(2, mwFitResult, mwGoF, mwTrain, mwTest, mwFlag);
//    // Get result
//    double fitResult[4];
//    double gof[5];
//    mwFitResult.GetData(fitResult, 4);
//    mwGoF.GetData(gof, 5);
//    createFitTerminate();
//
//    // Transform to vector
//    coef[0] = fitResult[0];
//    coef[1] = fitResult[1];
//    coef[2] = fitResult[2];
//    coef[3] = fitResult[3];
//    coef[4] = gof[0];
//    coef[5] = gof[1];
//    coef[6] = gof[2];
//    coef[7] = gof[3];
//    coef[8] = gof[4];
//    return coef;
//}

vector<float> exp2fit(vector<float> &trainVal, vector<float> &groundTruth, double learningRate) {
    vector<float> coef(9);

    return coef;
}