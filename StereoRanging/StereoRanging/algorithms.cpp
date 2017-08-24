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

    // ��ȡͼ�񣬻�ȡͼ�������㣬������
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

    for(auto it = imageNames.begin(); it != imageNames.end(); ++it) {
        vector<KeyPoint> keyPoints;
        Mat descriptor;

        image = imread(*it);
        if(image.empty()) {
            continue;
        }

        // ż�������ڴ����ʧ�ܵĴ���
        sift->detectAndCompute(image, noArray(), keyPoints, descriptor);
        //// ��������٣����ų���ͼ��
        //if(keyPoints.size() <= 10) {
        //    continue;
        //}

        // ����������λ�û�ȡ����ֵ
        vector<Vec3b> colors(keyPoints.size());
        for(int i = 0; i < keyPoints.size(); ++i) {
            Point2f& p = keyPoints[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }

        keyPoints4All.push_back(keyPoints);
        descriptor4All.push_back(descriptor);
        colors4All.push_back(colors);
    }

    //// ������ͼ��û������㹻��������㣬�򽫷��ص��б�Ϊͼ���б�ĳ���
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

    // ��ȡͼ�񣬻�ȡͼ�������㣬������
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

    for(auto it = images.begin(); it != images.end(); ++it) {
        vector<KeyPoint> keyPoints;
        Mat descriptor;

        image = *it;
        // ż�������ڴ����ʧ�ܵĴ���
        sift->detectAndCompute(image, noArray(), keyPoints, descriptor);
        //// ��������٣����ų���ͼ��
        //if(keyPoints.size() <= 10) {
        //    continue;
        //}

        // ����������λ�û�ȡ����ֵ
        vector<Vec3b> colors(keyPoints.size());
        for(int i = 0; i < keyPoints.size(); ++i) {
            Point2f& p = keyPoints[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }

        keyPoints4All.push_back(keyPoints);
        descriptor4All.push_back(descriptor);
        colors4All.push_back(colors);
    }

    //// ������ͼ��û������㹻��������㣬�򽫷��ص��б�Ϊͼ���б�ĳ���
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

    //��ȡ����Ratio Test����Сƥ��ľ���
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
        //�ų�������Ratio Test�ĵ��ƥ��������ĵ�
        if(knnMatches[r][0].distance > 0.6 * knnMatches[r][1].distance ||
           knnMatches[r][0].distance > 5 * max(minDist, 10.0f)) {
            continue;
        }
        //����ƥ���
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

inline Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
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

inline void imresize(Mat &src, int height) {
    double ratio = src.rows * 1.0 / height;
    int width = static_cast<int>(src.cols * 1.0 / ratio);
    resize(src, src, Size(width, height));
}

