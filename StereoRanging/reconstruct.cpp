#include <iostream>
#include <fstream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>

#include "main.hpp"
#include "reconstruct.hpp"

using namespace cv;
using namespace std;


/**
 * usingBoard：定义所使用的标定板
 * @param boardNum   [Input]标定板编号
 *                      标定板1：9x7，35mm，白色边缘
 *                      标定板2：12x9，30mm，黑色边缘
 *                      标定板3：10x9，90mm，白色边缘
 *                      标定板4：13x9，30mm，白色边缘
 * @param boardSize  [Output]返回标定板棋盘格内角点数目
 * @param squareSize [Output]返回标定板方格边长
 */
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


/**
 * 计算标定板角点位置
 * @param boardSize  [input]标定板内角点size
 * @param squareSize [input]标定板方格边长
 * @param corners    [output]返回的角点标称位置
 */
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners) {
    for(int i = 0; i < boardSize.height; ++i) {
        for(int j = 0; j < boardSize.width; ++j) {
            corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
        }
    }
}


/**
 * 计算投影误差
 * @param objectPoints  
 * @param imagePoints   
 * @param rvecs         
 * @param tvecs         
 * @param cameraMatrix  
 * @param distCoeffs    
 * @param perViewErrors 
 */
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


/**
 * 固定主点坐标到指定值
 * @param K     [InputOutputArray] 内参数矩阵(CV_64FC1)
 * @param point [InputArray] 要固定的主点坐标位置
 * @return      是否成功操作
 */
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


/**
 * extractFeatures 对图像列表提取SIFT特征
 * @param imageNames     [input]图像名称列表
 * @param keyPoints4All  [output]所有图像特征点位置列表
 * @param descriptor4All [output]所有图像特征描述子列表
 * @param colors4All     [output]所有图像特征点像素列表
 */
void extractFeatures(vector<string>& imageNames,
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


/**
* extractFeatures 对图像列表提取SIFT特征
* @param images         [input]图像列表
* @param keyPoints4All  [output]所有图像特征点位置列表
* @param descriptor4All [output]所有图像特征描述子列表
* @param colors4All     [output]所有图像特征点像素列表
*/
void extractFeatures(vector<Mat>& images,
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


/**
 * matchFeatures 特征匹配
 * @param query   图像1特征点的特征描述
 * @param train   图像2特征点的特征描述
 * @param matches 匹配点对
 */
void matchFeatures(Mat& query, Mat& train, vector<DMatch>& matches) {
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


/**
 * getMatchedPoints 获取匹配点对的位置
 * @param p1      
 * @param p2      
 * @param matches 
 * @param out_p1  
 * @param out_p2  
 */
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


/**
 * getMatchedColors 获取匹配点对的像素值
 * @param c1      
 * @param c2      
 * @param matches 
 * @param out_c1  
 * @param out_c2  
 */
void getMatchedColors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches,
                      vector<Vec3b>& out_c1, vector<Vec3b>& out_c2) {
    out_c1.clear();
    out_c2.clear();
    for(int i = 0; i < matches.size(); ++i) {
        out_c1.push_back(c1[matches[i].queryIdx]);
        out_c2.push_back(c2[matches[i].trainIdx]);
    }
}


/**
 * 求本征矩阵，并分解出相机双目外参（位姿关系）（内参数使用相同矩阵）
 * @param K    [InputArray] 内参数矩阵
 * @param R    [OutputArray] camera2 对 camera1 的旋转矩阵
 * @param T    [OutputArray] camera2 对 camera1 的平移向量
 * @param p1   [InputArray] 同名点对在 camera1 图像上的点坐标
 * @param p2   [InputArray] 同名点对在 camera2 图像上的点坐标
 * @param mask [InputOutputArray] 匹配点对flag，接受的匹配点（inliers）返回正值
 */
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


/**
 * 求本征矩阵，并分解出相机双目外参（位姿关系）（内参数使用不同矩阵）
 * @param K1   [InputArray] camera1 内参数矩阵
 * @param K2   [InputArray] camera2 内参数矩阵
 * @param R    [OutputArray] camera2 对 camera1 的旋转矩阵
 * @param T    [OutputArray] camera2 对 camera1 的平移向量
 * @param p1   [InputArray] 同名点对在 camera1 图像上的点坐标
 * @param p2   [InputArray] 同名点对在 camera2 图像上的点坐标
 * @param mask [InputOutputArray] 匹配点对flag，接受的匹配点（inliers）返回正值
 */
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


/**
 * 去除不匹配点对（位置）
 * @param p1   [InputOutputArray]
 * @param mask [InputArray]
 */
void maskoutPoints(vector<Point2f>& p1, Mat& mask) {
    vector<Point2f> p1_copy = p1;
    p1.clear();

    for(int i = 0; i < mask.rows; ++i) {
        if(mask.at<uchar>(i) > 0) {
            p1.push_back(p1_copy[i]);
        }
    }
}


/**
 * 去除不匹配点对（像素值）
 * @param p1   [InputOutputArray]
 * @param mask [InputArray]
 */
void maskoutColors(vector<Vec3b>& p1, Mat& mask) {
    vector<Vec3b> p1_copy = p1;
    p1.clear();
    
    for(int i = 0; i < mask.rows; ++i) {
        if(mask.at<uchar>(i) > 0) {
            p1.push_back(p1_copy[i]);
        }
    }
}


/**
 * 三角化重建空间点（内参数使用相同矩阵）
 * @param K  [InputArray] 内参数矩阵
 * @param R  [InputArray] camera2 对 camera1 的旋转矩阵
 * @param T  [InputArray] camera2 对 camera1 的平移向量
 * @param p1 [InputArray] 同名点对在 camera1 图像上的点坐标
 * @param p2 [InputArray] 同名点对在 camera2 图像上的点坐标
 * @param structure [OutputArray] 重建出的空间点（齐次坐标）
 */
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
    proj1 = fK*proj1;
    proj2 = fK*proj2;

    // 三角重建
    triangulatePoints(proj1, proj2, p1, p2, structure);
}


/**
 * 三角化重建空间点（内参数使用不同矩阵）
 * @param K1 [InputArray] camera1 内参数矩阵
 * @param K2 [InputArray] camera2 内参数矩阵
 * @param R  [InputArray] camera2 对 camera1 的旋转矩阵
 * @param T  [InputArray] camera2 对 camera1 的平移向量
 * @param p1 [InputArray] 同名点对在 camera1 图像上的点坐标
 * @param p2 [InputArray] 同名点对在 camera2 图像上的点坐标
 * @param structure [OutputArray] 重建出的空间点（齐次坐标）
 */
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
    proj1 = fK1*proj1;
    proj2 = fK2*proj2;

    // 三角重建
    triangulatePoints(proj1, proj2, p1, p2, structure);
}


/**
 * 三角化重建空间点（内参数使用相同矩阵）
 * @param K  [InputArray] 内参数矩阵
 * @param R1 [InputArray] camera1 对世界坐标系的旋转矩阵
 * @param T1 [InputArray] camera1 对世界坐标系的平移向量
 * @param R2 [InputArray] camera2 对世界坐标系的旋转矩阵
 * @param T2 [InputArray] camera2 对世界坐标系的平移向量
 * @param p1 [InputArray] 同名点对在 camera1 图像上的点坐标
 * @param p2 [InputArray] 同名点对在 camera2 图像上的点坐标
 * @param structure [OutputArray] 重建出的空间点（空间坐标）
 */
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
    proj1 = fK*proj1;
    proj2 = fK*proj2;

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


/**
 * toPoints3D 将齐次坐标转换为空间坐标。数据为float型（CV_32FC1）。
 * @param points4D [InputArray] 齐次坐标点，4xN
 * @param points3D [OutputArray] 空间坐标点
 */
void toPoints3D(Mat& points4D, Mat& points3D) {
    points3D = Mat::zeros(3, points4D.size().width, CV_32FC1);
    for(int i = 0; i < points4D.size().width; i++) {
        points3D.at<float>(0, i) = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        points3D.at<float>(1, i) = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        points3D.at<float>(2, i) = points4D.at<float>(2, i) / points4D.at<float>(3, i);
    }
}


/**
 * saveStructure 保存相机位姿和点云坐标到文件
 * @param fileName  
 * @param rotations 
 * @param motions   
 * @param structure 
 * @param colors    
 */
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


/**
 * ranging 计算中位线距离
 * @param structure 点在世界坐标系下的坐标（单位：mm）
 * @param R 旋转矩阵
 * @param t 平移向量
 * @return 计算出的距离（单位：m）
 */
vector<double> ranging(Mat structure, Mat R, Mat t) {
    vector<double> range;
    // 输入大小有误，返回空
    if(structure.size().height != 3 || R.size() != Size(3, 3) || t.size() != Size(1, 3)) {
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
    
