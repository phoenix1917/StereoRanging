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


/**
 * usingBoard��������ʹ�õı궨��
 * @param boardNum   [Input]�궨����
 *                      �궨��1��9x7��35mm����ɫ��Ե
 *                      �궨��2��12x9��30mm����ɫ��Ե
 *                      �궨��3��10x9��90mm����ɫ��Ե
 *                      �궨��4��13x9��30mm����ɫ��Ե
 * @param boardSize  [Output]���ر궨�����̸��ڽǵ���Ŀ
 * @param squareSize [Output]���ر궨�巽��߳�
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
        cout << "�޶�Ӧ�궨��" << endl;
        break;
    }
}


/**
 * ����궨��ǵ�λ��
 * @param boardSize  [input]�궨���ڽǵ�size
 * @param squareSize [input]�궨�巽��߳�
 * @param corners    [output]���صĽǵ���λ��
 */
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners) {
    for(int i = 0; i < boardSize.height; ++i) {
        for(int j = 0; j < boardSize.width; ++j) {
            corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
        }
    }
}


/**
 * ����ͶӰ���
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
 * �̶��������굽ָ��ֵ
 * @param K     [InputOutputArray] �ڲ�������(CV_64FC1)
 * @param point [InputArray] Ҫ�̶�����������λ��
 * @return      �Ƿ�ɹ�����
 */
bool fixPrinciplePoint(Mat& K, Point2f point) {
    if(K.size() == Size(3, 3)) {
        K.at<double>(0, 2) = point.x;
        K.at<double>(1, 2) = point.y;
        return true;
    } else {
        cout << "�ڲ��������С����ȷ������" << endl;
        return false;
    }
}


/**
 * �������󣬲��ֽ�����˫Ŀ��Σ�λ�˹�ϵ�����ڲ���ʹ����ͬ����
 * @param K    [InputArray] �ڲ�������
 * @param R    [OutputArray] camera2 �� camera1 ����ת����
 * @param T    [OutputArray] camera2 �� camera1 ��ƽ������
 * @param p1   [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
 * @param p2   [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
 * @param mask [InputOutputArray] ƥ����flag�����ܵ�ƥ��㣨inliers��������ֵ
 */
bool findTransform(Mat& K, Mat& R, Mat& T,
                   vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask) {
    //�����ڲξ����ȡ����Ľ���͹������꣨�������꣩
    double focalLength = 0.5 * (K.at<double>(0) + K.at<double>(4));
    Point2d principlePoint(K.at<double>(2), K.at<double>(5));

    //����ƥ�����ȡ��������ʹ��RANSAC����һ���ų�ʧ���
    Mat E = findEssentialMat(p1, p2, focalLength, principlePoint, RANSAC, 0.999, 1.0, mask);
    if(E.empty()) {
        return false;
    }

    double feasibleCount = countNonZero(mask);
    cout << (int)feasibleCount << " -in- " << p1.size() << endl;
    //����RANSAC���ԣ�outlier��������50%ʱ������ǲ��ɿ���
    if(feasibleCount <= 15 || (feasibleCount / p1.size()) < 0.6) {
        return false;
    }

    //�ֽⱾ�����󣬻�ȡ��Ա任������inliers��Ŀ
    int passCount = recoverPose(E, p1, p2, R, T, focalLength, principlePoint, mask);

    //ͬʱλ���������ǰ���ĵ������Ҫ�㹻��
    if(((double)passCount) / feasibleCount < 0.7) {
        return false;
    }

    return true;
}


/**
 * �������󣬲��ֽ�����˫Ŀ��Σ�λ�˹�ϵ�����ڲ���ʹ�ò�ͬ����
 * @param K1   [InputArray] camera1 �ڲ�������
 * @param K2   [InputArray] camera2 �ڲ�������
 * @param R    [OutputArray] camera2 �� camera1 ����ת����
 * @param T    [OutputArray] camera2 �� camera1 ��ƽ������
 * @param p1   [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
 * @param p2   [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
 * @param mask [InputOutputArray] ƥ����flag�����ܵ�ƥ��㣨inliers��������ֵ
 */
bool findTransform(Mat& K1, Mat& K2, Mat& R, Mat& T,
                   vector<Point2f>& p1, vector<Point2f>& p2, Mat& mask) {
    //�����ڲξ����ȡ����Ľ���͹������꣨�������꣩
    double focalLength = 0.5 * (K1.at<double>(0) + K1.at<double>(4) +
                                K2.at<double>(0) + K2.at<double>(4));
    Point2d principlePoint((K1.at<double>(2) + K2.at<double>(2)) / 2,
                           (K1.at<double>(5) + K2.at<double>(5)) / 2);

    //����ƥ�����ȡ��������ʹ��RANSAC����һ���ų�ʧ���
    Mat E = findEssentialMat(p1, p2, focalLength, principlePoint, RANSAC, 0.999, 1.0, mask);
    if(E.empty()) {
        return false;
    }

    double feasibleCount = countNonZero(mask);
    cout << (int)feasibleCount << " -in- " << p1.size() << endl;
    //����RANSAC���ԣ�outlier��������50%ʱ������ǲ��ɿ���
    if(feasibleCount <= 15 || (feasibleCount / p1.size()) < 0.6) {
        return false;
    }

    //�ֽⱾ�����󣬻�ȡ��Ա任������inliers��Ŀ
    int passCount = recoverPose(E, p1, p2, R, T, focalLength, principlePoint, mask);

    //ͬʱλ���������ǰ���ĵ������Ҫ�㹻��
    if(((double)passCount) / feasibleCount < 0.7) {
        return false;
    }

    return true;
}


/**
 * ȥ����ƥ���ԣ�λ�ã�
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
 * ȥ����ƥ���ԣ�����ֵ��
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
 * ���ǻ��ؽ��ռ�㣨�ڲ���ʹ����ͬ����
 * @param K  [InputArray] �ڲ�������
 * @param R  [InputArray] camera2 �� camera1 ����ת����
 * @param T  [InputArray] camera2 �� camera1 ��ƽ������
 * @param p1 [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
 * @param p2 [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
 * @param structure [OutputArray] �ؽ����Ŀռ�㣨������꣩
 */
void reconstruct(Mat& K, Mat& R, Mat& T,
                 vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure) {
    // ���������ͶӰ���󣨵�Ӧ�Ծ���K[R T]��triangulatePointsֻ֧��float�ͣ�CV_32FC1��
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    // ��ʼ��camera1��ͶӰ����
    // ��������ϵ������camera1�����������ϵ��
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    // ��ʼ��camera2��ͶӰ����
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);

    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK * proj1;
    proj2 = fK * proj2;

    // �����ؽ�
    triangulatePoints(proj1, proj2, p1, p2, structure);
}


/**
 * ���ǻ��ؽ��ռ�㣨�ڲ���ʹ�ò�ͬ����
 * @param K1 [InputArray] camera1 �ڲ�������
 * @param K2 [InputArray] camera2 �ڲ�������
 * @param R  [InputArray] camera2 �� camera1 ����ת����
 * @param T  [InputArray] camera2 �� camera1 ��ƽ������
 * @param p1 [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
 * @param p2 [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
 * @param structure [OutputArray] �ؽ����Ŀռ�㣨������꣩
 */
void reconstruct(Mat& K1, Mat& K2, Mat& R, Mat& T,
                 vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure) {
    // ���������ͶӰ���󣨵�Ӧ�Ծ���K[R T]��triangulatePointsֻ֧��float�ͣ�CV_32FC1��
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    // ��ʼ��camera1��ͶӰ����
    // ��������ϵ������camera1�����������ϵ��
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    // ��ʼ��camera2��ͶӰ����
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);

    Mat fK1, fK2;
    K1.convertTo(fK1, CV_32FC1);
    K2.convertTo(fK2, CV_32FC1);
    proj1 = fK1 * proj1;
    proj2 = fK2 * proj2;

    // �����ؽ�
    triangulatePoints(proj1, proj2, p1, p2, structure);
}


/**
 * ���ǻ��ؽ��ռ�㣨�ڲ���ʹ����ͬ����
 * @param K  [InputArray] �ڲ�������
 * @param R1 [InputArray] camera1 ����������ϵ����ת����
 * @param T1 [InputArray] camera1 ����������ϵ��ƽ������
 * @param R2 [InputArray] camera2 ����������ϵ����ת����
 * @param T2 [InputArray] camera2 ����������ϵ��ƽ������
 * @param p1 [InputArray] ͬ������� camera1 ͼ���ϵĵ�����
 * @param p2 [InputArray] ͬ������� camera2 ͼ���ϵĵ�����
 * @param structure [OutputArray] �ؽ����Ŀռ�㣨�ռ����꣩
 */
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2,
                 vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure) {
    //���������ͶӰ���󣨵�Ӧ�Ծ���K[R T]��triangulatePointsֻ֧��float��
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

    //�����ؽ�
    Mat structure4D;
    triangulatePoints(proj1, proj2, p1, p2, structure4D);

    structure.clear();
    structure.reserve(structure4D.cols);
    for(int i = 0; i < structure4D.cols; ++i) {
        //������꣬�������һ��Ԫ��תΪ�ռ�����
        Mat_<float> col = structure4D.col(i);
        col /= col(3);
        structure.push_back(Point3f(col(0), col(1), col(2)));
    }
}


/**
 * toPoints3D ���������ת��Ϊ�ռ����ꡣ����Ϊfloat�ͣ�CV_32FC1����
 * @param points4D [InputArray] �������㣬4xN
 * @param points3D [OutputArray] �ռ������
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
 * saveStructure �������λ�˺͵������굽�ļ�
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
 * ranging ͨ��R��t������λ�߾���
 * @param structure ������������ϵ�µ����꣨��λ��mm��
 * @param R ��ת����
 * @param t ƽ������
 * @return ������ľ��루��λ��m��
 */
vector<double> ranging(Mat structure, Mat R, Mat t) {
    vector<double> range;
    // �����С���󣬷��ؿ�
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
    

/**
 * ranging ͨ��K��t������λ�߾���
 * @param points1 ��1��ͼ���ϵĲ�������
 * @param points2 ��2��ͼ���϶�Ӧ��ͬ��������
 * @param K1 ��1��ͼ���Ӧ������ڲ���
 * @param K2 ��2��ͼ���Ӧ������ڲ���
 * @param t ƽ������
 * @return ������ľ��루��λ��m��
 */
vector<double> ranging(Mat points1, Mat points2, Mat K1, Mat K2, Mat t) {
    vector<double> range;
    vector<Point3f> points3D1, points3D2;
    Mat invertK1, invertK2;
    // �����С���󣬷��ؿ�
    if(K1.size() != Size(3, 3) || K2.size() != Size(3, 3) || t.size() != Size(1, 3)) {
        cout << "Ranging error: wrong input sizes." << endl;
        return range;
    }

    for(int i = 0; i < points1.size().width; i++) {
        
    }
}


/**
 * points2DtoPoints3D ��ͼ������ӳ�䵽���������ϵ�Ŀռ�����
 * @param K        [InputArray] ����ڲ�������
 * @param points   [InputArray] ͼ�����꣨����Σ�
 * @param points3D [OutputArray] ���������ϵ�µĿռ����꣨����Σ�
 */
void points2DToPoints3D(Mat K, vector<Point2f> points, Mat points3D) {
    Mat invertK;
    for(int i = 0; i < points.size(); i++) {

    }
}

