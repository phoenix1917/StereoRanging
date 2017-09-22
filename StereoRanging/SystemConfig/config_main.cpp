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
#include "config_main.hpp"

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
    // ������ͼ������
    vector<Mat> imageSetL, imageSetR;
    // �궨ͼ��Ĵ�С
    Size calibImgSize;
    // �궨ͼ����
    int calibImgCount;
    // �ҵ�ȫ���ǵ��ͼ����
    int acceptedCount = 0;
    // �ҵ�ȫ���ǵ��ͼ����
    vector<int> acceptedImages;
    // ÿ��ͼ��ʹ�õı궨��Ľǵ����ά����
    vector<vector<Point3f>> objectPoints(1);
    // ���б궨ͼ��Ľǵ�
    vector<vector<Point2f>> allCornersL, allCornersR;
    // �ڲ�������ͻ���ϵ��
    Mat cameraMatrixL = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsL = Mat::zeros(5, 1, CV_64F);
    Mat cameraMatrixR = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsR = Mat::zeros(5, 1, CV_64F);
    // ˫Ŀ��Σ��������󣬻�������
    Mat R, T, E, F;
    // ��������
    vector<float> coef;

    // ��ȡ��Ŀͼ��
    while(getline(finL, fileName)) {
        Mat img = imread(fileName);
        imageSetL.push_back(img);
    }
    calibImgSize = imageSetL[0].size();
    calibImgCount = imageSetL.size();
    // ��ȡ��Ŀͼ��
    while(getline(finR, fileName)) {
        Mat img = imread(fileName);
        imageSetR.push_back(img);
    }
    if(calibImgCount != imageSetR.size()) {
        cout << "ͼ�����Ŀ��һ�£�����" << endl << endl;
        system("pause");
        return 0;
    }

    // ��ͼ����ȡ�ǵ�
    for(int i = 0; i < calibImgCount; i++) {
        bool foundAllCornersL = false, foundAllCornersR = false;
        vector<Point2f> cornerBufL, cornerBufR;
        Mat viewL, viewR, grayL, grayR;
        int filtering = 2;

        switch(filtering) {
        case 1:
            // ��ֵ�˲�
            medianBlur(imageSetL[i], viewL, 3);
            medianBlur(imageSetR[i], viewR, 3);
            break;
        case 2:
            // ˫���˲�
            bilateralFilter(imageSetL[i], viewL, 10, 20, 5);
            bilateralFilter(imageSetR[i], viewR, 10, 20, 5);
            break;
        default:
            viewL = imageSetL[i].clone();
            viewR = imageSetR[i].clone();
            break;
        }
        procCLAHE(viewL, imageSetL[i], 2.0, Size(2, 2));
        procCLAHE(viewR, imageSetR[i], 2.0, Size(2, 2));
        viewL = imageSetL[i].clone();
        viewR = imageSetR[i].clone();
        cvtColor(viewL, grayL, CV_RGB2GRAY);
        cvtColor(viewR, grayR, CV_RGB2GRAY);

        // ������ǿ��ͼ��
        if(doSaveEnhancedImg) {
            imwrite(pathEnhanced + "L_" + num2str(i + 1) + ".jpg", viewL);
            imwrite(pathEnhanced + "R_" + num2str(i + 1) + ".jpg", viewR);
        }

        // Ѱ�����̸���ڽǵ�λ��
        // flags:
        // CV_CALIB_CB_ADAPTIVE_THRESH
        // CV_CALIB_CB_NORMALIZE_IMAGE
        // CV_CALIB_CB_FILTER_QUADS
        // CALIB_CB_FAST_CHECK
        foundAllCornersL = findChessboardCorners(grayL, boardSize, cornerBufL,
                                                 CV_CALIB_CB_NORMALIZE_IMAGE);
        foundAllCornersR = findChessboardCorners(grayR, boardSize, cornerBufR,
                                                 CV_CALIB_CB_NORMALIZE_IMAGE);

        if(showCornerExt) {
            // �����ڽǵ㡣�����ȫ���ǵ㣬����չʾ����δ��������Ƽ���ĵ�
            drawChessboardCorners(viewL, boardSize, Mat(cornerBufL), foundAllCornersL);
            drawChessboardCorners(viewR, boardSize, Mat(cornerBufR), foundAllCornersR);
            imshow("corners-leftcam", viewL);
            imshow("corners-rightcam", viewR);
            waitKey();
        }

        if(foundAllCornersL && foundAllCornersR) {
            // ��¼����ͼ���±�
            acceptedCount += 1;
            acceptedImages.push_back(i);
            viewL = imageSetL[i].clone();
            viewR = imageSetR[i].clone();
            // Ѱ�������ؼ��ǵ㣬ֻ�ܴ���Ҷ�ͼ
            cornerSubPix(grayL, cornerBufL, Size(10, 10), Size(-1, -1),
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            cornerSubPix(grayR, cornerBufR, Size(10, 10), Size(-1, -1),
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            allCornersL.push_back(cornerBufL);
            allCornersR.push_back(cornerBufR);
            if(showCornerExt) {
                // ���Ƶ�����Ľǵ�
                drawChessboardCorners(viewL, boardSize, Mat(cornerBufL), true);
                drawChessboardCorners(viewR, boardSize, Mat(cornerBufR), true);
                imshow("corners-leftcam", viewL);
                imshow("corners-rightcam", viewR);
                waitKey();
            } else {
                cout << ">>>" << i + 1;
            }
        }
    }
    if(!showCornerExt) {
        cout << endl << endl;
    }
    destroyWindow("corners-leftcam");
    destroyWindow("corners-rightcam");

    // �����ȡ���ͳ��
    if(acceptedCount <= 3) {
        cout << "�ǵ���ʧ��" << endl;
        system("pause");
        return 0;
    } else {
        cout << "ʹ�� " << acceptedCount << " ��ͼ����б궨��" << endl;
        foutL << "ʹ�� " << acceptedCount << " ��ͼ����б궨��" << endl;
        foutR << "ʹ�� " << acceptedCount << " ��ͼ����б궨��" << endl;
        for(auto iter = acceptedImages.cbegin(); iter != acceptedImages.cend(); ) {
            cout << (*iter) + 1;
            foutL << (*iter) + 1;
            foutR << (*iter) + 1;
            ++iter;
            if(iter != acceptedImages.cend()) {
                cout << ", ";
                foutL << ", ";
                foutR << ", ";
            } else {
                cout << endl << endl;
                foutL << endl << endl;
                foutR << endl << endl;
            }
        }
    }

    // �������ѽ���ͼ�񣬳�ʼ���궨���Ͻǵ����ά����
    calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);
    objectPoints.resize(allCornersL.size(), objectPoints[0]);

    // ��ʼ���ڲ�������
    cameraMatrixL = initCameraMatrix2D(objectPoints, allCornersL, calibImgSize);
    cameraMatrixR = initCameraMatrix2D(objectPoints, allCornersR, calibImgSize);
    cout << "�ڲ���Ԥ����" << endl;
    cout << "cameraMatrix_L = " << endl << cameraMatrixL << endl << endl;
    cout << "cameraMatrix_R = " << endl << cameraMatrixR << endl << endl;
    cout << "------------------------------------------" << endl << endl;
    foutL << "�ڲ���Ԥ����" << endl;
    foutL << "cameraMatrix = " << endl << cameraMatrixL << endl << endl;
    foutR << "�ڲ���Ԥ����" << endl;
    foutR << "cameraMatrix = " << endl << cameraMatrixR << endl << endl;

    if(doSingleCalib) {
        // ÿ��ͼ�����ת�����ƽ������
        vector<Mat> rVecsL, tVecsL, rVecsR, tVecsR;
        // �ڲ������
        Mat stdDevIntrinsicsL, stdDevIntrinsicsR;
        // ��������
        Mat stdDevExtrinsicsL, stdDevExtrinsicsR;
        // ÿ��ͼ�����ͶӰ���
        vector<double> perViewErrorsL, perViewErrorsR;

        // ��Ŀ�궨
        // flags:
        // CV_CALIB_USE_INTRINSIC_GUESS        ʹ��Ԥ�����ڲ�������
        // CV_CALIB_FIX_PRINCIPAL_POINT        �̶���������
        // CV_CALIB_FIX_ASPECT_RATIO           �̶�����ȣ�ֻ����fy
        // CV_CALIB_ZERO_TANGENT_DIST          �������������(p1, p2)
        // CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 �̶��������K1-K6����
        // CV_CALIB_RATIONAL_MODEL             ����8��������ģ��(K4-K6)����д�����5����
        // CALIB_THIN_PRISM_MODEL              ����S1-S4��������д�򲻼���
        // CALIB_FIX_S1_S2_S3_S4               �̶�S1-S4����ֵ
        // CALIB_TILTED_MODEL                  ����(tauX, tauY)��������д�򲻼���
        // CALIB_FIX_TAUX_TAUY                 �̶�(tauX, tauY)����ֵ

        // ��Ŀ
        double reprojectionErrorL = calibrateCamera(objectPoints, allCornersL, calibImgSize,
                                                    cameraMatrixL, distCoeffsL, rVecsL, tVecsL,
                                                    stdDevIntrinsicsL, stdDevExtrinsicsL, perViewErrorsL,
                                                    CV_CALIB_USE_INTRINSIC_GUESS +
                                                    CV_CALIB_FIX_PRINCIPAL_POINT);
        cout << "��Ŀ";
        printCalibResults(cameraMatrixL, distCoeffsL, reprojectionErrorL, stdDevIntrinsicsL);
        cout << "------------------------------------------" << endl << endl;

        // ��Ŀ
        double reprojectionErrorR = calibrateCamera(objectPoints, allCornersR, calibImgSize,
                                                    cameraMatrixR, distCoeffsR, rVecsR, tVecsR,
                                                    stdDevIntrinsicsR, stdDevExtrinsicsR, perViewErrorsR,
                                                    CV_CALIB_USE_INTRINSIC_GUESS +
                                                    CV_CALIB_FIX_PRINCIPAL_POINT);
        cout << "��Ŀ";
        printCalibResults(cameraMatrixR, distCoeffsR, reprojectionErrorR, stdDevIntrinsicsR);
        cout << "------------------------------------------" << endl << endl;

        // ����궨������ļ�
        printCalibResults(cameraMatrixL, distCoeffsL, reprojectionErrorL, stdDevIntrinsicsL, stdDevExtrinsicsL, perViewErrorsL, foutL);
        printCalibResults(cameraMatrixR, distCoeffsR, reprojectionErrorR, stdDevIntrinsicsR, stdDevExtrinsicsR, perViewErrorsR, foutR);
    } else {
        // �ֶ������ڲ���
        cout << "�����ڲ�����" << endl;
        cout << "��Ŀ����(fx, fy)�� ";
        cin >> cameraMatrixL.at<double>(0, 0) >> cameraMatrixL.at<double>(1, 1);
        cout << "��Ŀ��������(Cx, Cy)�� ";
        cin >> cameraMatrixL.at<double>(0, 2) >> cameraMatrixL.at<double>(1, 2);
        cout << endl << endl;
        cout << "��Ŀ����(fx, fy)�� ";
        cin >> cameraMatrixR.at<double>(0, 0) >> cameraMatrixR.at<double>(1, 1);
        cout << "��Ŀ��������(Cx, Cy)�� ";
        cin >> cameraMatrixR.at<double>(0, 2) >> cameraMatrixR.at<double>(1, 2);
        cout << endl << "------------------------------------------" << endl << endl;

        // TEST:
        // ָ���ڲ���
        //cameraMatrixL.at<double>(0, 0) = 8131.6;
        //cameraMatrixL.at<double>(1, 1) = 8854.1;
        //cameraMatrixL.at<double>(0, 2) = 360;
        //cameraMatrixL.at<double>(1, 2) = 288;

        //cameraMatrixR.at<double>(0, 0) = 8112.8;
        //cameraMatrixR.at<double>(1, 1) = 8831.9;
        //cameraMatrixR.at<double>(0, 2) = 360;
        //cameraMatrixR.at<double>(1, 2) = 288;

        //cout << "cameraMatrixL = " << endl;
        //cout << cameraMatrixL << endl << endl;
        //cout << "cameraMatrixR = " << endl;
        //cout << cameraMatrixR << endl << endl;
        //cout << "------------------------------------------" << endl << endl;
    }

    if(doStereoCalib) {
        // ����궨
        // flags:
        // CV_CALIB_FIX_INTRINSIC              �̶��ڲ����ͻ���ģ�ͣ�ֻ����(R, T, E, F)
        // CV_CALIB_USE_INTRINSIC_GUESS        ʹ��Ԥ�����ڲ�������
        // CV_CALIB_FIX_PRINCIPAL_POINT        �̶���������
        // CV_CALIB_FIX_FOCAL_LENGTH           �̶�����
        // CV_CALIB_FIX_ASPECT_RATIO           �̶�����ȣ�ֻ����fy
        // CV_CALIB_SAME_FOCAL_LENGTH          �̶�x, y���򽹾��Ϊ1
        // CV_CALIB_ZERO_TANGENT_DIST          �������������(p1, p2)
        // CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 �̶��������K1-K6����
        // CV_CALIB_RATIONAL_MODEL             ����8��������ģ��(K4-K6)����д�����5����
        // CALIB_THIN_PRISM_MODEL              ����S1-S4��������д�򲻼���
        // CALIB_FIX_S1_S2_S3_S4               �̶�S1-S4����ֵ
        // CALIB_TILTED_MODEL                  ����(tauX, tauY)��������д�򲻼���
        // CALIB_FIX_TAUX_TAUY                 �̶�(tauX, tauY)����ֵ
        double reprojErrorStereo = stereoCalibrate(objectPoints, allCornersL, allCornersR,
                                                   cameraMatrixL, distCoeffsL,
                                                   cameraMatrixR, distCoeffsR,
                                                   calibImgSize, R, T, E, F,
                                                   CALIB_FIX_INTRINSIC);

        //CALIB_USE_INTRINSIC_GUESS +
        //CALIB_FIX_PRINCIPAL_POINT

        // Rodrigues�任�������ʱ����ת�ĽǶȣ�
        Mat rod;
        Rodrigues(R, rod);
        rod.at<double>(0, 0) = rod.at<double>(0, 0) / CV_PI * 180;
        rod.at<double>(1, 0) = rod.at<double>(1, 0) / CV_PI * 180;
        rod.at<double>(2, 0) = rod.at<double>(2, 0) / CV_PI * 180;

        // ���˫Ŀ�궨���
        cout << "˫Ŀ�궨�����" << endl;
        cout << "R = " << endl << R << endl;
        cout << "Rodrigues = " << endl << rod << endl;
        cout << "t = " << endl << T << endl;
        cout << "E = " << endl << E << endl;
        cout << "F = " << endl << F << endl;
        cout << "reprojection error = " << endl << reprojErrorStereo << endl;
        cout << endl << "------------------------------------------" << endl << endl;
        foutStereo << "R = " << endl << R << endl;
        foutStereo << "Rodrigues = " << endl << rod << endl;
        foutStereo << "t = " << endl << T << endl;
        foutStereo << "E = " << endl << E << endl;
        foutStereo << "F = " << endl << F << endl;
        foutStereo << "reprojection error = " << endl << reprojErrorStereo << endl;

    } else {
        R = Mat::eye(3, 3, CV_64F);
        T = Mat::zeros(3, 1, CV_64F);
        // �ֶ����������
        cout << "�����������" << endl;
        cout << "R = " << endl;
        cin >> R.at<double>(0, 0) >> R.at<double>(0, 1) >> R.at<double>(0, 2);
        cin >> R.at<double>(1, 0) >> R.at<double>(1, 1) >> R.at<double>(1, 2);
        cin >> R.at<double>(2, 0) >> R.at<double>(2, 1) >> R.at<double>(2, 2);
        cout << endl << "t = " << endl;
        cin >> T.at<double>(0, 0) >> T.at<double>(1, 0) >> T.at<double>(2, 0);
        // ���Rodrigues�任���
        Mat rod;
        Rodrigues(R, rod);
        rod.at<double>(0, 0) = rod.at<double>(0, 0) / CV_PI * 180;
        rod.at<double>(1, 0) = rod.at<double>(1, 0) / CV_PI * 180;
        rod.at<double>(2, 0) = rod.at<double>(2, 0) / CV_PI * 180;
        cout << endl << "Rodrigues = " << endl << rod << endl;
    }

    // ���ڲ���ͼ��Ĵ�С
    Size trainImgSize;
    // ���ڲ���ͼ����
    int trainImgCount;
    // ���ͼ���ϵ�������λ��
    vector<Point2f> objectPointsL, objectPointsR;
    // ���ͼ���ϵ�����������ֵ
    vector<Vec3b> objectColorsL, objectColorsR;
    // ���ǻ��ؽ�������������ϵ����
    Mat structure, structure3D;
    // ������λ�߾���
    vector<double> dist;

    // ѵ��
    // ��ȡ��Ŀѵ��ͼ��
    while(getline(fTrainL, fileName)) {
        Mat img = imread(fileName);
        trainSetL.push_back(img);
    }
    trainImgSize = trainSetL[0].size();
    trainImgCount = trainSetL.size();
    // ��ȡ��Ŀѵ��ͼ��
    while(getline(fTrainR, fileName)) {
        Mat img = imread(fileName);
        trainSetR.push_back(img);
    }

    // ���ڱ�������ͼ��Ĳ��ֵ
    vector<float> trainVal(trainImgCount);
    // ѵ��ͼ���е�Ŀ����ʵ����
    vector<float> groundTruth;

    // ��ȡground truth
    while(getline(fTrainGT, fileName)) {
        groundTruth.push_back(atof(fileName.c_str()));
    }

    for(int i = 0; i < trainImgCount; i++) {
        //// ��ֵ�˲�
        //medianBlur(trainSetL[i], tempEnhanceL, 3);
        //medianBlur(trainSetR[i], tempEnhanceR, 3);
        //trainSetL[i] = tempEnhanceL.clone();
        //trainSetR[i] = tempEnhanceR.clone();

        // �������
        if(doDistortionCorrect) {
            undistort(trainSetL[i].clone(), trainSetL[i], cameraMatrixL, distCoeffsL);
            undistort(trainSetR[i].clone(), trainSetR[i], cameraMatrixR, distCoeffsR);
        }

        if(doEnhance) {
            // CLAHE��ǿ
            procCLAHE(trainSetL[i], tempEnhanceL, clipL / 10.0, Size(gridXL, gridYL));
            procCLAHE(trainSetR[i], tempEnhanceR, clipR / 10.0, Size(gridXR, gridYR));
            namedWindow("enhance_leftcam");
            createTrackbar("Clip", "enhance_leftcam", &clipL, MAX_CLIP_LIMIT, onEnhanceTrackbarL_Train, (void *)&i);
            createTrackbar("Grid X", "enhance_leftcam", &gridXL, MAX_GRID_SIZE_X, onEnhanceTrackbarL_Train, (void *)&i);
            createTrackbar("Grid Y", "enhance_leftcam", &gridYL, MAX_GRID_SIZE_Y, onEnhanceTrackbarL_Train, (void *)&i);
            setMouseCallback("enhance_leftcam", onEnhanceMouseL_Train, (void *)&i);
            imshow("enhance_leftcam", tempEnhanceL);

            namedWindow("enhance_rightcam");
            createTrackbar("Clip", "enhance_rightcam", &clipR, MAX_CLIP_LIMIT, onEnhanceTrackbarR_Train, (void *)&i);
            createTrackbar("Grid X", "enhance_rightcam", &gridXR, MAX_GRID_SIZE_X, onEnhanceTrackbarR_Train, (void *)&i);
            createTrackbar("Grid Y", "enhance_rightcam", &gridYR, MAX_GRID_SIZE_Y, onEnhanceTrackbarR_Train, (void *)&i);
            setMouseCallback("enhance_rightcam", onEnhanceMouseR_Train, (void *)&i);
            imshow("enhance_rightcam", tempEnhanceR);

            cout << "��" << i + 1 << "��ͼ��" << endl;
            cout << "��ǿͼ��" << endl ;
            waitKey();
        }

        imshow("train_leftcam", trainSetL[i]);
        imshow("train_rightcam", trainSetR[i]);

        // ��ͼ��ѡȡROI
        setMouseCallback("train_leftcam", onMouseL_ROI_Train, (void *)&i);
        setMouseCallback("train_rightcam", onMouseR_ROI_Train, (void *)&i);
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

        switch(feature) {
        case SIFT:
            // �ֲ�����ƥ��
            extractSIFTFeatures(rois, keyPoints4All, descriptor4All, colors4All);
            matchSIFTFeatures(descriptor4All[0], descriptor4All[1], matches);
            getMatchedPoints(keyPoints4All[0], keyPoints4All[1], matches, objectPointsL, objectPointsR);
            getMatchedColors(colors4All[0], colors4All[1], matches, objectColorsL, objectColorsR);
            break;
        case GMS:
            gmsMatch(roiImgL, roiImgR, objectPointsL, objectPointsR);
            break;
        default:
            break;
        }

        if(objectPointsL.size() == 0) {
            cout << "û���ҵ�ƥ��㣬�ֶ�ѡ��һ��" << endl;
            imshow("train_leftcam", trainSetL[i]);
            imshow("train_rightcam", trainSetR[i]);
            // �ֶ�ѡ��һ���
            targetL = Point(0, 0);
            targetR = Point(0, 0);
            setMouseCallback("train_leftcam", onMouseL_Train, (void *)&i);
            setMouseCallback("train_rightcam", onMouseR_Train, (void *)&i);
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
            // ��ROI����ָ���ԭͼ����
            for(int i = 0; i < objectPointsL.size(); i++) {
                objectPointsL[i] += Point2f(roiL.x, roiL.y);
                objectPointsR[i] += Point2f(roiR.x, roiR.y);
            }
        }

        // �ؽ�����
        reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                    objectPointsL, objectPointsR, structure);
        toPoints3D(structure, structure3D);
        // ��λ�߾���
        dist = ranging(structure3D, R, T);

        // �������
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
        cout << "Ŀ�����(����ѵ��) " << range << " m" << endl ;
        trainVal[i] = range;
        // ground truth
        cout << "����ʵ�ʾ��룺 " << groundTruth[i] << " m" << endl << endl;
        waitKey();
        destroyAllWindows();
    }

    cout << "��ʼѵ������ģ��" << endl;
    cout << "----------------------------------" << endl;
    switch(fit) {
    case Poly:
        // ����ʽ���
        coef = polyfit2(trainVal, groundTruth, 4);
        break;
    case Exp2:
        // ����ָ�����
        coef = exp2fit(trainVal, groundTruth, 0.01);
        break;
    default:
        break;
    }
    printCoef(coef, fit);
    saveTrainResults(trainResult, cameraMatrixL, distCoeffsL, cameraMatrixR,
                     distCoeffsR, R, T, trainVal, groundTruth, fit, coef);
    cout << "----------------------------------" << endl;
    cout << "���" << endl;

    system("pause");
    return 0;
}
#endif


void onMouseL_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetL[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetL = Point(x, y);
        circle(frame, targetL, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetL, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("train_leftcam", frame);
    }
}

void onMouseL_ROI_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetL[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetL = Point(x, y);
        getROI(frame, targetL, roiSize, roiL, roiImgL);
        rectangle(frame, roiL, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("train_leftcam", frame);
    }
}

void onMouseR_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetR[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetR = Point(x, y);
        circle(frame, targetR, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetR, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("train_rightcam", frame);
    }
}

void onMouseR_ROI_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetR[*i].clone();
        // ��¼��ǰλ�õ����꣬��һ����
        targetR = Point(x, y);
        getROI(frame, targetR, roiSize, roiR, roiImgR);
        rectangle(frame, roiR, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("train_rightcam", frame);
    }
}

void onEnhanceTrackbarL_Train(int pos, void *userdata) {
    int *i = (int *)userdata;
    if(clipL && gridXL && gridYL) {
        double clipVal = double(clipL) / 10;
        procCLAHE(trainSetL[*i], tempEnhanceL, clipVal, Size(gridXL, gridYL));
        imshow("enhance_leftcam", tempEnhanceL);
    }
}

void onEnhanceTrackbarR_Train(int pos, void *userdata) {
    int *i = (int *)userdata;
    if(clipR && gridXR && gridYR) {
        double clipVal = double(clipR) / 10;
        procCLAHE(trainSetR[*i], tempEnhanceR, clipVal, Size(gridXR, gridYR));
        imshow("enhance_rightcam", tempEnhanceR);
    }
}

void onEnhanceMouseL_Train(int event, int x, int y, int flags, void* param) {
    int *i = (int *)param;

    switch(event) {
    case EVENT_LBUTTONDOWN:
        imshow("enhance_leftcam", trainSetL[*i]);
        break;
    case EVENT_LBUTTONUP:
        imshow("enhance_leftcam", tempEnhanceL);
        break;
    case EVENT_RBUTTONUP:
        tempEnhanceL.copyTo(trainSetL[*i]);
        destroyWindow("enhance_leftcam");
        break;
    default:
        break;
    }
}

void onEnhanceMouseR_Train(int event, int x, int y, int flags, void* param) {
    int *i = (int *)param;

    switch(event) {
    case EVENT_LBUTTONDOWN:
        imshow("enhance_rightcam", trainSetR[*i]);
        break;
    case EVENT_LBUTTONUP:
        imshow("enhance_rightcam", tempEnhanceR);
        break;
    case EVENT_RBUTTONUP:
        tempEnhanceR.copyTo(trainSetR[*i]);
        destroyWindow("enhance_rightcam");
        break;
    }
}

