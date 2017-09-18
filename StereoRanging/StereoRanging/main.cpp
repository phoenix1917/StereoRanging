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
#include "algorithms.hpp"

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
    // 读入后的图像序列
    vector<Mat> imageSetL, imageSetR;
    // 标定图像的大小
    Size calibImgSize;
    // 标定图像数
    int calibImgCount;
    // 找到全部角点的图像数
    int acceptedCount = 0;
    // 找到全部角点的图像编号
    vector<int> acceptedImages;
    // 每幅图像使用的标定板的角点的三维坐标
    vector<vector<Point3f>> objectPoints(1);
    // 所有标定图像的角点
    vector<vector<Point2f>> allCornersL, allCornersR;
    // 内参数矩阵和畸变系数
    Mat cameraMatrixL = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsL = Mat::zeros(5, 1, CV_64F);
    Mat cameraMatrixR = Mat::eye(3, 3, CV_64F);
    Mat distCoeffsR = Mat::zeros(5, 1, CV_64F);
    // 双目外参，本征矩阵，基础矩阵
    Mat R, T, E, F;
    // 补偿参数
    Mat coef;
    
    // 读取左目图像
    while(getline(finL, fileName)) {
        Mat img = imread(fileName);
        imageSetL.push_back(img);
    }
    calibImgSize = imageSetL[0].size();
    calibImgCount = imageSetL.size();
    // 读取右目图像
    while(getline(finR, fileName)) {
        Mat img = imread(fileName);
        imageSetR.push_back(img);
    }
    if(calibImgCount != imageSetR.size()) {
        cout << "图像对数目不一致，请检查" << endl << endl;
        system("pause");
        return 0;
    }

    // 逐图像提取角点
    for(int i = 0; i < calibImgCount; i++) {
        bool foundAllCornersL = false, foundAllCornersR = false;
        vector<Point2f> cornerBufL, cornerBufR;
        Mat viewL, viewR, grayL, grayR;
        int filtering = 2;

        switch(filtering) {
        case 1:
            // 中值滤波
            medianBlur(imageSetL[i], viewL, 3);
            medianBlur(imageSetR[i], viewR, 3);
            break;
        case 2:
            // 双边滤波
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

        // 保存增强的图像
        if(doSaveEnhancedImg) {
            imwrite(pathEnhanced + "L_" + num2str(i + 1) + ".jpg", viewL);
            imwrite(pathEnhanced + "R_" + num2str(i + 1) + ".jpg", viewR);
        }

        // 寻找棋盘格的内角点位置
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
            // 绘制内角点。若检出全部角点，连线展示；若未检出，绘制检出的点
            drawChessboardCorners(viewL, boardSize, Mat(cornerBufL), foundAllCornersL);
            drawChessboardCorners(viewR, boardSize, Mat(cornerBufR), foundAllCornersR);
            imshow("corners-leftcam", viewL);
            imshow("corners-rightcam", viewR);
            waitKey();
        }

        if(foundAllCornersL && foundAllCornersR) {
            // 记录这组图像下标
            acceptedCount += 1;
            acceptedImages.push_back(i);
            viewL = imageSetL[i].clone();
            viewR = imageSetR[i].clone();
            // 寻找亚像素级角点，只能处理灰度图
            cornerSubPix(grayL, cornerBufL, Size(10, 10), Size(-1, -1),
                            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            cornerSubPix(grayR, cornerBufR, Size(10, 10), Size(-1, -1),
                            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            allCornersL.push_back(cornerBufL);
            allCornersR.push_back(cornerBufR);
            if(showCornerExt) {
                // 绘制调整后的角点
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

    // 输出提取结果统计
    if(acceptedCount <= 3) {
        cout << "角点检测失败" << endl;
        system("pause");
        return 0;
    } else {
        cout << "使用 " << acceptedCount << " 幅图像进行标定：" << endl;
        foutL << "使用 " << acceptedCount << " 幅图像进行标定：" << endl;
        foutR << "使用 " << acceptedCount << " 幅图像进行标定：" << endl;
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
    
    // 对所有已接受图像，初始化标定板上角点的三维坐标
    calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);
    objectPoints.resize(allCornersL.size(), objectPoints[0]);
    
    // 初始化内参数矩阵
    cameraMatrixL = initCameraMatrix2D(objectPoints, allCornersL, calibImgSize);
    cameraMatrixR = initCameraMatrix2D(objectPoints, allCornersR, calibImgSize);
    cout << "内参数预估：" << endl;
    cout << "cameraMatrix_L = " << endl << cameraMatrixL << endl << endl;
    cout << "cameraMatrix_R = " << endl << cameraMatrixR << endl << endl;
    cout << "------------------------------------------" << endl << endl;
    foutL << "内参数预估：" << endl;
    foutL << "cameraMatrix = " << endl << cameraMatrixL << endl << endl;
    foutR << "内参数预估：" << endl;
    foutR << "cameraMatrix = " << endl << cameraMatrixR << endl << endl;

    if(doSingleCalib) {
        // 每幅图像的旋转矩阵和平移向量
        vector<Mat> rVecsL, tVecsL, rVecsR, tVecsR;
        // 内参数误差
        Mat stdDevIntrinsicsL, stdDevIntrinsicsR;
        // 外参数误差
        Mat stdDevExtrinsicsL, stdDevExtrinsicsR;
        // 每幅图像的重投影误差
        vector<double> perViewErrorsL, perViewErrorsR;

        // 单目标定
        // flags:
        // CV_CALIB_USE_INTRINSIC_GUESS        使用预估的内参数矩阵
        // CV_CALIB_FIX_PRINCIPAL_POINT        固定主点坐标
        // CV_CALIB_FIX_ASPECT_RATIO           固定焦距比，只计算fy
        // CV_CALIB_ZERO_TANGENT_DIST          不计算切向畸变(p1, p2)
        // CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 固定径向畸变K1-K6参数
        // CV_CALIB_RATIONAL_MODEL             计算8参数畸变模型(K4-K6)，不写则计算5参数
        // CALIB_THIN_PRISM_MODEL              计算S1-S4参数，不写则不计算
        // CALIB_FIX_S1_S2_S3_S4               固定S1-S4参数值
        // CALIB_TILTED_MODEL                  计算(tauX, tauY)参数，不写则不计算
        // CALIB_FIX_TAUX_TAUY                 固定(tauX, tauY)参数值
        
        // 左目
        double reprojectionErrorL = calibrateCamera(objectPoints, allCornersL, calibImgSize,
                                                    cameraMatrixL, distCoeffsL, rVecsL, tVecsL,
                                                    stdDevIntrinsicsL, stdDevExtrinsicsL, perViewErrorsL,
                                                    CV_CALIB_USE_INTRINSIC_GUESS +
                                                    CV_CALIB_FIX_PRINCIPAL_POINT);
        cout << "左目";
        printCalibResults(cameraMatrixL, distCoeffsL, reprojectionErrorL, stdDevIntrinsicsL);
        cout << "------------------------------------------" << endl << endl;

        // 右目
        double reprojectionErrorR = calibrateCamera(objectPoints, allCornersR, calibImgSize,
                                                    cameraMatrixR, distCoeffsR, rVecsR, tVecsR,
                                                    stdDevIntrinsicsR, stdDevExtrinsicsR, perViewErrorsR,
                                                    CV_CALIB_USE_INTRINSIC_GUESS + 
                                                    CV_CALIB_FIX_PRINCIPAL_POINT);
        cout << "右目";
        printCalibResults(cameraMatrixR, distCoeffsR, reprojectionErrorR, stdDevIntrinsicsR);
        cout << "------------------------------------------" << endl << endl;
        
        // 输出标定结果到文件
        printCalibResults(cameraMatrixL, distCoeffsL, reprojectionErrorL, stdDevIntrinsicsL, stdDevExtrinsicsL, perViewErrorsL, foutL);
        printCalibResults(cameraMatrixR, distCoeffsR, reprojectionErrorR, stdDevIntrinsicsR, stdDevExtrinsicsR, perViewErrorsR, foutR);
    } else {
        // 手动输入内参数
        cout << "输入内参数：" << endl;
        cout << "左目焦距(fx, fy)： ";
        cin >> cameraMatrixL.at<double>(0, 0) >> cameraMatrixL.at<double>(1, 1);
        cout << "左目主点坐标(Cx, Cy)： ";
        cin >> cameraMatrixL.at<double>(0, 2) >> cameraMatrixL.at<double>(1, 2);
        cout << endl << endl;
        cout << "右目焦距(fx, fy)： ";
        cin >> cameraMatrixR.at<double>(0, 0) >> cameraMatrixR.at<double>(1, 1);
        cout << "右目主点坐标(Cx, Cy)： ";
        cin >> cameraMatrixR.at<double>(0, 2) >> cameraMatrixR.at<double>(1, 2);
        cout << endl << "------------------------------------------" << endl << endl;

        // 指定内参数
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
        // 立体标定
        // flags:
        // CV_CALIB_FIX_INTRINSIC              固定内参数和畸变模型，只计算(R, T, E, F)
        // CV_CALIB_USE_INTRINSIC_GUESS        使用预估的内参数矩阵
        // CV_CALIB_FIX_PRINCIPAL_POINT        固定主点坐标
        // CV_CALIB_FIX_FOCAL_LENGTH           固定焦距
        // CV_CALIB_FIX_ASPECT_RATIO           固定焦距比，只计算fy
        // CV_CALIB_SAME_FOCAL_LENGTH          固定x, y方向焦距比为1
        // CV_CALIB_ZERO_TANGENT_DIST          不计算切向畸变(p1, p2)
        // CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 固定径向畸变K1-K6参数
        // CV_CALIB_RATIONAL_MODEL             计算8参数畸变模型(K4-K6)，不写则计算5参数
        // CALIB_THIN_PRISM_MODEL              计算S1-S4参数，不写则不计算
        // CALIB_FIX_S1_S2_S3_S4               固定S1-S4参数值
        // CALIB_TILTED_MODEL                  计算(tauX, tauY)参数，不写则不计算
        // CALIB_FIX_TAUX_TAUY                 固定(tauX, tauY)参数值
        double reprojErrorStereo = stereoCalibrate(objectPoints, allCornersL, allCornersR,
                                                   cameraMatrixL, distCoeffsL,
                                                   cameraMatrixR, distCoeffsR,
                                                   calibImgSize, R, T, E, F,
                                                   CALIB_FIX_INTRINSIC);

        //CALIB_USE_INTRINSIC_GUESS +
        //CALIB_FIX_PRINCIPAL_POINT


        // Rodrigues变换（输出逆时针旋转的角度）
        Mat rod;
        Rodrigues(R, rod);
        rod.at<double>(0, 0) = rod.at<double>(0, 0) / CV_PI * 180;
        rod.at<double>(1, 0) = rod.at<double>(1, 0) / CV_PI * 180;
        rod.at<double>(2, 0) = rod.at<double>(2, 0) / CV_PI * 180;

        // 输出双目标定结果
        cout << "双目标定结果：" << endl;
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
        // 手动输入外参数
        cout << "输入外参数：" << endl;
        cout << "R = " << endl;
        cin >> R.at<double>(0, 0) >> R.at<double>(0, 1) >> R.at<double>(0, 2);
        cin >> R.at<double>(1, 0) >> R.at<double>(1, 1) >> R.at<double>(1, 2);
        cin >> R.at<double>(2, 0) >> R.at<double>(2, 1) >> R.at<double>(2, 2);
        cout << endl << "t = " << endl;
        cin >> T.at<double>(0, 0) >> T.at<double>(1, 0) >> T.at<double>(2, 0);
        // 输出Rodrigues变换结果
        Mat rod;
        Rodrigues(R, rod);
        rod.at<double>(0, 0) = rod.at<double>(0, 0) / CV_PI * 180;
        rod.at<double>(1, 0) = rod.at<double>(1, 0) / CV_PI * 180;
        rod.at<double>(2, 0) = rod.at<double>(2, 0) / CV_PI * 180;
        cout << endl << "Rodrigues = " << endl << rod << endl;
    }


    // 用于测距的图像的大小
    Size trainImgSize, testImgSize;
    // 用于测距的图像数
    int trainImgCount, testImgCount;
    // 测距图像上的特征点位置
    vector<Point2f> objectPointsL, objectPointsR;
    // 测距图像上的特征点像素值
    vector<Vec3b> objectColorsL, objectColorsR;
    // 三角化重建出的世界坐标系坐标
    Mat structure, structure3D;
    // 测距点中位线距离
    vector<double> dist;

    if(doTrain) {
        // 训练
        // 读取左目训练图像
        while(getline(fTrainL, fileName)) {
            Mat img = imread(fileName);
            trainSetL.push_back(img);
        }
        trainImgSize = trainSetL[0].size();
        trainImgCount = trainSetL.size();
        // 读取右目训练图像
        while(getline(fTrainR, fileName)) {
            Mat img = imread(fileName);
            trainSetR.push_back(img);
        }

        // 用于保存所有图像的测距值
        vector<float> trainVal(trainImgCount);
        // 训练图像中的目标真实距离
        vector<float> groundTruth(trainImgCount);

        for(int i = 0; i < trainImgCount; i++) {
            //// 中值滤波
            //medianBlur(trainSetL[i], tempEnhanceL, 3);
            //medianBlur(trainSetR[i], tempEnhanceR, 3);
            //trainSetL[i] = tempEnhanceL.clone();
            //trainSetR[i] = tempEnhanceR.clone();

            // 畸变矫正
            if(doDistortionCorrect) {
                undistort(trainSetL[i].clone(), trainSetL[i], cameraMatrixL, distCoeffsL);
                undistort(trainSetR[i].clone(), trainSetR[i], cameraMatrixR, distCoeffsR);
            }

            if(doEnhance) {
                // CLAHE增强
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

                cout << "增强图像" << endl << endl;
                waitKey();
            }

            imshow("train_leftcam", trainSetL[i]);
            imshow("train_rightcam", trainSetR[i]);

            // 逐图像选取ROI
            setMouseCallback("train_leftcam", onMouseL_ROI_Train, (void *)&i);
            setMouseCallback("train_rightcam", onMouseR_ROI_Train, (void *)&i);
            cout << "在第" << i + 1 << "组图像中各选择一个ROI进行特征匹配：" << endl;
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

            switch(type) {
            case SIFT:
                // 局部特征匹配
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
                cout << "没有找到匹配点，手动选择一组" << endl;
                imshow("train_leftcam", trainSetL[i]);
                imshow("train_rightcam", trainSetR[i]);
                // 手动选择一组点
                targetL = Point(0, 0);
                targetR = Point(0, 0);
                setMouseCallback("train_leftcam", onMouseL_Train, (void *)&i);
                setMouseCallback("train_rightcam", onMouseR_Train, (void *)&i);
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

            // 重建坐标
            reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                        objectPointsL, objectPointsR, structure);
            toPoints3D(structure, structure3D);
            // 中位线距离
            dist = ranging(structure3D, R, T);

            // 输出距离
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
            cout << "目标距离(用于训练) " << range << " m" << endl << endl;
            trainVal[i] = range;
            waitKey();

            // 记录ground truth
            cout << "输入测距点实际距离： ";
            cin >> groundTruth[i];
            cout << endl;

            destroyAllWindows();
        }

        cout << "开始训练补偿模型……" << endl;
        
        vector<Point2f> trainPoints;
        for(int i = 0; i < trainVal.size(); i++) {
            trainPoints.push_back(Point2f(trainVal[i], groundTruth[i]));
        }
        coef = polyfit2(trainPoints, 4);

        //cout << "补偿模型：" << endl;
        //cout << "f(x) = " << coef.at<float>(0) << "+" << coef.at<float>(1) << "x+" << coef.at<float>(2) << "x^2+" << coef.at<float>(3) << "x^3+" << coef.at<float>(4) << "x^4" << endl;
        cout << "完成" << endl;

        //// MEX code
        //if(!createFitInitialize()) {
        //    cout << "拟合初始化失败" << endl;
        //    return -1;
        //}
        //// mwArray
        //mwArray mwFitResult(4, 1, mxDOUBLE_CLASS);
        //mwArray mwGoF(5, 1, mxDOUBLE_CLASS);
        //mwArray mwTrain(trainImgCount, 1, mxDOUBLE_CLASS);
        //mwArray mwTest(trainImgCount, 1, mxDOUBLE_CLASS);
        //mwArray mwFlag(1, 1, mxLOGICAL_CLASS);
        //mxLogical mxFlag = false;
        //// Set data
        //mwTrain.SetData(&trainVal[0], trainImgCount);
        //mwTest.SetData(&groundTruth[0], trainImgCount);
        //mwFlag.SetLogicalData(&mxFlag, 1);
        //// Fit
        //createFit(2, mwFitResult, mwGoF, mwTrain, mwTest, mwFlag);
        //// Get result
        //double fitResult[4];
        //double gof[5];
        //mwFitResult.GetData(fitResult, 4);
        //mwGoF.GetData(gof, 5);
        //createFitTerminate();
        //cout << "补偿模型：" << endl;
        //cout << "f(x) = " << fitResult[0] << "*exp(" << fitResult[1] << "*x) + " << fitResult[2] << "*exp(" << fitResult[3] << "*x)" << endl;
    }


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
            
            cout << "增强图像" << endl << endl;
            waitKey();
        }

        imshow("Ranging_leftcam", testSetL[i]);
        imshow("Ranging_rightcam", testSetR[i]);
        
        // 逐图像选取ROI
        setMouseCallback("Ranging_leftcam", onMouseL_ROI, (void *)&i);
        setMouseCallback("Ranging_rightcam", onMouseR_ROI, (void *)&i);
        cout << "在第" << i + 1 << "组图像中各选择一个ROI进行特征匹配：" << endl;
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

        switch(type) {
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
            if(doTrain) {
                // 二阶指数
                //range = compensate(37.2, 0.01229, -38.83, -0.01031, range);
                // 多项式
                range = compensatePoly(coef, range);
                cout << "目标距离(补偿) " << range << " m" << endl;
            } else {
                cout << "目标距离(未补偿) " << range << " m" << endl;
            }
            // foutTest << num2str(i + 1) + ": " << range << " m" << endl;
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

void onMouseL_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetL[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetL = Point(x, y);
        circle(frame, targetL, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetL, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("train_leftcam", frame);
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

void onMouseL_ROI_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetL[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetL = Point(x, y);
        getROI(frame, targetL, roiSize, roiL, roiImgL);
        rectangle(frame, roiL, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("train_leftcam", frame);
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

void onMouseR_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetR[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetR = Point(x, y);
        circle(frame, targetR, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetR, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("train_rightcam", frame);
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

void onMouseR_ROI_Train(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = trainSetR[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetR = Point(x, y);
        getROI(frame, targetR, roiSize, roiR, roiImgR);
        rectangle(frame, roiR, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("train_rightcam", frame);
    }
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

void onEnhanceTrackbarL(int pos, void *userdata) {
    int *i = (int *)userdata;
    if(clipL && gridXL && gridYL) {
        double clipVal = double(clipL) / 10;
        procCLAHE(testSetL[*i], tempEnhanceL, clipVal, Size(gridXL, gridYL));
        imshow("enhance_leftcam", tempEnhanceL);
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

void onEnhanceTrackbarR(int pos, void *userdata) {
    int *i = (int *)userdata;
    if(clipR && gridXR && gridYR) {
        double clipVal = double(clipR) / 10;
        procCLAHE(testSetR[*i], tempEnhanceR, clipVal, Size(gridXR, gridYR));
        imshow("enhance_rightcam", tempEnhanceR);
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

string num2str(int num) {
    ostringstream s1;
    s1 << num;
    string outStr = s1.str();
    return(outStr);
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
