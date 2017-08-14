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

// 显示角点提取结果
bool showCornerExt = true;
// 进行单目标定（true通过单目标定确定内参，false输入内参）
bool doSingleCalib = true;
// 测距方式（true通过双目标定确定外参，false通过特征提取获得匹配点计算位姿）
bool doStereoCalib = true;
// 手动选点（用于双目标定方式，true手动选点，false选择ROI中心进行局部特征提取）
bool manualPoints = true;
// 基线距离（用于特征提取方式，规定以mm为单位）
double baselineDist = 1420;
// ROI大小（横向半径，纵向半径）
Size roiSize = Size(30, 40);

// 读入的测距图像序列
vector<Mat> rangingSetL, rangingSetR;
// 用于记录测距图像中的同名点
Point2f targetL, targetR;
// 用于记录测距图像中的ROI
Rect roiL, roiR;
// 用于特征匹配的ROI图像
Mat roiImgL, roiImgR;
// 使用的标定板棋盘格内角点的个数
Size boardSize;
// 标定板上每个方格的大小
float squareSize;

int main() {
    // 定义所使用的标定板
    usingBoard(3, boardSize, squareSize);
    // 加载标定所用图像文件的路径
    ifstream finL("20170810_calib1_L.txt");
    ifstream finR("20170810_calib1_R.txt");
    // 加载测距所用的图像文件路径
    ifstream finRangingL("20170810_ranging1_L.txt");
    ifstream finRangingR("20170810_ranging1_R.txt");
    // 保存标定结果的文件
    ofstream foutL("20170810_calib_result1_L.txt");
    ofstream foutR("20170810_calib_result1_R.txt");
    ofstream foutStereo("20170810_stereo_result1.txt");

    // 每行读入的图像路径
    string fileName;
    // 读入后的图像序列
    vector<Mat> imageSetL, imageSetR;
    // 每幅图像的大小
    Size imageSize;
    // 输入图像数
    int imgCount;
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

    // 用于测距的图像的大小
    Size rangingImgSize;
    // 用于测距的图像数
    int rangingImgCount;
    // 测距图像上的特征点位置
    vector<Point2f> objectPointsL, objectPointsR;
    // 测距图像上的特征点像素值
    vector<Vec3b> objectColorsL, objectColorsR;

    // 读取左目图像
    while(getline(finL, fileName)) {
        Mat img = imread(fileName);
        imageSetL.push_back(img);
    }
    imageSize = imageSetL[0].size();
    imgCount = imageSetL.size();
    // 读取右目图像
    while(getline(finR, fileName)) {
        Mat img = imread(fileName);
        imageSetR.push_back(img);
    }
    if(imgCount != imageSetR.size()) {
        cout << "图像对数目不一致，请检查" << endl << endl;
        system("pause");
        return 0;
    }

    // 逐图像提取角点
    for(int i = 0; i < imgCount; i++) {
        bool foundAllCornersL = false, foundAllCornersR = false;
        vector<Point2f> cornerBufL, cornerBufR;
        Mat viewL, viewR, grayL, grayR;
        viewL = imageSetL[i].clone();
        viewR = imageSetR[i].clone();
        cvtColor(imageSetL[i], grayL, CV_RGB2GRAY);
        cvtColor(imageSetR[i], grayR, CV_RGB2GRAY);

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
    cameraMatrixL = initCameraMatrix2D(objectPoints, allCornersL, imageSize);
    cameraMatrixR = initCameraMatrix2D(objectPoints, allCornersR, imageSize);
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
        double reprojectionErrorL = calibrateCamera(objectPoints, allCornersL, imageSize,
                                                    cameraMatrixL, distCoeffsL, rVecsL, tVecsL,
                                                    stdDevIntrinsicsL, stdDevExtrinsicsL, perViewErrorsL,
                                                    CV_CALIB_USE_INTRINSIC_GUESS +
                                                    CV_CALIB_FIX_PRINCIPAL_POINT);
        cout << "左目";
        printCalibResults(cameraMatrixL, distCoeffsL, reprojectionErrorL, stdDevIntrinsicsL);
        cout << "------------------------------------------" << endl << endl;

        // 右目
        double reprojectionErrorR = calibrateCamera(objectPoints, allCornersR, imageSize,
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
        string useInit;
        cout << "使用预估值？ [y]/[other]   ";
        cin >> useInit;
        if(useInit.compare("y") != 0) {
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
        }
    }

    // 理想主点坐标
    Point ppIdeal = Point(imageSize.width / 2, imageSize.height / 2);
    // 匹配点对flag, 重建出的世界坐标系坐标（齐次）
    Mat mask, structure, structure3D;
    // 测距点中位线距离
    vector<double> dist;

    // 读取左目测距图像
    while(getline(finRangingL, fileName)) {
        Mat img = imread(fileName);
        rangingSetL.push_back(img);
    }
    rangingImgSize = rangingSetL[0].size();
    rangingImgCount = rangingSetL.size();
    // 读取右目测距图像
    while(getline(finRangingR, fileName)) {
        Mat img = imread(fileName);
        rangingSetR.push_back(img);
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
                                                   imageSize, R, T, E, F,
                                                   CALIB_USE_INTRINSIC_GUESS + 
                                                   CALIB_FIX_PRINCIPAL_POINT);
        // 输出双目标定结果
        cout << "双目标定结果：" << endl;
        cout << "R = " << endl << R << endl;
        cout << "t = " << endl << T << endl;
        cout << "E = " << endl << E << endl;
        cout << "F = " << endl << F << endl;
        cout << "reprojection error = " << endl << reprojErrorStereo << endl;
        cout << endl << "------------------------------------------" << endl << endl;
        foutStereo << "R = " << endl << R << endl;
        foutStereo << "t = " << endl << T << endl;
        foutStereo << "E = " << endl << E << endl;
        foutStereo << "F = " << endl << F << endl;
        foutStereo << "reprojection error = " << endl << reprojErrorStereo << endl;
        namedWindow("Ranging_leftcam");
        namedWindow("Ranging_rightcam");

        for(int i = 0; i < rangingImgCount; i++) {
            imshow("Ranging_leftcam", rangingSetL[i]);
            imshow("Ranging_rightcam", rangingSetR[i]);
            if(manualPoints) {
                // 逐图像选取同名点
                targetL = Point(0, 0);
                targetR = Point(0, 0);
                setMouseCallback("Ranging_leftcam", onMouseL, (void *)&i);
                setMouseCallback("Ranging_rightcam", onMouseR, (void *)&i);
                cout << "在第" << i + 1 << "组图像中各选择一个匹配点：" << endl;
                waitKey();
                cout << "目标点：L(" << targetL.x << ", " << targetL.y << ")   ";
                cout << "R(" << targetR.x << ", " << targetR.y << ")" << endl;
                cout << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;
                // 重建坐标
                objectPointsL.clear();
                objectPointsR.clear();
                objectPointsL.push_back(targetL);
                objectPointsR.push_back(targetR);
                reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                            objectPointsL, objectPointsR, structure);
                toPoints3D(structure, structure3D);
                // 中位线距离
                dist = ranging(structure3D, R, T);
                cout << "测距点深度 " << dist[0] << " m" << endl << endl;
                cout << "------------------------------------------" << endl << endl;
            } else {
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
                // 局部特征匹配
                vector<vector<KeyPoint>> keyPoints4All;
                vector<Mat> descriptor4All;
                vector<vector<Vec3b>> colors4All;
                vector<DMatch> matches;
                vector<Mat> rois = {roiImgL, roiImgR};
                extractFeatures(rois, keyPoints4All, descriptor4All, colors4All);
                matchFeatures(descriptor4All[0], descriptor4All[1], matches);
                getMatchedPoints(keyPoints4All[0], keyPoints4All[1], matches, objectPointsL, objectPointsR);
                getMatchedColors(colors4All[0], colors4All[1], matches, objectColorsL, objectColorsR);
                // 将ROI坐标恢复到原图像中
                if(objectPointsL.size() == 0) {
                    cout << "没有找到匹配点" << endl;
                    continue;
                }
                for(int i = 0; i < objectPointsL.size(); i++) {
                    objectPointsL[i] += Point2f(roiL.x, roiL.y);
                    objectPointsR[i] += Point2f(roiR.x, roiR.y);
                }
                // 重建坐标
                reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                            objectPointsL, objectPointsR, structure);
                toPoints3D(structure, structure3D);
                // 中位线距离
                dist = ranging(structure3D, R, T);
                for(auto iter = dist.cbegin(); iter < dist.cend(); ++iter) {
                    cout << "测距点深度 " << *iter << " m" << endl;
                }
                cout << endl;
            }
        }
    } else {
        // 基于重建的方式
        for(int i = 0; i < rangingImgCount; i++) {
            vector<vector<KeyPoint>> keyPoints4All;
            vector<Mat> descriptor4All;
            vector<vector<Vec3b>> colors4All;
            vector<DMatch> matches;
            vector<Mat> rois = { rangingSetL[i], rangingSetR[i] };
            Mat matchImg;

            cout << "第" << i + 1 << "组图像：" << endl;

            // 全图特征匹配
            extractFeatures(rois, keyPoints4All, descriptor4All, colors4All);
            matchFeatures(descriptor4All[0], descriptor4All[1], matches);
            // 绘制匹配图
            if(i != 0) {
                destroyWindow("matches" + num2str(i));
            }
            drawMatches(rangingSetL[i], keyPoints4All[0],
                        rangingSetR[i], keyPoints4All[1],
                        matches, matchImg, Scalar::all(-1), Scalar(97, 98, 255));
            imshow("matches" + num2str(i + 1), matchImg);
            waitKey();
            drawMatches(rangingSetL[i], keyPoints4All[0],
                        rangingSetR[i], keyPoints4All[1],
                        matches, matchImg, Scalar::all(-1), Scalar::all(-1), 
                        std::vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            imshow("matches" + num2str(i + 1), matchImg);
            waitKey();

            getMatchedPoints(keyPoints4All[0], keyPoints4All[1], matches, objectPointsL, objectPointsR);
            getMatchedColors(colors4All[0], colors4All[1], matches, objectColorsL, objectColorsR);

            // 计算E，分解出R、t
            bool foundE = findTransform(cameraMatrixL, cameraMatrixR, R, T,
                                        objectPointsL, objectPointsR, mask);
            if(foundE) {
                cout << "R = " << endl << R << endl << endl;
                cout << "t = " << endl << T << endl << endl;
                // 去除不匹配点对
                maskoutPoints(objectPointsL, mask);
                maskoutPoints(objectPointsR, mask);
                // 重建坐标
                T *= baselineDist;
                reconstruct(cameraMatrixL, cameraMatrixR, R, T,
                            objectPointsL, objectPointsR, structure);
                toPoints3D(structure, structure3D);
                // 中位线距离
                dist = ranging(structure3D, R, T);
                for(auto iter = dist.cbegin(); iter < dist.cend(); ++iter) {
                    cout << "测距点深度 " << *iter << " m" << endl;
                }
                cout << endl;
            } else {
                cout << "本征矩阵求解失败" << endl << endl;
            }
        }
    }

    system("pause");
    return 0;
}


/**
 * 同名点选取鼠标回调事件（左）
 * @param event 鼠标操作事件类型
 *          enum cv::MouseEventTypes
 *          EVENT_MOUSEMOVE     滑动
 *          EVENT_LBUTTONDOWN   左键按下
 *          EVENT_RBUTTONDOWN   右键按下
 *          EVENT_MBUTTONDOWN   中键按下
 *          EVENT_LBUTTONUP     左键释放
 *          EVENT_RBUTTONUP     右键释放
 *          EVENT_MBUTTONUP     中键释放
 *          EVENT_LBUTTONDBLCLK 左键双击
 *          EVENT_RBUTTONDBLCLK 右键双击
 *          EVENT_MBUTTONDBLCLK 中键双击
 *          EVENT_MOUSEWHEEL    滚轮上下滑动
 *          EVENT_MOUSEHWHEEL   滚轮左右滑动
 * @param x     鼠标位于窗口的x坐标位置（窗口左上角默认为原点，向右为x轴，向下为y轴）
 * @param y     鼠标位于窗口的y坐标位置
 * @param flags 鼠标拖拽及键鼠联合事件标志位
 *          enum cv::MouseEventFlags
 *          EVENT_FLAG_LBUTTON  左键拖拽
 *          EVENT_FLAG_RBUTTON  右键拖拽
 *          EVENT_FLAG_MBUTTON  中键拖拽
 *          EVENT_FLAG_CTRLKEY  Ctrl键按下
 *          EVENT_FLAG_SHIFTKEY Shift键按下
 *          EVENT_FLAG_ALTKEY   Alt键按下
 * @param param 自定义数据
 */
void onMouseL(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = rangingSetL[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetL = Point(x, y);
        circle(frame, targetL, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetL, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("Ranging_leftcam", frame);
    }
}


/**
 * onMouseL 局部特征提取ROI选取事件（左）
 * @param event 鼠标操作事件类型
 * @param x     鼠标位于窗口的x坐标位置
 * @param y     鼠标位于窗口的y坐标位置
 * @param flags 鼠标拖拽及键鼠联合事件标志位
 * @param param 自定义数据
 */
void onMouseL_ROI(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = rangingSetL[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetL = Point(x, y);
        getROI(frame, targetL, roiSize, roiL, roiImgL);
        rectangle(frame, roiL, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("Ranging_leftcam", frame);
    }
}


/**
 * 同名点选取鼠标回调事件（右）
 * @param event 鼠标操作事件类型
 * @param x     鼠标位于窗口的x坐标位置
 * @param y     鼠标位于窗口的y坐标位置
 * @param flags 鼠标拖拽及键鼠联合事件标志位
 * @param param 自定义数据
 */
void onMouseR(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = rangingSetR[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetR = Point(x, y);
        circle(frame, targetR, 2, Scalar(97, 98, 255), CV_FILLED, LINE_AA, 0);
        circle(frame, targetR, 20, Scalar(75, 83, 171), 2, LINE_AA, 0);
        imshow("Ranging_rightcam", frame);
    }
}


/**
 * onMouseL 局部特征提取ROI选取事件（右）
 * @param event 鼠标操作事件类型
 * @param x     鼠标位于窗口的x坐标位置
 * @param y     鼠标位于窗口的y坐标位置
 * @param flags 鼠标拖拽及键鼠联合事件标志位
 * @param param 自定义数据
 */
void onMouseR_ROI(int event, int x, int y, int flags, void *param) {
    if(event == EVENT_LBUTTONUP) {
        int *i = (int *)param;
        Mat frame = rangingSetR[*i].clone();
        // 记录当前位置的坐标，画一个点
        targetR = Point(x, y);
        getROI(frame, targetR, roiSize, roiR, roiImgR);
        rectangle(frame, roiR, Scalar(97, 98, 255), 1, LINE_AA, 0);
        imshow("Ranging_rightcam", frame);
    }
}


/**
 * getROI 根据选取的点在输入图像上生成ROI区域，并返回ROI图像
 * @param img     [input]输入图像
 * @param center  [input]输入的ROI中心点
 * @param roiSize [input]ROI的大小（半径）
 * @param roi     [output]ROI区域位置
 * @param roiImg  [output]ROI图像
 */
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
    roiImg = img(roi);
}


/**
 * 数字转字符串
 * @param num     输入数字
 * @return outStr 返回对应的字符串
 */
string num2str(int num) {
    ostringstream s1;
    s1 << num;
    string outStr = s1.str();
    return(outStr);
}


/**
 * 输出标定结果到控制台。
 * 输出内参数矩阵、畸变向量、总重投影误差；
 * 输出焦距及误差、主点坐标及误差、畸变向量及误差；
 * 输出畸变向量其他值的误差、外参数误差、每幅图像的重投影误差。
 * @param cameraMatrix
 * @param distCoeffs
 * @param reprojectionError
 * @param stdDevIntrinsics
 * @param stdDevExtrinsics
 * @param perViewErrors
 */
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


/**
 * 输出标定结果到文件。
 * 输出内参数矩阵、畸变向量、总重投影误差；
 * 输出焦距及误差、主点坐标及误差、畸变向量及误差；
 * 输出畸变向量其他值的误差、外参数误差、每幅图像的重投影误差。
 * @param cameraMatrix
 * @param distCoeffs
 * @param reprojectionError
 * @param stdDevIntrinsics
 * @param stdDevExtrinsics
 * @param perViewErrors
 * @param fout
 */
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


/**
 * 输出标定结果到控制台。
 * 输出内参数矩阵、畸变向量、总重投影误差；
 * 输出焦距及误差、主点坐标及误差、畸变向量及误差；
 * @param cameraMatrix
 * @param distCoeffs
 * @param reprojectionError
 * @param stdDevIntrinsics
 */
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


/**
 * 输出标定结果到文件。
 * 输出内参数矩阵、畸变向量、总重投影误差；
 * 输出焦距、主点坐标、畸变向量；
 * @param cameraMatrix
 * @param distCoeffs
 * @param reprojectionError
 */
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


