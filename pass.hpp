#define _CRT_SECURE_NO_WARNINGS

//#define JIAOZHENG     //控制用不用矫正
#define NOMINMAX
#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "io.h"
#include<Windows.h>
#include <stdio.h>
#include <stack>
#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/surface/mls.h>
#include <pcl/common/transforms.h>
#include <chrono>
using namespace cv;
using namespace std;
void log(std::string  s,int line) {
    std::cout << s <<"发生在"<< line<< endl;
}

//*****************************************************布尔量**********************************************************//
bool phase_jiaozheng = false;


//*****************************************************行模板相位图**********************************************************//
//π
float pi = 3.14159265358979323846;

//列模板
Mat template_lie;
//行模板
Mat template_hang;

//物体投影图片
vector<vector<Mat>> object_phase_images;
//列的连续相位
extern vector<Mat>  object_true_phase_images_lie;

//相位图的频率
float f[3] = { 70,64,59 };

// 摄像机单映射矩阵 
Mat camera_parameter = Mat::zeros(3, 4, CV_32FC1);//创建Mat类矩阵，定义初始化值全部是0，矩阵大小和txt一致
//投影仪单映射矩阵 
Mat projector_parameter = Mat::zeros(3, 4, CV_32FC1);
// 摄像机畸变矩阵 
Mat camera_distortion = Mat::zeros(1, 5, CV_32FC1);
// 投影仪畸变矩阵 
Mat projector_distortion = Mat::zeros(1, 5, CV_32FC1);
// 摄像机内参矩阵 
Mat camera_matix = Mat::zeros(3, 3, CV_32FC1);
// 投影仪内参矩阵
Mat projector_matix = Mat::zeros(3, 3, CV_32FC1);

//.pcd文件变量名
char pcd_path[100];
//比较字符串用于判断
char compare_path[100];
//标志圆定义的旋转平移矩阵
float angle = 0;	float X = 0;	float Y = 0;	float Z = 0;
//标志圆投影图片
vector<Mat> location_circle_images;//圆对应的点位

//函数声明
extern void matrix_circle(Mat& srcImage, Mat& srcImage2, float& angle, float& X, float& Y, float& Z, int number);
void true_phase(Mat& org_phase, Mat a[4], int m_rows, int m_cols);
void difference_frequency(Mat& high_frequency, Mat& low_frequency, Mat& output, int m_rows, int m_cols);
void Phase_unwrapping(Mat& synthesis_phase_before, Mat& synthesis_phase_after, Mat& continuous_phase, float f_before, float f_after, int m_rows, int m_cols);
Mat continus_phase_hang(Mat object_hang[3][4]);
int little_vaule(Mat imge, float vaule, int flag);
Mat phase(Mat b[6][4]);

//*************************************线程********************************************//
//物体投影图真实相位线程
HANDLE object_phaseCapThread = NULL;
DWORD object_phaseCapThreadID;
static void WINAPI object_phaseCapThreadRun();

void WINAPI object_phaseCapThreadRun()
{
    //*****************************************************行模板相位图**********************************************************//
    char path1[100];
    vector<Mat> translate_picture;
    for (int j = 12; j < 24; j++)
    {
        sprintf(path1, "..//picture//模板图//%d.bmp", j + 1);
        Mat imageInput = imread(path1, 2);
        imageInput.convertTo(imageInput, CV_32FC1);
        translate_picture.push_back(imageInput);
    }
    Mat translate_hang[3][4];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            translate_hang[i][j] = translate_picture[i * 4 + j];
        }
    }
    template_hang = continus_phase_hang(translate_hang);
    imwrite("理想相位.bmp", template_hang);
    //*****************************************************物体图相位图**********************************************************//
    for (int t = 0; t < object_phase_images.size(); t++)
    {
        Mat object[6][4];
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                object[(i + 3)%6][j] = object_phase_images[t][i * 4 + j];
            }
        }
        Mat continus_fai1 = phase(object);
        object_true_phase_images_lie.push_back(continus_fai1);//列真实相位
    }

    return;
}
//该函数目的是还原某一个点的包裹相位
void true_phase(Mat& org_phase, Mat a[4], int m_rows, int m_cols)	//计算截断相位
{
    for (int k = 0; k < m_rows; k++)
    {
        float* inData = org_phase.ptr<float>(k);
        const float* a0_data = a[0].ptr<float>(k);
        const float* a1_data = a[1].ptr<float>(k);
        const float* a2_data = a[2].ptr<float>(k);
        const float* a3_data = a[3].ptr<float>(k);

        for (int q = 0; q < m_cols; q++)
        {
            //上面是opencv反正切，下面是c++反正切，结果不同
            inData[q] = atan2((a3_data[q] - a1_data[q]), (a0_data[q] - a2_data[q]));
        }
    }
}
//该函数目的是将两个频率的相位合成
void difference_frequency(Mat& high_frequency, Mat& low_frequency, Mat& output, int m_rows, int m_cols)//相位差频
{
    for (int k = 0; k < m_rows; k++)
    {
        const float* high_frequency_data = high_frequency.ptr<float>(k);
        const float* low_frequency_data = low_frequency.ptr<float>(k);
        float* output_data = output.ptr<float>(k);

        for (int q = 0; q < m_cols; q++)
        {
            if (high_frequency_data[q] >= low_frequency_data[q])
            {
                output_data[q] = high_frequency_data[q] - low_frequency_data[q];
            }
            else
            {
                output_data[q] = high_frequency_data[q] + 2.0 * pi - low_frequency_data[q];
            }
        }

    }
}

void ConnectedCountBySeedFill11(const cv::Mat& _binImg, cv::Mat& _lableImg, int &iConnectedAreaCount)
{
    //拓宽1个像素的原因是：如果连通域在边缘，运行此函数会异常崩溃，所以需要在周围加一圈0值，确保连通域不在边上
    //==========图像周围拓宽1个像素============================================
    int top, bottom;      //【添加边界后的图像尺寸】
    int leftImage, rightImage;
    int borderType = BORDER_CONSTANT; //BORDER_REPLICATE
    //【初始化参数】
    top = (int)(1); bottom = (int)(1);
    leftImage = (int)(1); rightImage = (int)(1);
    Mat _binImg2, _binImg3;
    _binImg.copyTo(_binImg2);
    //初始化参数value
    Scalar value(0); //填充值
    //创建图像边界
    copyMakeBorder(_binImg2, _binImg3, top, bottom, leftImage, rightImage, borderType, value);

    //==========图像周围拓宽1个像素============================================

    // connected component analysis (4-component)
    // use seed filling algorithm
    // 1. begin with a foreground pixel and push its foreground neighbors into a stack;
    // 2. pop the top pixel on the stack and label it with the same label until the stack is empty
    //
    // foreground pixel: _binImg(x,y) = 1
    // background pixel: _binImg(x,y) = 0

    if (_binImg3.empty() ||
        _binImg3.type() != CV_8UC1)
    {
        return;
    }

    _lableImg.release();
    _binImg3.convertTo(_lableImg, CV_32SC1);
    int icount = 0;
    int label = 1; // start by 2

    int rows = _binImg3.rows - 1;
    int cols = _binImg3.cols - 1;
    for (int i = 1; i < rows - 1; i++)
    {
        int* data = _lableImg.ptr<int>(i);  //取一行数据
        for (int j = 1; j < cols - 1; j++)
        {
            if (data[j] == 1)  //像素不为0
            {
                stack<std::pair<int, int>> neighborPixels;   //新建一个栈
                neighborPixels.push(std::pair<int, int>(i, j));   // 像素坐标: <i,j> ，以该像素为起点，寻找连通域
                ++label; // 开始一个新标签，各连通域区别的标志
                while (!neighborPixels.empty())
                {
                    // 获取堆栈中的顶部像素并使用相同的标签对其进行标记
                    std::pair<int, int> curPixel = neighborPixels.top();
                    int curX = curPixel.first;
                    int curY = curPixel.second;
                    _lableImg.at<int>(curX, curY) = label; //对图像对应位置的点进行标记

                    // 弹出顶部像素  （顶部像素出栈）
                    neighborPixels.pop();

                    // 加入8邻域点
                    if (_lableImg.at<int>(curX, curY - 1) == 1)
                    {// 左点
                        neighborPixels.push(std::pair<int, int>(curX, curY - 1)); //左边点入栈
                    }

                    if (_lableImg.at<int>(curX, curY + 1) == 1)
                    {// 右点
                        neighborPixels.push(std::pair<int, int>(curX, curY + 1)); //右边点入栈
                    }

                    if (_lableImg.at<int>(curX - 1, curY) == 1)
                    {// 上点
                        neighborPixels.push(std::pair<int, int>(curX - 1, curY)); //上边点入栈
                    }

                    if (_lableImg.at<int>(curX + 1, curY) == 1)
                    {// 下点
                        neighborPixels.push(std::pair<int, int>(curX + 1, curY)); //下边点入栈
                    }
                    ////===============补充到8连通域======================================================
                    //if (_lableImg.at<int>(curX - 1, curY - 1) ==1)
                    //{// 左上点
                    //	neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1)); //左上点入栈
                    //}

                    //if (_lableImg.at<int>(curX - 1, curY + 1) ==1)
                    //{// 右上点
                    //	neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1)); //右上点入栈
                    //}

                    //if (_lableImg.at<int>(curX + 1, curY - 1) ==1)
                    //{// 左下点
                    //	neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1)); //左下点入栈
                    //}

                    //if (_lableImg.at<int>(curX + 1, curY + 1) ==1)
                    //{// 右下点
                    //	neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1)); //右下点入栈
                    //}
                    ////===============补充到8连通域======================================================
                }
            }
        }
    }
    iConnectedAreaCount = label - 1; //因为label从2开始计数的
    int a = 0;
}

void Phase_unwrapping(Mat& synthesis_phase_before, Mat& synthesis_phase_after, Mat& continuous_phase, float f_before, float f_after, int m_rows, int m_cols)//相位展开函数
{
    Mat k_picture = Mat::zeros(m_rows, m_cols, CV_32FC1);
    for (int i = 0; i < m_rows; i++)
    {
        const float*synthesis_phase_before_data = synthesis_phase_before.ptr<float>(i);
        const float*synthesis_phase_after_data = synthesis_phase_after.ptr<float>(i);
        float*test_data = k_picture.ptr<float>(i);

        for (int j = 0; j < m_cols; j++)
        {
            test_data[j] = round(((f_after / f_before) *synthesis_phase_after_data[j] - synthesis_phase_before_data[j]) / (2.0*pi));
        }
    }

    for (int i = 0; i < m_rows; i++)
    {
        const float*synthesis_phase_before_data = synthesis_phase_before.ptr<float>(i);
        const float*test_data = k_picture.ptr<float>(i);
        float*continuous_phase_data = continuous_phase.ptr<float>(i);

        for (int j = 0; j < m_cols; j++)
        {
            continuous_phase_data[j] = synthesis_phase_before_data[j] + 2.0 * pi * test_data[j];// 0.2 * pi * test_data[j]出来的结果比2 * pi * test_data[j]平整
        }
    }
}


void Phase_unwrapping111(Mat& synthesis_phase_before, Mat& synthesis_phase_after, Mat& continuous_phase, float f_before, float f_after, int m_rows, int m_cols)//相位展开函数
{
    Mat k_picture = Mat::zeros(m_rows, m_cols, CV_32FC1);
    for (int i = 0; i < m_rows; i++)
    {
        const float*synthesis_phase_before_data = synthesis_phase_before.ptr<float>(i);
        const float*synthesis_phase_after_data = synthesis_phase_after.ptr<float>(i);
        float*test_data = k_picture.ptr<float>(i);

        for (int j = 0; j < m_cols; j++)
        {
            test_data[j] = round(((f_after / f_before) *synthesis_phase_after_data[j] - synthesis_phase_before_data[j]) / (2.0*pi));
        }
    }

    Mat ert123; Mat* ert123_ip = &ert123;
    k_picture.copyTo(ert123);
    normalize(ert123, ert123, 0, 1, NORM_MINMAX);

    vector<int> sign_cols;
    for (int i = 1; i < m_rows; i++)
    {
        if ((k_picture.at<float>(i, 0) - k_picture.at<float>(i - 1, 0)) == 1)
        {
            sign_cols.push_back(i);
        }
    }

    //图形学处理
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat result = Mat::zeros(m_rows, m_cols, CV_32FC1);
    //漫水填充
    Point seed;	seed.x = 2591;	seed.y = 1943;
    //当能级为0的时候
    Mat k_translate;
    k_picture.copyTo(k_translate);
    Rect rect1(0, 0, k_picture.cols, 100);
    k_translate = k_translate(rect1);
    threshold(k_translate, k_translate, k_picture.at<float>(0, 0), 255, THRESH_BINARY);
    for (int k = 0; k < k_translate.rows; k++)
    {
        const float*input_data = k_translate.ptr<float>(k);
        float*output_data = result.ptr<float>(k);
        for (int q = 0; q < k_translate.cols; q++)
        {
            if (output_data[q] == 0 && input_data[q] == 0)
            {
                output_data[q] = k_picture.at<float>(0, 0) + 1;
            }
        }
    }
    //当能级为其他的时候
    int dsa = 0;
    for (int i = 1; i < sign_cols.size() - 1; i++)
    {
        int rect_x = 0;
        int height = 0;
        if (sign_cols[i] - 200 <= 0)
        {
            rect_x = 0;
            height = sign_cols[i] + 100;
        }
        else if ((sign_cols[i] + 200) >= 1944)
        {
            rect_x = sign_cols[i] - 200;
            height = 1943 - rect_x;
        }
        else
        {
            rect_x = sign_cols[i] - 200;
            height = 400;
        }


        Rect rect(0, rect_x, k_picture.cols, height);
        Mat k_picture_translate; Mat* k_picture_translate_ip = &k_picture_translate;
        k_picture.copyTo(*k_picture_translate_ip);
        *k_picture_translate_ip = k_picture_translate(rect);

        ////在二值化之前去掉过小的值
        //for (int q = 150; q < k_picture_translate.rows; q++)
        //{
        //	float*data1 = k_picture_translate.ptr<float>(q);
        //	for (int k = 0; k < k_picture_translate.cols; k++)
        //	{
        //		if (data1[k] < (k_picture.at<float>(sign_cols[i] - 10, 0)-3))
        //		{
        //			data1[k] = k_picture.at<float>(sign_cols[i] - 10, 0)+1;
        //		}
        //	}
        //}

        if (i == 1)
        {
            dsa = k_picture.at<float>(sign_cols[i] - 1, 0) - 1;
        }
        dsa = dsa + 1;

        //二值化、开运算
        threshold(*k_picture_translate_ip, *k_picture_translate_ip, k_picture.at<float>(sign_cols[i] - 10, 0), 1, THRESH_BINARY_INV);
        //找到连通域，去掉小连通域的干扰,并将中型连通域进行两部分处理
        Mat bcd;	int a;
        k_picture_translate.convertTo(*k_picture_translate_ip, CV_8UC1);
        ConnectedCountBySeedFill11(*k_picture_translate_ip, bcd, a);
        Rect rect123(1, 1, k_picture.cols, height);
        bcd = bcd(rect123);
        Mat bcd_big = Mat::zeros(bcd.size(), CV_8UC1);//中型连通域
        for (int k = 1; k < a + 2; k++)
        {
            vector<Point> points;
            for (int q = 0; q < bcd.rows; q++)
            {
                int*data12345 = bcd.ptr<int>(q);
                for (int t = 0; t < bcd.cols; t++)
                {
                    int sdf = data12345[t];
                    if (sdf == k)
                    {
                        Point a;
                        a.x = q;
                        a.y = t;
                        points.push_back(a);
                    }
                }
                if (points.size() > 5000)
                {
                    break;
                }
            }
            if (points.size() < 500)
            {
                for (int p = 0; p < points.size(); p++)
                {
                    bcd.at<int>(points[p].x, points[p].y) = 0;
                }
            }
            else if (points.size() > 500 && points.size() < 5000)
            {
                for (int p = 0; p < points.size(); p++)
                {
                    bcd.at<int>(points[p].x, points[p].y) = 0;
                    bcd_big.at<uchar>(points[p].x, points[p].y) = 255;
                }
            }
        }
        bcd.convertTo(bcd, CV_8UC1);
        for (int q = 0; q < bcd.rows; q++)
        {
            uchar*data = bcd.ptr<uchar>(q);
            for (int t = 0; t < bcd.cols; t++)
            {
                if (data[t] > 0)
                {
                    data[t] = 0;
                }
                else
                {
                    data[t] = 1;
                }
            }
        }
        a = 0;
        ConnectedCountBySeedFill11(bcd, bcd, a);
        bcd = bcd(rect123);
        for (int k = 1; k < a + 2; k++)
        {
            vector<Point> points;
            for (int q = 0; q < bcd.rows; q++)
            {
                int*data12345 = bcd.ptr<int>(q);
                for (int t = 0; t < bcd.cols; t++)
                {
                    int sdf = data12345[t];
                    if (sdf == k)
                    {
                        Point a;
                        a.x = q;
                        a.y = t;
                        points.push_back(a);
                    }
                }
                if (points.size() > 500)
                {
                    break;
                }
            }
            if (points.size() < 500)
            {
                for (int p = 0; p < points.size(); p++)
                {
                    bcd.at<int>(points[p].x, points[p].y) = 0;
                }
            }
        }
        bcd.convertTo(bcd, CV_32FC1);
        *k_picture_translate_ip = bcd;
        for (int q = 0; q < k_picture_translate.rows; q++)
        {
            float*data = k_picture_translate.ptr<float>(q);
            for (int t = 0; t < k_picture_translate.cols; t++)
            {
                if (data[t] > 0)
                {
                    data[t] = 255;
                }
            }
        }

        //由下往上找最简单的边界点
        vector<Point> border_points;
        for (int t = 0; t < k_picture_translate.cols; t++)
        {
            for (int q = k_picture_translate.rows - 1; q > 1; q--)
            {
                const float*data = k_picture_translate.ptr<float>(q);
                const float*data_before = k_picture_translate.ptr<float>(q - 1);
                if (data[t] > data_before[t])
                {
                    Point a;
                    a.x = t;
                    a.y = q;
                    border_points.push_back(a);
                    break;
                }
            }
        }
        Mat test = Mat::zeros(k_picture_translate.size(), CV_8UC1);
        for (int q = 0; q < border_points.size(); q++)
        {
            test.at<uchar>(border_points[q].y, border_points[q].x) = 255;
        }
        //找到相距过远的端点
        vector<Point> daundian_points;
        for (int q = 1; q < border_points.size(); q++)
        {
            if (border_points[q].x < 2591)
            {
                int a = test.at<uchar>(border_points[q].y, border_points[q].x) + test.at<uchar>(border_points[q].y, border_points[q].x - 1) + test.at<uchar>(border_points[q].y, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y - 1, border_points[q].x) + test.at<uchar>(border_points[q].y - 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y - 1, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y + 1, border_points[q].x) + test.at<uchar>(border_points[q].y + 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y + 1, border_points[q].x + 1);

                if (a == 255 * 2 || a == 255)
                {
                    if (abs(border_points[q].y - border_points[q + 1].y) > 10 || abs(border_points[q].y - border_points[q - 1].y) > 10)
                    {
                        if (daundian_points.size() > 2)//防止有三个点满足要求
                        {
                            if ((border_points[q].x - daundian_points[daundian_points.size() - 1].x) == 1 && (border_points[q].x - daundian_points[daundian_points.size() - 2].x) == 2)
                            {
                                continue;
                            }
                        }
                        test.at<uchar>(border_points[q].y, border_points[q].x) = 150;
                        daundian_points.push_back(border_points[q]);
                    }
                }
            }
        }
        vector<Point> daundian_buzu_points;
        if (daundian_points.size() != 0)
        {
            //将端点进行补足
            for (int q = 0; q < daundian_points.size(); q++)
            {
                stack<pair<int, int>> neighborPixels;   //新建一个栈
                neighborPixels.push(pair<int, int>(daundian_points[q].y, daundian_points[q].x));   // 像素坐标: <i,j> ，以该像素为起点，寻找连通域
                int sign_up = 0;//拐点向上的标志
                while (!neighborPixels.empty() && sign_up != 5)
                {
                    // 获取堆栈中的顶部像素并使用相同的标签对其进行标记
                    pair<int, int> curPixel = neighborPixels.top();
                    int curX = curPixel.first;
                    int curY = curPixel.second;
                    test.at<uchar>(curX, curY) = 255; //对图像对应位置的点进行标记

                    // 弹出顶部像素  （顶部像素出栈）
                    neighborPixels.pop();

                    // 加入8邻域点
                    if (k_picture_translate.at<float>(curX, curY - 1) == 255)
                    {// 左点
                        int a = k_picture_translate.at<float>(curX, curY - 1 - 1) + k_picture_translate.at<float>(curX, curY - 1 + 1) + k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY - 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX, curY - 1) == 150)//保证和下一个端点碰到时中断
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY - 1)); //左边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX, curY + 1) == 255)
                    {// 右点
                        int a = k_picture_translate.at<float>(curX, curY + 1 - 1) + k_picture_translate.at<float>(curX, curY + 1 + 1) + k_picture_translate.at<float>(curX - 1, curY + 1) + k_picture_translate.at<float>(curX + 1, curY + 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX, curY + 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY + 1)); //右边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX - 1, curY) == 255)
                    {// 上点
                        int a = k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX - 1, curY + 1) + k_picture_translate.at<float>(curX - 1 - 1, curY) + k_picture_translate.at<float>(curX - 1 + 1, curY);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX - 1, curY) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX - 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY)); //上边点入栈
                                sign_up++;
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY) == 255)
                    {// 下点
                        int a = k_picture_translate.at<float>(curX + 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY + 1) + k_picture_translate.at<float>(curX + 1 - 1, curY) + k_picture_translate.at<float>(curX + 1 + 1, curY);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX + 1, curY) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX + 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY)); //下边点入栈
                            }
                        }
                    }
                    if (k_picture_translate.at<float>(curX - 1, curY - 1) == 255)
                    {// 左上点
                        int a = k_picture_translate.at<float>(curX - 1, curY - 1 - 1) + k_picture_translate.at<float>(curX - 1, curY - 1 + 1) + k_picture_translate.at<float>(curX - 1 - 1, curY - 1) + k_picture_translate.at<float>(curX - 1 + 1, curY - 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX - 1, curY - 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX - 1, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY - 1)); //左边点入栈
                                sign_up++;
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX - 1, curY + 1) == 255)
                    {// 右上点
                        int a = k_picture_translate.at<float>(curX - 1, curY + 1 - 1) + k_picture_translate.at<float>(curX - 1, curY + 1 + 1) + k_picture_translate.at<float>(curX - 1 - 1, curY + 1) + k_picture_translate.at<float>(curX - 1 + 1, curY + 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX - 1, curY + 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX - 1, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY + 1)); //右边点入栈
                                sign_up++;
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY - 1) == 255)
                    {// 左下点
                        int a = k_picture_translate.at<float>(curX + 1, curY - 1 - 1) + k_picture_translate.at<float>(curX + 1, curY - 1 + 1) + k_picture_translate.at<float>(curX + 1 - 1, curY - 1) + k_picture_translate.at<float>(curX + 1 + 1, curY - 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX + 1, curY - 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX + 1, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY - 1)); //上边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY + 1) == 255)
                    {// 右下点
                        int a = k_picture_translate.at<float>(curX + 1, curY + 1 - 1) + k_picture_translate.at<float>(curX + 1, curY + 1 + 1) + k_picture_translate.at<float>(curX + 1 - 1, curY + 1) + k_picture_translate.at<float>(curX + 1 + 1, curY + 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX + 1, curY + 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX + 1, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY + 1)); //下边点入栈
                            }
                        }
                    }
                    //将拐点标志变为灰色，像素值设为150
                    if (sign_up >= 5 || neighborPixels.empty())
                    {
                        test.at<uchar>(curX, curY) = 150;
                        Point a;
                        a.x = curY;
                        a.y = curX;
                        daundian_buzu_points.push_back(a);
                        break;
                    }
                }
            }

            //将补足的端点进行连线
            if (daundian_buzu_points.size() % 2 == 0)//判断偶数
            {
                for (int q = 0; q < daundian_buzu_points.size(); q = q + 2)
                {
                    line(test, Point(daundian_buzu_points[q].x, daundian_buzu_points[q].y), Point(daundian_buzu_points[q + 1].x, daundian_buzu_points[q + 1].y), Scalar(255), 1);
                }
            }
            else//若是奇数，需要去掉一个点
            {
                for (int q = 1; q < daundian_buzu_points.size() - 1; q++)
                {
                    int a = abs((daundian_buzu_points[q].x - daundian_buzu_points[q - 1].x) - (daundian_buzu_points[q + 1].x - daundian_buzu_points[q].x));
                    if (a < 50)
                    {
                        line(test, Point(daundian_buzu_points[q - 1].x, daundian_buzu_points[q - 1].y), Point(daundian_buzu_points[q + 1].x, daundian_buzu_points[q + 1].y), Scalar(255), 1);
                        q = q + 2;
                    }
                    else
                    {
                        line(test, Point(daundian_buzu_points[q - 1].x, daundian_buzu_points[q - 1].y), Point(daundian_buzu_points[q].x, daundian_buzu_points[q].y), Scalar(255), 1);
                        q++;
                    }
                }
            }
        }
        //找到不连续的近距离端点进行补全
        vector<Point> daundian_short_points;
        for (int q = 1; q < border_points.size(); q++)
        {
            if (border_points[q].x < 2591)
            {
                int a = test.at<uchar>(border_points[q].y, border_points[q].x) + test.at<uchar>(border_points[q].y, border_points[q].x - 1) + test.at<uchar>(border_points[q].y, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y - 1, border_points[q].x) + test.at<uchar>(border_points[q].y - 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y - 1, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y + 1, border_points[q].x) + test.at<uchar>(border_points[q].y + 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y + 1, border_points[q].x + 1);

                if (a == 255 * 2 || a == 255)
                {
                    if (abs(border_points[q].y - border_points[q + 1].y) < 10 || abs(border_points[q].y - border_points[q - 1].y) < 10)
                    {
                        daundian_short_points.push_back(border_points[q]);
                    }
                }
            }
        }
        if (daundian_short_points.size() != 0)
        {
            //将端点进行补足
            for (int q = 0; q < daundian_short_points.size(); q++)
            {
                stack<pair<int, int>> neighborPixels;   //新建一个栈
                neighborPixels.push(pair<int, int>(daundian_short_points[q].y, daundian_short_points[q].x));   // 像素坐标: <i,j> ，以该像素为起点，寻找连通域
                while (!neighborPixels.empty())
                {
                    // 获取堆栈中的顶部像素并使用相同的标签对其进行标记
                    pair<int, int> curPixel = neighborPixels.top();
                    int curX = curPixel.first;
                    int curY = curPixel.second;
                    test.at<uchar>(curX, curY) = 255; //对图像对应位置的点进行标记

                    // 弹出顶部像素  （顶部像素出栈）
                    neighborPixels.pop();

                    // 加入8邻域点
                    if (k_picture_translate.at<float>(curX, curY - 1) == 255)
                    {// 左点
                        int a = k_picture_translate.at<float>(curX, curY - 1) + k_picture_translate.at<float>(curX, curY - 1 - 1) + k_picture_translate.at<float>(curX, curY - 1 + 1) +
                                k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX - 1, curY - 1 - 1) + k_picture_translate.at<float>(curX - 1, curY - 1 + 1) +
                                k_picture_translate.at<float>(curX + 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY - 1 - 1) + k_picture_translate.at<float>(curX + 1, curY - 1 + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY - 1)); //左边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX, curY + 1) == 255)
                    {// 右点
                        int a = k_picture_translate.at<float>(curX, curY + 1) + k_picture_translate.at<float>(curX, curY + 1 - 1) + k_picture_translate.at<float>(curX, curY + 1 + 1) +
                                k_picture_translate.at<float>(curX - 1, curY + 1) + k_picture_translate.at<float>(curX - 1, curY + 1 - 1) + k_picture_translate.at<float>(curX - 1, curY + 1 + 1) +
                                k_picture_translate.at<float>(curX + 1, curY + 1) + k_picture_translate.at<float>(curX + 1, curY + 1 - 1) + k_picture_translate.at<float>(curX + 1, curY + 1 + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY + 1)); //右边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX - 1, curY) == 255)
                    {// 上点
                        int a = k_picture_translate.at<float>(curX - 1, curY) + k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX - 1, curY + 1) +
                                k_picture_translate.at<float>(curX - 1 - 1, curY) + k_picture_translate.at<float>(curX - 1 - 1, curY - 1) + k_picture_translate.at<float>(curX - 1 - 1, curY + 1) +
                                k_picture_translate.at<float>(curX - 1 + 1, curY) + k_picture_translate.at<float>(curX - 1 + 1, curY - 1) + k_picture_translate.at<float>(curX - 1 + 1, curY + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX - 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY)); //上边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY) == 255)
                    {// 下点
                        int a = k_picture_translate.at<float>(curX + 1, curY) + k_picture_translate.at<float>(curX + 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY + 1) +
                                k_picture_translate.at<float>(curX + 1 - 1, curY) + k_picture_translate.at<float>(curX + 1 - 1, curY - 1) + k_picture_translate.at<float>(curX + 1 - 1, curY + 1) +
                                k_picture_translate.at<float>(curX + 1 + 1, curY) + k_picture_translate.at<float>(curX + 1 + 1, curY - 1) + k_picture_translate.at<float>(curX + 1 + 1, curY + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX + 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY)); //下边点入栈
                            }
                        }
                    }
                }
            }
        }
        //还原到原来大小
        Mat dst123 = Mat::zeros(k_picture.size(), CV_8UC3);
        for (int k = 0; k < test.rows; k++)
        {
            uchar*intput_data = test.ptr<uchar>(k);
            for (int q = 0; q < test.cols; q++)
            {
                if (intput_data[q] == 255 || intput_data[q] == 150)
                {
                    dst123.at<Vec3b>(k + rect_x, q)[0] = 255;
                }
            }
        }
        //填充
        floodFill(dst123, seed, 255, NULL, Scalar(0), Scalar(0), FLOODFILL_FIXED_RANGE);

        for (int k = 0; k < m_rows; k++)
        {
            float*output_data = result.ptr<float>(k);
            for (int q = 0; q < m_cols; q++)
            {
                if (output_data[q] == 0 && dst123.at<Vec3b>(k, q)[0] == 0)
                {
                    output_data[q] = k_picture.at<float>(sign_cols[i] - 10, 0) + 1;
                }
            }
        }

    }
    //当能级为最后一个的时候
    k_picture.copyTo(k_translate);
    threshold(k_translate, k_translate, k_picture.at<float>(1934, 2582), 255, THRESH_BINARY);
    for (int k = 0; k < k_translate.rows; k++)
    {
        const float*input_data = k_translate.ptr<float>(k);
        float*output_data = result.ptr<float>(k);
        for (int q = 0; q < k_translate.cols; q++)
        {
            if (output_data[q] == 0 && input_data[q] == 0)
            {
                output_data[q] = k_picture.at<float>(1934, 2582) + 1;
            }
        }
    }
    for (int k = 0; k < m_rows; k++)
    {
        float*output_data = result.ptr<float>(k);
        for (int q = 0; q < m_cols; q++)
        {
            output_data[q] = output_data[q] - 1;
        }
    }
    k_picture = result;




    ////取一行列数据用MATLAB进行查看
    //ofstream fout("hang.txt");
    //for (int j = 0; j < k_picture.cols; j++)
    //{
    //	fout << j << " " << k_picture.at<float>(1500, j) << "\n" << endl;
    //}
    //ofstream fout4("lie.txt");
    //for (int j = 0; j < k_picture.rows; j++)
    //{
    //	fout4 << j << " " << k_picture.at<float>(j, 0) << "\n" << endl;
    //}

    //normalize(k_picture, k_picture, 0, 1, CV_MINMAX);
    //ert123 = k_picture;
    //normalize(*ert123_ip, *ert123_ip, 0, 255, CV_MINMAX);
    //imwrite("能级图.bmp", *ert123_ip);

    for (int i = 0; i < m_rows; i++)
    {
        const float*synthesis_phase_before_data = synthesis_phase_before.ptr<float>(i);
        const float*test_data = k_picture.ptr<float>(i);
        float*continuous_phase_data = continuous_phase.ptr<float>(i);

        for (int j = 0; j < m_cols; j++)
        {
            continuous_phase_data[j] = synthesis_phase_before_data[j] + 2.0 * pi * test_data[j];// 0.2 * pi * test_data[j]出来的结果比2 * pi * test_data[j]平整
        }
    }
}

void Phase_unwrapping_tongji111(Mat& synthesis_phase_before, Mat& synthesis_phase_after, Mat& continuous_phase, float f_before, float f_after, int m_rows, int m_cols, float wavelength)//相位展开函数
{
    Mat k_picture = Mat::zeros(m_rows, m_cols, CV_32FC1);
    for (int i = 0; i < m_rows; i++)
    {
        const float*synthesis_phase_before_data = synthesis_phase_before.ptr<float>(i);
        const float*synthesis_phase_after_data = synthesis_phase_after.ptr<float>(i);
        float*test_data = k_picture.ptr<float>(i);

        for (int j = 0; j < m_cols; j++)
        {
            test_data[j] = round(((f_after / f_before) *synthesis_phase_after_data[j] - synthesis_phase_before_data[j]) / (2.0*pi));
        }
    }


    Mat ert123;	Mat* ert123_ip = &ert123;
    k_picture.copyTo(*ert123_ip);

    //找第一列的能级行标志
    vector<int> sign_cols;
    for (int i = 1; i < m_rows; i++)
    {
        const float* data_before = k_picture.ptr<float>(i - 1);
        const float* data = k_picture.ptr<float>(i);
        if ((data[0] - data_before[0]) == 1)
        {
            sign_cols.push_back(i);
        }
    }
    //图形学处理
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat result = Mat::zeros(m_rows, m_cols, CV_32FC1);
    //漫水填充
    Point seed;	seed.x = 2591;	seed.y = 1943;
    //当能级为0的时候
    Mat k_translate; Mat* k_translate_ip = &k_translate;
    k_picture.copyTo(*k_translate_ip);
    Rect rect1(0, 0, k_picture.cols, 300);
    *k_translate_ip = k_translate(rect1);
    threshold(*k_translate_ip, *k_translate_ip, 0, 255, THRESH_BINARY);
    for (int k = 0; k < k_translate.rows; k++)
    {
        const float*input_data = k_translate.ptr<float>(k);
        float*output_data = result.ptr<float>(k);
        for (int q = 0; q < k_translate.cols; q++)
        {
            if (output_data[q] == 0 && input_data[q] == 0)
            {
                output_data[q] = 1;
            }
        }
    }
    //当能级为其他的时候
    for (int i = 0; i < sign_cols.size(); i++)
    {
        int rect_x = sign_cols[i] - 150;
        int height = 0;
        if (i == sign_cols.size() - 1)
        {
            height = 1944 - rect_x;
        }
        else
        {
            height = 300;
        }

        Rect rect(0, rect_x, k_picture.cols, height);
        Mat k_picture_translate; Mat* k_picture_translate_ip = &k_picture_translate;
        k_picture.copyTo(*k_picture_translate_ip);
        *k_picture_translate_ip = k_picture_translate(rect);

        //去掉过大的值
        for (int q = 0; q < k_picture_translate.rows; q++)
        {
            float*data1 = k_picture_translate.ptr<float>(q);
            for (int k = 0; k < k_picture_translate.cols; k++)
            {
                if (data1[k]<k_picture.at<float>(sign_cols[i] - 10, 0) || data1[k]>(k_picture.at<float>(sign_cols[i] - 10, 0) + 1))
                {
                    data1[k] = k_picture.at<float>(sign_cols[i] - 10, 0) + 1;
                }
            }
        }


        //二值化、开运算
        threshold(*k_picture_translate_ip, *k_picture_translate_ip, k_picture.at<float>(sign_cols[i] - 10, 0), 1, THRESH_BINARY_INV);
        //找到连通域，去掉小连通域的干扰
        Mat bcd;	int a;
        k_picture_translate.convertTo(*k_picture_translate_ip, CV_8UC1);
        ConnectedCountBySeedFill11(*k_picture_translate_ip, bcd, a);
        Rect rect123(1, 1, k_picture.cols, height);
        bcd = bcd(rect123);
        for (int k = 1; k < a + 2; k++)
        {
            vector<Point> points;
            for (int q = 0; q < bcd.rows; q++)
            {
                int*data12345 = bcd.ptr<int>(q);
                for (int t = 0; t < bcd.cols; t++)
                {
                    int sdf = data12345[t];
                    if (sdf == k)
                    {
                        Point a;
                        a.x = q;
                        a.y = t;
                        points.push_back(a);
                    }
                }
                if (points.size() > 500)
                {
                    break;
                }
            }
            if (points.size() < 500)
            {
                for (int p = 0; p < points.size(); p++)
                {
                    bcd.at<int>(points[p].x, points[p].y) = 0;
                }
            }
        }
        bcd.convertTo(bcd, CV_8UC1);
        for (int q = 0; q < bcd.rows; q++)
        {
            uchar*data = bcd.ptr<uchar>(q);
            for (int t = 0; t < bcd.cols; t++)
            {
                if (data[t] > 0)
                {
                    data[t] = 0;
                }
                else
                {
                    data[t] = 1;
                }
            }
        }
        a = 0;
        ConnectedCountBySeedFill11(bcd, bcd, a);
        bcd = bcd(rect123);
        for (int k = 1; k < a + 2; k++)
        {
            vector<Point> points;
            for (int q = 0; q < bcd.rows; q++)
            {
                int*data12345 = bcd.ptr<int>(q);
                for (int t = 0; t < bcd.cols; t++)
                {
                    int sdf = data12345[t];
                    if (sdf == k)
                    {
                        Point a;
                        a.x = q;
                        a.y = t;
                        points.push_back(a);
                    }
                }
                if (points.size() > 500)
                {
                    break;
                }
            }
            if (points.size() < 500)
            {
                for (int p = 0; p < points.size(); p++)
                {
                    bcd.at<int>(points[p].x, points[p].y) = 0;
                }
            }
        }
        bcd.convertTo(bcd, CV_32FC1);
        *k_picture_translate_ip = bcd;
        for (int q = 0; q < k_picture_translate.rows; q++)
        {
            float*data = k_picture_translate.ptr<float>(q);
            for (int t = 0; t < k_picture_translate.cols; t++)
            {
                if (data[t] > 0)
                {
                    data[t] = 255;
                }
            }
        }

        //由下往上找最简单的边界点
        vector<Point> border_points;
        for (int t = 0; t < k_picture_translate.cols; t++)
        {
            for (int q = k_picture_translate.rows - 1; q > 1; q--)
            {
                const float*data = k_picture_translate.ptr<float>(q);
                const float*data_before = k_picture_translate.ptr<float>(q - 1);
                if (data[t] > data_before[t])
                {
                    Point a;
                    a.x = t;
                    a.y = q;
                    border_points.push_back(a);
                    break;
                }
            }
        }
        Mat test = Mat::zeros(k_picture_translate.size(), CV_8UC1);
        for (int q = 0; q < border_points.size(); q++)
        {
            test.at<uchar>(border_points[q].y, border_points[q].x) = 255;
        }
        //找到相距过远的端点
        vector<Point> daundian_points;
        for (int q = 1; q < border_points.size(); q++)
        {
            if (border_points[q].x < 2591)
            {
                int a = test.at<uchar>(border_points[q].y, border_points[q].x) + test.at<uchar>(border_points[q].y, border_points[q].x - 1) + test.at<uchar>(border_points[q].y, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y - 1, border_points[q].x) + test.at<uchar>(border_points[q].y - 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y - 1, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y + 1, border_points[q].x) + test.at<uchar>(border_points[q].y + 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y + 1, border_points[q].x + 1);

                if (a == 255 * 2 || a == 255)
                {
                    if (abs(border_points[q].y - border_points[q + 1].y) > 10 || abs(border_points[q].y - border_points[q - 1].y) > 10)
                    {
                        if (daundian_points.size() > 2)//防止有三个点满足要求
                        {
                            if ((border_points[q].x - daundian_points[daundian_points.size() - 1].x) == 1 && (border_points[q].x - daundian_points[daundian_points.size() - 2].x) == 2)
                            {
                                continue;
                            }
                        }
                        test.at<uchar>(border_points[q].y, border_points[q].x) = 150;
                        daundian_points.push_back(border_points[q]);
                    }
                }
            }
        }
        vector<Point> daundian_buzu_points;
        if (daundian_points.size() != 0)
        {
            //将端点进行补足
            for (int q = 0; q < daundian_points.size(); q++)
            {
                stack<pair<int, int>> neighborPixels;   //新建一个栈
                neighborPixels.push(pair<int, int>(daundian_points[q].y, daundian_points[q].x));   // 像素坐标: <i,j> ，以该像素为起点，寻找连通域
                int sign_up = 0;//拐点向上的标志
                while (!neighborPixels.empty() && sign_up != 5)
                {
                    // 获取堆栈中的顶部像素并使用相同的标签对其进行标记
                    pair<int, int> curPixel = neighborPixels.top();
                    int curX = curPixel.first;
                    int curY = curPixel.second;
                    test.at<uchar>(curX, curY) = 255; //对图像对应位置的点进行标记

                    // 弹出顶部像素  （顶部像素出栈）
                    neighborPixels.pop();

                    // 加入8邻域点
                    if (k_picture_translate.at<float>(curX, curY - 1) == 255)
                    {// 左点
                        int a = k_picture_translate.at<float>(curX, curY - 1 - 1) + k_picture_translate.at<float>(curX, curY - 1 + 1) + k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY - 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX, curY - 1) == 150)//保证和下一个端点碰到时中断
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY - 1)); //左边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX, curY + 1) == 255)
                    {// 右点
                        int a = k_picture_translate.at<float>(curX, curY + 1 - 1) + k_picture_translate.at<float>(curX, curY + 1 + 1) + k_picture_translate.at<float>(curX - 1, curY + 1) + k_picture_translate.at<float>(curX + 1, curY + 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX, curY + 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY + 1)); //右边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX - 1, curY) == 255)
                    {// 上点
                        int a = k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX - 1, curY + 1) + k_picture_translate.at<float>(curX - 1 - 1, curY) + k_picture_translate.at<float>(curX - 1 + 1, curY);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX - 1, curY) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX - 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY)); //上边点入栈
                                sign_up++;
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY) == 255)
                    {// 下点
                        int a = k_picture_translate.at<float>(curX + 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY + 1) + k_picture_translate.at<float>(curX + 1 - 1, curY) + k_picture_translate.at<float>(curX + 1 + 1, curY);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX + 1, curY) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX + 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY)); //下边点入栈
                            }
                        }
                    }
                    if (k_picture_translate.at<float>(curX - 1, curY - 1) == 255)
                    {// 左上点
                        int a = k_picture_translate.at<float>(curX - 1, curY - 1 - 1) + k_picture_translate.at<float>(curX - 1, curY - 1 + 1) + k_picture_translate.at<float>(curX - 1 - 1, curY - 1) + k_picture_translate.at<float>(curX - 1 + 1, curY - 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX - 1, curY - 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX - 1, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY - 1)); //左边点入栈
                                sign_up++;
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX - 1, curY + 1) == 255)
                    {// 右上点
                        int a = k_picture_translate.at<float>(curX - 1, curY + 1 - 1) + k_picture_translate.at<float>(curX - 1, curY + 1 + 1) + k_picture_translate.at<float>(curX - 1 - 1, curY + 1) + k_picture_translate.at<float>(curX - 1 + 1, curY + 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX - 1, curY + 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX - 1, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY + 1)); //右边点入栈
                                sign_up++;
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY - 1) == 255)
                    {// 左下点
                        int a = k_picture_translate.at<float>(curX + 1, curY - 1 - 1) + k_picture_translate.at<float>(curX + 1, curY - 1 + 1) + k_picture_translate.at<float>(curX + 1 - 1, curY - 1) + k_picture_translate.at<float>(curX + 1 + 1, curY - 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX + 1, curY - 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX + 1, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY - 1)); //上边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY + 1) == 255)
                    {// 右下点
                        int a = k_picture_translate.at<float>(curX + 1, curY + 1 - 1) + k_picture_translate.at<float>(curX + 1, curY + 1 + 1) + k_picture_translate.at<float>(curX + 1 - 1, curY + 1) + k_picture_translate.at<float>(curX + 1 + 1, curY + 1);
                        if (a != 255 * 4)
                        {
                            if (test.at<uchar>(curX + 1, curY + 1) == 150)
                            {
                                test.at<uchar>(curX, curY) = 150;
                                Point a;
                                a.x = curY;
                                a.y = curX;
                                daundian_buzu_points.push_back(a);
                                break;
                            }
                            if (test.at<uchar>(curX + 1, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY + 1)); //下边点入栈
                            }
                        }
                    }
                    //将拐点标志变为灰色，像素值设为150
                    if (sign_up >= 5 || neighborPixels.empty())
                    {
                        test.at<uchar>(curX, curY) = 150;
                        Point a;
                        a.x = curY;
                        a.y = curX;
                        daundian_buzu_points.push_back(a);
                        break;
                    }
                }
            }

            //将补足的端点进行连线
            if (daundian_buzu_points.size() % 2 == 0)//判断偶数
            {
                for (int q = 0; q < daundian_buzu_points.size(); q = q + 2)
                {
                    line(test, Point(daundian_buzu_points[q].x, daundian_buzu_points[q].y), Point(daundian_buzu_points[q + 1].x, daundian_buzu_points[q + 1].y), Scalar(255), 1);
                }
            }
        }
        //找到不连续的近距离端点进行补全
        vector<Point> daundian_short_points;
        for (int q = 1; q < border_points.size(); q++)
        {
            if (border_points[q].x < 2591)
            {
                int a = test.at<uchar>(border_points[q].y, border_points[q].x) + test.at<uchar>(border_points[q].y, border_points[q].x - 1) + test.at<uchar>(border_points[q].y, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y - 1, border_points[q].x) + test.at<uchar>(border_points[q].y - 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y - 1, border_points[q].x + 1) +
                        test.at<uchar>(border_points[q].y + 1, border_points[q].x) + test.at<uchar>(border_points[q].y + 1, border_points[q].x - 1) + test.at<uchar>(border_points[q].y + 1, border_points[q].x + 1);

                if (a == 255 * 2 || a == 255)
                {
                    if (abs(border_points[q].y - border_points[q + 1].y) < 10 || abs(border_points[q].y - border_points[q - 1].y) < 10)
                    {
                        daundian_short_points.push_back(border_points[q]);
                    }
                }
            }
        }
        if (daundian_short_points.size() != 0)
        {
            //将端点进行补足
            for (int q = 0; q < daundian_short_points.size(); q++)
            {
                stack<pair<int, int>> neighborPixels;   //新建一个栈
                neighborPixels.push(pair<int, int>(daundian_short_points[q].y, daundian_short_points[q].x));   // 像素坐标: <i,j> ，以该像素为起点，寻找连通域
                while (!neighborPixels.empty())
                {
                    // 获取堆栈中的顶部像素并使用相同的标签对其进行标记
                    pair<int, int> curPixel = neighborPixels.top();
                    int curX = curPixel.first;
                    int curY = curPixel.second;
                    test.at<uchar>(curX, curY) = 255; //对图像对应位置的点进行标记

                    // 弹出顶部像素  （顶部像素出栈）
                    neighborPixels.pop();

                    // 加入8邻域点
                    if (k_picture_translate.at<float>(curX, curY - 1) == 255)
                    {// 左点
                        int a = k_picture_translate.at<float>(curX, curY - 1) + k_picture_translate.at<float>(curX, curY - 1 - 1) + k_picture_translate.at<float>(curX, curY - 1 + 1) +
                                k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX - 1, curY - 1 - 1) + k_picture_translate.at<float>(curX - 1, curY - 1 + 1) +
                                k_picture_translate.at<float>(curX + 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY - 1 - 1) + k_picture_translate.at<float>(curX + 1, curY - 1 + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX, curY - 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY - 1)); //左边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX, curY + 1) == 255)
                    {// 右点
                        int a = k_picture_translate.at<float>(curX, curY + 1) + k_picture_translate.at<float>(curX, curY + 1 - 1) + k_picture_translate.at<float>(curX, curY + 1 + 1) +
                                k_picture_translate.at<float>(curX - 1, curY + 1) + k_picture_translate.at<float>(curX - 1, curY + 1 - 1) + k_picture_translate.at<float>(curX - 1, curY + 1 + 1) +
                                k_picture_translate.at<float>(curX + 1, curY + 1) + k_picture_translate.at<float>(curX + 1, curY + 1 - 1) + k_picture_translate.at<float>(curX + 1, curY + 1 + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX, curY + 1) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX, curY + 1)); //右边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX - 1, curY) == 255)
                    {// 上点
                        int a = k_picture_translate.at<float>(curX - 1, curY) + k_picture_translate.at<float>(curX - 1, curY - 1) + k_picture_translate.at<float>(curX - 1, curY + 1) +
                                k_picture_translate.at<float>(curX - 1 - 1, curY) + k_picture_translate.at<float>(curX - 1 - 1, curY - 1) + k_picture_translate.at<float>(curX - 1 - 1, curY + 1) +
                                k_picture_translate.at<float>(curX - 1 + 1, curY) + k_picture_translate.at<float>(curX - 1 + 1, curY - 1) + k_picture_translate.at<float>(curX - 1 + 1, curY + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX - 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX - 1, curY)); //上边点入栈
                            }
                        }
                    }

                    if (k_picture_translate.at<float>(curX + 1, curY) == 255)
                    {// 下点
                        int a = k_picture_translate.at<float>(curX + 1, curY) + k_picture_translate.at<float>(curX + 1, curY - 1) + k_picture_translate.at<float>(curX + 1, curY + 1) +
                                k_picture_translate.at<float>(curX + 1 - 1, curY) + k_picture_translate.at<float>(curX + 1 - 1, curY - 1) + k_picture_translate.at<float>(curX + 1 - 1, curY + 1) +
                                k_picture_translate.at<float>(curX + 1 + 1, curY) + k_picture_translate.at<float>(curX + 1 + 1, curY - 1) + k_picture_translate.at<float>(curX + 1 + 1, curY + 1);
                        if (a != 255 * 9)
                        {
                            if (test.at<uchar>(curX + 1, curY) == 0)
                            {
                                neighborPixels.push(pair<int, int>(curX + 1, curY)); //下边点入栈
                            }
                        }
                    }
                }
            }
        }
        //还原到原来大小
        Mat dst123 = Mat::zeros(k_picture.size(), CV_8UC3);
        for (int k = 0; k < test.rows; k++)
        {
            uchar*intput_data = test.ptr<uchar>(k);
            for (int q = 0; q < test.cols; q++)
            {
                if (intput_data[q] == 255 || intput_data[q] == 150)
                {
                    dst123.at<Vec3b>(k + rect_x, q)[0] = 255;
                }
            }
        }
        //填充
        floodFill(dst123, seed, 255, NULL, Scalar(0), Scalar(0), FLOODFILL_FIXED_RANGE);

        for (int k = 0; k < m_rows; k++)
        {
            float*output_data = result.ptr<float>(k);
            for (int q = 0; q < m_cols; q++)
            {
                if (output_data[q] == 0 && dst123.at<Vec3b>(k, q)[0] == 0)
                {
                    output_data[q] = k_picture.at<float>(sign_cols[i] - 10, 0) + 1;
                }
            }
        }
    }
    //当能级为最后一个的时候
    k_picture.copyTo(*k_translate_ip);
    threshold(*k_translate_ip, *k_translate_ip, k_picture.at<float>(1934, 2582), 255, THRESH_BINARY);
    for (int k = 0; k < k_translate.rows; k++)
    {
        const float*input_data = k_translate.ptr<float>(k);
        float*output_data = result.ptr<float>(k);
        for (int q = 0; q < k_translate.cols; q++)
        {
            if (output_data[q] == 0 && input_data[q] == 0)
            {
                output_data[q] = k_picture.at<float>(1934, 2582) + 1;
            }
        }
    }
    for (int k = 0; k < m_rows; k++)
    {
        float*output_data = result.ptr<float>(k);
        for (int q = 0; q < m_cols; q++)
        {
            output_data[q] = output_data[q] - 1;
        }
    }
    k_picture = result;

    ////取一行列数据用MATLAB进行查看
    //ofstream fout("hang.txt");
    //for (int j = 0; j < k_picture.cols; j++)
    //{
    //	fout << j << " " << k_picture.at<float>(1500, j) << "\n" << endl;
    //}
    //ofstream fout4("lie.txt");
    //for (int j = 0; j < k_picture.rows; j++)
    //{
    //	fout4 << j << " " << k_picture.at<float>(j, 1609) << "\n" << endl;
    //}

    //normalize(k_picture, k_picture, 0, 1, CV_MINMAX);
    //ert123 = k_picture;
    //normalize(*ert123_ip, *ert123_ip, 0, 255, CV_MINMAX);
    //imwrite("能级图.bmp", *ert123_ip);


    for (int i = 0; i < m_rows; i++)
    {
        const float*synthesis_phase_before_data = synthesis_phase_before.ptr<float>(i);
        const float*test_data = k_picture.ptr<float>(i);
        float*continuous_phase_data = continuous_phase.ptr<float>(i);

        for (int j = 0; j < m_cols; j++)
        {
            continuous_phase_data[j] = synthesis_phase_before_data[j] + 2.0 * pi * test_data[j];// 0.2 * pi * test_data[j]出来的结果比2 * pi * test_data[j]平整
        }
    }
}

Mat phase(Mat b[6][4])
{
    Mat object[6][4];
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            b[i][j].convertTo(object[i][j], CV_32FC1);//转成64位图
            GaussianBlur(object[i][j], object[i][j], Size(5, 5), 0, 0);
        }
    }

    int object_rows = object[0][0].rows;
    int object_cols = object[0][0].cols;

    //******************第二步***********************//
    //物体图截断相位求解及伽马矫正.求了六组包裹函数

    Mat wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f1, object[0], object_rows, object_cols);
    Mat wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f2, object[1], object_rows, object_cols);
    Mat wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f3, object[2], object_rows, object_cols);

    Mat wrapped_phase_f11 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f11, object[3], object_rows, object_cols);
    Mat wrapped_phase_f22 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f22, object[4], object_rows, object_cols);
    Mat wrapped_phase_f33 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f33, object[5], object_rows, object_cols);

    /*
    for (int i = 0; i < object_rows; i++)
    {
        float* data1 = wrapped_phase_f1.ptr<float>(i);
        float* data11 = wrapped_phase_f11.ptr<float>(i);
        float* data2 = wrapped_phase_f2.ptr<float>(i);
        float* data22 = wrapped_phase_f22.ptr<float>(i);
        float* data3 = wrapped_phase_f3.ptr<float>(i);
        float* data33 = wrapped_phase_f33.ptr<float>(i);
        for (int j = 0; j < object_cols; j++)
        {
            if (data1[j] < data11[j])
            {
                data1[j] = (data1[j] + data11[j] - (pi / 4.0)) / 2.0;
            }
            if (data1[j] > data11[j])
            {
                data1[j] = (data1[j] + data11[j] - (pi / 4.0) + 2 * pi) / 2.0;
            }

            if (data2[j] < data22[j])
            {
                data2[j] = (data2[j] + data22[j] - (pi / 4.0)) / 2.0;
            }
            if (data2[j] > data22[j])
            {
                data2[j] = (data2[j] + data22[j] - (pi / 4.0) + 2 * pi) / 2.0;
            }

            if (data3[j] < data33[j])
            {
                data3[j] = (data3[j] + data33[j] - (pi / 4.0)) / 2.0;
            }
            if (data3[j] > data33[j])
            {
                data3[j] = (data3[j] + data33[j] - (pi / 4.0) + 2 * pi) / 2.0;
            }
        }
    }
    */

    //******************第三步***********************//
    //物体图三频外差

    Mat object_12 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f2, object_12, object_rows, object_cols);
    Mat object_23 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(wrapped_phase_f2, wrapped_phase_f3, object_23, object_rows, object_cols);
    Mat object_13 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f3, object_13, object_rows, object_cols);
    Mat object_123 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(object_12, object_23, object_123, object_rows, object_cols);

    //******************第四步***********************//
    //差频解包裹(按照师姐论文推)
    //物体图差频解包裹
    float f_12 = f[0] - f[1];//f1和f2的频差
    float f_23 = f[1] - f[2];
    float f_13 = f[0] - f[2];
    float f_123 = f_12 - f_23;

#ifdef JIAOZHENG
    Mat synthesis_wrapped_phase_f13 = Mat::zeros(object_rows, object_cols, CV_32FC1);
	Phase_unwrapping_tongji111(object_13, object_123, synthesis_wrapped_phase_f13, (1.0 / f_13), (1.0 / f_123), object_rows, object_cols, f_13);
	Mat synthesis_wrapped_phase_f12 = Mat::zeros(object_rows, object_cols, CV_32FC1);
	Phase_unwrapping_tongji111(object_12, object_123, synthesis_wrapped_phase_f12, (1.0 / f_12), (1.0 / f_123), object_rows, object_cols, f_12);
	Mat synthesis_wrapped_phase_f23 = Mat::zeros(object_rows, object_cols, CV_32FC1);
	Phase_unwrapping_tongji111(object_23, object_123, synthesis_wrapped_phase_f23, (1.0 / f_23), (1.0 / f_123), object_rows, object_cols, f_23);

	Mat synthesis_wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_32FC1);
	Phase_unwrapping111(wrapped_phase_f3, synthesis_wrapped_phase_f13, synthesis_wrapped_phase_f3, 1.0 / float(f[2]), (1.0 / f_13), object_rows, object_cols);
	Mat synthesis_wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_32FC1);
	Phase_unwrapping111(wrapped_phase_f1, synthesis_wrapped_phase_f12, synthesis_wrapped_phase_f1, 1.0 / float(f[0]), (1.0 / f_12), object_rows, object_cols);
	Mat synthesis_wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_32FC1);
	Phase_unwrapping111(wrapped_phase_f2, synthesis_wrapped_phase_f23, synthesis_wrapped_phase_f2, 1.0 / float(f[1]), (1.0 / f_23), object_rows, object_cols);
#else
    Mat synthesis_wrapped_phase_f13 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(object_13, object_123, synthesis_wrapped_phase_f13, (1.0 / f_13), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f12 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(object_12, object_123, synthesis_wrapped_phase_f12, (1.0 / f_12), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f23 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(object_23, object_123, synthesis_wrapped_phase_f23, (1.0 / f_23), (1.0 / f_123), object_rows, object_cols);

    Mat synthesis_wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(wrapped_phase_f3, synthesis_wrapped_phase_f13, synthesis_wrapped_phase_f3, 1.0 / float(f[2]), (1.0 / f_13), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(wrapped_phase_f1, synthesis_wrapped_phase_f12, synthesis_wrapped_phase_f1, 1.0 / float(f[0]), (1.0 / f_12), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(wrapped_phase_f2, synthesis_wrapped_phase_f23, synthesis_wrapped_phase_f2, 1.0 / float(f[1]), (1.0 / f_23), object_rows, object_cols);
#endif

    //返回真实相位
    //这段主要负责对相位图平均化
    Mat synthesis_wrapped_phase = Mat::zeros(object_rows, object_cols, CV_32FC1);
    for (int k = 0; k < synthesis_wrapped_phase_f1.rows; k++)
    {
        const float* data_1 = synthesis_wrapped_phase_f1.ptr<float>(k);
        const float* data_2 = synthesis_wrapped_phase_f2.ptr<float>(k);
        const float* data_3 = synthesis_wrapped_phase_f3.ptr<float>(k);
        float* out_data = synthesis_wrapped_phase.ptr<float>(k);
        for (int q = 0; q < synthesis_wrapped_phase_f1.cols; q++)
        {
            out_data[q] = (data_1[q] + data_2[q] + data_3[q]) / 3.0;
        }
    }

    if (phase_jiaozheng == true)
    {
        for (int j = 0; j < object_cols; j++)
        {
            for (int i = 1; i < object_rows; i++)
            {
                float* data_1 = synthesis_wrapped_phase.ptr<float>(i - 1);
                float* data_2 = synthesis_wrapped_phase.ptr<float>(i);
                if ((data_1[j] - data_2[j]) > 2 * pi)
                {
                    data_2[j] = data_2[j] + 2 * pi;
                }
            }
        }
    }

    return synthesis_wrapped_phase;
}
Mat continus_phase_hang(Mat object_hang[3][4])
{
    int object_rows = object_hang[0][0].rows;
    int object_cols = object_hang[0][0].cols;

    //******************第二步***********************//
    //物体图截断相位求解
    Mat wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f1, object_hang[0], object_rows, object_cols);
    Mat wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f2, object_hang[1], object_rows, object_cols);
    Mat wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    true_phase(wrapped_phase_f3, object_hang[2], object_rows, object_cols);

    //******************第三步***********************//
    float f_12 = f[0] - f[1];//f1和f2的频差
    float f_23 = f[1] - f[2];
    float f_13 = f[0] - f[2];
    float f_123 = f_12 - f_23;


    //物体图三频外差
    Mat object_12 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f2, object_12, object_rows, object_cols);
    Mat object_23 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(wrapped_phase_f2, wrapped_phase_f3, object_23, object_rows, object_cols);
    Mat object_13 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f3, object_13, object_rows, object_cols);
    Mat object_123 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    difference_frequency(object_12, object_23, object_123, object_rows, object_cols);


    //******************第四步***********************//
    //差频解包裹(按照师姐论文推)
    //物体图差频解包裹
    Mat synthesis_wrapped_phase_f12 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(object_12, object_123, synthesis_wrapped_phase_f12, (1.0 / f_12), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f23 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(object_23, object_123, synthesis_wrapped_phase_f23, (1.0 / f_23), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f13 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(object_13, object_123, synthesis_wrapped_phase_f13, (1.0 / f_13), (1.0 / f_123), object_rows, object_cols);

    Mat synthesis_wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(wrapped_phase_f1, synthesis_wrapped_phase_f12, synthesis_wrapped_phase_f1, 1.0 / float(f[0]), (1.0 / f_12), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(wrapped_phase_f2, synthesis_wrapped_phase_f23, synthesis_wrapped_phase_f2, 1.0 / float(f[1]), (1.0 / f_23), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_32FC1);
    Phase_unwrapping(wrapped_phase_f3, synthesis_wrapped_phase_f13, synthesis_wrapped_phase_f3, 1.0 / float(f[2]), (1.0 / f_13), object_rows, object_cols);

    //求平均去误差
    Mat synthesis_wrapped_phase = Mat::zeros(object_rows, object_cols, CV_32FC1);
    for (int k = 0; k < synthesis_wrapped_phase_f1.rows; k++)
    {
        const float* data_1 = synthesis_wrapped_phase_f1.ptr<float>(k);
        const float* data_2 = synthesis_wrapped_phase_f2.ptr<float>(k);
        const float* data_3 = synthesis_wrapped_phase_f3.ptr<float>(k);
        float* out_data = synthesis_wrapped_phase.ptr<float>(k);
        for (int q = 0; q < synthesis_wrapped_phase_f1.cols; q++)
        {
            out_data[q] = (data_1[q] + data_2[q] + data_3[q]) / 3.0;
        }
    }

    GaussianBlur(synthesis_wrapped_phase, synthesis_wrapped_phase, Size(17, 17), 0, 0);
    GaussianBlur(synthesis_wrapped_phase, synthesis_wrapped_phase, Size(17, 17), 0, 0);

    return synthesis_wrapped_phase;
}
//寻找最接近value的值；
int little_vaule(Mat imge, float vaule, int flag)
{
    if (flag == 0)//行
    {
        const float* imge_data = imge.ptr<float>(0);

        float translate_vaule = 0;
        int a = 0;
        for (int i = 0; i < imge.cols; i++)
        {
            if (i == 0)
            {
                translate_vaule = abs(vaule - imge_data[i]);
            }
            else
            {
                float b = abs(vaule - imge_data[i]);
                if (b < translate_vaule)
                {
                    translate_vaule = b;
                    a = i;
                }
            }
        }

        a = a * (1280.0 / 912.0);

        return a;
    }
    else if (flag == 1)//列
    {
        float translate_vaule = 0;
        int a = 0;
        for (int i = 0; i < imge.rows; i++)
        {
            if (i == 0)
            {
                translate_vaule = abs(vaule - imge.at<float>(i, 0));
            }
            else
            {
                float b = abs(vaule - imge.at<float>(i, 0));
                if (b < translate_vaule)
                {
                    translate_vaule = b;
                    a = i;
                }
            }
        }

        a = a * (800.0 / 1140.0);

        return a;
    }
}

void three_dimensional_coordinate(Mat& object_true_phase_images_lie, int roi_y, int roi_x, Mat& depth_image_x, Mat& depth_image_y, Mat& depth_image_z)
{
    int a = roi_y;
    int b = roi_x;

    int count = 0;
    vector <Point3f> points;
    vector <Point3f> translation_points;//过渡

    for (int i = 0; i < object_true_phase_images_lie.rows; i = i + 2)
    {
        for (int j = 0; j < object_true_phase_images_lie.cols; j = j + 2)
        {
            Mat before_matrix = Mat::zeros(3, 3, CV_32FC1);
            Mat after_matrix = Mat::zeros(3, 1, CV_32FC1);
            Mat result_matrix = Mat::zeros(3, 1, CV_32FC1);
            Mat result_matrix_1 = Mat::zeros(4, 1, CV_32FC1);
            float abc = little_vaule(template_hang, object_true_phase_images_lie.at<float>(i, j), 1);

            before_matrix.at<float>(0, 0) = -camera_parameter.at<float>(2, 0)*(j + a) + camera_parameter.at<float>(0, 0);
            before_matrix.at<float>(0, 1) = -camera_parameter.at<float>(2, 1)*(j + a) + camera_parameter.at<float>(0, 1);
            before_matrix.at<float>(0, 2) = -camera_parameter.at<float>(2, 2)*(j + a) + camera_parameter.at<float>(0, 2);

            before_matrix.at<float>(1, 0) = -camera_parameter.at<float>(2, 0)*(i + b) + camera_parameter.at<float>(1, 0);
            before_matrix.at<float>(1, 1) = -camera_parameter.at<float>(2, 1)*(i + b) + camera_parameter.at<float>(1, 1);
            before_matrix.at<float>(1, 2) = -camera_parameter.at<float>(2, 2)*(i + b) + camera_parameter.at<float>(1, 2);

            before_matrix.at<float>(2, 0) = -projector_parameter.at<float>(2, 0)*abc + projector_parameter.at<float>(1, 0);
            before_matrix.at<float>(2, 1) = -projector_parameter.at<float>(2, 1)*abc + projector_parameter.at<float>(1, 1);
            before_matrix.at<float>(2, 2) = -projector_parameter.at<float>(2, 2)*abc + projector_parameter.at<float>(1, 2);

            after_matrix.at<float>(0, 0) = -camera_parameter.at<float>(0, 3) + camera_parameter.at<float>(2, 3)*(j + a);
            after_matrix.at<float>(1, 0) = -camera_parameter.at<float>(1, 3) + camera_parameter.at<float>(2, 3)*(i + b);
            after_matrix.at<float>(2, 0) = -projector_parameter.at<float>(1, 3) + projector_parameter.at<float>(2, 3)*abc;

            solve(before_matrix, after_matrix, result_matrix, DECOMP_SVD);
            //世界坐标系
            depth_image_x.at<float>(i, j) = result_matrix.at<float>(0, 0);
            depth_image_y.at<float>(i, j) = result_matrix.at<float>(1, 0);
            depth_image_z.at<float>(i, j) = result_matrix.at<float>(2, 0);
        }
    }
    for (int i = 0; i < depth_image_z.rows; i = i + 2)
    {
        for (int j = 0; j < depth_image_z.cols; j = j + 2)
        {
            if (depth_image_x.at<float>(i, j) != 0 || depth_image_y.at<float>(i, j) != 0 || depth_image_z.at<float>(i, j) != 0)
            {
                Point3f three_dimensional_point;
                three_dimensional_point.x = depth_image_x.at<float>(i, j);
                three_dimensional_point.y = depth_image_y.at<float>(i, j);
                three_dimensional_point.z = depth_image_z.at<float>(i, j);
                translation_points.push_back(three_dimensional_point);
                count++;
            }
        }
    }

    //精简三维点
    int count_points = 0;
    for (int i = 0; i < count; i++)
    {
        points.push_back(translation_points[i]);
        count_points++;
    }

    //输出三维点到txt文件
    ofstream f_out("3dpoints.txt");
    for (int j = 0; j < translation_points.size(); j++)
    {
        f_out << translation_points[j].x << " " << translation_points[j].y << " " << translation_points[j].z << "\t" << endl;
    }
    f_out.close();

    //pcl
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = count_points;
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);
    //把三维点保存为pcd文件
    for (int i = 0; i < count_points; i++)
    {
        cloud.points[i].x = points[i].x;
        cloud.points[i].y = points[i].y;
        cloud.points[i].z = points[i].z;
    }
    pcl::io::savePCDFileASCII(pcd_path, cloud);
    std::cerr << "Saved " << pcd_path << cloud.points.size() << std::endl;

    //不是位置1的时候旋转平移
    if (strcmp(compare_path, pcd_path) != 0)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::io::loadPCDFile(pcd_path, *source_cloud);
        // 定义三维点云的旋转矩阵
        Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
        float theta = 0; // 角度
        if (angle < 0)
        {
            theta = M_PI * (angle / 180.0); // 角度
        }
        else if (angle > 0)
        {
            theta = -M_PI * (abs(-360.0 + angle) / 180.0); // 角度
        }
        transform_1(0, 0) = cos(theta);
        transform_1(0, 1) = -sin(theta);
        transform_1(1, 0) = sin(theta);
        transform_1(1, 1) = cos(theta);
        transform_1(0, 3) = X;
        transform_1(1, 3) = Y;
        transform_1(2, 3) = Z;

        // 执行变换，并将结果保存在新创建的 transformed_cloud 中
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*source_cloud, *transformed_cloud, transform_1);
        //保存变换的点
        pcl::PCDWriter writer;
        writer.write(pcd_path, *transformed_cloud);
    }

}

Point3f three_dimensional_coordinate_1(Mat& object_true_phase_images_lie, int roi_y, int roi_x, int x, int y)
{
    int a = roi_y;
    int b = roi_x;

    Point3f abcd;

    Mat before_matrix = Mat::zeros(3, 3, CV_32FC1);
    Mat after_matrix = Mat::zeros(3, 1, CV_32FC1);
    Mat result_matrix = Mat::zeros(3, 1, CV_32FC1);
    float abc = little_vaule(template_hang, object_true_phase_images_lie.at<float>(y, x), 1);//投影仪的列坐标

    before_matrix.at<float>(0, 0) = -camera_parameter.at<float>(2, 0)*(x + a) + camera_parameter.at<float>(0, 0);
    before_matrix.at<float>(0, 1) = -camera_parameter.at<float>(2, 1)*(x + a) + camera_parameter.at<float>(0, 1);
    before_matrix.at<float>(0, 2) = -camera_parameter.at<float>(2, 2)*(x + a) + camera_parameter.at<float>(0, 2);

    before_matrix.at<float>(1, 0) = -camera_parameter.at<float>(2, 0)*(y + b) + camera_parameter.at<float>(1, 0);
    before_matrix.at<float>(1, 1) = -camera_parameter.at<float>(2, 1)*(y + b) + camera_parameter.at<float>(1, 1);
    before_matrix.at<float>(1, 2) = -camera_parameter.at<float>(2, 2)*(y + b) + camera_parameter.at<float>(1, 2);

    before_matrix.at<float>(2, 0) = -projector_parameter.at<float>(2, 0)*abc + projector_parameter.at<float>(1, 0);
    before_matrix.at<float>(2, 1) = -projector_parameter.at<float>(2, 1)*abc + projector_parameter.at<float>(1, 1);
    before_matrix.at<float>(2, 2) = -projector_parameter.at<float>(2, 2)*abc + projector_parameter.at<float>(1, 2);

    after_matrix.at<float>(0, 0) = -camera_parameter.at<float>(0, 3) + camera_parameter.at<float>(2, 3)*(x + a);
    after_matrix.at<float>(1, 0) = -camera_parameter.at<float>(1, 3) + camera_parameter.at<float>(2, 3)*(y + b);
    after_matrix.at<float>(2, 0) = -projector_parameter.at<float>(1, 3) + projector_parameter.at<float>(2, 3)*abc;

    solve(before_matrix, after_matrix, result_matrix, DECOMP_SVD);
    abcd.x = result_matrix.at<float>(0, 0);
    abcd.y = result_matrix.at<float>(1, 0);
    abcd.z = result_matrix.at<float>(2, 0);

    return abcd;
}

Mat polyfit(vector<Point2f>& in_point, int n)
{
    int size = in_point.size();
    //所求未知数个数
    int x_num = n + 1;
    //构造矩阵U和Y
    Mat mat_u(size, x_num, CV_64F);
    Mat mat_y(size, 1, CV_64F);

    for (int i = 0; i < mat_u.rows; ++i)
        for (int j = 0; j < mat_u.cols; ++j)
        {
            mat_u.at<float>(i, j) = pow(in_point[i].x, j);
        }

    for (int i = 0; i < mat_y.rows; ++i)
    {
        mat_y.at<float>(i, 0) = in_point[i].y;
    }

    //矩阵运算，获得系数矩阵K
    Mat mat_k(x_num, 1, CV_64F);
    mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
    //cout << mat_k << endl;
    return mat_k;
}


int main()
{
    auto begin = chrono::steady_clock::now();
    //***********************读取内参矩阵和八个参数****************************/
    fstream file1, file2, file3, file4;//创建文件流对象
    file1.open("..\\document\\摄像机单应性矩阵.txt");
    file2.open("..\\document\\投影仪单应性矩阵.txt");
    file3.open("..\\document\\畸变系数.txt");
    file4.open("..\\document\\内参矩阵.txt");

    //将txt文件数据写入到mat矩阵中
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            file1 >> camera_parameter.at<float>(i, j);
            std::cout << camera_parameter.at<float>(i, j) << " ";
        }
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            file2 >> projector_parameter.at<float>(i, j);
            std::cout << projector_parameter.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    log("输出内参矩阵完成,",__LINE__);
    log("下一步", __LINE__);
    for (int j = 0; j < 5; j++)
    {
        file3 >> camera_distortion.at<float>(0, j);
    }
    for (int j = 0; j < 5; j++)
    {
        file3 >> projector_distortion.at<float>(0, j);
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            file4 >> camera_matix.at<float>(i, j);
        }
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            file4 >> projector_matix.at<float>(i, j);
        }
    }
    //**************************************求物体图的连续相位****************************/
    // 提取存储在给定目录中的物体投影图像的路径
    //读入物体图
    for (int i = 0; i < 1; i++)
    {
        char translate_path[100];
        char translate_path1[100];
        char calibration_path[100];
        char location_circle_path[100];
        vector<Mat> translate_picture;
        string s= "..\\pic\\物体图\\螺纹";
        sprintf(translate_path, "J:\\source\\repos\\wxproject\\pic\\物体图\\3_23_2", i + 1);
        for (int j = 0; j < 24; j++)
        {
            sprintf(translate_path1, "\\%d.bmp", translate_path,j + 1);
            sprintf(calibration_path, "%s\\%d.bmp", translate_path, j+1);
            Mat imageInput = imread(calibration_path, 2);
            if (imageInput.empty()) {
                log("物体图没有读入",__LINE__);
            }
            imageInput.convertTo(imageInput, CV_32FC1);
            translate_picture.push_back(imageInput);
        }
        object_phase_images.push_back(translate_picture);
        sprintf(location_circle_path, "%s\\位置%d.bmp", translate_path, i + 1);
        Mat image = imread(location_circle_path, 2);
        if (image.empty()) {
            log("位置图读取错误",__LINE__);
        }
        location_circle_images.push_back(image);
    }
    //创建线程
    object_phaseCapThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)object_phaseCapThreadRun, NULL, 0, &object_phaseCapThreadID);

    /************************************************求取像点三维坐标****************************************************/
    //等待线程返回
    WaitForSingleObject(object_phaseCapThread, INFINITE);
    CloseHandle(object_phaseCapThread);
    object_phaseCapThread = NULL;
    //上面其实只做了两步操作，一个是得到列相同横条纹的相位图，一个是得到物体的相位图
    //求取三维点
    Rect rect(457, 842, 1575, 906);
    Mat roi_image_lie;

    for (int i = 0; i < object_true_phase_images_lie.size(); i++)
    {
        imwrite("test.bmp", object_true_phase_images_lie[i]);
        //存储路径
        sprintf(pcd_path, "..\\document\\part_%d.pcd", i + 1);
        if (i == 0)
        {
            strcpy(compare_path, pcd_path);
        }
        //提取ROI的图
        roi_image_lie = object_true_phase_images_lie[i](rect);
        //通过标志圆旋转平移
        if (i != 0)
        {
            matrix_circle(location_circle_images[0], location_circle_images[i], angle, X, Y, Z, i);
        }

        Mat depth_image_x = Mat::zeros(roi_image_lie.rows, roi_image_lie.cols, CV_32FC1);
        Mat depth_image_y = Mat::zeros(roi_image_lie.rows, roi_image_lie.cols, CV_32FC1);
        Mat depth_image_z = Mat::zeros(roi_image_lie.rows, roi_image_lie.cols, CV_32FC1);
        three_dimensional_coordinate(roi_image_lie, rect.x, rect.y, depth_image_x, depth_image_y, depth_image_z);
    }
    normalize(roi_image_lie, roi_image_lie, 0, 1, NORM_MINMAX);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - begin;
    std::cout << "time: " << elapsed.count() << "us" << std::endl;
    std::cout << "程序执行完成" << endl;

}