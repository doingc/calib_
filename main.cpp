#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "io.h"
#include<Windows.h>
#include <stdio.h>
#include<thread>
#include <opencv2/highgui.hpp>
#include <stdio.h>
using namespace cv;
using namespace std;
//*************************************全局变量********************************************//
//π
double pi = 3.14159265358979323846;

//带相位的标定图片
std::vector<std::vector<Mat>> phase_images;
//列连续相位
std::vector<Mat> true_phase_images_f1_lie;
//行连续相位
std::vector<Mat> true_phase_images_f1_hang;

//列模板，这个的源文件来自于模板图
Mat template_lie;
//行模板
Mat template_hang;

//相位图的频率
double f[3] = { 70,64,59 };
static int countp = 1;
//标定图片的宽高
int biaoding_picture_rows;
int biaoding_picture_cols;

Size image_size;  /* 图像的尺寸 */
Size projector_image_size;  /* 投影仪DMD的尺寸 */
Size board_size = Size(11, 8);    /* 标定板上每行、列的角点数 */
vector<Point2f> image_points_buf;  /* 缓存相机标定每幅图像上检测到的角点 */
vector<vector<Point2f>> image_points_seq; /* 保存相机标定检测到的所有角点 */
vector<vector<Point2f>> projector_image_points_seq; /* 保存投影仪标定检测到的所有角点 */

//计算投影仪坐标点会用到
int heng_row, heng_col;
double ata_x, ata_y;

// 摄像机内参数矩阵 
Mat cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));
Mat projectorMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));

//函数声明
void true_phase(Mat& org_phase, Mat a[4], int m_rows, int m_cols);//计算包裹相位
void Phase_unwrapping(Mat& synthesis_phase_before, Mat& synthesis_phase_after, Mat& continuous_phase, double f_before, double f_after, int m_rows, int m_cols);
void difference_frequency(Mat& high_frequency, Mat& low_frequency, Mat& output, int m_rows, int m_cols);
Mat continus_phase_lie(Mat object_lie[3][4]);
Mat continus_phase_hang(Mat object_hang[3][4]);

int little_vaule(Mat imge, double vaule, int flag);
double mean_vaule(Mat imge, double vaule, int flag);
double nihe_line(Mat& image, int vaule, int zong, int a);

//*************************************线程********************************************//
void log(std::string s) {
    std::cout << s;
}
//求解行标定图真实相位线程
HANDLE phaseCapThread_hang = NULL;
DWORD phaseCapThreadID_hang;
static void WINAPI phaseCapThreadRun_hang();
//该函数主要用来求解hang相位的真实相位
void WINAPI phaseCapThreadRun_hang()
{
    //计算行标定图相位
    char path[100];
    //读入模板图
    vector<Mat> translate_picture;//模板图
    for (int j = 12; j < 24; j++)
    {
        sprintf(path, "..\\pic\\模板图\\%d.bmp", j + 1);
        Mat imageInput = imread(path, 2);
        if (imageInput.empty()) {
            std::cout << "模板图读入出错" << endl;
        }
        imageInput.convertTo(imageInput, CV_64FC1);
        translate_picture.push_back(imageInput);
    }
    Mat translate_hang[3][4];//模板图
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            translate_hang[i][j] = translate_picture[i * 4 + j];
        }
    }
    template_hang = continus_phase_hang(translate_hang);
    for (int t = 0; t < phase_images.size(); t++)
    {
        //将vector转化为数组，标定板图像
        Mat object_hang[3][4];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                object_hang[i][j] = phase_images[t][i * 4 + j + 13];
                GaussianBlur(object_hang[i][j], object_hang[i][j], Size(17, 17), 0, 0);
            }
        }
        Mat continus_fai1 = continus_phase_hang(object_hang);
        true_phase_images_f1_hang.push_back(continus_fai1);
    }
    log("模板图行完成");
}

//求解列标定图真实相位线程
HANDLE phaseCapThread_lie = NULL;
DWORD phaseCapThreadID_lie;
static void WINAPI phaseCapThreadRun_lie();

void WINAPI phaseCapThreadRun_lie()
{
    char path[100];
    vector<Mat> translate_picture;
    for (int j = 0; j < 12; j++)
    {
        sprintf(path, "..\\pic\\模板图\\%d.bmp", j + 1);
        Mat imageInput = imread(path, 2);
        if (imageInput.empty()) {
            std::cout << "error in 126";
            exit(-1);
        }
        imageInput.convertTo(imageInput, CV_64FC1);
        translate_picture.push_back(imageInput);
    }
    Mat translate_lie[3][4];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            translate_lie[i][j] = translate_picture[i * 4 + j];
        }
    }

    template_lie = continus_phase_lie(translate_lie);
    for (int t = 0; t < phase_images.size(); t++)
    {
        //将vector转化为数组
        Mat object_lie[3][4];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                object_lie[i][j] = phase_images[t][i * 4 + j + 1];
                GaussianBlur(object_lie[i][j], object_lie[i][j], Size(17, 17), 0, 0);
            }
        }
        Mat continus_fai1 = continus_phase_lie(object_lie);
        true_phase_images_f1_lie.push_back(continus_fai1);
    }
    log("模板图相位列完成");
}


//求解标定图角点线程
HANDLE calibration_CapThread = NULL;
DWORD calibration_CapThreadID;
static void WINAPI calibration_CapThreadRun();

void WINAPI calibration_CapThreadRun()
{
    //读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
    for (int i = 0; i < phase_images.size(); i++)
    {
        Mat imageInput;
        phase_images[i][0].convertTo(imageInput, CV_8UC1);
        if (imageInput.empty()) {
            std::cout << "test" << endl;
        }
        image_size.width = biaoding_picture_cols;
        image_size.height = biaoding_picture_rows;
        std::cout << "标定第" << i+1 << "张图片" << endl;
        /* 提取角点 */
        if (0 == findChessboardCornersSB(imageInput, board_size, image_points_buf))
        {
            cout << i << "\t " << "can not find chessboard corners!\n" << endl;
        }
        else
        {
            /* 亚像素精确化 */
            //cornerSubPix(imageInput, image_points_buf, Size(11, 11), Size(-1, -1), TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            //find4QuadCornerSubpix(imageInput, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化
            image_points_seq.emplace_back(image_points_buf);  //保存亚像素角点
            /* 在图像上显示角点位置 */
            Mat imageInput1;
            phase_images[i][0].convertTo(imageInput1, CV_8UC3);
            drawChessboardCorners(imageInput1, board_size, image_points_buf, false); //用于在图片中标记角点
            char path_jiao[100];
            sprintf(path_jiao, "..\\pic\\角点图\\camera_%d.bmp", i);
            imwrite(path_jiao, imageInput1);
        }
    }
    log("相机角点图标定完成");
}

//求解投影仪标定线程
HANDLE projectorCapThread = NULL;
DWORD projectorCapThreadID;
static void WINAPI projectorCapThreadRun();

void WINAPI projectorCapThreadRun()
{
    //projector_image_size.width = 912;
    //projector_image_size.height = 1140;
    projector_image_size.width = 1280;
    projector_image_size.height = 800;

    for (int i = 0; i < phase_images.size(); i++)
    {
        //投影仪中的标定转化坐标
        vector<Point2f> projector_image_points_buf;  /* 保存投影仪标定每幅图像上检测到的角点 */
        for (int j = 0; j < image_points_seq[0].size(); j++)
        {
            Point2d pojector_translate_point;
            double a1 = nihe_line(true_phase_images_f1_lie[i], int(image_points_seq[i][j].y), int(image_points_seq[i][j].x), 0);
            double a2 = nihe_line(true_phase_images_f1_hang[i], int(image_points_seq[i][j].x), int(image_points_seq[i][j].y), 1);
            pojector_translate_point.x = little_vaule(template_lie, a1, 0);
            pojector_translate_point.y = little_vaule(template_hang, a2, 1);
            projector_image_points_buf.push_back(pojector_translate_point);
        }
        projector_image_points_seq.push_back(projector_image_points_buf);
        fstream outfile;

        //测试
        Mat test = Mat::zeros(800, 1280, CV_8UC3);
        drawChessboardCorners(test, board_size, projector_image_points_buf, true);
        char path_jiao[100];
        sprintf(path_jiao, "..\\pic\\角点图\\projector_%d.bmp", i);



        imwrite(path_jiao, test);
        std::cout<<"投影仪角点成功找到" + std::to_string(i);
    }
    log("投影仪角点图标定完成");
}
//*************************************函数********************************************//
void true_phase(Mat& org_phase, Mat a[4], int m_rows, int m_cols)	//计算截断相位
{
    for (int k = 0; k < m_rows; k++)
    {
        double* inData = org_phase.ptr<double>(k);
        const double* a0_data = a[0].ptr<double>(k);
        const double* a1_data = a[1].ptr<double>(k);
        const double* a2_data = a[2].ptr<double>(k);
        const double* a3_data = a[3].ptr<double>(k);
        for (int q = 0; q < m_cols; q++)
        {
            inData[q] = atan2((a3_data[q] - a1_data[q]), (a0_data[q] - a2_data[q]));
        }
    }

}
//计算相位差
void difference_frequency(Mat& high_frequency, Mat& low_frequency, Mat& output, int m_rows, int m_cols)//相位差频
{
    for (int k = 0; k < m_rows; k++)
    {
        const double* high_frequency_data = high_frequency.ptr<double>(k);
        const double* low_frequency_data = low_frequency.ptr<double>(k);
        double* output_data = output.ptr<double>(k);

        for (int q = 0; q < m_cols; q++)
        {
            //
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
//相位解包函数
void Phase_unwrapping(Mat& synthesis_phase_before, Mat& synthesis_phase_after, Mat& continuous_phase, double f_before, double f_after, int m_rows, int m_cols)//相位展开函数
{
    Mat k_picture = Mat::zeros(m_rows, m_cols, CV_64FC1);
    for (int i = 0; i < m_rows; i++)
    {
        const double* synthesis_phase_before_data = synthesis_phase_before.ptr<double>(i);
        const double* synthesis_phase_after_data = synthesis_phase_after.ptr<double>(i);
        double* test_data = k_picture.ptr<double>(i);

        for (int j = 0; j < m_cols; j++)
        {
            test_data[j] = round(((f_after / f_before) * synthesis_phase_after_data[j] - synthesis_phase_before_data[j]) / (2.0 * pi));
        }
    }
    for (int i = 0; i < m_rows; i++)
    {
        const double* synthesis_phase_before_data = synthesis_phase_before.ptr<double>(i);
        const double* test_data = k_picture.ptr<double>(i);
        double* continuous_phase_data = continuous_phase.ptr<double>(i);

        for (int j = 0; j < m_cols; j++)
        {
            continuous_phase_data[j] = synthesis_phase_before_data[j] + 2.0 * pi * test_data[j];// 0.2 * pi * test_data[j]出来的结果比2 * pi * test_data[j]平整
        }
    }
}

Mat continus_phase_lie(Mat object_lie[3][4])
{
    int object_rows = object_lie[0][0].rows;
    int object_cols = object_lie[0][0].cols;

    //******************第二步***********************//
    //物体图截断相位求解
    Mat wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    true_phase(wrapped_phase_f1, object_lie[0], object_rows, object_cols);
    Mat wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    true_phase(wrapped_phase_f2, object_lie[1], object_rows, object_cols);
    Mat wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    true_phase(wrapped_phase_f3, object_lie[2], object_rows, object_cols);

    //******************第三步***********************//
    double f_12 = f[0] - f[1];//f1和f2的频差
    double f_23 = f[1] - f[2];
    double f_13 = f[0] - f[2];
    double f_123 = f_12 - f_23;


    //物体图三频外差
    Mat object_12 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f2, object_12, object_rows, object_cols);
    Mat object_23 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    difference_frequency(wrapped_phase_f2, wrapped_phase_f3, object_23, object_rows, object_cols);
    Mat object_13 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f3, object_13, object_rows, object_cols);
    Mat object_123 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    difference_frequency(object_12, object_23, object_123, object_rows, object_cols);


    //******************第四步***********************//
    //差频解包裹(按照师姐论文推)
    //物体图差频解包裹
    Mat synthesis_wrapped_phase_f12 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(object_12, object_123, synthesis_wrapped_phase_f12, (1.0 / f_12), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f23 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(object_23, object_123, synthesis_wrapped_phase_f23, (1.0 / f_23), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f13 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(object_13, object_123, synthesis_wrapped_phase_f13, (1.0 / f_13), (1.0 / f_123), object_rows, object_cols);

    Mat synthesis_wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(wrapped_phase_f1, synthesis_wrapped_phase_f12, synthesis_wrapped_phase_f1, 1.0 / double(f[0]), (1.0 / f_12), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(wrapped_phase_f2, synthesis_wrapped_phase_f23, synthesis_wrapped_phase_f2, 1.0 / double(f[1]), (1.0 / f_23), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(wrapped_phase_f3, synthesis_wrapped_phase_f13, synthesis_wrapped_phase_f3, 1.0 / double(f[2]), (1.0 / f_13), object_rows, object_cols);


    //求平均去误差
    Mat synthesis_wrapped_phase = Mat::zeros(object_rows, object_cols, CV_64FC1);
    for (int k = 0; k < synthesis_wrapped_phase_f1.rows; k++)
    {
        const double* data_1 = synthesis_wrapped_phase_f1.ptr<double>(k);
        const double* data_2 = synthesis_wrapped_phase_f2.ptr<double>(k);
        const double* data_3 = synthesis_wrapped_phase_f3.ptr<double>(k);
        double* out_data = synthesis_wrapped_phase.ptr<double>(k);
        for (int q = 0; q < synthesis_wrapped_phase_f1.cols; q++)
        {
            out_data[q] = (data_1[q] + data_2[q] + data_3[q]) / 3.0;

        }

    }

    GaussianBlur(synthesis_wrapped_phase, synthesis_wrapped_phase, Size(17, 17), 0, 0);
    GaussianBlur(synthesis_wrapped_phase, synthesis_wrapped_phase, Size(17, 17), 0, 0);

    return synthesis_wrapped_phase;
}
//求解真实相位
Mat continus_phase_hang(Mat object_hang[3][4])
{
    int object_rows = object_hang[0][0].rows;
    int object_cols = object_hang[0][0].cols;

    //******************第二步***********************//
    //物体图截断相位求解
    Mat wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    true_phase(wrapped_phase_f1, object_hang[0], object_rows, object_cols);//这幅图是截断相位
    Mat wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    true_phase(wrapped_phase_f2, object_hang[1], object_rows, object_cols);
    Mat wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    true_phase(wrapped_phase_f3, object_hang[2], object_rows, object_cols);

    //******************第三步***********************//
    double f_12 = f[0] - f[1];//f1和f2的频差
    double f_23 = f[1] - f[2];
    double f_13 = f[0] - f[2];
    double f_123 = f_12 - f_23;


    //物体图三频外差，这一步求解的是相位差
    Mat object_12 = Mat::zeros(object_rows, object_cols, CV_64FC1);//1,2的相位差
    difference_frequency(wrapped_phase_f1, wrapped_phase_f2, object_12, object_rows, object_cols);
    Mat object_23 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    difference_frequency(wrapped_phase_f2, wrapped_phase_f3, object_23, object_rows, object_cols);
    Mat object_13 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    difference_frequency(wrapped_phase_f1, wrapped_phase_f3, object_13, object_rows, object_cols);
    Mat object_123 = Mat::zeros(object_rows, object_cols, CV_64FC1);//1,2,3的相位差
    difference_frequency(object_12, object_23, object_123, object_rows, object_cols);


    //******************第四步***********************//
    //差频解包裹(按照师姐论文推)
    //物体图差频解包裹，这一步是合成相位
    Mat synthesis_wrapped_phase_f12 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(object_12, object_123, synthesis_wrapped_phase_f12, (1.0 / f_12), (1.0 / f_123), object_rows, object_cols);//相位解包
    Mat synthesis_wrapped_phase_f23 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(object_23, object_123, synthesis_wrapped_phase_f23, (1.0 / f_23), (1.0 / f_123), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f13 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(object_13, object_123, synthesis_wrapped_phase_f13, (1.0 / f_13), (1.0 / f_123), object_rows, object_cols);

    Mat synthesis_wrapped_phase_f1 = Mat::zeros(object_rows, object_cols, CV_64FC1);//恢复了f1的真实相位
    Phase_unwrapping(wrapped_phase_f1, synthesis_wrapped_phase_f12, synthesis_wrapped_phase_f1, 1.0 / double(f[0]), (1.0 / f_12), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f2 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(wrapped_phase_f2, synthesis_wrapped_phase_f23, synthesis_wrapped_phase_f2, 1.0 / double(f[1]), (1.0 / f_23), object_rows, object_cols);
    Mat synthesis_wrapped_phase_f3 = Mat::zeros(object_rows, object_cols, CV_64FC1);
    Phase_unwrapping(wrapped_phase_f3, synthesis_wrapped_phase_f13, synthesis_wrapped_phase_f3, 1.0 / double(f[2]), (1.0 / f_13), object_rows, object_cols);

    //求平均去误差
    Mat synthesis_wrapped_phase = Mat::zeros(object_rows, object_cols, CV_64FC1);
    for (int k = 0; k < synthesis_wrapped_phase_f1.rows; k++)
    {
        const double* data_1 = synthesis_wrapped_phase_f1.ptr<double>(k);
        const double* data_2 = synthesis_wrapped_phase_f2.ptr<double>(k);
        const double* data_3 = synthesis_wrapped_phase_f3.ptr<double>(k);
        double* out_data = synthesis_wrapped_phase.ptr<double>(k);
        for (int q = 0; q < synthesis_wrapped_phase_f1.cols; q++)
        {
            out_data[q] = (data_1[q] + data_2[q] + data_3[q]) / 3.0;
        }
    }

    GaussianBlur(synthesis_wrapped_phase, synthesis_wrapped_phase, Size(17, 17), 0, 0);
    GaussianBlur(synthesis_wrapped_phase, synthesis_wrapped_phase, Size(17, 17), 0, 0);

    return synthesis_wrapped_phase;
}

int little_vaule(Mat imge, double vaule, int flag)
{
    if (flag == 0)//行
    {
        const double* imge_data = imge.ptr<double>(0);

        double translate_vaule = 0;
        int a = 0;
        for (int i = 0; i < imge.cols; i++)
        {
            if (i == 0)
            {
                translate_vaule = abs(vaule - imge_data[i]);
            }
            else
            {
                double b = abs(vaule - imge_data[i]);
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
        double translate_vaule = 0;
        int a = 0;
        for (int i = 0; i < imge.rows; i++)
        {
            if (i == 0)
            {
                translate_vaule = abs(vaule - imge.at<double>(i, 0));
            }
            else
            {
                double b = abs(vaule - imge.at<double>(i, 0));
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

double mean_vaule(Mat imge, double vaule, int flag)
{
    double a = 0;
    if (flag == 0)//lie
    {

        for (int i = 0; i < imge.rows; i++)
        {
            a = a + imge.at<double>(i, int(vaule));
        }

        a = a / double(imge.rows);


    }
    else if (flag == 1)//hang
    {


        for (int i = 0; i < imge.cols; i++)
        {
            a = a + imge.at<double>(int(vaule), i);
        }

        a = a / double(imge.cols);

    }
    return a;
}
//拟合某条直线
double nihe_line(Mat& image, int vaule, int zong, int a)
{
    if (a == 0)//纵
    {
        int before = 0;
        int after = 0;
        vector<Point2d>points;
        if ((zong - 200) < 0)
        {
            before = 0;
        }
        else
        {
            before = zong - 200;
        }
        if ((zong + 200) > 2591)
        {
            after = 2591;
        }
        else
        {
            after = zong + 200;
        }
        for (int i = 500; i < 1500; i++)
        {
            double b = image.at<double>(vaule, i);
            points.push_back(Point2d(i, b));
        }
        //构建A矩阵
        int N = 2;
        Mat A = Mat::zeros(N, N, CV_64FC1);

        for (int row = 0; row < A.rows; row++)
        {
            for (int col = 0; col < A.cols; col++)
            {
                for (int k = 0; k < points.size(); k++)
                {
                    A.at<double>(row, col) = A.at<double>(row, col) + pow(points[k].x, row + col);
                }
            }
        }
        //构建B矩阵
        Mat B = Mat::zeros(N, 1, CV_64FC1);
        for (int row = 0; row < B.rows; row++)
        {

            for (int k = 0; k < points.size(); k++)
            {
                B.at<double>(row, 0) = B.at<double>(row, 0) + pow(points[k].x, row) * points[k].y;
            }
        }
        //A*X=B
        Mat X;
        solve(A, B, X, DECOMP_LU);
        // y = b + ax;
        double y = X.at<double>(0, 0) + X.at<double>(1, 0) * zong;
        return y;
    }
    else if (a == 1)//横
    {
        int before = 0;
        int after = 0;
        vector<Point2d>points;
        if ((zong - 200) < 0)
        {
            before = 0;
        }
        else
        {
            before = zong - 200;
        }
        if ((zong + 200) > 1943)
        {
            after = 1943;
        }
        else
        {
            after = zong + 200;
        }
        for (int i = 600; i < 1400; i++)
        {
            points.push_back(Point2d(i, image.at<double>(i, vaule)));
        }
        //构建A矩阵
        int N = 2;
        Mat A = Mat::zeros(N, N, CV_64FC1);

        for (int row = 0; row < A.rows; row++)
        {
            for (int col = 0; col < A.cols; col++)
            {
                for (int k = 0; k < points.size(); k++)
                {
                    A.at<double>(row, col) = A.at<double>(row, col) + pow(points[k].x, row + col);
                }
            }
        }
        //构建B矩阵
        Mat B = Mat::zeros(N, 1, CV_64FC1);
        for (int row = 0; row < B.rows; row++)
        {

            for (int k = 0; k < points.size(); k++)
            {
                B.at<double>(row, 0) = B.at<double>(row, 0) + pow(points[k].x, row) * points[k].y;
            }
        }
        //A*X=B
        Mat X;
        solve(A, B, X, DECOMP_LU);
        // y = b + ax;
        double y = X.at<double>(0, 0) + X.at<double>(1, 0) * zong;
        return y;
    }
}

int main()
{
    // 读入图片到phase_images;
    int m = 10;//n是一共多少组标定图
    int n = 25;
    for (int i = 0; i < m; i++)
    {
        char translate_path[100];//第i组棋盘格路径图
        char translate_path1[100];//第i张图
        char calibration_path[100];//具体标定路径图
        vector<Mat> translate_picture;//单标定图像
        sprintf(translate_path, "..\\pic\\标定图\\%d", i + 1);
        for (int j = 0; j < n; j++)
        {
            sprintf(translate_path1, "\\%d.bmp", j);
            sprintf(calibration_path, "%s%s", translate_path, translate_path1);
            Mat imageInput = imread(calibration_path, 2);
            if (imageInput.empty()) {
                std::cout << "读入标定图出错";
                exit(-1);
            }

            //读取宽和高
            if (i == 0 && j == 0)
            {
                biaoding_picture_cols = imageInput.cols;
                biaoding_picture_rows = imageInput.rows;
            }

            imageInput.convertTo(imageInput, CV_64FC1);
            translate_picture.push_back(imageInput);
        }
        phase_images.push_back(translate_picture);
    }
    std::cout << "成功读入所有的标定图片到-->phase_images" << endl;
    //创建线程
    phaseCapThread_hang = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)phaseCapThreadRun_hang, NULL, 0, &phaseCapThreadID_hang);//行的相位图
    phaseCapThread_lie = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)phaseCapThreadRun_lie, NULL, 0, &phaseCapThreadID_lie);//列的相位图
    calibration_CapThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)calibration_CapThreadRun, NULL, 0, &calibration_CapThreadID);//图像角点

    //**************************************求相机的内参和外参****************************/
    //ofstream fout("caliberation_result.txt");  /* 保存标定结果的文件 */
    ofstream parameter1_fout("..\\document\\摄像机单应性矩阵.txt");  /* 保存摄像机单应性矩阵 */
    ofstream parameter2_fout("..\\document\\投影仪单应性矩阵.txt");  /* 保存投影仪单应性矩阵 */
    ofstream parameter3_fout("..\\document\\畸变系数.txt");  /* 保存投影仪和相机畸变系数 */
    ofstream parameter4_fout("..\\document\\内参矩阵.txt");  /* 保存投影仪和相机内参矩阵 */
    //ofstream parameter5_fout("J:\\毕业文件\\document\\RT矩阵.txt");	/* 将之后的世界坐标系的点云转为相机坐标系下的点云 */

    //摄像机和投影仪标定共用的
    /*棋盘三维信息*/
    Size square_size = Size(0.3, 0.3);  /* 实际测量得到的标定板上每个棋盘格的大小 */
    vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */

    //摄像机内外参数
    Mat distCoeffs = Mat(1, 5, CV_64FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    vector<Mat> tvecsMat;  /* 每幅图像的平移向量 */
    vector<Mat> rvecsMat; /* 每幅图像的旋转向量 */
    //投影仪内外参数
    Mat projector_distCoeffs = Mat(1, 5, CV_64FC1, Scalar::all(0)); /* 投影仪的5个畸变系数：k1,k2,p1,p2,k3 */
    vector<Mat> projector_tvecsMat;  /* 每幅图像的平移向量 */
    vector<Mat> projector_rvecsMat; /* 每幅图像的旋转向量 */

    /* 初始化标定板上角点的三维坐标，这里实际上创造了一个z=0的平面以左上角为中心的 */
    int i, j, t;
    for (t = 0; t < phase_images.size(); t++)
    {
        vector<Point3f> tempPointSet;
        for (i = 0; i < board_size.height; i++)
        {
            for (j = 0; j < board_size.width; j++)
            {
                Point3f realPoint;
                /* 假设标定板放在世界坐标系中z=0的平面上 */
                realPoint.x = j * square_size.width;
                realPoint.y = i * square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        object_points.push_back(tempPointSet);
    }
    std::cout << 1 << endl;
    //等待线程返回
    WaitForSingleObject(calibration_CapThread, INFINITE);

    CloseHandle(calibration_CapThread);
    std::cout << "等待标定返回" << endl;
    calibration_CapThread = NULL;
    std::cout << 2 << endl;
    WaitForSingleObject(phaseCapThread_hang, INFINITE);
    CloseHandle(phaseCapThread_hang);
    phaseCapThread_hang = NULL;
    std::cout << "关闭了行相位计算线程" << endl;
    std::cout << 3 << endl;
    WaitForSingleObject(phaseCapThread_lie, INFINITE);
    CloseHandle(phaseCapThread_lie);
    phaseCapThread_lie = NULL;
    std::cout << "关闭了列相位计算线程" << endl;
    //开始投影仪角点采取线程
    projectorCapThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)projectorCapThreadRun, NULL, 0, &projectorCapThreadID);
    std::cout << "投影仪开始标定" << endl;
    /* 摄像机开始标定 */
    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

    //求摄像机的单应性矩阵（以左相机建立世界坐标）
    Mat rotation_matrix = Mat::zeros(3, 3, CV_64FC1); //保存每幅图像的旋转矩阵
    Rodrigues(rvecsMat[0], rotation_matrix);//Rodrigues这个函数将会使rotation_matrix变成64位
    Mat RT_matrix = Mat::zeros(3, 4, CV_64FC1);//RT矩阵
    hconcat(rotation_matrix, tvecsMat[0], RT_matrix);
    std::cout << 5 << endl;
    Mat camera_result = Mat::zeros(3, 4, CV_64FC1);
    camera_result = cameraMatrix * RT_matrix;
    for (int i = 0; i < camera_result.rows; i++)
    {
        for (int j = 0; j < camera_result.cols; j++)
        {
            parameter1_fout << camera_result.at<double>(i, j) << "\t";
        }
        parameter1_fout << std::endl;
    }
    parameter1_fout.close();

    for (int i = 0; i < distCoeffs.rows; i++)
    {
        for (int j = 0; j < distCoeffs.cols; j++)
        {
            parameter3_fout << distCoeffs.at<double>(i, j) << "\t";
        }
        parameter3_fout << std::endl;
    }

    for (int i = 0; i < cameraMatrix.rows; i++)
    {
        for (int j = 0; j < cameraMatrix.cols; j++)
        {
            parameter4_fout << cameraMatrix.at<double>(i, j) << "\t";
        }
        parameter4_fout << std::endl;
    }

    WaitForSingleObject(projectorCapThread, INFINITE);
    CloseHandle(projectorCapThread);
    projectorCapThread = NULL;

    calibrateCamera(object_points, projector_image_points_seq, projector_image_size, projectorMatrix, projector_distCoeffs, projector_rvecsMat, projector_tvecsMat, 0);

    //求投影仪的单应性矩阵
    Mat projector_rotation_matrix = Mat::zeros(3, 3, CV_64FC1); //保存每幅图像的旋转矩阵
    Rodrigues(projector_rvecsMat[0], projector_rotation_matrix);//Rodrigues这个函数将会使rotation_matrix变成64位
    Mat projector_RT_matrix = Mat::zeros(3, 4, CV_64FC1);
    hconcat(projector_rotation_matrix, projector_tvecsMat[0], projector_RT_matrix);

    Mat projector_camera_result = Mat::zeros(3, 4, CV_64FC1);
    projector_camera_result = projectorMatrix * projector_RT_matrix;
    for (int i = 0; i < projector_camera_result.rows; i++)
    {
        for (int j = 0; j < projector_camera_result.cols; j++)
        {
            parameter2_fout << projector_camera_result.at<double>(i, j) << "\t";
        }
        parameter2_fout << std::endl;
    }
    parameter2_fout.close();

    for (int i = 0; i < projector_distCoeffs.rows; i++)
    {
        for (int j = 0; j < projector_distCoeffs.cols; j++)
        {
            parameter3_fout << projector_distCoeffs.at<double>(i, j) << "\t";
        }
        parameter3_fout << std::endl;
    }
    parameter3_fout.close();

    for (int i = 0; i < projectorMatrix.rows; i++)
    {
        for (int j = 0; j < projectorMatrix.cols; j++)
        {
            parameter4_fout << projectorMatrix.at<double>(i, j) << "\t";
        }
        parameter4_fout << std::endl;
    }
    parameter4_fout.close();
    std::cout << "程序完成运行";
    return 0;
}