#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <cstring>
#include <cxcore.h>
#include <highgui.h>

#define SADFull 0
#define SADFullLimit 20
#define SADFullRange 1

#define SADCorner 1
#define CornerMAX 200
#define Block_S 2
#define Block_L 10
#define SADCornerLimit 100
 
#define N 1024

using namespace cv;
using namespace std;

int height, width;
CvPoint OpticalFlow[N][N];
 

int main(int, char** argv)
{
	//读入部分采用C++,联想笔记本在用Iplimage读入时缺少驱动
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return -1;
	Mat frame_mat;


	IplImage *frame,*frame_pre=0,*frame_show,*frame_gray;
	while (1)
	{
		cap >> frame_mat;
		resize(frame_mat, frame_mat, Size(frame_mat.cols/2, frame_mat.rows/2));


		frame = &IplImage(frame_mat);
		height = frame->height;
		width = frame->width;
		frame_show = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
		if (!frame) break;

		cvSmooth(frame, frame);
		

		if (frame_pre != NULL)
		{
			//cvSetImageROI(frame, cvRect(0, frame->height / 2, frame->width, frame->height / 2));// 为图像设置ROI区域  
			//cvAddS(frame, cvScalar(100, 100, 100), frame);// 对图像做与运算  
			//cvAdd(pre_frame, frame, frame_temp);
			//cvResetImageROI(frame);// 释放ROI区域  

			uchar* data_pre = (uchar *)frame_pre->imageData;
			uchar* data = (uchar *)frame->imageData;
			uchar* data_show = (uchar *)frame_show->imageData;
			int step = frame->widthStep / sizeof(uchar);
			int channels = frame->nChannels;

         #if  (SADFull)

			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
				{
					int big = INT_MAX;
					for (int dx = -SADFullRange; dx <= SADFullRange; dx++)
						for (int dy = -SADFullRange; dy <= SADFullRange; dy++)
						{
						      int pos_x = dx + i, pos_y = dy + j;
							  if (pos_x < 0 || pos_x >= height || pos_y < 0 || pos_y >= width)
							  {
								  continue;
							  }

							  double dis = 0;
							  for (int k = 0; k < 3; k++)
							  {
								  dis += pow(data[pos_x*step + pos_y*channels + k] - data_pre[i*step + j*channels + k], 2);
							  }

							  dis = sqrt(dis);
							  if (dis < big)
							  {
								  big = dis;
								  OpticalFlow[i][j] = Point(dx, dy);
							  }

						}
					if (big > SADFullLimit)
					{
						for (int k = 0; k < 3; k++)
							data_show[i*step + j*channels + k] = data[i*step + j*channels + k];
					}
					else
					{
						for (int k = 0; k < 3; k++)
							data_show[i*step + j*channels + k] = 255;
					}
				      
				}
         #endif 

         #if (SADCorner)

			IplImage *frame_eig, *frame_temp;
			//cvCornerHarris(pre_frame,)
			frame_gray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
			frame_eig = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, 1);
			frame_temp = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, 1);
			frame_show = cvCloneImage(frame_pre);
			
			cvCvtColor(frame_pre, frame_gray, CV_BGR2GRAY);
			int num_corners = CornerMAX;
			CvPoint2D32f *corners;
			corners =(CvPoint2D32f *)malloc(sizeof(CvPoint2D32f)*(num_corners+10));


			cvGoodFeaturesToTrack(frame_gray, frame_eig, frame_temp, corners,&num_corners,0.01,10);


			for (int i = 0; i<num_corners; i++)
			{
				int x = corners[i].x, y = corners[i].y;
				OpticalFlow[x][y] = cvPoint(0, 0);
				if (x - Block_S < 0 || x + Block_S >= width || y - Block_S < 0 || y + Block_S >= height) continue;

				int big = INT_MAX;

				for (int j = -Block_L; j <= Block_L; j++)
					for (int k = -Block_L; k <= Block_L; k++)
					{
					     int pos_x = x + j, pos_y = y + k;
					     if (pos_x < 0 || pos_x >= width || pos_y < 0 || pos_y >= height) continue;
				    	if (pos_x - Block_S < 0 || pos_x + Block_S >= width || pos_y - Block_S < 0 || pos_y + Block_S >= height) continue;
				       	double dis_block = 0;
			     		for (int h1 = -Block_S; h1 <= Block_S; h1++)
				    		for (int h2 = -Block_S; h2 <= Block_S; h2++)
				    		{
					    	double dis = 0;
					    	for (int p = 0; p < 3; p++)
						    	dis += pow(data[(pos_y + h2)*step + (pos_x + h1)*channels + p] - data_pre[(y + h2)*step + (x + h1)*channels + p], 2);
					    	dis_block += sqrt(dis);
						}
				    	if (dis_block < big)
				     	{
					    	big = dis_block;
					    	OpticalFlow[x][y] = CvPoint(j, k);
				    	}
					}

				//if (big > SADCornerLimit)
				{
					cvLine(frame_show, cvPoint(x, y), cvPoint(x + OpticalFlow[x][y].x, y + OpticalFlow[x][y].y), CV_RGB(0, 255, 0), 1, 8);
				}



				cvCircle(frame_show, cvPoint((int)corners[i].x, (int)corners[i].y), 1, CV_RGB(0, 0, 255), 2, 8);
				//fprintf(fp,"\t%f,%f\n",corners[i].x,corners[i].y);  
			}

         #endif 

			//cvSmooth(frame_show, frame_show);
			cvNamedWindow("result",0);// 创建一个窗口 
			cvShowImage("result", frame_show);// 在窗口中显示图像  
			cvWaitKey(20);// 延时 
		}
		frame_pre = cvCloneImage(frame);
		//frame_temp = cvCloneImage(frame);
	}

	system("pause");
	return 0;
}
