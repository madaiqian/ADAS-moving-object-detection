
#include <iostream>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include <cstring>

using namespace cv;
using namespace std;

#define UNKNOWN_FLOW_THRESH 1e9

const int N = 1500;
int flag[N][N];

Mat prevgray, gray, flow, cflow, frame, pre_frame, img_scale, img_temp, mask = Mat(Size(1, 300), CV_8UC1);;
Size dsize;
vector<Point2f> prepoint, nextpoint;
vector<Point2f> F_prepoint, F_nextpoint;
vector<uchar> state;
vector<float> err;
double dis[N];
int cal = 0;
int width = 100, height = 100;
int rec_width = 40;
int Harris_num = 0;
int flag2 = 0;

///调参点
string file_name = //"C:\\Users\\Administrator\\Desktop\\datasets\\123.avi";
//"C:\\Users\\Administrator\\Desktop\\VID_20151130_201047.mp4";
"C:\\Users\\Administrator\\Desktop\\mod_datasets\\set00\\V001.seq";
//"C:\\Users\\Administrator\\Desktop\\datasets\\set08\\V004.seq";


double vehicle_speed = 1;
double limit_of_check = 2120;
double scale = 1; //设置缩放倍数
int margin = 4; //帧间隔
double limit_dis_epi =2; //距离极线的距离
/// 

// 将int 转换成string 
string itos(int i)
{
	stringstream s;
	s << i;
	return s.str();
}


bool ROI_mod(int x1, int y1)
{
	if (x1 >= width / 16 && x1 <= width - width / 16 && y1 >= height / 3 && y1 <= height - height / 6) return 1;
	return 0;
}

void ready()
{
	//图像预处理
	Harris_num = 0;
	F_prepoint.clear();
	F_nextpoint.clear();
	height = frame.rows*scale;
	width = frame.cols*scale;
	dsize = Size(frame.cols*scale, frame.rows*scale);
	img_scale = Mat(dsize, CV_32SC3);
	img_temp = Mat(dsize, CV_32SC3);
	resize(frame, img_scale, dsize);
	resize(frame, img_temp, dsize);
	cvtColor(img_scale, gray, CV_BGR2GRAY);
	//框框大小
	rec_width = frame.cols / 15;

	cout << " cal :   " << cal << endl;
	cout << "行: " << img_scale.rows << "    列: " << img_scale.cols << endl;
	//equalizeHist(gray, gray); //直方图均衡
	return;
}

void optical_flow_check()
{
	int limit_edge_corner = 5;
	for (int i = 0; i < state.size(); i++)
		if (state[i] != 0)
		{

		   int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
		   int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
		   int x1 = prepoint[i].x, y1 = prepoint[i].y;
		   int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
		   if ((x1 < limit_edge_corner || x1 >= gray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= gray.cols - limit_edge_corner
			|| y1 < limit_edge_corner || y1 >= gray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= gray.rows - limit_edge_corner))
		   {
			   state[i] = 0;
			   continue;
		   }
		double sum_check = 0;
		for (int j = 0; j < 9; j++)
			sum_check += abs(prevgray.at<uchar>(y1 + dy[j], x1 + dx[j]) - gray.at<uchar>(y2 + dy[j], x2 + dx[j]));
		if (sum_check>limit_of_check) state[i] = 0;

		if (state[i])
		 {
			Harris_num++;
			F_prepoint.push_back(prepoint[i]);
			F_nextpoint.push_back(nextpoint[i]);
		 }
		}
	return;
}

bool stable_judge()
{
	int stable_num = 0;
	double limit_stalbe = 0.5;
	for (int i = 0; i < state.size(); i++)
		if (state[i])
		{
		if (sqrt((prepoint[i].x - nextpoint[i].x)*(prepoint[i].x - nextpoint[i].x) + (prepoint[i].y - nextpoint[i].y)*(prepoint[i].y - nextpoint[i].y)) < limit_stalbe) stable_num++;
		}
	if (stable_num*1.0 / Harris_num > 0.2) return 1;
	return 0;
}


int main(int, char**)
{
	VideoCapture cap;
	//cap.open(0);
	cap.open(file_name);

	if (!cap.isOpened())
		return -1;


	for (;;)
	{
		double t = (double)cvGetTickCount();

		cap >> frame;
		if (frame.empty()) break;
		cal++;
		//if (cal <= 3000) continue;

		//图像预处理
		ready();

		//隔margin帧处理一次
		if (cal % margin != 0)
		{
			continue;
		}


		if (prevgray.data)
		{
			//calcOpticalFlowPyrLK光流
			goodFeaturesToTrack(prevgray, prepoint, 200, 0.01, 8, Mat(), 3, true, 0.04);
			cornerSubPix(prevgray, prepoint, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
			calcOpticalFlowPyrLK(prevgray, gray, prepoint, nextpoint, state, err, Size(22, 22), 5, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

			optical_flow_check();

			if (stable_judge())
			{
				//stable_way();
				cout << 1 << endl;
			}
			else
			{
				cout << 0 << endl;
			}

			//找角点
			for (int i = 0; i < state.size(); i++)
			{

				double x1 = prepoint[i].x, y1 = prepoint[i].y;
				double x2 = nextpoint[i].x, y2 = nextpoint[i].y;
				if (state[i] != 0)
				{

					//画出所有角点
					circle(img_scale, nextpoint[i], 3, Scalar(255, 0, 255));
					circle(pre_frame, prepoint[i], 2, Scalar(255, 0, 255));
				}
			}
			cout << Harris_num << endl;



			//-----------------------------计算 F-Matrix
				Mat F = Mat(3, 3, CV_32FC1);
				//F = findFundamentalMat(F_prepoint, F_nextpoint, mask, FM_RANSAC, 2, 0.99);

				double ppp = 110;
				Mat L = Mat(1000, 3, CV_32FC1);
				while (ppp > 5)
				{
					vector<Point2f> F2_prepoint, F2_nextpoint;
					F2_prepoint.clear();
					F2_nextpoint.clear();
					ppp = 0;
					F = findFundamentalMat(F_prepoint, F_nextpoint, mask, FM_RANSAC, 0.1, 0.99);
					//cout << F << endl;
					//computeCorrespondEpilines(F_prepoint,1,F,L);
					for (int i = 0; i < mask.rows; i++)
					{
						if (mask.at<uchar>(i, 0) == 0);
						else
						{
							///circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
							double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
							double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
							double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
							double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B);
							cout << "------:" << dd << "   " << F_prepoint[i].x << "   " << F_prepoint[i].y << endl;
							//cout << "A:  " << A << "   B: " << B << "   C:  " << C << endl;
							ppp += dd;
							if (dd > 0.1)
							{
								circle(pre_frame, F_prepoint[i], 6, Scalar(255, 0, 0), 3);
							}
							else
							{
								F2_prepoint.push_back(F_prepoint[i]);
								F2_nextpoint.push_back(F_nextpoint[i]);
							}
						}
					}

					F_prepoint = F2_prepoint;
					F_nextpoint = F2_nextpoint;
					cout << "--------------       " << ppp << "      ---------------" << endl;
				}









				//T：异常角点集
				vector<Point2f> T;
				T.clear();



				for (int i = 0; i < prepoint.size(); i++)
				{
					if (state[i] != 0)
					{
						double A = F.at<double>(0, 0)*prepoint[i].x + F.at<double>(0, 1)*prepoint[i].y + F.at<double>(0, 2);
						double B = F.at<double>(1, 0)*prepoint[i].x + F.at<double>(1, 1)*prepoint[i].y + F.at<double>(1, 2);
						double C = F.at<double>(2, 0)*prepoint[i].x + F.at<double>(2, 1)*prepoint[i].y + F.at<double>(2, 2);
						double dd = fabs(A*nextpoint[i].x + B*nextpoint[i].y + C) / sqrt(A*A + B*B);

						//画光流
						int x1 = (int)prepoint[i].x, y1 = (int)prepoint[i].y;
						int x2 = (int)nextpoint[i].x, y2 = (int)nextpoint[i].y;
						//if (sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)) < limit_flow) continue;
						line(img_scale, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar{ 255, 255, 0 }, 2);
						line(pre_frame, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar{ 0, 255, 0 }, 1);


						//距离的极线阈值
						if (dd <= limit_dis_epi) continue;
						cout << "dis: " << dd << endl;
						dis[T.size()] = dd;
						T.push_back(nextpoint[i]);


						//画异常角点
						//circle(img_scale, nextpoint[i], 7, Scalar(255, 255, 255),3);
						circle(pre_frame, prepoint[i], 3, Scalar(255, 255, 255), 2);

						//画极线
						if (fabs(B) < 0.0001)
						{
							double xx = C / A, yy = 0;
							double xxx = C / A, yyy = gray.cols;
							line(pre_frame, Point(xx, yy), Point(xxx, yyy), Scalar::all(-1), 0.01);
							flag2++;
							continue; 
						}
						double xx = 0, yy = -C / B;
						double xxx = gray.cols, yyy = -(C + A*gray.cols) / B;
						if (fabs(yy) > 12345 || fabs(yyy) > 12345)
						{
							yy = 0;
							xx = -C / A;
							yyy = gray.rows;
							xxx = -(C + B*yyy) / A;
						}
						line(img_scale, Point(xx, yy), Point(xxx, yyy), Scalar::all(-1), 0.01);
						line(pre_frame, Point(xx, yy), Point(xxx, yyy), Scalar::all(-1), 0.01);


					}
				}


				//画框(一） 异常点
				/*
				int flag[1000];
				memset(flag, 0, sizeof(flag));
				for (int i = 0; i < T.size();i++)
				if (!flag[i] && IOI(T[i].x,T[i].y) )
				{
				int num_t = 0;
				for (int j = 0; j < T.size(); j++)
				{
				double dd = sqrt((T[j].x - T[i].x)*(T[j].x - T[i].x) + (T[j].y - T[i].y)*(T[j].y - T[i].y));
				if (j!=i && dd < rec_width) num_t++;
				}
				if (num_t < 5) continue;
				rectangle(frame, Point(T[i].x * (1 / scale) - rec_width, T[i].y * (1 / scale) + rec_width), Point(T[i].x * (1 / scale) + rec_width, T[i].y * (1 / scale) - rec_width), Scalar(255, 255, 255), 3);
				for (int j = 0; j < T.size(); j++)
				{
				double dd = sqrt((T[j].x - T[i].x)*(T[j].x - T[i].x) + (T[j].y - T[i].y)*(T[j].y - T[i].y));
				if (dd < rec_width) flag[j] = 1;
				}

				}
				*/
				if (1)
				{
					//画框(二） 枚举 mod
					int tt = 10;
					double flag_meiju[100][100];
					memset(flag_meiju, 0, sizeof(flag_meiju));
					for (int i = 0; i < gray.rows / tt; i++)
						for (int j = 0; j < gray.cols / tt; j++)
						{
						double x1 = i*tt + tt / 2;
						double y1 = j*tt + tt / 2;
						for (int k = 0; k < T.size(); k++)
							if (ROI_mod(T[k].x, T[k].y) && sqrt((T[k].x - y1)*(T[k].x - y1) + (T[k].y - x1)*(T[k].y - x1)) < tt*sqrt(2)) flag_meiju[i][j]++;//flag_meiju[i][j] += dis[k];
						}
					double mm = 0;
					int mark_i = 0, mark_j = 0;
					for (int i = 0; i < gray.rows / tt; i++)
						for (int j = 0; j < gray.cols / tt; j++)
							if (ROI_mod(j*tt, i*tt) && flag_meiju[i][j] > mm)
							{
						mark_i = i;
						mark_j = j;
						mm = flag_meiju[i][j];
						if (mm < 2) continue;
						rectangle(frame, Point(mark_j*tt / scale - rec_width, mark_i*tt / scale + rec_width), Point(mark_j*tt / scale + rec_width, mark_i*tt / scale - rec_width), Scalar(0, 255, 255), 3);

							}
					if (mm > 1111) rectangle(frame, Point(mark_j*tt / scale - rec_width, mark_i*tt / scale + rec_width), Point(mark_j*tt / scale + rec_width, mark_i*tt / scale - rec_width), Scalar(0, 255, 255), 3);
					else
					{
						//画框(三） 
						/*
						memset(flag_meiju, 0, sizeof(flag_meiju));
						for (int i = 0; i < gray.rows / tt; i++)
						for (int j = 0; j < gray.cols / tt; j++)
						{
						double x1 = i*tt + tt / 2;
						double y1 = j*tt + tt / 2;
						for (int k = 0; k < T.size(); k++)
						if (ROI_obscale(T[k].x, T[k].y) && sqrt((T[k].x - y1)*(T[k].x - y1) + (T[k].y - x1)*(T[k].y - x1)) < tt*sqrt(2)) flag_meiju[i][j] ++;
						}
						mm = 0;
						mark_i = 0, mark_j = 0;
						for (int i = 0; i < gray.rows / tt; i++)
						for (int j = 0; j < gray.cols / tt; j++)
						if (flag_meiju[i][j] > mm)
						{
						mark_i = i;
						mark_j = j;
						mm = flag_meiju[i][j];
						}
						//rectangle(frame, Point(mark_j*tt / scale - rec_width, mark_i*tt / scale + rec_width), Point(mark_j*tt / scale + rec_width, mark_i*tt / scale - rec_width), Scalar(255, 0, 0), 3);
						*/
					}
				}
				//绘制ROI
				rectangle(frame, Point(width / 16 / scale, height * 5 / 6 / scale), Point((width - width / 16) / scale, height / 3 / scale), Scalar(255, 0, 0), 1, 0);


				//输出结果图
				string a = itos(cal / margin), b = ".jpg";
				imwrite("F:\\data\\result2_" + a + b, pre_frame);
				imwrite("F:\\data\\result3_" + a + b, frame);
				cvNamedWindow("img_scale", 0);
				imshow("img_scale", img_scale);
				cvNamedWindow("pre", 0);
				imshow("pre", pre_frame);
				cvNamedWindow("frame", 0);
				imshow("frame", frame);
			
		}


		if (waitKey(27) >= 0)
			break;
		std::swap(prevgray, gray);
		resize(img_temp, pre_frame, dsize);
		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
		cout << "-----" << flag2 << endl;
	}
	return 0;
}
