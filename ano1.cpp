#include "stdafx.h"

#include "dzo/convolution.h"
#include "opencv2/imgproc.hpp"

// x, y gradient, sobel
static void gradient()
{
	cv::Mat img;
	cv::Mat input = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE);
	input.convertTo(img, CV_32FC1, 1 / 255.0);

	cv::Mat gradient(img.rows, img.cols, CV_32FC1);

	// backward x
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			gradient.at<float>(y, x) = img.at<float>(y, x) - img.at<float>(y, x - 1);
		}
	}

	//cv::imshow("Backward x", gradient_x);

	// forward x
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			gradient.at<float>(y, x) = img.at<float>(y, x) - img.at<float>(y, x + 1);
		}
	}

	//cv::imshow("Forward x", gradient_x);

	// central x, magnitude
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			float gradX = (img.at<float>(y, x - 1) - img.at<float>(y, x + 1)) / 2;
			float gradY = (img.at<float>(y - 1, x) - img.at<float>(y + 1, x)) / 2;

			gradient.at<float>(y, x) = sqrt(gradX * gradX + gradY * gradY);
		}
	}

	//cv::imshow("Central x", gradient_x);

	cv::Mat sobel_x_img = img.clone();
	cv::Mat sobel_y_img = img.clone();

	double maskX[3][3] = {
		-1.0, 0.0, 1.0
		-2.0, 0.0, 2.0,
		-1.0, 0.0, 1.0
	};
	convolution<float, 3>(img, sobel_x_img, maskX);
	cv::imshow("Sobel x", sobel_x_img);

	double maskY[3][3] = {
		-1.0, -2.0, -1.0,
		0.0, 0.0, 0.0,
		1.0, 2.0, 1.0
	};
	convolution<float, 3>(img, sobel_y_img, maskY);
	cv::imshow("Sobel y", sobel_y_img);

	cv::Mat magnitudeImg = img.clone();
	cv::magnitude(sobel_x_img, sobel_y_img, magnitudeImg);
	cv::imshow("Magnitude", magnitudeImg);
	
	cv::waitKey(0);
}
static void laplace()
{
	cv::Mat img = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32FC1, 1 / 255.0);

	cv::imshow("Original", img);

	cv::Mat gradient(img.rows, img.cols, CV_32FC1);
	cv::Mat gradientBlur = gradient.clone();
	cv::Mat laplaceColored(img.rows, img.cols, CV_32FC3);

	// laplace
	double laplace[3][3] = {
		0.0, 1.0, 0.0,
		1.0, -4.0, 1.0,
		0.0, 1.0, 0.0
	};
	convolution<float, 3>(img, gradient, laplace);
	cv::imshow("Laplace", gradient);

	// colorize from laplace
	for (int i = 0; i < gradient.rows; i++)
	{
		for (int j = 0; j < gradient.cols; j++)
		{
			float lapValue = gradient.at<float>(i, j);
			if (fabs(lapValue) < 0.005f)
			{
				laplaceColored.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
			}
			else if (lapValue > 0)
			{
				laplaceColored.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 1.0f, 0.0f);
			}
			else if (lapValue < 0)
			{
				laplaceColored.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 1.0f);
			}
		}
	}

	cv::imshow("Colored", laplaceColored);

	// gaussian
	cv::GaussianBlur(img, gradient, cv::Size(3, 3), 5.0);
	cv::imshow("Gaussian", gradient);

	// gaussian + laplace
	convolution<float, 3>(gradient, gradientBlur, laplace);
	cv::imshow("Gaussian + Laplace", gradientBlur);
}

void cviko2();
void cviko3();
void cviko4();

int main(int argc, char* argv[])
{
	cviko4();

	cv::waitKey(0);

	return 0;
}
