// Matice jsou uloženy po řádcích

#include "stdafx.h"

#include "operator/convolution.h"
#include "operator/anisotropic.h"
#include "operator/fft.h"
#include "operator/distortion.h"
#include "operator/histogram.h"

cv::Mat genCircle(float radius)
{
	cv::Mat img(64, 64, CV_64FC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (cv::norm(cv::Vec2i(i, j), cv::Vec2i(32, 32)) <= radius)
			{
				img.at<double>(i, j) = 1.0f;
			}
			else img.at<double>(i, j) = 0.0f;
		}
	}

	return img;
}
void sinusGen()
{
	cv::Mat sinImg(64, 64, CV_64FC1);

	for (int x = 0; x < sinImg.cols; x++)
	{
		double scale = x * M_PI_4 / 4;//(x / (double)sinImg.cols) *2 * M_PI;
		double sinValue = sin(scale);
		sinValue = (sinValue + 1.0) * 0.5; // normalize

		for (int y = 0; y < sinImg.rows; y++)
		{
			sinImg.at<double>(y, x) = sinValue;
		}
	}
}

void cviko1()
{
	/*cv::cvtColor(src_8uc3_img, src_8uc1_img, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	src_8uc1_img.convertTo(src_32fc1_img, CV_32FC1, 1.0f / 255.0f);	// škálovací faktor

	cv::Mat gradient = cv::Mat(50, 256, CV_8UC3);

	for (int y = 0; y < gradient.rows; y++)
	{
	for (int x = 0; x < gradient.cols; x++)
	{
	cv::Vec3b& p = gradient.at<cv::Vec3b>(y, x);
	p[0] = p[1] = p[2] = static_cast<uchar>((x / (float) gradient.cols) * 255);
	}
	}*/
}
void cviko7()
{
	cv::Mat img = cv::imread("images/lena64_bars.png", CV_LOAD_IMAGE_GRAYSCALE), moon;
	cv::imshow("Orig", img);
	double scaleMN = 1.0 / sqrt(img.rows * img.cols);
	img.convertTo(moon, CV_64FC1, (1.0 / 255.0) * scaleMN);

	cv::Mat circle = genCircle(64.0f);

	for (int i = 30; i < 34; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			if (j <= 30 || j >= 34)
			{
				circle.at<double>(i, j) = 0.0f;
			}
		}
	}

	cv::imshow("Circle", circle);

	cv::Mat coef = dft(moon);
	//visualizeDft(coef);

	applyMask(circle, coef);

	cv::Mat restored = idft(coef);

	cv::imshow("Restored", restored);
	cv::moveWindow("Restored", 250, 250);
}
void cviko8()
{
	/*img = cv::imread("images/distorted_window.jpg", CV_LOAD_IMAGE_COLOR), result;
	img.copyTo(result);

	cv::imshow("Original", img);
	cv::imshow("Undistorted", result);

	cv::createTrackbar("K1", "Undistorted", &k1Scaler, 1000, [](int pos, void* data) {
		undistort(img, result, k1Scaler, k2Scaler);
		cv::imshow("Original", img);
		cv::imshow("Undistorted", result);
	});
	cv::createTrackbar("K2", "Undistorted", &k2Scaler, 1000, [](int pos, void* data) {
		undistort(img, result, k1Scaler, k2Scaler);
		cv::imshow("Original", img);
		cv::imshow("Undistorted", result);
	});

	while (true)
		cv::waitKey(0);*/
}

typedef double pixel_t;
typedef unsigned int uint;

int main(int argc, char* argv[])
{
	cv::Mat img = cv::imread("images/uneq.jpg", cv::ImreadModes::IMREAD_GRAYSCALE);

	cv::imshow("Original", img);

	int histogram[256];
	get_histogram(img, histogram);
	equalize_histogram(img, histogram);

	cv::imshow("Equalized", img);
	cv::waitKey(0);

	return 0;
}

