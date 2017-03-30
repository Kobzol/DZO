// Matice jsou uloženy po řádcích

#include "stdafx.h"

#include "dzo/convolution.h"
#include "dzo/anisotropic.h"
#include "dzo/fft.h"
#include "dzo/distortion.h"
#include "dzo/histogram.h"
#include "dzo/transformation.h"
#include "dzo/enhancement.h"

typedef double pixel_t;
typedef unsigned int uint;

void save(const cv::Mat& img, float scale = 1.0f)
{
	cv::Mat saved;
	img.convertTo(saved, CV_8UC3, scale);
	cv::imwrite("image.jpg", saved);
}

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

static cv::Mat img, result;
static int k1Scaler = 0, k2Scaler = 0;

static void cviko1()
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
static void cviko7()
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
static void cviko8()
{
	img = cv::imread("images/distorted_window.jpg", CV_LOAD_IMAGE_COLOR), result;
	img.copyTo(result);

	cv::imshow("Original", img);
	cv::imshow("Undistorted", result);

	cv::createTrackbar("K1", "Undistorted", &k1Scaler, 1000, [](int pos, void* data) {
		undistort(img, result, k1Scaler / 100.0f, k2Scaler / 100.0f);
		cv::imshow("Original", img);
		cv::imshow("Undistorted", result);
	});
	cv::createTrackbar("K2", "Undistorted", &k2Scaler, 1000, [](int pos, void* data) {
		undistort(img, result, k1Scaler / 100.0f, k2Scaler / 100.0f);
		cv::imshow("Original", img);
		cv::imshow("Undistorted", result);
	});

	while (true)
		cv::waitKey(0);
}
static void cviko9()
{
	cv::Mat img = cv::imread("images/uneq.jpg", cv::ImreadModes::IMREAD_GRAYSCALE);

	cv::imshow("Original", img);

	int histogram[256];
	get_histogram(img, histogram);
	equalize_histogram(img, histogram);

	int max = histogram[0];
	int min = histogram[0];
	for (int i = 1; i < 256; i++)
	{
		if (histogram[i] > max)
		{
			max = histogram[i];
		}
		if (histogram[i] < min)
		{
			min = histogram[i];
		}
	}

	cv::Mat histogramImg(256, 256, CV_32FC1);
	for (int i = 0; i < histogramImg.cols; i++)
	{
		float height = static_cast<float>(histogram[i] - min);
		height /= (max - min);

		if (height < 0.0f) height = 0.0f;

		for (int j = 0; j < histogramImg.rows; j++)
		{
			if (j < height * histogramImg.rows)
			{
				histogramImg.at<float>(histogramImg.rows - j - 1, i) = 1.0f;
			}
			else histogramImg.at<float>(histogramImg.rows - j - 1, i) = 0.0f;
		}
	}

	int minCdf = cdf(histogram, 0);
	int maxCdf = minCdf;

	for (int i = 1; i < 256; i++)
	{
		int cdfValue = cdf(histogram, i);
		if (cdfValue > maxCdf)
		{
			maxCdf = cdfValue;
		}
		if (cdfValue < minCdf)
		{
			minCdf = cdfValue;
		}
	}

	cv::Mat cdfImg(256, 256, CV_32FC1);
	for (int i = 0; i < cdfImg.rows; i++)
	{
		float height = static_cast<float>(cdf(histogram, i) - minCdf);
		height /= (maxCdf - minCdf);

		if (height < 0.0f) height = 0.0f;

		for (int j = 0; j < cdfImg.rows; j++)
		{
			if (j < height * cdfImg.rows)
			{
				cdfImg.at<float>(cdfImg.rows - j - 1, i) = 1.0f;
			}
			else cdfImg.at<float>(cdfImg.rows - j - 1, i) = 0.0f;
		}
	}

	cv::imshow("Histogram", histogramImg);
	cv::imshow("CDF", cdfImg);

	cv::imshow("Equalized", img);
	cv::waitKey(0);
}
static void cviko10()
{
	cv::Mat vsb = cv::imread("images/vsb.jpg", cv::IMREAD_ANYCOLOR);
	cv::Mat flag = cv::imread("images/flag.png", cv::IMREAD_ANYCOLOR);

	cv::Mat transformation = cv::Mat::eye(3, 3, CV_64FC1);

	float h = (float)flag.rows;
	float w = (float)flag.cols;

	float h_left_offset = 188;
	float w_right_offset = 167;

	cv::Point2f points[4] = {
		{ 0, 0 },
		{ h - h_left_offset, -5 },
		{ h - h_left_offset - 12.0f, (w - w_right_offset) + 2 },
		{ -30.0f, w - w_right_offset }
	};

	perspectiveManual(transformation, flag.rows, flag.cols, points);
	translate(transformation, 106.0f, 70.0f);

	transform<cv::Vec3b>(vsb, flag, transformation);

	cv::imshow("Transformed", vsb);
	save(vsb);
	cv::waitKey(0);
}

std::vector<float> trace(cv::Mat& img)
{
	std::vector<float> column(img.rows, 0.0f);

	for (int i = 0; i < img.rows; i++)
	{
		float rowValue = 0.0f;
		for (int j = 0; j < img.cols; j++)
		{
			rowValue += img.at<float>(i, j);
		}

		column[i] = rowValue / img.cols;
	}

	return column;
}
void setFromColumn(cv::Mat& img, std::vector<float>& column)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<float>(i, j) = column[i];
		}
	}
}
void addToResult(cv::Mat& inputImg, cv::Mat& resultImg)
{
	for (int i = 0; i < resultImg.rows; i++)
	{
		for (int j = 0; j < resultImg.cols; j++)
		{
			resultImg.at<float>(i, j) += inputImg.at<float>(i, j) / 360.0;
		}
	}
}

int _main(int argc, char* argv[])
{
	cv::Mat img = cv::imread("images/test02.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat flImg(img.rows, img.cols, CV_32FC1), rotatedImg(img.rows, img.cols, CV_32FC1), resultImg(img.rows, img.cols, CV_32FC1), testImg(img.rows, img.cols, CV_32FC1);
	img.convertTo(flImg, CV_32FC1, 1.0 / 255.0);

	for (int i = 0; i < resultImg.rows; i++)
	{
		for (int j = 0; j < resultImg.cols; j++)
		{
			resultImg.at<float>(i, j) = 0.0;
		}
	}

	std::vector<float> projections[360];
	cv::Point2f center(flImg.rows / 2.0f, flImg.cols / 2.0f);

	for (int i = 0; i < 360; i++)
	{
		cv::Mat r = cv::getRotationMatrix2D(center, i, 1.0);
		cv::warpAffine(flImg, rotatedImg, r, flImg.size());

		cv::imshow("Rotated", rotatedImg);
		cv::waitKey(10);

		projections[i] = trace(rotatedImg);
	}

	for (int i = 0; i < 360; i++)
	{
		setFromColumn(testImg, projections[i]);

		cv::Mat r = cv::getRotationMatrix2D(center, 359 - i, 1.0);
		cv::warpAffine(testImg, rotatedImg, r, testImg.size());

		cv::imshow("Rotated", rotatedImg);
		cv::waitKey(10);

		addToResult(rotatedImg, resultImg);
	}

	double min, max;
	cv::minMaxIdx(resultImg, &min, &max);

	for (int i = 0; i < resultImg.rows; i++)
	{
		for (int j = 0; j < resultImg.cols; j++)
		{
			resultImg.at<float>(i, j) = simple_contrast_enhancement<float>(resultImg.at<float>(i, j), min, max);
		}
	}

	cv::imshow("Test", resultImg);
	cv::waitKey(0);

	return 0;
}
