#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>

#include <opencv2/opencv.hpp>

double base(int width, int height, int m, int k, int n, int l)
{
	return 2 * M_PI * (((m * k) / (double)height) + ((n * l) / (double)width));
}

void changeQuadrants(cv::Mat& img)
{
	int width = img.cols;
	int height = img.rows;

	for (int k = 0; k < height / 2; k++)
	{
		for (int l = 0; l < width; l++)
		{
			int x = (l < width / 2) ? (l + width / 2) : (l - width / 2);
			int y = k + height / 2;
			std::swap(img.at<cv::Vec2d>(k, l), img.at<cv::Vec2d>(y, x));
		}
	}
}

void visualizeDft(const cv::Mat& coefficients)
{
	cv::Mat powerImg(coefficients.rows, coefficients.cols, CV_64FC1);
	cv::Mat phaseImg;
	powerImg.copyTo(phaseImg);

	int width = coefficients.cols;
	int height = coefficients.rows;

	for (int k = 0; k < height; k++)
	{
		for (int l = 0; l < width; l++)
		{
			cv::Vec2d coef = coefficients.at<cv::Vec2d>(k, l);
			double real = coef[0];
			double comp = coef[1];

			double spectrum = sqrt(real * real + comp * comp);
			double power = spectrum * spectrum;
			double phase = atan(comp / real);
			double lg = log2(power);
			phaseImg.at<double>(k, l) = phase;
			powerImg.at<double>(k, l) = lg;
		}
	}

	double min, max;
	cv::minMaxLoc(powerImg, &min, &max);
	double scale = max - min;

	for (int k = 0; k < height / 2; k++)
	{
		for (int l = 0; l < width; l++)
		{
			int x = (l < width / 2) ? (l + width / 2) : (l - width / 2);
			int y = k + height / 2;
			std::swap(powerImg.at<double>(k, l), powerImg.at<double>(y, x));
			powerImg.at<double>(k, l) = (powerImg.at<double>(k, l) - min) / scale;
			powerImg.at<double>(y, x) = (powerImg.at<double>(y, x) - min) / scale;
		}
	}

	cv::imshow("Phase", phaseImg);
	cv::imshow("Power", powerImg);
}
cv::Mat dft(const cv::Mat& input)
{
	cv::Mat coeficients(input.rows, input.cols, CV_64FC2);

	int width = input.cols;
	int height = input.rows;

	double scaleMN = 1.0 / sqrt(input.rows * input.cols);
	double sumf = 0, sumF = 0;

	for (int k = 0; k < height; k++)
	{
		for (int l = 0; l < width; l++)
		{
			double real = 0.0, comp = 0.0;

			for (int m = 0; m < height; m++)
			{
				for (int n = 0; n < width; n++)
				{
					double pixel = input.at<double>(m, n);
					double arg = -base(width, height, m, k, n, l);
					real += std::cos(arg) * pixel;
					comp += std::sin(arg) * pixel;
				}
			}

			double value = input.at<double>(k, l) / scaleMN;
			sumf += value * value;
			double len = sqrt(real * real + comp * comp);
			sumF += len * len;

			coeficients.at<cv::Vec2d>(k, l) = cv::Vec2d(real, comp);
		}
	}

	assert(abs(sumf - sumF) < 0.1f);

	return coeficients;
}
cv::Mat idft(const cv::Mat& coefficients)
{
	cv::Mat result(coefficients.rows, coefficients.cols, CV_64FC1);

	int width = coefficients.cols;
	int height = coefficients.rows;

	double scale = 1.0 / sqrt(width * height);

	for (int k = 0; k < height; k++)
	{
		for (int l = 0; l < width; l++)
		{
			double real = 0.0;

			for (int m = 0; m < height; m++)
			{
				for (int n = 0; n < width; n++)
				{
					cv::Vec2d coef = coefficients.at<cv::Vec2d>(m, n);
					double baseReal = scale * std::cos(base(width, height, m, k, n, l));
					double baseComplex = scale * std::sin(base(width, height, m, k, n, l));
					real += (coef[0] * baseReal) - (coef[1] * baseComplex);		// realna cast komplexniho cisla
				}
			}

			result.at<double>(k, l) = real;
		}
	}

	//cv::normalize(result, result, 0, 1, cv::NormTypes::NORM_MINMAX);

	return result;
}
void applyMask(const cv::Mat& mask, cv::Mat& coefficients)
{
	changeQuadrants(coefficients);

	for (int x = 0; x < mask.rows; x++)
	{
		for (int y = 0; y < mask.cols; y++)
		{
			if (mask.at<double>(x, y) < 0.5)
			{
				coefficients.at<cv::Vec2d>(x, y) = cv::Vec2d(0.0, 0.0);
			}
		}
	}

	changeQuadrants(coefficients);
}
void applyMaskCallback(std::function<double(cv::Mat&, int x, int y)> callback, cv::Mat& coefficients)
{
	changeQuadrants(coefficients);

	for (int x = 0; x < coefficients.rows; x++)
	{
		for (int y = 0; y < coefficients.cols; y++)
		{
			coefficients.at<cv::Vec2d>(x, y) *= callback(coefficients, x, y);
		}
	}

	changeQuadrants(coefficients);
}