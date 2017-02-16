#pragma once

#include <opencv2/opencv.hpp>

static double getDec(double number)
{
	double _;
	return modf(number, &_);
}

template <typename T>
T linearInterpolation(cv::Mat& img, double x, double y)
{
	return img.at<T>(std::floor(x), std::floor(y));
}

template <typename T>
T bilinearInterpolation(cv::Mat& img, double x, double y)
{
	// ul, ur, ll, lr
	cv::Point2d points[4] = {
		cv::Point2d(std::floor(x), std::floor(y)),
		cv::Point2d(std::floor(x), std::ceil(y)),
		cv::Point2d(std::ceil(x), std::floor(y)),
		cv::Point2d(std::ceil(x), std::ceil(y))
	};

	double coefs[4] = {
		(1 - getDec(x)) * (1 - getDec(y)),
		(1 - getDec(x)) * (getDec(y)),
		(getDec(x)) * (1 - getDec(y)),
		(getDec(x)) * (getDec(y))
	};

	T result;
	for (int i = 0; i < 4; i++)
	{
		cv::Point2d p = points[i];
		if (p.x >= 0 && p.x < img.rows && p.y >= 0 && p.y < img.cols)
		{
			result += coefs[i] * img.at<T>(static_cast<int>(p.x), static_cast<int>(p.y));
		}
	}

	return result;
}
