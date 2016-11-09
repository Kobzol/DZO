#pragma once

#include <opencv2/opencv.hpp>

double taylor(double r, float k1, float k2)
{
	return 1 + k1 * r + k2 * r * r;
}
float getDec(float number)
{
	float _;
	return modf(number, &_);
}
cv::Vec3b bilinearInterpolation(cv::Mat& img, float x, float y)
{
	// ul, ur, ll, lr
	cv::Point2f points[4] = {
		cv::Point2f(std::floor(x), std::floor(y)),
		cv::Point2f(std::floor(x), std::ceil(y)),
		cv::Point2f(std::ceil(x), std::floor(y)),
		cv::Point2f(std::ceil(x), std::ceil(y))
	};

	float coefs[4] = {
		(1 - getDec(x)) * (1 - getDec(y)),
		(1 - getDec(x)) * (getDec(y)),
		(getDec(x)) * (1 - getDec(y)),
		(getDec(x)) * (getDec(y))
	};

	cv::Vec3b result(0, 0, 0);

	for (int i = 0; i < 4; i++)
	{
		cv::Point2f p = points[i];
		if (p.x >= 0 && p.x < img.rows && p.y >= 0 && p.y < img.cols)
		{
			result += coefs[i] * img.at<cv::Vec3b>(p.x, p.y);
		}
	}

	return result;
}
void undistort(cv::Mat& img, cv::Mat& result, float k1, float k2)
{
	k1 /= 100.0f;
	k2 /= 100.0f;

	float halfX = result.rows * 0.5f;
	float halfY = result.cols * 0.5f;
	float R = sqrt(halfX * halfX + halfY * halfY);

	for (int xn = 0; xn < result.rows; xn++)
	{
		for (int yn = 0; yn < result.cols; yn++)
		{
			float xd = xn - halfX;
			float yd = yn - halfY;

			float xDot = xd / R;
			float yDot = yd / R;

			float r2 = xDot * xDot + yDot * yDot;
			float theta = 1.0 / taylor(r2, k1, k2);

			xd *= theta;
			yd *= theta;
			xd += halfX;
			yd += halfY;

			if (xd >= 0 && xd < img.rows && yd >= 0 && yd < img.cols)
			{
				result.at<cv::Vec3b>(xn, yn) = bilinearInterpolation(img, xd, yd);
			}
			else result.at<cv::Vec3b>(xn, yn) = cv::Vec3b(0, 0, 0);
		}
	}
}
