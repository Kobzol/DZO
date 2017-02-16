#pragma once

#include <opencv2/opencv.hpp>

#include "interpolation.h"

float taylor(float r, float k1, float k2)
{
	return 1.0f + k1 * r + k2 * r * r;
}

void undistort(cv::Mat& img, cv::Mat& result, float k1, float k2)
{
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
			float theta = 1.0f / taylor(r2, k1, k2);

			xd *= theta;
			yd *= theta;
			xd += halfX;
			yd += halfY;

			if (xd >= 0 && xd < img.rows && yd >= 0 && yd < img.cols)
			{
				result.at<cv::Vec3b>(xn, yn) = bilinearInterpolation<cv::Vec3b>(img, xd, yd);
			}
			else result.at<cv::Vec3b>(xn, yn) = cv::Vec3b(0, 0, 0);
		}
	}
}
