#pragma once

#include <opencv2/opencv.hpp>

#include "interpolation.h"

void translate(cv::Mat& transformation, double x, double y)
{
	cv::Mat translation = cv::Mat(cv::Matx33d(
		1, 0, -x,
		0, 1, -y,
		0, 0, 1
	));

	transformation = transformation * translation;
}
void scale(cv::Mat& transformation, double x, double y, double z = 1.0f)
{
	cv::Mat scale = cv::Mat(cv::Matx33d(
		1.0 / x, 0, 0,
		0, 1.0 / y, 0,
		0, 0, 1.0 / z
	));

	transformation = transformation * scale;
}
void rotate(cv::Mat& transformation, double angle, cv::Point2d center = cv::Point2d(0.0, 0.0))
{
	double radians = -angle * (M_PI / 180.0);
	cv::Mat rotation3x3 = cv::Mat(cv::Matx33d(
		std::cos(radians), -std::sin(radians), 0,
		std::sin(radians), std::cos(radians), 0,
		0, 0, 1
	));

	translate(transformation, -center.x, -center.y);
	transformation = transformation * rotation3x3;
	translate(transformation, center.x, center.y);
}
void perspective(cv::Mat& transformation, int height, int width, cv::Point2f points[4])
{
	cv::Point2f from[4] = {
		cv::Point2f(0.0f, 0.0f),
		cv::Point2f((float) height, 0.0f),
		cv::Point2f((float) height, (float) width),
		cv::Point2f(0.0f, (float) width)
	};

	cv::Mat pers = cv::getPerspectiveTransform(points, from);
	transformation = transformation * pers;
}

cv::Vec2d transformPos(const cv::Vec2d& position, const cv::Mat& transformation)
{
	cv::Mat homPos = cv::Mat(cv::Matx31d(position.val[0], position.val[1], 1.0f));
	cv::Mat transformed = transformation * homPos;

	double x = transformed.at<double>(0, 0);
	double y = transformed.at<double>(1, 0);
	double w = transformed.at<double>(2, 0);

	return cv::Vec2d(x / w, y / w);
}

template <typename T>
void transform(cv::Mat& output, cv::Mat& input, const cv::Mat& transformation)
{
	for (int x = 0; x < output.rows; x++)
	{
		for (int y = 0; y < output.cols; y++)
		{
			cv::Vec2d transformedPos = transformPos(cv::Vec2d(x, y), transformation);
			
			if (transformedPos.val[0] >= 0 &&
				transformedPos.val[0] < input.rows &&
				transformedPos.val[1] >= 0 &&
				transformedPos.val[1] < input.cols)
			{
				output.at<T>(x, y) = bilinearInterpolation<cv::Vec3b>(input, transformedPos.val[0], transformedPos.val[1]);
			}
		}
	}
}
