#include "stdafx.h"
#include "segmentation.h"

#include <queue>

std::ostream& operator<<(std::ostream& os, Feature& feature)
{
	os << feature.f1 << ", " << feature.f2;
	return os;
}

bool isValid(cv::Mat& img, int i, int j)
{
	return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

void threshold(cv::Mat& source, cv::Mat& dest, float threshold)
{
	for (int i = 0; i < source.rows; i++)
	{
		for (int j = 0; j < source.cols; j++)
		{
			if (source.at<uchar>(i, j) >= threshold)
			{
				dest.at<uchar>(i, j) = THRESHOLD_MARK;
			}
			else dest.at<uchar>(i, j) = 0;
		}
	}
}
static void floodFill(cv::Mat& mask, cv::Mat& visited, int i, int j, Object& object)
{
	std::queue<cv::Point> queue;
	queue.push({ i, j });

	while (!queue.empty())
	{
		cv::Point p = queue.front();
		queue.pop();
		i = p.x, j = p.y;

		if (!isValid(mask, i, j)) continue;
		if (visited.at<uchar>(i, j) == 0 && mask.at<uchar>(i, j) == THRESHOLD_MARK)
		{
			visited.at<uchar>(i, j) = 255;
			object.points.emplace_back(i, j);

			for (int x = -1; x < 2; x++)
			{
				for (int y = -1; y < 2; y++)
				{
					if (x == 0 && y == 0) continue;
					if (isValid(mask, i + x, j + y) && visited.at<uchar>(i + x, j + y) == 0)
					{
						queue.push({ i + x, j + y });
					}
				}
			}
		}
	}
}
std::vector<Object> index(cv::Mat& mask)
{
	std::vector<Object> objects;

	cv::Mat visited = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);

	for (int i = 0; i < mask.rows; i++)
	{
		for (int j = 0; j < mask.cols; j++)
		{
			if (mask.at<uchar>(i, j) == THRESHOLD_MARK && visited.at<uchar>(i, j) == 0)
			{
				Object object;
				floodFill(mask, visited, i, j, object);
				if (object.points.size() > 30)
				{
					objects.push_back(object);
				}
			}
		}
	}

	return objects;
}

int getArea(const Object& object)
{
	return object.points.size();
}
cv::Point2f centerOfMass(const Object& object)
{
	int area = getArea(object);

	float x = 0.0, y = 0.0;
	for (const cv::Point& point : object.points)
	{
		x += point.x;
		y += point.y;
	}

	return cv::Point2f(x / area, y / area);
}
std::vector<cv::Point> getPerimeter(cv::Mat& img, Object& object)
{
	cv::Point2f center = centerOfMass(object);

	std::vector<cv::Point> perimeter;
	cv::Point surround[4] = {
		{ -1, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 0, -1 }
	};

	for (cv::Point& p : object.points)
	{
		bool per = false;
		for (int i = 0; i < 4; i++)
		{
			int x = surround[i].x + p.x;
			int y = surround[i].y + p.y;
			if (isValid(img, x, y))
			{
				if (img.at<uchar>(x, y) != img.at<uchar>(center.x, center.y))
				{
					per = true;
					break;
				}
			}
		}

		if (per)
		{
			perimeter.push_back(p);
		}
	}

	return perimeter;
}

static double moment(int p, int q, Object& object)
{
	double sum = 0.0;
	cv::Point2f center = centerOfMass(object);

	for (auto& point : object.points)
	{
		sum += std::pow(point.x - center.x, p) * std::pow(point.y - center.y, q);
	}

	return sum;
}
double momentMinMax(Object& object)
{
	double mom2_0 = moment(2, 0, object);
	double mom0_2 = moment(0, 2, object);
	double mom1_1 = moment(1, 1, object);

	double umin = 0.5 * (mom2_0 + mom0_2) - 0.5 * std::sqrt(4 * (mom1_1 * mom1_1) + std::pow((mom2_0 - mom0_2), 2));
	double umax = 0.5 * (mom2_0 + mom0_2) + 0.5 * std::sqrt(4 * (mom1_1 * mom1_1) + std::pow((mom2_0 - mom0_2), 2));

	return umin / umax;
}
