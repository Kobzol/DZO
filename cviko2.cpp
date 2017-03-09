#include "stdafx.h"

#include <queue>

#include "dzo/convolution.h"

#define VISITED (100.0f)

static int getQuadrant(float angle)
{
	float absAngle = angle < 0 ? angle + M_PI : angle;
	if (absAngle < M_PI_4) return 1;
	if (absAngle < M_PI_2) return 2;
	if (absAngle < (M_PI * (3.0f / 4.0f))) return 3;

	return 4;
}
static float getAlpha(int quadrant, float angle)
{
	angle = angle < 0 ? angle + M_PI : angle;
	angle = fmod(angle, M_PI_4);

	float alpha = 0.0f;

	if (quadrant % 2 == 1)
	{
		alpha = atan(angle);
	}
	else if (quadrant % 2 == 0)
	{
		angle = M_PI_4 - angle;
		alpha = atan(angle);
		alpha = 1 - alpha;
	}

	return alpha;
}
static void crossFillEdges(cv::Mat& img, cv::Mat& gradient, int y, int x, float lowThreshold, float highThreshold)
{
	if (gradient.at<cv::Vec2f>(y, x).val[0] == VISITED) return;

	std::queue<std::tuple<int, int>> nodes;
	nodes.push(std::make_tuple(y, x));

	while (!nodes.empty())
	{
		std::tuple<int, int> pos = nodes.front();
		nodes.pop();

		if (gradient.at<cv::Vec2f>(y, x).val[0] == VISITED) continue;
		gradient.at<cv::Vec2f>(y, x).val[0] = VISITED;

		y = std::get<0>(pos);
		x = std::get<1>(pos);

		for (int i = -1; i < 2; i++)
		{
			for (int j = -1; j < 2; j++)
			{
				if (i == 0 && j == 0) continue;

				float neighbourGrad = gradient.at<cv::Vec2f>(y + i, x + j).val[1];
				if (neighbourGrad >= lowThreshold && neighbourGrad < highThreshold && gradient.at<cv::Vec2f>(y + i, x + j).val[0] != VISITED)
				{
					gradient.at<cv::Vec2f>(y + i, x + j).val[1] = highThreshold + 0.5f;
					img.at<float>(y + i, x + j) = 1.0f;
					nodes.push(std::make_tuple(y + i, x + j));
				}
			}
		}
	}
}
static void laplaceThreshold(cv::Mat& img, cv::Mat& result, int low, int high)
{
	cv::Mat gradient(img.rows, img.cols, CV_32FC2);

	float lowValue = low / 100.0f;
	float highValue = high / 100.0f;

	// compute and store gradient
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			float gradX = (img.at<float>(y, x - 1) - img.at<float>(y, x + 1)) / 2;
			float gradY = (img.at<float>(y - 1, x) - img.at<float>(y + 1, x)) / 2;
			float angle = atan2(gradY, gradX + 0.001f);
			float gradValue = sqrt(gradX * gradX + gradY * gradY);

			gradient.at<cv::Vec2f>(y, x) = cv::Vec2f(angle, gradValue);
		}
	}

	// check threshold only
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			float gradValue = gradient.at<cv::Vec2f>(y, x).val[1];

			if (gradValue >= highValue)
			{
				result.at<float>(y, x) = 1.0f;
			}
			else if (gradValue >= lowValue)
			{
				gradValue -= lowValue;
				gradValue /= (highValue - lowValue);
				result.at<float>(y, x) = gradValue;
			}
		}
	}

	cv::imshow("ThresholdOnly", result);

	// check threshold and apply edge reduction
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			float angle = gradient.at<cv::Vec2f>(y, x).val[0] + M_PI_2;
			float gradValue = gradient.at<cv::Vec2f>(y, x).val[1];
			float e_plus = 0.0f, e_minus = 0.0f;

			int quadrant = getQuadrant(angle);
			float alpha = getAlpha(quadrant, angle);

			switch (quadrant)
			{
			case 1:
				e_plus = alpha * gradient.at<cv::Vec2f>(y - 1, x + 1).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y, x + 1).val[1];
				e_minus = alpha * gradient.at<cv::Vec2f>(y + 1, x - 1).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y, x - 1).val[1];
				break;
			case 2:
				e_plus = alpha * gradient.at<cv::Vec2f>(y - 1, x).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y - 1, x).val[1];
				e_minus = alpha * gradient.at<cv::Vec2f>(y + 1, x).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y + 1, x - 1).val[1];
				break;
			case 3:
				e_plus = alpha * gradient.at<cv::Vec2f>(y - 1, x - 1).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y - 1, x).val[1];
				e_minus = alpha * gradient.at<cv::Vec2f>(y + 1, x + 1).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y + 1, x).val[1];
				break;
			case 4:
				e_plus = alpha * gradient.at<cv::Vec2f>(y, x - 1).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y - 1, x - 1).val[1];
				e_minus = alpha * gradient.at<cv::Vec2f>(y, x + 1).val[1] + (1 - alpha) * gradient.at<cv::Vec2f>(y + 1, x + 1).val[1];
				break;
			}

			if (gradValue > e_plus && gradValue > e_minus && gradValue >= highValue)
			{
				result.at<float>(y, x) = 1.0f;
			}
			else result.at<float>(y, x) = 0.0f;
		}
	}

	cv::imshow("ThresholdReduction", result);

	// crossfill edges
	for (int y = 1; y < img.rows - 1; y++)
	{
		for (int x = 1; x < img.cols - 1; x++)
		{
			crossFillEdges(result, gradient, y, x, lowValue, highValue);
		}
	}
}
static void double_thresholding()
{
	static cv::Mat img = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32FC1, 1 / 255.0);
	cv::imshow("Crossfill", img);

	static int lowThreshold = 6, highThreshold = 7;
	static cv::Mat laplace = img.clone();

	cv::createTrackbar("Low threshold", "Crossfill", &lowThreshold, 100, [](int pos, void* threshold) {
		*((int*)threshold) = pos;
		laplaceThreshold(img, laplace, lowThreshold, highThreshold);
		cv::imshow("Crossfill", laplace);
	}, &lowThreshold);
	cv::createTrackbar("High threshold", "Crossfill", &highThreshold, 100, [](int pos, void* threshold) {
		*((int*)threshold) = pos;
		laplaceThreshold(img, laplace, lowThreshold, highThreshold);
		cv::imshow("Crossfill", laplace);
	}, &highThreshold);

	laplaceThreshold(img, laplace, lowThreshold, highThreshold);
	cv::imshow("Crossfill", laplace);
}

void cviko2()
{
	double_thresholding();
}
