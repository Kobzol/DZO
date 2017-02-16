#pragma once

#include "../stdafx.h"

#include <chrono>

const double E = 2.718281828459045;

inline int other(int x)
{
	return 1 - x;
}

double norm(double value)
{
	return value;
}
double g(double sigma, double value)
{
	double n = abs(value);
	double exponent = -((std::pow(n, 2) / (std::pow(sigma, 2))));

	return exp(exponent);
}

void anisotropicFiltering(cv::Mat& img, double sigma, double lambda, int iterations)
{
	cv::Mat tmp;
	img.copyTo(tmp);
	cv::Mat* images[2] = { &tmp, &img };
	int activeImg = 0;

	double total = 0.0;

	for (int t = 0; t < iterations; t++)
	{
		auto start = std::chrono::steady_clock::now();

		int oth = other(activeImg);

		for (int y = 1; y < images[activeImg]->rows - 1; y++)
		{
			for (int x = 1; x < images[activeImg]->cols - 1; x++)
			{
				double orig = images[oth]->at<double>(y, x);
				double n = images[oth]->at<double>(y - 1, x) - orig;
				double s = images[oth]->at<double>(y + 1, x) - orig;
				double w = images[oth]->at<double>(y, x - 1) - orig;
				double e = images[oth]->at<double>(y, x + 1) - orig;

				double cN = g(sigma, norm(n));
				double cS = g(sigma, norm(s));
				double cW = g(sigma, norm(w));
				double cE = g(sigma, norm(e));

				double result = orig + lambda * (cN * n + cS * s + cE * e + cW * w);
				images[activeImg]->at<double>(y, x) = result;
			}
		}

		auto end = (std::chrono::steady_clock::now() - start);
		total += std::chrono::duration_cast<std::chrono::milliseconds>(end).count();

		//cv::imshow("Iteration", *(images[activeImg]));
		//cv::waitKey(1);

		std::cout << t << std::endl;

		activeImg = other(activeImg);
	}

	std::cout << total / iterations << std::endl;
}