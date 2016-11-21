#pragma once

#include <opencv2/opencv.hpp>

void get_histogram(cv::Mat& img, int* histogram)
{
	std::memset(histogram, 0, sizeof(int) * 256);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			histogram[img.at<unsigned char>(i, j)]++;
		}
	}
}
int cdf(int* histogram, int i)
{
	int sum = 0;

	for (int j = 0; j <= i; j++)
	{
		sum += histogram[j];
	}

	return sum;
}
void equalize_histogram(cv::Mat& img, int* histogram)
{
	int* min = std::find_if(histogram, histogram + 256, [](int x) -> bool { return x != 0; });

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			unsigned char& pixel = img.at<unsigned char>(i, j);
			float difference = static_cast<float>(cdf(histogram, pixel) - *min);
			difference /= (img.rows * img.cols) - *min;
			difference *= 255;

			pixel = static_cast<unsigned char>(std::round(difference));
		}
	}
}
