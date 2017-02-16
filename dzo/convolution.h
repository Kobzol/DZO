#pragma once

#include <opencv2/core.hpp>

template <typename pixel_t, unsigned int S>
void convolution(cv::Mat& original, cv::Mat& resultImg, double kernel[S][S], double scale = 1.0)
{
	original.copyTo(resultImg);

	int offset = S / 2;
	for (int y = offset; y < original.rows - offset; y++)
	{
		for (int x = offset; x < original.cols - offset; x++)
		{
			pixel_t result = pixel_t();
			for (int i = -offset; i < (offset + 1); i++)
			{
				for (int j = -offset; j < (offset + 1); j++)
				{
					pixel_t pix = original.at<pixel_t>(y + i, x + j);
					pix *= kernel[(i + offset)][(j + offset)];
					result += pix;
				}
			}

			result /= scale;
			resultImg.at<pixel_t>(y, x) = result;
		}
	}
}
