#pragma once

#include <opencv2/core.hpp>

template <typename T, unsigned int S>
void convolution(cv::Mat& original, cv::Mat& resultImg, T kernel[S][S])
{
	original.copyTo(resultImg);

	int offset = S / 2;
	T scale = 0;
	for (int i = 0; i < S; i++)
	{
		for (int j = 0; j < S; j++)
		{
			scale += kernel[i][j];
		}
	}

	for (int y = offset; y < original.rows - offset; y++)
	{
		for (int x = offset; x < original.cols - offset; x++)
		{
			cv::Vec3i result = cv::Vec3i(0, 0, 0);
			for (int i = -offset; i < (offset + 1); i++)
			{
				for (int j = -offset; j < (offset + 1); j++)
				{
					pixel_t pix = original.at<pixel_t>(y + i, x + j);
					cv::Vec3i c = pix;
					c *= kernel[(i + offset)][(j + offset)];
					result += c;
				}
			}

			if (scale == 0)
			{
				scale = 1;
			}

			result /= scale;
			resultImg.at<pixel_t>(y, x) = result;
		}
	}
}
