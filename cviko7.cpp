#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <ctime>

#define K_GAUSSIAN (5)

#define INITIAL_SIGMA (30.0)

static double LAMBDA = 2.5;
static double ALPHA = 0.01;
static double SIGMA_THRESHOLD = 6.0;

//#define PRINT_PIXEL

#define SQR(x) ((x) * (x))

struct Gaussian
{
	Gaussian(double mean, double sigma, double probability)
		: mean(mean), sigma(sigma), probability(probability)
	{

	}

	double mean;
	double sigma;
	double probability;

	double calculateDensity(double value)
	{
		double base = 1.0 / (sigma * std::sqrt(2 * M_PI));
		double exponent = -(SQR(value - mean)) / (2 * SQR(sigma));

		return base * exp(exponent) * this->probability;
	}
	void update(double value, double density, double sum)
	{
		if (sigma < SIGMA_THRESHOLD) return;

		double pkx = density / sum;
		this->probability = (1 - ALPHA) * probability + ALPHA * (pkx);
		
		double ro = (ALPHA * pkx) / probability;

		this->mean = (1 - ro) * mean + ro * value;
		this->sigma = (1 - ro) * SQR(sigma) + ro * (SQR(value - mean));
		this->sigma = std::sqrt(this->sigma);
	}
};
struct PixelMog
{
	std::vector<Gaussian> gaussians;
};

using MogMap = std::vector<PixelMog>;

static int pixelX = 300, pixelY = 300;

static int lambdaInt = 250;
static int alphaInt = 1;
static int sigmaInt = 600;

static void updateGaussians(const cv::Mat& frame, MogMap& mogMap, cv::Mat& output)
{
#pragma omp parallel for
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			PixelMog& mog = mogMap[i * frame.cols + j];

			uchar pixel = frame.at<cv::Vec3b>(i, j).val[0];
			double densities[K_GAUSSIAN];
			double densitySum = 0.0;

			int maximumIndex = 0;
			double densityMax = -10000.0;

			for (int g = 0; g < K_GAUSSIAN; g++)
			{
				double density = mog.gaussians[g].calculateDensity(pixel);
				densities[g] = density;
				densitySum += density;

				if (density > densityMax)
				{
					densityMax = density;
					maximumIndex = g;
				}
			}

			mog.gaussians[maximumIndex].update(pixel, densities[maximumIndex], densitySum);

			double probSum = 0.0;
			for (int g = 0; g < K_GAUSSIAN; g++)
			{
				probSum += mog.gaussians[g].probability;
			}

			for (int g = 0; g < K_GAUSSIAN; g++)
			{
				mog.gaussians[g].probability /= probSum;
			}

			std::sort(mog.gaussians.begin(), mog.gaussians.end(), [](const Gaussian& g1, const Gaussian& g2)
			{
				return g1.probability > g2.probability;
			});

#ifdef PRINT_PIXEL
			if (i == pixelX && j == pixelY)
			{
				std::cerr << mog.gaussians[0].mean << " " << mog.gaussians[0].sigma << std::endl;
			}
#endif

			if (fabs((double) pixel - mog.gaussians[0].mean) > mog.gaussians[0].sigma * LAMBDA)
			{
				output.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
			else output.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
		}
	}
}

void cviko7()
{
	srand((unsigned int) time(nullptr));

	cv::VideoCapture capture("videos/dt_passat.mpg");
	if (!capture.isOpened())
	{
		throw "video not found";
	}

	cv::Mat frame, output;
	capture.read(frame);

	output = frame.clone();

	MogMap mogs(frame.rows * frame.cols);
	
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			for (int g = 0; g < K_GAUSSIAN; g++)
			{
				double mean = (g / (float) K_GAUSSIAN) * 255 + (INITIAL_SIGMA / 2.0);
				mogs[i * frame.cols + j].gaussians.emplace_back(mean, INITIAL_SIGMA, 1.0 / K_GAUSSIAN);
			}
		}
	}

	cv::namedWindow("MOG", 1);
	cv::setMouseCallback("MOG", [](int event, int x, int y, int flags, void* userdata)
	{
		if (event == cv::MouseEventTypes::EVENT_LBUTTONDOWN)
		{
			pixelX = y;
			pixelY = x;
		}
	}, NULL);

	cv::namedWindow("video", 1);

	cv::createTrackbar("Lambda", "video", &lambdaInt, 400, [](int pos, void*data)
	{
		LAMBDA = lambdaInt / 100.0;
	});
	cv::createTrackbar("Alpha", "video", &alphaInt, 100, [](int pos, void*data)
	{
		ALPHA = alphaInt / 100.0;
	});
	cv::createTrackbar("Sigma", "video", &sigmaInt, 600, [](int pos, void*data)
	{
		SIGMA_THRESHOLD = sigmaInt / 100.0;
	});

	while (capture.read(frame))
	{
		updateGaussians(frame, mogs, output);

		cv::imshow("video", frame);
		cv::imshow("MOG", output);
		cv::waitKey(10);
	}
}
