#include "stdafx.h"

#include <ctime>
#include "ano/segmentation.h"

#define THRESHOLD_LIMIT (128)

void cviko3()
{
	cv::Mat img = cv::imread("images/train.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask = img.clone();
	cv::Mat outline = cv::Mat(img.rows, img.cols, CV_8UC1).setTo(cv::Scalar(0));
	cv::Mat colored = cv::Mat::zeros(img.size(), CV_8UC3);

	threshold(img, mask, THRESHOLD_LIMIT);

	cv::Mat thresholded = mask.clone();
	std::vector<Object> objects = index(mask);
	
	cv::RNG rng(static_cast<unsigned int>(time(nullptr)));
	int i = 0;
	for (Object& obj : objects)
	{
		cv::Point center = centerOfMass(obj);
		int area = getArea(obj);
		std::vector<cv::Point> perimeter = getPerimeter(thresholded, obj);

		cv::Vec3b color = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		for (cv::Point& p : obj.points)
		{
			colored.at<cv::Vec3b>(p.x, p.y) = color;
		}

		for (cv::Point& p : perimeter)
		{
			colored.at<cv::Vec3b>(p.x, p.y) = cv::Vec3b(255, 255, 255);
		}

		std::string areaText = std::to_string(i) + ": " + std::to_string(area);
		std::cerr << areaText << std::endl;

		cv::putText(colored, areaText, cv::Point(center.y - 20, center.x), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Vec3b(255, 255, 255));
		i++;
	}
	
	cv::imshow("Img", img);
	cv::imshow("Colored", colored);
}
