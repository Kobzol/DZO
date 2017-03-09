#include "stdafx.h"

#include <ctime>
#include "ano/segmentation.h"

#define THRESHOLD_LIMIT (128)
#define AREA_SCALE (100)

std::vector<Feature> getSampleObjects()
{
	cv::Mat img = cv::imread("images/train.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask = img.clone();
	cv::Mat colored = cv::Mat::zeros(img.size(), CV_8UC3);

	threshold(img, mask, THRESHOLD_LIMIT);

	cv::Mat thresholded = mask.clone();
	std::vector<Object> objects = index(mask);

	std::vector<Feature> features(3);

	cv::RNG rng(static_cast<unsigned int>(time(nullptr)));
	int i = 0;

	for (Object& obj : objects)
	{
		cv::Point center = centerOfMass(obj);
		int area = getArea(obj);
		std::vector<cv::Point> perimeter = getPerimeter(thresholded, obj);

		double f1 = (perimeter.size() * perimeter.size()) / (double)(AREA_SCALE * area);
		double f2 = momentMinMax(obj);

		features[i / 4].f1 += f1;
		features[i / 4].f2 += f2;

		cv::Vec3b color = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		for (cv::Point& p : obj.points)
		{
			colored.at<cv::Vec3b>(p.x, p.y) = color;
		}

		for (cv::Point& p : perimeter)
		{
			colored.at<cv::Vec3b>(p.x, p.y) = cv::Vec3b(255, 255, 255);
		}

		colored.at<cv::Vec3b>(center.x, center.y) = cv::Vec3b(255, 0, 0);

		std::string areaText = std::to_string(i) + ": " + std::to_string(area);
		std::cerr << f1 << " " << f2 << std::endl;

		cv::putText(colored, areaText, cv::Point(center.y - 20, center.x), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Vec3b(255, 255, 255));
		i++;
	}

	cv::imshow("Colored", colored);

	for (auto& feature : features)
	{
		feature.f1 /= 4.0;
		feature.f2 /= 4.0;
	}

	return features;
}
std::vector<std::pair<Object, Feature>> getObjects(std::string path, cv::Mat& colored)
{
	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	cv::Mat mask = img.clone();
	colored = cv::Mat::zeros(img.size(), CV_8UC3);

	threshold(img, mask, THRESHOLD_LIMIT);

	cv::Mat thresholded = mask.clone();
	std::vector<Object> objects = index(mask);

	std::vector<std::pair<Object, Feature>> features;

	cv::RNG rng(static_cast<unsigned int>(time(nullptr)));
	int i = 0;

	for (Object& obj : objects)
	{
		cv::Point center = centerOfMass(obj);
		int area = getArea(obj);
		std::vector<cv::Point> perimeter = getPerimeter(thresholded, obj);

		std::cerr << "Area: " << area << ", perimeter: " << perimeter.size() << std::endl;

		double f1 = (perimeter.size() * perimeter.size()) / (double)(AREA_SCALE * area);
		double f2 = momentMinMax(obj);

		features.emplace_back(obj, Feature(f1, f2));

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
		std::cerr << f1 << " " << f2 << std::endl;
		i++;
	}

	return features;
}

void cviko4()
{
	cv::Mat colored;

	// square, star, rectangle
	std::vector<Feature> features = getSampleObjects();
	std::vector<std::pair<Object, Feature>> objects = getObjects("images/test02.png", colored);

	std::cerr << std::endl;

	for (auto& p : objects)
	{
		Feature feature = p.second;

		int minIndex = 0;
		for (int i = 0; i < features.size(); i++)
		{
			if (features[i].distance(feature) <= features[minIndex].distance(feature))
			{
				minIndex = i;
			}
			assert(features[i].distance(feature) == feature.distance(features[i]));
			std::cerr << feature << " to " << features[i] << " = " << features[i].distance(feature) << std::endl;
		}

		std::cerr << std::endl;

		cv::Point center = centerOfMass(p.first);
		cv::putText(colored, std::to_string(minIndex), cv::Point(center.y, center.x), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Vec3b(0, 0, 255));
	}

	cv::imshow("Colored", colored);
}
