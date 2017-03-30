#include "stdafx.h"

#include <ctime>
#include "ano/segmentation.h"
#include "ano/kmeans.h"

#define THRESHOLD_LIMIT (128)
#define AREA_SCALE (100)

static std::vector<Feature> getSampleObjects()
{
	cv::Mat img = cv::imread("images/train.png", cv::IMREAD_GRAYSCALE);
	cv::Mat mask = img.clone();
	cv::Mat colored = cv::Mat::zeros(img.size(), CV_8UC3);

	threshold(img, mask, THRESHOLD_LIMIT);

	cv::Mat thresholded = mask.clone();
	std::vector<Object> objects = index(mask);

	std::vector<Feature> features;

	cv::RNG rng(static_cast<unsigned int>(time(nullptr)));
	int i = 0;

	for (Object& obj : objects)
	{
		cv::Point center = centerOfMass(obj);
		int area = getArea(obj);
		std::vector<cv::Point> perimeter = getPerimeter(thresholded, obj);

		double f1 = (perimeter.size() * perimeter.size()) / (double)(AREA_SCALE * area);
		double f2 = momentMinMax(obj);

		features.emplace_back(f1, f2);

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

	KMeans kmeans;
	std::vector<std::vector<Feature>> clusters = kmeans.segment(features, 3);
	std::vector<Feature> results;

	for (int i = 0; i < clusters.size(); i++)
	{
		results.push_back(Feature(0.0, 0.0));

		for (int f = 0; f < clusters[i].size(); f++)
		{
			results[i].f1 += clusters[i][f].f1;
			results[i].f2 += clusters[i][f].f2;
		}

		results[i].f1 /= clusters[i].size();
		results[i].f2 /= clusters[i].size();
	}

	return results;
}
static std::vector<std::pair<Object, Feature>> getObjects(std::string path, cv::Mat& colored)
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

void cviko5()
{
	cv::Mat colored;

	// square, star, rectangle
	std::vector<Feature> features = getSampleObjects();
	std::vector<std::pair<Object, Feature>> objects = getObjects("images/test01.png", colored);

	std::cerr << std::endl;

	for (auto& p : objects)
	{
		Feature feature = p.second;

		int minIndex = 0;
		for (int i = 0; i < features.size(); i++)
		{
			if (features[i].distance(feature) < features[minIndex].distance(feature))
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
