#pragma once

#include <vector>
#include <ostream>
#include <opencv2/opencv.hpp>

class Feature
{
public:
	Feature()
	{

	}
	Feature(double f1, double f2) : f1(f1), f2(f2)
	{

	}

	double distance(const Feature& feature)
	{
		return std::pow(feature.f1 - this->f1, 2) + std::pow(feature.f2 - this->f2, 2);
	}

	double f1 = 0.0;
	double f2 = 0.0;
};

struct Object
{
	std::vector<cv::Point> points;
	Feature feature;
};

std::ostream& operator<<(std::ostream& os, Feature& feature);

static bool isValid(cv::Mat& img, int i, int j);

void threshold(cv::Mat& source, cv::Mat& dest, float threshold);
std::vector<Object> index(cv::Mat& mask);
int getArea(const Object& object);
cv::Point2f centerOfMass(const Object& object);
std::vector<cv::Point> getPerimeter(cv::Mat& img, Object& object);

double moment(int p, int q, Object& object);
double momentMinMax(Object& object);

std::vector<Object> getObjects(cv::Mat img, int thresholdLimit);
