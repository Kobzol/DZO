#include "stdafx.h"

#include "ano/segmentation.h"
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

void cviko6()
{
	Mat img = cv::imread("images/train.png", IMREAD_GRAYSCALE);
	std::vector<Object> objects = getObjects(img, 128);

	Mat layerSize(1, 3, CV_32SC1);
	layerSize.at<int>(0) = 2;
	layerSize.at<int>(1) = 4;
	layerSize.at<int>(2) = 3;

	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	Ptr<ANN_MLP> net = ANN_MLP::create();
	net->setLayerSizes(layerSize);
	net->setTrainMethod(ANN_MLP::BACKPROP);
	net->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	net->setBackpropMomentumScale(0.5f);
	net->setBackpropWeightScale(0.5f);
	net->setTermCriteria(criteria);

	Mat trainClasses;
	trainClasses.create(objects.size(), 2, CV_32FC1);
	for (int i = 0; i < objects.size(); i++)
	{
		trainClasses.at<float>(i, 0) = objects[i].feature.f1;
		trainClasses.at<float>(i, 1) = objects[i].feature.f2;
	}

	Mat responses(objects.size(), 3, CV_32FC1);
	for (int i = 0; i < objects.size(); i++)
	{
		responses.at<float>(i, 0) = i < 4 ? 1 : 0;
		responses.at<float>(i, 1) = i >= 4 && i < 8 ? 1 : 0;
		responses.at<float>(i, 2) = i >= 8 ? 1 : 0;
	}

	Ptr<TrainData> trainData = TrainData::create(trainClasses, ml::SampleTypes::ROW_SAMPLE, responses);
	net->train(trainData);

	img = cv::imread("images/test02.png", IMREAD_GRAYSCALE);
	std::vector<Object> newObjects = getObjects(img, 128);

	Mat testClasses;
	testClasses.create(newObjects.size(), 2, CV_32FC1);
	for (int i = 0; i < newObjects.size(); i++)
	{
		testClasses.at<float>(i, 0) = newObjects[i].feature.f1;
		testClasses.at<float>(i, 1) = newObjects[i].feature.f2;
	}

	Mat output;
	net->predict(testClasses, output);
}
