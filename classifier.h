#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>

#include "opencv2/core/core.hpp"
#include <opencv2/ml/ml.hpp>

#include "common.h"

#define SVM_UPPER 1
#define SVM_LOWER -1

class Classifier
{
public:
	Classifier();
	void addPosiviteFeatures(std::vector<double> features);
	void addNegativeFeatures(std::vector<double> features);
	void clearPositiveFeatures();
	void clearNegativeFeatures();

	bool setIntParameter(FaceVerifierIntParameters parameter, int value);
	bool setDoubleParameter(FaceVerifierDoubleParameters parameter, double value);

	void clearTrain();
	bool train();
	bool isTrained();
	double predict(std::vector<double> features);

	std::string errorMessage();

private:
	int minimumPositiveTrainImageCount;
	int maximumPositiveTrainImageCount;
	int minimumNegativeTrainImageCount;
	int maximumNegativeTrainImageCount;

	std::vector<std::vector<double> > positiveTrainFeatures;
	std::vector<std::vector<double> > negativeTrainFeatures;

	std::vector<double> trainLabels;
	std::vector<std::vector<double> > trainFeatures; // both positive and negative
	std::vector<std::vector<double> > scaledTrainFeatures; // both positive and negative

	cv::Mat trainFeaturesForSVM;
	cv::Mat trainLabelsForSVM;
	cv::Mat predictFeaturesForSVM;

	std::vector<double> scaleMaximum;
	std::vector<double> scaleMinimum;

	CvSVM svm;
	CvSVMParams svmParameters;
	double sigmoidA;
	double sigmoidB;

	bool trained;
	std::string errorString;

	void prepareDataForSVMTraining();
	void calculateScaleParameters();
	std::vector<double> scaleFeatures(std::vector<double> data);
	std::vector<std::vector<double> > scaleFeatures(std::vector<std::vector<double> > data);
	cv::Mat vectorToMat(std::vector<double> data);
	cv::Mat vectorToMat(std::vector<std::vector<double> > data);
	double sigmoidPredict(double signedDistance, double A, double B);
}; //class

#endif // CLASSIFIER_H
