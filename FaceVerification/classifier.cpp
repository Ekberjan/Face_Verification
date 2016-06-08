#include "stdafx.h"

#include "classifier.h"
#include "common.h"

#ifdef VERBOSE
#include <iostream>
#endif

Classifier::Classifier()
{
	trained = false;

	svmParameters.svm_type = CvSVM::C_SVC;
	svmParameters.kernel_type = CvSVM::RBF;
	svmParameters.term_crit = cvTermCriteria(CV_TERMCRIT_EPS, 100, 0.001);
	svmParameters.C = 10.0;
	svmParameters.gamma = 0.001953125;
	sigmoidA = -5.2;
	sigmoidB = -0.5;

	minimumPositiveTrainImageCount = 250;
	maximumPositiveTrainImageCount = 500;

	minimumNegativeTrainImageCount = 500;
	maximumNegativeTrainImageCount = 1000;
}

void Classifier::addPosiviteFeatures(std::vector<double> features)
{
	if (positiveTrainFeatures.size() == maximumPositiveTrainImageCount)
	{
		int removeIndex = rand() % maximumPositiveTrainImageCount;
		positiveTrainFeatures.erase(positiveTrainFeatures.begin() + removeIndex);
	}
	positiveTrainFeatures.push_back(features);
}

void Classifier::addNegativeFeatures(std::vector<double> features)
{
	if (negativeTrainFeatures.size() == maximumNegativeTrainImageCount)
	{
		int removeIndex = rand() % maximumNegativeTrainImageCount;
		negativeTrainFeatures.erase(negativeTrainFeatures.begin() + removeIndex);
	}
	negativeTrainFeatures.push_back(features);
}

void Classifier::clearPositiveFeatures()
{
	positiveTrainFeatures.clear();
}

void Classifier::clearNegativeFeatures()
{
	negativeTrainFeatures.clear();
}

bool Classifier::setIntParameter(FaceVerifierIntParameters parameter, int value)
{
	bool returnValue = true;
	switch (parameter) {
	case ClassifierParameterMinimumTrainPositiveImageCount:
		minimumPositiveTrainImageCount = value;
		break;
	case ClassifierParameterMaximumTrainPositiveImageCount:
		maximumPositiveTrainImageCount = value;
		break;
	case ClassifierParameterMinimumTrainNegativeImageCount:
		minimumNegativeTrainImageCount = value;
		break;
	case ClassifierParameterMaximumTrainNegativeImageCount:
		maximumNegativeTrainImageCount = value;
		break;
	default:
		errorString = "Classifier::setIntParameter() : There is no such int parameter! - ";
		errorString.append(std::to_string(parameter));
		returnValue = false;
		break;
	}
	return returnValue;
}

bool Classifier::setDoubleParameter(FaceVerifierDoubleParameters parameter, double value)
{
	bool returnValue = true;
	switch (parameter) {
	case ClassifierParameterC:
		svmParameters.C = value;
		break;
	case ClassifierParameterGamma:
		svmParameters.gamma = value;
		break;
	case ClassifierParameterSigmoidA:
		sigmoidA = value;
		break;
	case ClassifierParameterSigmoidB:
		sigmoidB = value;
		break;
	default:
		errorString = "Classifier::setDoubleParameter() : There is no such double parameter! - ";
		errorString.append(std::to_string(parameter));
		returnValue = false;
		break;
	}
	return returnValue;
}

void Classifier::clearTrain()
{
	trained = false;
	trainFeatures.clear();
	scaledTrainFeatures.clear();
	trainLabels.clear();
}

bool Classifier::train()
{
	if (negativeTrainFeatures.size() == 0)
	{
		std::cout << "Classifier::train() : Negative train features NOT set!" << std::endl;
		return false;
	}
	if (negativeTrainFeatures.size() < minimumNegativeTrainImageCount)
	{
		std::cout << "Classifier::train() : Number of negative train features (" << std::to_string(negativeTrainFeatures.size())
			<< ") is not sufficient (" << std::to_string(minimumNegativeTrainImageCount) << ")!" << std::endl;
		return false;
	}
	if (positiveTrainFeatures.size() < minimumPositiveTrainImageCount)
	{
		std::cout << "Classifier::train() : Number of positive train features (" << std::to_string(positiveTrainFeatures.size()) << ") is not sufficient ("
			<< std::to_string(minimumPositiveTrainImageCount) << ")!" << std::endl;
		return false;
	}
	prepareDataForSVMTraining();
	trained = svm.train(trainFeaturesForSVM, trainLabelsForSVM, cv::Mat(), cv::Mat(), svmParameters);
	if (!trained)
	{
		std::cout << "Classifier::train() : CvSVM failed to train!" << std::endl;
	}
	return trained;
}

bool Classifier::isTrained()
{
	return trained;
}

double Classifier::predict(std::vector<double> features)
{
	if (!trained)
	{
		return FACEVERIFIER_PREDICT_RESULT_SYSTEM_NOT_TRAINED;
	}
	std::vector<double> scaledFeatures = scaleFeatures(features);
	predictFeaturesForSVM = vectorToMat(scaledFeatures);

	//double result = svm.predict(predictFeaturesForSVM, false); //if we use this line svm will return a decision
	//return result;

	double signedDistance = svm.predict(predictFeaturesForSVM, true); //if we use this line svm will return signed distance to the margin
	double score = sigmoidPredict(signedDistance, sigmoidA, sigmoidB); //if we use this line signed distance will be converted to score
	return score;
}

std::string Classifier::errorMessage()
{
	return errorString;
}

void Classifier::prepareDataForSVMTraining()
{
	trainFeatures.clear();
	trainLabels.clear();

	int positiveTrainSampleStep = floor(((double)positiveTrainFeatures.size()) / double(minimumPositiveTrainImageCount));
	int negativeTrainSampleStep = floor(((double)negativeTrainFeatures.size()) / double(minimumNegativeTrainImageCount));

	for (int i = 0; i < positiveTrainFeatures.size(); i += positiveTrainSampleStep)
	{
		trainFeatures.push_back(positiveTrainFeatures[i]);
		trainLabels.push_back(1.0);
	}

	for (int i = 0; i < negativeTrainFeatures.size(); i += negativeTrainSampleStep)
	{
		trainFeatures.push_back(negativeTrainFeatures[i]);
		trainLabels.push_back(-1.0);
	}

	calculateScaleParameters();
	scaledTrainFeatures = scaleFeatures(trainFeatures);
	trainFeaturesForSVM = vectorToMat(scaledTrainFeatures);
	trainLabelsForSVM = vectorToMat(trainLabels);
}

void Classifier::calculateScaleParameters()
{
	for (int i = 0; i < trainFeatures.size(); i++)
	{
		std::vector<double> trainFeature = trainFeatures[i];
		if (i == 0)
		{
			scaleMinimum = trainFeature;
			scaleMaximum = trainFeature;
		}
		else
		{
			for (int j = 0; j < trainFeature.size(); j++)
			{
				if (trainFeature[j] < scaleMinimum[j])
				{
					scaleMinimum[j] = trainFeature[j];
				}
				else if (trainFeature[j] > scaleMaximum[j])
				{
					scaleMaximum[j] = trainFeature[j];
				}
			}
		}
	}
}

std::vector<double> Classifier::scaleFeatures(std::vector<double> data)
{
	std::vector<double> dataScaled;
	for (int j = 0; j < data.size(); j++)
	{
		if (scaleMinimum[j] == scaleMaximum[j])
		{
			dataScaled.push_back(scaleMinimum[j]);
		}
		else if (data[j] == scaleMinimum[j])
		{
			dataScaled.push_back(SVM_LOWER);
		}
		else if (data[j] == scaleMaximum[j])
		{
			dataScaled.push_back(SVM_UPPER);
		}
		else
		{
			dataScaled.push_back(SVM_LOWER + (SVM_UPPER - SVM_LOWER) * (data[j] - scaleMinimum[j]) / (scaleMaximum[j] - scaleMinimum[j]));
		}
	}
	return dataScaled;
}

std::vector<std::vector<double> > Classifier::scaleFeatures(std::vector<std::vector<double> > data)
{
	std::vector<std::vector<double> > dataScaled;

	for (int i = 0; i < data.size(); i++)
	{
		std::vector<double> dataRow = data[i];
		std::vector<double> dataRowScaled;
		for (int j = 0; j < dataRow.size(); j++)
		{
			if (scaleMinimum[j] == scaleMaximum[j])
			{
				dataRowScaled.push_back(scaleMinimum[j]);
			}
			else if (dataRow[j] == scaleMinimum[j])
			{
				dataRowScaled.push_back(SVM_LOWER);
			}
			else if (dataRow[j] == scaleMaximum[j])
			{
				dataRowScaled.push_back(SVM_UPPER);
			}
			else
			{
				dataRowScaled.push_back(SVM_LOWER + (SVM_UPPER - SVM_LOWER) * (dataRow[j] - scaleMinimum[j]) / (scaleMaximum[j] - scaleMinimum[j]));
			}
		}
		dataScaled.push_back(dataRowScaled);
	}
	return dataScaled;
}

cv::Mat Classifier::vectorToMat(std::vector<double> data)
{
	int size = data.size();
	//    float *labels = new float[size];
	//    std::vector<double>::iterator it;
	//    int i = 0;
	//    for(it = data.begin(); it < data.end(); it++)
	//    {
	//        labels[i] = (float)*it;
	//        ++i;
	//    }

	//    cv::Mat label(size, 1, CV_32FC1, labels);
	//    delete labels;
	//    return label;

	cv::Mat mat(size, 1, CV_32F);
	for (int i = 0; i < size; ++i)
	{
		mat.at<float>(i, 0) = data[i];
	}
	return mat;
}

cv::Mat Classifier::vectorToMat(std::vector<std::vector<double> > data)
{
	int height = data.size();
	std::vector<double> vectorData = data[0];
	int width = vectorData.size();

	//    float *dataInFloat = new float[width*height];

	//    std::vector<double>::iterator widthIterator;
	//    std::vector<std::vector<double> >::iterator heightIterator;
	//    int i = 0;
	//    for(heightIterator = data.begin(); heightIterator < data.end(); heightIterator++ )
	//    {
	//        std::vector<double> oneData = *heightIterator;
	//        for(widthIterator = oneData.begin(); widthIterator < oneData.end(); widthIterator++)
	//        {
	//            dataInFloat[i] = (float)*widthIterator;
	//            ++i;
	//        }
	//    }

	//    cv::Mat mat(height, width, CV_32FC1, dataInFloat);

	//    delete dataInFloat;
	//    return mat;

	cv::Mat mat(height, width, CV_32F);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			mat.at<float>(i, j) = data[i][j];
		}
	}

	return mat;
}

double Classifier::sigmoidPredict(double signedDistance, double A, double B)
{
	double fApB = signedDistance*A + B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
	{
		return 1.0 - (exp(-fApB) / (1.0 + exp(-fApB)));
	}
	else
	{
		return 1.0 - (1.0 / (1.0 + exp(fApB)));
	}
}
