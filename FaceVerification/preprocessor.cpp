#include "stdafx.h"

#include "preprocessor.h"

#include "opencv2\imgproc\imgproc.hpp"

#ifdef VERBOSE
#include <iostream>
#endif

Preprocessor::Preprocessor()
{
	//these are default parameters
	crop = false;
	cropPercentageLeft = 0;
	cropPercentageRight = 0;
	cropPercentageTop = 0;
	cropPercentageBottom = 0;
	smooth = false;
	smoothingFactor = 12.0;
	resize = true;
	imageSize = 64;
	applyHistogramEqualization = false;
}

std::vector<cv::Mat> Preprocessor::preprocess(std::vector<cv::Mat> &images)
{
	std::vector<cv::Mat> preprocessedImages;
	for (int i = 0; i < images.size(); ++i)
	{
		preprocessedImages.push_back(preprocess(images[i]));
	}
	return preprocessedImages;
}

cv::Mat Preprocessor::preprocess(const cv::Mat &image)
{
	cv::Mat preprocessedImage;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, preprocessedImage, CV_BGR2GRAY);
	}
	else
	{
		preprocessedImage = image.clone();
	}

	if (crop)
	{
		int x1 = (int)((double)(preprocessedImage.cols)*cropPercentageLeft / 100.0);
		int x2 = preprocessedImage.cols - (int)((double)(preprocessedImage.cols)*cropPercentageRight / 100.0);
		int y1 = (int)((double)(preprocessedImage.rows)*cropPercentageTop / 100.0);
		int y2 = preprocessedImage.rows - (int)((double)(preprocessedImage.rows)*cropPercentageBottom / 100.0);

		preprocessedImage = preprocessedImage.rowRange(y1, y2).colRange(x1, x2);
	}
	if (smooth)
	{
		int ksize = (int)((double)(preprocessedImage.rows) / smoothingFactor);
		ksize = (ksize % 2 != 0) ? ksize : ksize + 1;
		cv::GaussianBlur(preprocessedImage, preprocessedImage, cv::Size(ksize, ksize), 0);
	}
	if (resize)
	{
		cv::resize(preprocessedImage, preprocessedImage, cv::Size(imageSize, imageSize));
	}
	if (applyHistogramEqualization)
	{
		cv::equalizeHist(preprocessedImage, preprocessedImage);
	}
	return preprocessedImage;
}

bool Preprocessor::setIntParameter(FaceVerifierIntParameters parameter, int value)
{
	bool returnValue = true;
	switch (parameter) {
	case PreprocessorParameterCroppingPercentageLeft:
		cropPercentageLeft = value;
		break;
	case PreprocessorParameterCroppingPercentageRight:
		cropPercentageRight = value;
		break;
	case PreprocessorParameterCroppingPercentageTop:
		cropPercentageTop = value;
		break;
	case PreprocessorParameterCroppingPercentageBottom:
		cropPercentageBottom = value;
		break;
	case PreprocessorParameterNewSize:
		imageSize = value;
		break;
	default:
		errorString = "Preprocessor::setIntParameter() : There is no such int parameter! - ";
		errorString.append(std::to_string(parameter));
		returnValue = false;
		break;
	}
	return returnValue;
}

bool Preprocessor::setDoubleParameter(FaceVerifierDoubleParameters parameter, double value)
{
	bool returnValue = true;
	switch (parameter) {
	case PreprocessorParameterSmoothingFactor:
		smoothingFactor = value;
		break;
	default:
		errorString = "Preprocessor::setDoubleParameter() : There is no such double parameter! - ";
		errorString.append(std::to_string(parameter));
		returnValue = false;
		break;
	}
	return returnValue;
}

bool Preprocessor::setBoolParameter(FaceVerifierBoolParameters parameter, bool value)
{
	bool returnValue = true;
	switch (parameter) {
	case PreprocessorParameterCropping:
		crop = value;
		break;
	case PreprocessorParameterSmoothing:
		smooth = value;
		break;
	case PreprocessorParameterResizing:
		resize = value;
		break;
	case PreprocessorParameterHistogramEqualization:
		applyHistogramEqualization = value;
		break;
	default:
		errorString = "Preprocessor::setBoolParameter() : There is no such bool parameter! - ";
		errorString.append(std::to_string(parameter));
		returnValue = false;
		break;
	}
	return returnValue;
}

std::string Preprocessor::errorMessage()
{
	return errorString;
}
