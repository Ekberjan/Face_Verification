#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "opencv2/core/core.hpp"

#include "common.h"

class Preprocessor
{
public:
	Preprocessor();
	std::vector<cv::Mat> preprocess(std::vector<cv::Mat>& images);
	cv::Mat preprocess(const cv::Mat& image);

	bool setIntParameter(FaceVerifierIntParameters parameter, int value);
	bool setDoubleParameter(FaceVerifierDoubleParameters parameter, double value);
	bool setBoolParameter(FaceVerifierBoolParameters parameter, bool value);

	std::string errorMessage();
private:
	bool crop;
	int cropPercentageLeft;
	int cropPercentageRight;
	int cropPercentageTop;
	int cropPercentageBottom;
	bool smooth;
	double smoothingFactor;
	bool resize;
	int imageSize;
	bool applyHistogramEqualization;

	std::string errorString;



	
	cv::Rect eyeRegion;


}; //class

#endif // PREPROCESSOR_H
