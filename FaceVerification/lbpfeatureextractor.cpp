#include "stdafx.h"

#include "lbpfeatureextractor.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "opencv2/imgproc/imgproc.hpp"

#ifdef VERBOSEW
#include <iostream>
#endif

LBPFeatureExtractor::LBPFeatureExtractor()
{
	lbpType = LBPType::OLBP;
	blockSize = 8;
	binCount = 8;
	imageSize = 64;
	featureVectorLength = std::pow(imageSize / blockSize, 2) * binCount;
}

std::vector<double> LBPFeatureExtractor::getFeature(const cv::Mat &image)
{
	cv::Mat lbpImage;
	if (lbpType == LBPType::OLBP)
	{
		lbpImage = OLBP(image);
	}
	else if (lbpType == LBPType::ELBP)
	{
		lbpImage = ELBP(image);
	}
	else if (lbpType == LBPType::VARLBP)
	{
		lbpImage = VARLBP(image);
	}

	return histogram(lbpImage);
}

bool LBPFeatureExtractor::setIntParameter(FaceVerifierIntParameters parameter, int value)
{
	bool returnValue = true;
	switch (parameter) {
	case LBPFeatureExtractorParameterLBPType:
	{
												LBPType temp = lbpType;
												lbpType = static_cast<LBPType>(value);
												if (lbpType < LBPType_First ||
													lbpType > LBPType_Last)
												{
													lbpType = temp;
													errorString = "LBPFeatureExtractor::setIntParameter() : LBPFeatureExtractorParameterLBPType is out of bounds! - ";
													errorString.append(std::to_string(value));
													returnValue = false;
												}
												break;
	}
	case LBPFeatureExtractorParameterBlockSize:
	{
												  blockSize = value;
												  featureVectorLength = std::pow(imageSize / blockSize, 2) * binCount;
												  break;
	}
	case LBPFeatureExtractorParameterBinCount:
	{
												 binCount = value;
												 featureVectorLength = std::pow(imageSize / blockSize, 2) * binCount;
												 break;
	}
	case LBPFeatureExtractorParameterImageSize:
	{
												  imageSize = value;
												  featureVectorLength = std::pow(imageSize / blockSize, 2) * binCount;
												  break;
	}
	default:
	{
			   errorString = "LBPFeatureExtractor::setIntParameter() : There is no such int parameter! - ";
			   errorString.append(std::to_string(parameter));
			   returnValue = false;
			   break;
	}
	}

	return returnValue;
}

std::string LBPFeatureExtractor::errorMessage()
{
	return errorString;
}

template <typename _Tp>
void LBPFeatureExtractor::OLBP_(const cv::Mat& src_, cv::Mat& dst)
{
	cv::Mat src(src_.rows + 2, src_.cols + 2, src_.type(), cv::Scalar(0));
	src_.copyTo(src.rowRange(1, src_.rows + 1).colRange(1, src_.cols + 1));
	dst = cv::Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
	for (int i = 1; i<src.rows - 1; i++)
	{
		for (int j = 1; j<src.cols - 1; j++)
		{
			_Tp center = src.at<_Tp>(i, j);
			unsigned char code = 0;
			code |= (src.at<_Tp>(i - 1, j - 1) > center) << 7;
			code |= (src.at<_Tp>(i - 1, j) > center) << 6;
			code |= (src.at<_Tp>(i - 1, j + 1) > center) << 5;
			code |= (src.at<_Tp>(i, j + 1) > center) << 4;
			code |= (src.at<_Tp>(i + 1, j + 1) > center) << 3;
			code |= (src.at<_Tp>(i + 1, j) > center) << 2;
			code |= (src.at<_Tp>(i + 1, j - 1) > center) << 1;
			code |= (src.at<_Tp>(i, j - 1) > center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}

template <typename _Tp>
void LBPFeatureExtractor::ELBP_(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors)
{
	neighbors = std::max(std::min(neighbors, 31), 1); // set bounds...
	// Note: alternatively you can switch to the new OpenCV Mat_
	// type system to define an unsigned int matrix... I am probably
	// mistaken here, but I didn't see an unsigned int representation
	// in OpenCV's classic typesystem...
	dst = cv::Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
	for (int n = 0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius)* cos(2.0*M_PI*n / static_cast<float>(neighbors));
		float y = static_cast<float>(radius)* -sin(2.0*M_PI*n / static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				float t = w1*src.at<_Tp>(i + fy, j + fx) + w2*src.at<_Tp>(i + fy, j + cx) + w3*src.at<_Tp>(i + cy, j + fx) + w4*src.at<_Tp>(i + cy, j + cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) && (abs(t - src.at<_Tp>(i, j)) > std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

template <typename _Tp>
void LBPFeatureExtractor::VARLBP_(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors)
{
	std::max(std::min(neighbors, 31), 1); // set bounds
	dst = cv::Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32FC1); //! result
	// allocate some memory for temporary on-line variance calculations
	cv::Mat _mean = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
	cv::Mat _delta = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
	cv::Mat _m2 = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
	for (int n = 0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius)* cos(2.0*M_PI*n / static_cast<float>(neighbors));
		float y = static_cast<float>(radius)* -sin(2.0*M_PI*n / static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				float t = w1*src.at<_Tp>(i + fy, j + fx) + w2*src.at<_Tp>(i + fy, j + cx) + w3*src.at<_Tp>(i + cy, j + fx) + w4*src.at<_Tp>(i + cy, j + cx);
				_delta.at<float>(i, j) = t - _mean.at<float>(i, j);
				_mean.at<float>(i, j) = (_mean.at<float>(i, j) + (_delta.at<float>(i, j) / (1.0*(n + 1)))); // i am a bit paranoid
				_m2.at<float>(i, j) = _m2.at<float>(i, j) + _delta.at<float>(i, j) * (t - _mean.at<float>(i, j));
			}
		}
	}
	// calculate result
	for (int i = radius; i < src.rows - radius; i++) {
		for (int j = radius; j < src.cols - radius; j++) {
			dst.at<float>(i - radius, j - radius) = _m2.at<float>(i, j) / (1.0*(neighbors - 1));
		}
	}
}

template <typename _Tp>
std::vector<double> LBPFeatureExtractor::histogram_(const cv::Mat &src)
{
	std::vector<double> featureVector(featureVectorLength, 0);
	cv::Mat imageBlock(blockSize, blockSize, CV_8UC1);
	int index = 0;
	for (int i = 0; i < imageSize; i += blockSize)
	{
		for (int j = 0; j < imageSize; j += blockSize)
		{
			imageBlock = src.rowRange(j, j + blockSize).colRange(i, i + blockSize);
			std::vector<double> featureVectorBlock(binCount, 0);
			for (int ii = 0; ii < imageBlock.rows; ++ii)
			{
				for (int jj = 0; jj < imageBlock.cols; ++jj)
				{
					int bin = (int)ceil((imageBlock.at<_Tp>(ii, jj) + 1.0) / (256.0 / (double)binCount));
					featureVectorBlock[bin - 1] += 1.0 / (double)(blockSize * blockSize);
				}
			}
			for (int k = 0; k < binCount; ++k)
			{
				double n = featureVectorBlock[k];
				// int scaled = n * 10000000000;
				// n = static_cast<double>(scaled)/10000000000.0;
				featureVector[(index * binCount) + k] = n;
			}
			++index;
		}
	}

	return featureVector;
}

void LBPFeatureExtractor::OLBP(const cv::Mat &src_, cv::Mat &dst)
{
	cv::Mat src = src_.clone();;
	if (src.channels() > 1)
	{
		cv::cvtColor(src, src, CV_BGR2GRAY);
	}
	switch (src.type())
	{
	case CV_8SC1: OLBP_<char>(src, dst); break;
	case CV_8UC1: OLBP_<unsigned char>(src, dst); break;
	case CV_16SC1: OLBP_<short>(src, dst); break;
	case CV_16UC1: OLBP_<unsigned short>(src, dst); break;
	case CV_32SC1: OLBP_<int>(src, dst); break;
	case CV_32FC1: OLBP_<float>(src, dst); break;
	case CV_64FC1: OLBP_<double>(src, dst); break;
	}
}

void LBPFeatureExtractor::ELBP(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors)
{
	switch (src.type()) {
	case CV_8SC1: ELBP_<char>(src, dst, radius, neighbors); break;
	case CV_8UC1: ELBP_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1: ELBP_<short>(src, dst, radius, neighbors); break;
	case CV_16UC1: ELBP_<unsigned short>(src, dst, radius, neighbors); break;
	case CV_32SC1: ELBP_<int>(src, dst, radius, neighbors); break;
	case CV_32FC1: ELBP_<float>(src, dst, radius, neighbors); break;
	case CV_64FC1: ELBP_<double>(src, dst, radius, neighbors); break;
	}
}

void LBPFeatureExtractor::VARLBP(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors)
{
	switch (src.type()) {
	case CV_8SC1: VARLBP_<char>(src, dst, radius, neighbors); break;
	case CV_8UC1: VARLBP_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1: VARLBP_<short>(src, dst, radius, neighbors); break;
	case CV_16UC1: VARLBP_<unsigned short>(src, dst, radius, neighbors); break;
	case CV_32SC1: VARLBP_<int>(src, dst, radius, neighbors); break;
	case CV_32FC1: VARLBP_<float>(src, dst, radius, neighbors); break;
	case CV_64FC1: VARLBP_<double>(src, dst, radius, neighbors); break;
	}
}

cv::Mat LBPFeatureExtractor::OLBP(const cv::Mat &src)
{
	cv::Mat dst;
	OLBP(src, dst);
	return dst;
}

cv::Mat LBPFeatureExtractor::ELBP(const cv::Mat& src, int radius, int neighbors)
{
	cv::Mat dst;
	ELBP(src, dst, radius, neighbors);
	return dst;
}

cv::Mat LBPFeatureExtractor::VARLBP(const cv::Mat& src, int radius, int neighbors)
{
	cv::Mat dst;
	VARLBP(src, dst, radius, neighbors);
	return dst;
}

std::vector<double> LBPFeatureExtractor::histogram(const cv::Mat &src)
{
	switch (src.type()) {
	case CV_8SC1: return histogram_<char>(src); break;
	case CV_8UC1: return histogram_<unsigned char>(src); break;
	case CV_16SC1: return histogram_<short>(src); break;
	case CV_16UC1: return histogram_<unsigned short>(src); break;
	case CV_32SC1: return histogram_<int>(src); break;
	}
}
