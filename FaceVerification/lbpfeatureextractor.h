#ifndef LBPFEATUREEXTRACTOR_H
#define LBPFEATUREEXTRACTOR_H

#include <opencv2/core/core.hpp>
#include "common.h"

class LBPFeatureExtractor
{
public:
	LBPFeatureExtractor();
	std::vector<double> getFeature(const cv::Mat& image);

	bool setIntParameter(FaceVerifierIntParameters parameter, int value);

	std::string errorMessage();

private:
	LBPType lbpType;
	int blockSize;
	int binCount;
	int imageSize;
	int featureVectorLength;

	std::string errorString;

	//template function
	template <typename _Tp>
	void OLBP_(const cv::Mat& src, cv::Mat& dst);

	template <typename _Tp>
	void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

	template <typename _Tp>
	void VARLBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

	template <typename _Tp>
	std::vector<double> histogram_(const cv::Mat& src);

	// wrapper function
	void OLBP(const cv::Mat& src, cv::Mat& dst);
	void ELBP(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);
	void VARLBP(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

	// Mat return type function
	cv::Mat OLBP(const cv::Mat& src);
	cv::Mat ELBP(const cv::Mat& src, int radius = 1, int neighbors = 8);
	cv::Mat VARLBP(const cv::Mat& src, int radius = 1, int neighbors = 8);

	std::vector<double> histogram(const cv::Mat& src);
}; //class

#endif // LBPFEATUREEXTRACTOR_H
