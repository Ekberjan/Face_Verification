#ifndef FACEVERIFIER_H
#define FACEVERIFIER_H

#ifndef MAKEDLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#include "common.h"
#include "classifier.h"
#include "preprocessor.h"
#include "preprocessFace.h"
#include "lbpfeatureextractor.h"
#include "opencv2\objdetect\objdetect.hpp"

static class DLLEXPORT FaceVerifier
{
public:
	void initialize();
	void run();
	void trainNegativeImages();
	void trainPositiveImages();
	void setCameraId(std::string id);
	void predict(cv::Mat image);

	int getPositiveImageCount();
	int getNegativeImageCount();
	double getPredictionScore();

	cv::Mat getCapturedFrame();
	cv::Mat getProcessedFace();


private:
	int imageCount;
	int minNeighbors;
	int negativeImageCount;
	int positiveImageCount;
	int minPositiveTrainingCount;

	const int faceWidth = 120;
	const int faceHeight = faceWidth;

	double scaleFactor;
	double predictionScore;
	double genuineThreshold;

	bool trainPositiveStatus;
	bool trainNegativeStatus;
	bool initializationStatus;
	bool faceRecognitionStatus;
	bool needDetectFaceForTrainStatus;
	bool needEyeAlignmentPreprocessing;
	bool preprocessLeftAndRightSeparately;

	cv::Mat frame;

	cv::Size minSize;
	cv::Size maxSize;

	std::string cameraId;
	std::string getPath();
	std::string currentPath;
	std::string window_name;
	std::string faceCascadeFilename;
	std::string eyeCascadeFilename1;
	std::string eyeCascadeFilename2;
	std::string negativeImageList;
	std::string negativeImageName;
	std::vector<cv::Rect> faceAreas;
	std::vector<cv::Mat> faces;

	std::vector<cv::Rect> getFaceAreas(cv::Mat image);

	cv::Mat face;
	cv::Mat negativeFrame;

	Preprocessor preprocessor;
	preprocessFace eyeAlignmentPreprocess;
	LBPFeatureExtractor lbpFeatureExtractor;

	Classifier classifier;
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade1;
	cv::CascadeClassifier eyeCascade2;

	detectObject objectDetector;
	cv::Rect *detectedLeftEye;
	cv::Rect *detectedRightEye;
	cv::Rect faceRect;  // Position of detected face.
	cv::Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
	cv::Point leftEye, rightEye;    // Position of the detected eyes.
	cv::Point processedLeftEye, processedRightEye;    // Position of the detected eyes.
	cv::Rect searchedLeftNegativeEye, searchedRightNegativeEye;
	cv::Point leftNegativeEye, rightNegativeEye;    // Position of the detected eyes.
	cv::Mat preprocessedFace;
	cv::Mat preprocessedNegativeFace;
	cv::Rect eyeRegion;
	cv::Scalar eyeColor;

	void processEyes(cv::Mat processedFace);
	void processFace(cv::Mat frame, cv::Mat grayImage, bool isPredict);

};


#endif