#ifndef FACEVERIFICATION_H
#define FACEVERIFICATION_H

#include <fstream>
#include <iostream>
#include "opencv\highgui.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <opencv2\core\core.hpp>
#include "classifier.h"
#include "preprocessor.h"
#include "lbpfeatureextractor.h"
#include "opencv2\objdetect\objdetect.hpp"
#include <thread>
#include <chrono>
#include "preprocessFace.h"
#include "detectObject.h"

class FaceVerification
{
public:
	void initialize();
	void writeFile();
	void runFaceVerifier();
	void trainNegativeImages();

private:
	int negativeImageCount;
	int positiveImageCount;
	int minNeighbors;
	int imageCount;
	int minPositiveTrainingCount;
	double predictScore;
	double scaleFactor;
	double genuineThreshold;
	char imageName[1000];
	bool initializationStatus;
	bool trainStatus;
	bool faceRecognitionStatus;
	bool needDetectFaceForTrainStatus;
	cv::Size minSize;
	cv::Size maxSize;
	std::vector<cv::Rect> faceAreas;
	std::vector<cv::Rect> faceAreasMale;
	std::vector<cv::Rect> faceAreasFemale;
	std::ofstream myfile;
	std::string capturedImageName;
	std::string negativeImageList;
	std::string negativeImageName;
	std::vector<cv::Mat> faces;
	cv::Mat face;
	cv::Mat negativeFrame;
	Classifier classifier;
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade1;
	cv::CascadeClassifier eyeCascade2;
	Preprocessor preprocessor;
	LBPFeatureExtractor lbpFeatureExtractor;

	//char dirName[1000];
	char fileName[1000];
	char outputFileName[1000];
	int copyImageCount;

	// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
	const int faceWidth = 120;
	const int faceHeight = faceWidth;

	const char *faceCascadeFilename, *eyeCascadeFilename1, *eyeCascadeFilename2;
	bool preprocessLeftAndRightSeparately;
	bool needEyeAlignmentPreprocessing;
	std::vector<cv::Rect> eyeAreas;

	preprocessFace eyeAlignmentPreprocess;
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

	void processFace(cv::Mat frame, cv::Mat processedFace);
	void processEyes(cv::Mat inputFrame);

	std::string inputDirName;
	int inputImageCount;


}; // class

#endif