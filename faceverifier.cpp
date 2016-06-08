#include "stdafx.h"
#include "faceverifier.h"
#include <Windows.h>
#include <thread>
#include <fstream>

void FaceVerifier::initialize()
{
	imageCount = 0;
	negativeImageCount = 0;
	positiveImageCount = 0;
	predictionScore = 0.0;
	scaleFactor = 1.35;
	minNeighbors = 2;
	minSize = cv::Size(64, 64);
	maxSize = cv::Size(300, 300);
	minPositiveTrainingCount = 200;

	classifier.setIntParameter(FaceVerifierIntParameters::ClassifierParameterMinimumTrainPositiveImageCount, 200);
	classifier.setIntParameter(FaceVerifierIntParameters::ClassifierParameterMaximumTrainPositiveImageCount, 1000);
	classifier.setIntParameter(FaceVerifierIntParameters::ClassifierParameterMinimumTrainNegativeImageCount, 100);
	classifier.setIntParameter(FaceVerifierIntParameters::ClassifierParameterMaximumTrainNegativeImageCount, 1000);

	classifier.setDoubleParameter(FaceVerifierDoubleParameters::ClassifierParameterC, 10.0);
	classifier.setDoubleParameter(FaceVerifierDoubleParameters::ClassifierParameterGamma, 0.001953125);
	classifier.setDoubleParameter(FaceVerifierDoubleParameters::ClassifierParameterSigmoidA, -5.2);
	classifier.setDoubleParameter(FaceVerifierDoubleParameters::ClassifierParameterSigmoidB, -0.5);

	currentPath = getPath();
	window_name = "Capture - Face detection";
	faceCascadeFilename = currentPath + "\\Cascades\\lbpcascades\\lbpcascade_frontalface.xml";
	eyeCascadeFilename1 = currentPath + "\\Cascades\\haarcascades\\haarcascade_eye.xml";
	eyeCascadeFilename2 = currentPath + "\\Cascades\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
	negativeImageList = currentPath + "\\Negative_Images\\FileList.txt";

	faceCascade.load(faceCascadeFilename);
	eyeCascade1.load(eyeCascadeFilename1);
	eyeCascade2.load(eyeCascadeFilename2);
	initializationStatus = true;

	preprocessLeftAndRightSeparately = true;
	needEyeAlignmentPreprocessing = true;
	eyeColor = cv::Scalar(0, 255, 255);

	trainPositiveStatus = false;
	trainNegativeStatus = false;
	faceRecognitionStatus = false;
	needDetectFaceForTrainStatus = false;
	genuineThreshold = 0.95;
}

std::string FaceVerifier::getPath()
{
	std::string path;
	wchar_t buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, sizeof(buffer));
	std::wstring ws(buffer);
	std::string str(ws.begin(), ws.end());
	const size_t last_slash_idx = str.rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		path = str.substr(0, last_slash_idx);
	}

	return path;
}

void FaceVerifier::setCameraId(std::string id)
{
	cameraId = id;
}

void FaceVerifier::trainNegativeImages()
{
	faceRecognitionStatus = true;
	std::ifstream negativeImageFiles(negativeImageList);

	if (!negativeImageFiles.is_open())
	{
		std::string s = "FaceVerification::run() : Cannot open the provided negativeImageFiles.";
		/*System::String^ message = gcnew System::String(s.c_str());
		System::Console::WriteLine(message);*/
		std::cout << s << std::endl;

		return;
	}

	while (std::getline(negativeImageFiles, negativeImageName))
	{
		negativeFrame = cv::imread(negativeImageName);

		if (negativeFrame.channels() == 3)
		{
			cv::cvtColor(negativeFrame, negativeFrame, CV_BGR2GRAY);
		}
		cv::equalizeHist(negativeFrame, negativeFrame);

		negativeImageCount++;

		std::cout << "negativeImageCount: " << negativeImageCount << std::endl;

		cv::Mat preprocessedNegativeImage = preprocessor.preprocess(negativeFrame);
		std::vector<double> negativeFeatures = lbpFeatureExtractor.getFeature(preprocessedNegativeImage);
		classifier.addNegativeFeatures(negativeFeatures);
	}

	trainNegativeStatus = true;
}

void FaceVerifier::trainPositiveImages()
{
	if (!initializationStatus)
	{
		std::string s = "FaceVerification::run() : The system is not initialized properly. Please check it!";
		/*System::String^ message = gcnew System::String(s.c_str());
		System::Console::WriteLine(message);*/

		std::cout << s << std::endl;

		return;
	}
	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		std::string s = "FaceVerification::run() : Cannot capture image from camera. Please make sure that the camera is opened.";
		/*System::String^ message = gcnew System::String(s.c_str());
		System::Console::WriteLine(message);*/
		std::cout << s << std::endl;

		return;
	}

	//System::Threading::Thread::Sleep(2000);
	std::this_thread::sleep_for(std::chrono::seconds(2));

	cv::namedWindow("frame", 1);
	cv::namedWindow("preprocessed face", 1);

	while (positiveImageCount <= minPositiveTrainingCount)
	{
		std::cout << "positiveImageCount: " << positiveImageCount << std::endl;

		capture >> frame;
		cv::Mat grayImage;

		if (frame.channels() == 3)
		{
			cv::cvtColor(frame, grayImage, CV_BGR2GRAY);
		}
		else
		{
			grayImage = frame.clone();
		}

		faceCascade.detectMultiScale(grayImage, faceAreas, scaleFactor, minNeighbors, 0, minSize, maxSize);
		processFace(frame, grayImage, false);

		if (preprocessedFace.cols > 0 && preprocessedFace.rows > 0)
		{
			eyeAlignmentPreprocess.detectBothEyes(preprocessedFace, eyeCascade1, eyeCascade2, leftEye, rightEye, detectedLeftEye, detectedRightEye);

			if (leftEye.x >= 0) {   // Check if the eye was detected
				cv::circle(preprocessedFace, cv::Point(leftEye.x, leftEye.y), 6, eyeColor, 1, CV_AA);
			}
			if (rightEye.x >= 0) {   // Check if the eye was detected
				cv::circle(preprocessedFace, cv::Point(rightEye.x, rightEye.y), 6, eyeColor, 1, CV_AA);
			}

			cv::imshow("preprocessed face", preprocessedFace);
		} // end of if-condition

		cv::imshow("frame", frame);
		cv::waitKey(10);

	} // end of while-loop
}

//void FaceVerifier::trainPositiveImages(std::vector<double> features)
//{
//	classifier.addPosiviteFeatures(features);
//	positiveImageCount++;
//
//	if (positiveImageCount >= minPositiveTrainingCount)
//	{
//		trainPositiveStatus = true;
//	}
//}

void FaceVerifier::predict(cv::Mat frame)
{
	if (!trainNegativeStatus || !trainPositiveStatus)
	{
		std::cout << "The classifier is not trained properly. " << std::endl;
		return;
	}
		
	cv::Mat grayImage;

	if (frame.channels() == 3)
	{
		cv::cvtColor(frame, grayImage, CV_BGR2GRAY);
	}
	else
	{
		grayImage = frame.clone();
	}

	faceCascade.detectMultiScale(grayImage, faceAreas, scaleFactor, minNeighbors, 0, minSize, maxSize);
	processFace(frame, grayImage, true);

	if (preprocessedFace.cols > 0 && preprocessedFace.rows > 0)
	{
		eyeAlignmentPreprocess.detectBothEyes(preprocessedFace, eyeCascade1, eyeCascade2, leftEye, rightEye, detectedLeftEye, detectedRightEye);

		if (leftEye.x >= 0) {   // Check if the eye was detected
			cv::circle(preprocessedFace, cv::Point(leftEye.x, leftEye.y), 6, eyeColor, 1, CV_AA);
		}
		if (rightEye.x >= 0) {   // Check if the eye was detected
			cv::circle(preprocessedFace, cv::Point(rightEye.x, rightEye.y), 6, eyeColor, 1, CV_AA);
		}

		cv::imshow("preprocessed face", preprocessedFace);
	} // end of if-condition

	cv::imshow("frame", frame);
	cv::waitKey(10);

}

void FaceVerifier::processFace(cv::Mat frame, cv::Mat grayImage, bool isPredict)
{

	for (int i = 0; i < faceAreas.size(); i++)
	{
		cv::Rect face_i = faceAreas[i];
		cv::rectangle(frame, face_i, CV_RGB(0, 255, 0), 1);
	}

	for (std::vector<cv::Rect>::const_iterator r = faceAreas.begin(); r != faceAreas.end(); ++r)
	{
		cv::Rect faceArea(*r);
		face = frame(faceArea);
		cv::resize(face, face, cv::Size(120, 120));

		if (needEyeAlignmentPreprocessing)
		{
			preprocessedFace = eyeAlignmentPreprocess.getPreprocessedFace(grayImage, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

			if (leftEye.x >= 0) {   // Check if the eye was detected
				cv::circle(frame, cv::Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
			}
			if (rightEye.x >= 0) {   // Check if the eye was detected
				cv::circle(frame, cv::Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
			}
		}

		else
		{
			preprocessedFace = preprocessor.preprocess(face);
		}

		// Find a face and preprocess it to have a standard size and contrast & brightness.s
		if (preprocessedFace.rows > 0 && preprocessedFace.cols > 0)
		{
			std::vector<double> features = lbpFeatureExtractor.getFeature(preprocessedFace);

			if (!isPredict)
			{
				classifier.addPosiviteFeatures(features);
				positiveImageCount++;
				
				if (positiveImageCount >= minPositiveTrainingCount)
				{
					trainPositiveStatus = true;
				}
			}
			else
			{
				classifier.train();
				predictionScore = classifier.predict(features);
			}
		}

		faces.push_back(face);
	}
}

void FaceVerifier::processEyes(cv::Mat processedFace)
{
	if (processedFace.rows > 0 && processedFace.cols > 0)
	{
		eyeAlignmentPreprocess.detectBothEyes(processedFace, eyeCascade1, eyeCascade2, processedLeftEye, processedRightEye, &searchedLeftEye, &searchedRightEye);
		cv::Rect faceRectEye = cv::Rect(0, 0, processedFace.rows, processedFace.cols);
		objectDetector.detectLargestObject(processedFace, faceCascade, faceRectEye);

		if (processedLeftEye.x >= 0) {   // Check if the eye was detected
			cv::circle(processedFace, cv::Point(faceRectEye.x + leftEye.x, faceRectEye.y + leftEye.y), 6, eyeColor, 1, CV_AA);
		}
		if (processedRightEye.x >= 0) {   // Check if the eye was detected
			cv::circle(processedFace, cv::Point(faceRectEye.x + rightEye.x, faceRectEye.y + rightEye.y), 6, eyeColor, 1, CV_AA);
		}
	}
}

cv::Mat FaceVerifier::getCapturedFrame()
{
	return frame;
}

cv::Mat FaceVerifier::getProcessedFace()
{
	return preprocessedFace;
}

int FaceVerifier::getPositiveImageCount()
{
	return positiveImageCount;
}

int FaceVerifier::getNegativeImageCount()
{
	return negativeImageCount;
}

double FaceVerifier::getPredictionScore()
{
	return predictionScore;
}