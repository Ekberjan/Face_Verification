// FaceVerification.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FaceVerification.h"
#include "preprocessFace.h"

#include <string.h>

void FaceVerification::initialize()
{
	imageCount = 0;
	negativeImageCount = 0;
	positiveImageCount = 0;
	predictScore = 0.0;
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

	faceCascadeFilename = "C:/OpenCV-2.4.10/sources/data/lbpcascades/lbpcascade_frontalface.xml";
	eyeCascadeFilename1 = "C:/OpenCV-2.4.10/sources/data/haarcascades/haarcascade_eye.xml";
	eyeCascadeFilename2 = "C:/OpenCV-2.4.10/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
	negativeImageList = "D:/Workspace/C++/Old_Codes/Images/Negative_Images/FileList.txt";

	faceCascade.load(faceCascadeFilename);
	eyeCascade1.load(eyeCascadeFilename1);
	eyeCascade2.load(eyeCascadeFilename2);
	initializationStatus = true;

	preprocessLeftAndRightSeparately = true;
	needEyeAlignmentPreprocessing = true;
	eyeColor = cv::Scalar(0, 255, 255);

	trainStatus = false;
	faceRecognitionStatus = false;
	needDetectFaceForTrainStatus = false;
	genuineThreshold = 0.95;

	inputDirName = "D:/Workspace/C++/Images/BioID_DataBase/BioID-FaceDatabase-V1.2/PNGs/";
	inputImageCount = 1521;
}

void FaceVerification::trainNegativeImages()
{
	faceRecognitionStatus = true;
	std::ifstream negativeImageFiles(negativeImageList);

	if (!negativeImageFiles.is_open())
	{
		std::cout << "FaceVerification::run() : Cannot open the provided negativeImageFiles. " << std::endl;
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

		// TODO: check whether we do eye alignment for preprocessing, and
		// implement respective steps for training negative images
		/*
		if (needEyeAlignmentPreprocessing)
		{
			preprocessedNegativeFace = getPreprocessedFace(negativeFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftNegativeEye, &rightNegativeEye, &searchedLeftNegativeEye, &searchedRightNegativeEye);
			if (preprocessedNegativeFace.cols > 0 && preprocessedNegativeFace.rows > 0)
			{
				negativeImageCount++;
				std::vector<double> negativeFeatures = lbpFeatureExtractor.getFeature(preprocessedNegativeFace);
				classifier.addNegativeFeatures(negativeFeatures);
			}
		}

		else */
		{		
			negativeImageCount++;
			cv::Mat preprocessedNegativeImage = preprocessor.preprocess(negativeFrame);
			std::vector<double> negativeFeatures = lbpFeatureExtractor.getFeature(preprocessedNegativeImage);
			classifier.addNegativeFeatures(negativeFeatures);
		}
		
		std::cout << "Total negative image count: " << negativeImageCount << std::endl;
		
	}

}

void FaceVerification::runFaceVerifier()
{
	if (initializationStatus != true)
	{
		std::cout << "FaceVerification::run() : The system is not initialized properly. Please check it!" << std::endl;
		return;
	}
	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		std::cout << "FaceVerification::run() : Cannot capture image from camera. Please make sure that the camera is opened. " << std::endl;
		return;
	}

	cv::Mat frame, advertiseMat, predictionResultMat;
	cv::namedWindow("frame", 1);
	cv::namedWindow("preprocessed face", 1);	
	
	std::this_thread::sleep_for(std::chrono::seconds(2));

	while (true)
	//for (int i = 1; i <= inputImageCount; i++)
	{
		imageCount++;
		capture >> frame;
		//frame = cv::imread(inputDirName + std::to_string(i) + ".png");
		cv::Mat grayImage;

		if (frame.channels() == 3)
		{
			cv::cvtColor(frame, grayImage, CV_BGR2GRAY);
		}
		else
		{
			grayImage = frame.clone();
		}

		//cv::equalizeHist(grayImage, grayImage);
		faceCascade.detectMultiScale(grayImage, faceAreas, scaleFactor, minNeighbors, 0, minSize, maxSize);
		processFace(frame, grayImage);
		
		
		if (preprocessedFace.cols > 0 && preprocessedFace.rows > 0)
		{
			eyeAlignmentPreprocess.detectBothEyes(preprocessedFace, eyeCascade1, eyeCascade2, leftEye, rightEye, detectedLeftEye, detectedRightEye);
			
			if (leftEye.x >= 0) {   // Check if the eye was detected
				cv::circle(preprocessedFace, cv::Point(leftEye.x, leftEye.y), 6, eyeColor, 1, CV_AA);
			}
			if (rightEye.x >= 0) {   // Check if the eye was detected
				cv::circle(preprocessedFace, cv::Point(rightEye.x, rightEye.y), 6, eyeColor, 1, CV_AA);
			}

			//processEyes(preprocessedFace);


			
			/*
			cv::rectangle(preprocessedFace, upperFace, CV_RGB(0, 0, 255), 4);
			cv::rectangle(preprocessedFace, middleFace, CV_RGB(255, 0, 0), 4);
			cv::rectangle(preprocessedFace, bottomFace, CV_RGB(255, 255, 255), 4);*/

			//cv::imwrite(outputFaceWithBoundriesDir + std::to_string(outputFaceCount) + ".png", preprocessedFace);

			cv::imshow("preprocessed face", preprocessedFace);
		} 

		cv::imshow("frame", frame);
		cv::waitKey(10);
	}
}

void FaceVerification::processFace(cv::Mat frame, cv::Mat grayImage)
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

			if (faceRecognitionStatus == true && positiveImageCount <= minPositiveTrainingCount)
			{
				classifier.addPosiviteFeatures(features);
				positiveImageCount++;
				std::cout << "adding positive image to the classifier... " << std::endl;
			}
			else
			{
				classifier.train();
				trainStatus = true;

				predictScore = classifier.predict(features);
				std::cout << "prediction score: " << predictScore << std::endl;

				/*
				if (predictScore >= genuineThreshold)
				{
				predictionResultMat = cv::imread("D:/Workspace/C++/Images/Gender_Dataset/Right.jpg");
				cv::resize(predictionResultMat, predictionResultMat, cv::Size(640, 480));
				cv::imshow("predictionResult", predictionResultMat);

				std::cout << "same person. " << std::endl;

				}
				else
				{
				predictionResultMat = cv::imread("D:/Workspace/C++/Images/Gender_Dataset/Wrong.jpg");
				cv::resize(predictionResultMat, predictionResultMat, cv::Size(640, 480));
				cv::imshow("predictionResult", predictionResultMat);

				std::cout << "impostor " << std::endl;

				}*/

			}
		}
		//capturedImageName = "E:/Workspace/Images/CapturedFaces/" + std::to_string(imageCount+248) + ".jpg";
		//cv::imwrite( capturedImageName, face );
		//cv::imshow("face", face);
		faces.push_back(face);
	}
}

// process the face image to obtain eye region separately
void FaceVerification::processEyes(cv::Mat processedFace)
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


