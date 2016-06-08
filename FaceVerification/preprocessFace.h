/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************
*   by Shervin Emami, 5th Dec 2012
*   http://www.shervinemami.info/openCV.html
******************************************************************************
*   Ch8 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
// preprocessFace.h, by Shervin Emami (www.shervinemami.info) on 30th May 2012.
// Easily preprocess face images, for face recognition.
//////////////////////////////////////////////////////////////////////////////////////

#pragma once


#include <stdio.h>
#include <iostream>
#include <vector>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"
#include "detectObject.h"

class preprocessFace
{
public:
	/*
	// Remove the outer border of the face, so it doesn't include the background & hair.
	// Keeps the center of the rectangle at the same place, rather than just dividing all values by 'scale'.
	Rect scaleRectFromCenter(const Rect wholeFaceRect, float scale = 0.7f);
	*/

	// Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
	// or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
	// want to search eyes using 2 different cascades. For example, you could use a regular eye detector
	// as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
	// Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
	// Can also store the searched left & right eye regions if desired.
	void detectBothEyes(const cv::Mat &face, cv::CascadeClassifier &eyeCascade1, cv::CascadeClassifier &eyeCascade2, cv::Point &leftEye, cv::Point &rightEye, cv::Rect *searchedLeftEye = NULL, cv::Rect *searchedRightEye = NULL);

	// Histogram Equalizae seperately for the left and right sides of the face,
	// so that if there is a strong light on one side but not the other, it will still look OK.
	void equalizeLeftAndRightHalves(cv::Mat &faceImg);

	// Create a grayscale face image that has a standard size and contrast & brightness.
	// "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
	// If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
	// so that if there is a strong light on one side but not the other, it will still look OK.
	// Performs Face Preprocessing as a combination of:
	//  - geometrical scaling, rotation and translation using Eye Detection,
	//  - smoothing away image noise using a Bilateral Filter,
	//  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
	//  - removal of background and hair using an Elliptical Mask.
	// Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
	// If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
	// and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
	cv::Mat getPreprocessedFace(cv::Mat &srcImg, int desiredFaceWidth, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade1, cv::CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, cv::Rect *storeFaceRect = NULL, cv::Point *storeLeftEye = NULL, cv::Point *storeRightEye = NULL, cv::Rect *searchedLeftEye = NULL, cv::Rect *searchedRightEye = NULL);

	detectObject detector;

	cv::Mat eyeRegion, noseRegion, mouthRegion;

private:
	int inputFaceCount = 0;

	std::string inputImageDirName = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Original_Image/";
	std::string inputFaceDirName = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Face_Input/";	
	std::string outputFaceDirName = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Face_Processed/";
	std::string outputFaceWithBoundriesDir = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Face_with_Boundries/";
	std::string outputEyeDirName = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Eyes/";
	std::string outputNoseDirName = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Nose/";
	std::string outputMouthDirName = "D:/Workspace/C++/Gender_Classification/Images/BioID/Resized/64_by_64/Mouth/";
	cv::Size outputImgSize = cv::Size(32, 32);
	bool writeResults(cv::Mat &inputImage, cv::Mat &inputFace, cv::Mat &processedFace);
	bool needResults = false;

};