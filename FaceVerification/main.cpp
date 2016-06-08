#include "stdafx.h"
//#include "FaceVerification.h"
#include "faceverifier.h"
#include <thread>


int _tmain(int argc, _TCHAR* argv[])
{
	/*FaceVerification faceVerification;
	faceVerification.initialize();
	faceVerification.trainNegativeImages();
	faceVerification.runFaceVerifier();*/

	FaceVerifier faceVerifier;
	faceVerifier.initialize();
	faceVerifier.trainNegativeImages();	
	faceVerifier.trainPositiveImages();

	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		std::string s = "FaceVerification::run() : Cannot capture image from camera. Please make sure that the camera is opened.";		
		std::cout << s << std::endl;
		return 1;
	}

	std::this_thread::sleep_for(std::chrono::seconds(2));

	cv::namedWindow("frame", 1);
	cv::namedWindow("preprocessed face", 1);
	
	cv::Mat frame;
	//while (imageCount <= minPositiveTrainingCount)
	while (true)
	{
		capture >> frame;
		faceVerifier.predict(frame);
		double score = faceVerifier.getPredictionScore();
		std::cout << "prediction score: " << score << std::endl;
	}



	return 0;
}
