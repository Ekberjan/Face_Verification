#ifndef COMMON_H
#define COMMON_H

/**************************************************************************************/
// About face detector (Begin)
//
enum FaceDetectorType
{
	HaarFrontalFaceAlt = 0,
	HaarProfileFace,
	LbpFrontalFace,
	LbpProfileFace,
	FaceDetectrType_First = HaarFrontalFaceAlt, // do not use this as enum value!
	FaceDetectrType_Last = LbpProfileFace // do not use this as enum value!
};

enum FaceDetectorReturnType
{
	Grayscale = 0,
	Colored,
	FaceDetectorReturnType_First = Grayscale, // do not use this as enum value!
	FaceDetectorReturnType_Last = Colored // do not use this as enum value!
};
//
// About face detector (End)
/**************************************************************************************/




/**************************************************************************************/
// About preprocessor (Begin)
//
// nothing here, keep up the good work!
//
// About preprocessor (End)
/**************************************************************************************/




/**************************************************************************************/
// About lbp  feature extractor (Begin)
//
enum LBPType
{
	OLBP = 0,
	ELBP,
	VARLBP,
	LBPType_First = OLBP, // do not use this as enum value!
	LBPType_Last = VARLBP // do not use this as enum value!

};
//
// About lbp  feature extractor (End)
/**************************************************************************************/




/**************************************************************************************/
// About classifier (Begin)
//
// nothing here also, you look like you're keeping up the good work ;)
//
// About classifier (End)
/**************************************************************************************/

// predefined return values for predict
#define FACEVERIFIER_PREDICT_RESULT_FACE_NOT_FOUND 99999.0
#define FACEVERIFIER_PREDICT_RESULT_SYSTEM_NOT_TRAINED 99998.0
#define FACEVERIFIER_PREDICT_RESULT_SYSTEM_NOT_INITIALIZED 99997.0
#define FACEVERIFIER_PREDICT_RESULT_UNKNOWN_ERROR 99996.0

#define CLOTHVERIFIER_PREDICT_RESULT_MODEL_REGION_EMPTY 99999.0
#define CLOTHVERIFIER_PREDICT_RESULT_MODEL_REGION_NOT_VALID 99998.0
#define CLOTHVERIFIER_PREDICT_RESULT_SYSTEM_NOT_TRAINED 99997.0

enum FaceVerifierIntParameters
{
	//parameters about face verifier
	FaceVerifierParameterNumberOfImagesForRotationAngleDetection = 0,
	FaceVerifierParameterRotationAngleBegin,
	FaceVerifierParameterRotationAngleEnd,
	FaceVerifierParameterRotationAngleStep,
	FaceVerifierParameter_FirstInt = FaceVerifierParameterNumberOfImagesForRotationAngleDetection, // do not use this as enum value!
	FaceVerifierParameter_LastInt = FaceVerifierParameterRotationAngleStep, // do not use this as enum value!

	//parameters about face detector
	FaceDetectorParameterReturnType = 100,
	FaceDetectorParameterMinimumNeighbors,
	FaceDetectorParameterMinimumSize,
	FaceDetectorParameterMaximumSize,
	FaceDetectorParameterRotationAngle,
	FaceDetectorParameter_FirstInt = FaceDetectorParameterReturnType, // do not use this as enum value!
	FaceDetectorParameter_LastInt = FaceDetectorParameterRotationAngle, // do not use this as enum value!

	//parameters about preprocessor
	PreprocessorParameterCroppingPercentageLeft = 200,
	PreprocessorParameterCroppingPercentageRight,
	PreprocessorParameterCroppingPercentageTop,
	PreprocessorParameterCroppingPercentageBottom,
	PreprocessorParameterNewSize,
	PreprocessorParameter_FirstInt = PreprocessorParameterCroppingPercentageLeft, // do not use this as enum value!
	PreprocessorParameter_LastInt = PreprocessorParameterNewSize, // do not use this as enum value!

	//parameters about lbp feature extractor
	LBPFeatureExtractorParameterLBPType = 300,
	LBPFeatureExtractorParameterBlockSize,
	LBPFeatureExtractorParameterBinCount,
	LBPFeatureExtractorParameterImageSize,
	LBPFeatureExtractorParameter_FirstInt = LBPFeatureExtractorParameterLBPType, // do not use this as enum value!
	LBPFeatureExtractorParameter_LastInt = LBPFeatureExtractorParameterImageSize, // do not use this as enum value!

	//parameters about classifier
	ClassifierParameterMinimumTrainPositiveImageCount = 400,
	ClassifierParameterMaximumTrainPositiveImageCount,
	ClassifierParameterMinimumTrainNegativeImageCount,
	ClassifierParameterMaximumTrainNegativeImageCount,
	ClassifierParameter_FirstInt = ClassifierParameterMinimumTrainPositiveImageCount, // do not use this as enum value!
	ClassifierParameter_LastInt = ClassifierParameterMaximumTrainNegativeImageCount // do not use this as enum value!
};

enum FaceVerifierDoubleParameters
{
	//parameters about face verifier
	FaceVerifierParameter_FirstDouble = 0, // do not use this as enum value!
	FaceVerifierParameter_LastDouble = 0, // do not use this as enum value!

	//parameters about face detector
	FaceDetectorParameterScale = 100,
	FaceDetectorParameter_FirstDouble = FaceDetectorParameterScale, // do not use this as enum value!
	FaceDetectorParameter_LastDouble = FaceDetectorParameterScale, // do not use this as enum value!

	//parameters about preprocessor
	PreprocessorParameterSmoothingFactor = 200,
	PreprocessorParameter_FirstDouble = PreprocessorParameterSmoothingFactor, // do not use this as enum value!
	PreprocessorParameter_LastDouble = PreprocessorParameterSmoothingFactor, // do not use this as enum value!

	//parameters about lbp feature extractor
	LBPFeatureExtractorParameter_FirstDouble = 300, // do not use this as enum value!
	LBPFeatureExtractorParameter_LastDouble = 300, // do not use this as enum value!

	//parameters about classifier
	ClassifierParameterC = 400,
	ClassifierParameterGamma,
	ClassifierParameterSigmoidA,
	ClassifierParameterSigmoidB,
	ClassifierParameter_FirstDouble = ClassifierParameterC, // do not use this as enum value!
	ClassifierParameter_LastDouble = ClassifierParameterSigmoidB // do not use this as enum value!
};

enum FaceVerifierBoolParameters
{
	//parameters about face verifier
	FaceVerifierParameterRotationAngleDetection = 0,
	FaceVerifierParameter_FirstBool = FaceVerifierParameterRotationAngleDetection, // do not use this as enum value!
	FaceVerifierParameter_LastBool = FaceVerifierParameterRotationAngleDetection, // do not use this as enum value!

	//parameters about face detector
	FaceDetectorParameterFlip = 100,
	FaceDetectorParameter_FirstBool = FaceDetectorParameterFlip, // do not use this as enum value!
	FaceDetectorParameter_LastBool = FaceDetectorParameterFlip, // do not use this as enum value!

	//parameters about preprocessor
	PreprocessorParameterCropping = 200,
	PreprocessorParameterSmoothing,
	PreprocessorParameterResizing,
	PreprocessorParameterHistogramEqualization,
	PreprocessorParameter_FirstBool = PreprocessorParameterCropping, // do not use this as enum value!
	PreprocessorParameter_LastBool = PreprocessorParameterHistogramEqualization, // do not use this as enum value!

	//parameters about lbp feature extractor
	LBPFeatureExtractorParameter_FirstBool = 300, // do not use this as enum value!
	LBPFeatureExtractorParameter_LastBool = 300, // do not use this as enum value!

	//parameters about classifier
	ClassifierParameter_FirstBool = 400, // do not use this as enum value!
	ClassifierParameter_LastBool = 400 // do not use this as enum value!
};

enum ClothVerifierIntParameters
{
	//parameters about cloth verifier
	ClothVerifierIntParametersMinimumTrainImageCount = 0,
	ClothVerifierIntParametersGMMNumberOfMixtureComponents,
	ClothVerifierIntParametersInputImageWidth,
	ClothVerifierIntParametersInputImageHeight,
	ClothVerifierIntParametersGmmRegionWidth,
	ClothVerifierIntParametersGmmRegionHeight,
	ClothVerifierParameter_FirstInt = ClothVerifierIntParametersMinimumTrainImageCount,
	ClothVerifierParameter_LastInt = ClothVerifierIntParametersGmmRegionHeight,
};

enum ClothVerifierDoubleParameters
{
	//parameters about cloth verifier
	ClothVerifierDoubleParametersSquaredMahalanobisDistanceThreshold = 0,
	ClothVerifierDoubleParametersClothRegionD,
	ClothVerifierDoubleParametersGMMLearningRate,
	ClothVerifierParameter_FirstDouble = ClothVerifierDoubleParametersSquaredMahalanobisDistanceThreshold,
	ClothVerifierParameter_LastDouble = ClothVerifierDoubleParametersGMMLearningRate
};


enum ClothVerifierBooleanParameters
{
	//parameters about cloth verifier
	ClothVerifierBooleanParametersShadowDetection = 0,
	ClothVerifierBooleanParametersFlipStatus,
	ClothVerifierParameter_FirstBoolean = ClothVerifierBooleanParametersShadowDetection,
	ClothVerifierParameter_LastBoolean = ClothVerifierBooleanParametersFlipStatus
};


enum MotionBlurIntParameters
{
	MotionBlurIntParametersCountNumber = 0,
	MotionBlurIntParametersDistance,
	MotionBlurParameter_FirstInt = MotionBlurIntParametersCountNumber,
	MotionBlurParameter_LastInt = MotionBlurIntParametersDistance
};

enum MotionBlurDoubleParameters
{
	MotionBlurDoubleParametersAlpha = 0,
	MotionBlurDoubleParametersAngle,
	MotionBlurParameter_FirstDouble = MotionBlurDoubleParametersAlpha,
	MotionBlurParameter_LastDouble = MotionBlurDoubleParametersAngle
};

#endif // COMMON_H
