# Face-Verification
C++ based face verification using OpenCV. 

This is a project that performs face verification based on OpenCV. Few information before you use:

1. The facial image features are extracted based on Local Binary Patterns (LBP). The program first performs facial region detection on the input image, then eye detection (both eyes), and perform eye-alignment. Then performs feature extraction using LBP, and send the obtained feacture vector into classifier. 

2. The classification is performed using Support Vector Machines (SVM). 

3. You need to provide relevant negative images for training SVM. Currently, the negative image count is set as 300, you can change it in the code according to your own case. You need to provide negative image list txt file. Please refer to 'negativeImageList' member in the 'faceverifier.cpp' file for more information. 

4. You need to provide relevant positive images for training SVM. Currently, the positive images are those captured from the embedded camera from the host computer, and minimum positive sample size (image count) is set as 200. You can change the source of your positive images (such as from your disk folder instead of capturing live from camera), as well as sample size in the code. 

5. Of course, you need OpenCV for this code, so be sure that you have installed OpenCV on your computer, and provide necessary path to your solution. 

6. The final result is a prediction score between 0 and 1, in which 1 means the test image is totally the same as the positive training image, while 0 means exactly the reverse. 

7. In the relevant header file, you may see codes for 'DLL EXPORT'. This is for ease to let you embed your code into a dll file if necessary. You may do a search on the internet to better understand how to build dll files from source code as well. 

 
Best wishes, 

Ekberjan 

