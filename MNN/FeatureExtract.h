#ifndef _PREPROCESS_IMAGE_H_
#define _PREPROCESS_IMAGE_H_
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "FaceAlign.h""
namespace PREPROCESS_IMAGE
{

	//compare input features with labeled features, get classification result with minimal distance and class index
	struct class_info
	{
		double min_distance;
		int index;
	};


	class_info classify(const cv::Mat& img, const  cv::Mat& cmp);

	void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);

	cv::Mat alignToNcnn(cv::Mat& img, float* landmark);
	
	cv::Mat alignToMtcnn(const cv::Mat& img, float* landmark, bool is_xy);	

	cv::Mat alignToMtcnnExpand(const cv::Mat& img, float* landmark, bool is_xy, int resolution);
}
#endif
