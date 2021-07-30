#ifndef __IMAGE_SIMILARITY_H__
#define __IMAGE_SIMILARITY_H__
#include "../Basic/CGPBaseHeader.h"
namespace CGP
{
	namespace ImageSim
	{
		float ssim(const cv::Mat& im1, const cv::Mat& im2, int window = 7, float k1 = 0.01f, float k2 = 0.03f, float L = 255.f);
		float ssimWrapper(const cv::Mat& im1, const cv::Mat& im2, int window = 7, float k1 = 0.01f, float k2 = 0.03f, float L = 255.f);
		float matchShape(const cv::Mat& im1, const cv::Mat& im2);
		float matchShapeImage(const cv::Mat& im1, const cv::Mat& im2);
	}
}
#endif

