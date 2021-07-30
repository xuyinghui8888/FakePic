#ifndef _FACE_ALIGN_H_
#define _FACE_ALIGN_H_
#include <opencv2/opencv.hpp>

namespace PREPROCESS_IMAGE
{
	cv::Mat meanAxis0(const cv::Mat &src);

	cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B);

	cv::Mat varAxis0(const cv::Mat &src);

	int MatrixRank(cv::Mat M);

	//    References
	//    ----------
	//    .. [1] "Least-squares estimation of transformation parameters between two
	//    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
	//
	//    """
	//
	//    Anthor:Jack Yu
	cv::Mat similarTransform(cv::Mat src, cv::Mat dst);
}

#endif
