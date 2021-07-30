#include <cmath>
#include "Lighting.h"
#include "../FileIO/FileIO.h"
#include "../RT/RT.h"
#include "../Debug/DebugTools.h"

using namespace CGP;

cv::Mat ShLighting::calcRGBCoeff(const cv::Mat& init_mat_gamma, const cv::Mat& mat_norm)
{
	cv::Mat mat_gamma = init_mat_gamma;
	// 初始Gamma
	floatVec pf_init_gamma = { 0.8, 0, 0, 0, 0, 0, 0, 0, 0 };
	cv::Mat mat_init_gamma_init(1, 9, CV_32FC1, pf_init_gamma.data());

	cv::Mat mat_init_gamma;
	cv::vconcat(mat_init_gamma_init, mat_init_gamma_init, mat_init_gamma);
	cv::vconcat(mat_init_gamma_init, mat_init_gamma, mat_init_gamma);

	//std::cout << matInitGamma << std::endl;
	// 加上初始Gamma
	//mat_gamma = mat_gamma + mat_init_gamma;

	// 计算球谐函数的各常量
	float a0 = CGPI;
	float a1 = 2 * CGPI / sqrt(3.0);
	float a2 = 2 * CGPI / sqrt(8.0);
	float c0 = 1 / sqrt(4 * CGPI);
	float c1 = sqrt(3.0) / sqrt(4 * CGPI);
	float c2 = 3 * sqrt(5.0) / sqrt(12 * CGPI);

	// 
	cv::Mat mat_Y = cv::Mat(mat_norm.rows, 9, CV_32FC1);

	//std::cout << "matY: " << std::endl << matY << std::endl;

#pragma omp parallel for
	for (int y = 0; y < mat_Y.rows; y++)
	{
		mat_Y.at<float>(y, 0) = a0 * c0;
		mat_Y.at<float>(y, 1) = -a1 * c1 * mat_norm.at<float>(y, 1);
		mat_Y.at<float>(y, 2) = a1 * c1 * mat_norm.at<float>(y, 2);
		mat_Y.at<float>(y, 3) = -a1 * c1 * mat_norm.at<float>(y, 0);
		mat_Y.at<float>(y, 4) = a2 * c2 * mat_norm.at<float>(y, 0) * mat_norm.at<float>(y, 1);
		mat_Y.at<float>(y, 5) = -a2 * c2 * mat_norm.at<float>(y, 1) * mat_norm.at<float>(y, 2);
		mat_Y.at<float>(y, 6) = (a2 * c2 * 0.5 / sqrt(3.0) * (3.0 * mat_norm.at<float>(y, 2) * mat_norm.at<float>(y, 2) - 1));
		mat_Y.at<float>(y, 7) = (-a2 * c2 * mat_norm.at<float>(y, 0) * mat_norm.at<float>(y, 2));
		mat_Y.at<float>(y, 8) = (a2 * c2 * 0.5 * (mat_norm.at<float>(y, 0) * mat_norm.at<float>(y, 0) - mat_norm.at<float>(y, 1) * mat_norm.at<float>(y, 1)));
	}

	mat_gamma = mat_gamma.t();
	//cv::Mat matR_w = matY * matGamma.col(0);
	//cv::Mat matG_w = matY * matGamma.col(1);
	//cv::Mat matB_w = matY * matGamma.col(2);

	cv::Mat mat_RGB_w = mat_Y * mat_gamma;
	//cv::hconcat(matR_w, matG_w, matRGB_w);
	//cv::hconcat(matRGB_w, matB_w, matRGB_w);
	return mat_RGB_w;
}

void ShLighting::calcRGBCoeff(floatVec& light_coefs, float3E& norm, float3E& res)
{
	cv::Mat mat_gamma(3, 9, CV_32FC1, light_coefs.data());
	cv::Mat mat_norm(1, 3, CV_32FC1, norm.data());
	cv::Mat mat_rgb_weight = calcRGBCoeff(mat_gamma, mat_norm);
	res.x() = GETF(mat_rgb_weight, 0, 0);
	res.y() = GETF(mat_rgb_weight, 0, 1);
	res.z() = GETF(mat_rgb_weight, 0, 2);
}