#ifndef __LIGHTING_WRAPPER_H__
#define __LIGHTING_WRAPPER_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	//only due with triangle mesh
	namespace ShLighting
	{
		cv::Mat calcRGBCoeff(const cv::Mat& init_mat_gamma, const cv::Mat& mat_norm);
		void calcRGBCoeff(floatVec& light_coefs, float3E& norm, float3E& res);
	};
}
#endif