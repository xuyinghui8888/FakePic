#ifndef ROTATION_TRANSLATION_H
#define ROTATION_TRANSLATION_H
#include "../Basic/CGPBaseHeader.h"
namespace CGP
{
	namespace RT
	{
		cv::Mat toPrjMat(const cv::Mat& rotation_vector, const cv::Mat& translation_vector);
		cv::Mat toPrjMat(const double* const xi);
		void toPrjMat(const double* const xi, cv::Mat& J, cv::Mat& RT);
		cv::Mat toFullPrjMat(const cv::Mat& rotation_vector, const cv::Mat& translation_vector);
		//get scale for src * scale = dst
		double getScale(const float3Vec& src, const float3Vec& dst);
		//get scale for src * scale = dst
		void getScale3Dim(const float3Vec& src, const float3Vec& dst, float3E& res);
		void getTranslate(const float3Vec& src, const float3Vec& dst, float3E& trans);
		void getTranslate(const float3Vec& src, const intVec& src_set, const float3Vec& dst, const intVec& dst_set, float3E& trans);
		void getCenter(const float3Vec& points, float3E& center);
		void shiftCenter(const float3Vec&src, const float3E& shift, float3Vec& dst);
		void scaleInPlace(double scale, float3Vec& src);
		void scaleInPlace(double scale, float3Vec& src, float3E& src_center);
		void scaleInPlace(const float3E& scale, float3Vec& src);
		void scaleInPlace(const float3E& scale, const intVec& roi, float3Vec& src);
		void scaleInPlaceNoShift(double scale, float3Vec& src);
		void transformInPlace(const mat4f& t, float3Vec& src);
		void translateInPlace(const float3E& translate, float3Vec& src);
		void translateRoiCenterInPlace(const float3E& dst_center, const intVec& roi, float3Vec& src);
		void translateAndScaleInPlace(const float3E& translate, double scale, float3Vec& src);
		void scaleAndTranslateInPlace(double scale, const float3E& translate, float3Vec& src);
		void scaleAndTranslateInPlace(double scale, const float3E& scale_center, const float3E& translate, float3Vec& src);
		void scaleInCenterAndTranslateInPlace(double scale, const float3E& translate, float3Vec& src);
		void getRotationInPlace(const mat3f& rot, float3Vec& pos);
	}
}

#endif
