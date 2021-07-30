#include "RT.h"
#include "../Basic/SafeGuard.h"
#include "../Mesh/MeshTools.h"
using namespace CGP;
//100w times, Rodrigues 520ms, cvRodrigues2 450ms
cv::Mat RT::toPrjMat(const cv::Mat& rotation_vector, const cv::Mat& translation_vector)
{
	cv::Mat rMat;
	Rodrigues(rotation_vector, rMat);
	return (cv::Mat_<double>(3, 4) << rMat.at<double>(0, 0), rMat.at<double>(0, 1), rMat.at<double>(0, 2), translation_vector.at<double>(0),
		rMat.at<double>(1, 0), rMat.at<double>(1, 1), rMat.at<double>(1, 2), translation_vector.at<double>(1),
		rMat.at<double>(2, 0), rMat.at<double>(2, 1), rMat.at<double>(2, 2), translation_vector.at<double>(2));
}
//80times per frame

cv::Mat RT::toPrjMat(const double* const xi)
{
	cv::Mat rotation_vector = cv::Mat_<double>(3, 1);
	rotation_vector.at<double>(0, 0) = xi[0];
	rotation_vector.at<double>(1, 0) = xi[1];
	rotation_vector.at<double>(2, 0) = xi[2];
	cv::Mat rMat;
	cv::Rodrigues(rotation_vector, rMat);

	return (cv::Mat_<double>(3, 4) << rMat.at<double>(0, 0), rMat.at<double>(0, 1), rMat.at<double>(0, 2), xi[3],
		rMat.at<double>(1, 0), rMat.at<double>(1, 1), rMat.at<double>(1, 2), xi[4],
		rMat.at<double>(2, 0), rMat.at<double>(2, 1), rMat.at<double>(2, 2), xi[5]);
}
//600 times per frame

void RT::toPrjMat(const double* const xi, cv::Mat& J, cv::Mat& RT)
{
	static double oldXi[6];
	static double oldJ[72];
	static double oldRT[12];

	static const bool enableCache = true;
	if (enableCache) {
		if (memcmp(oldXi, xi, 6 * sizeof(double)) == 0) {
			memcpy(J.data, oldJ, 72 * sizeof(double));
			memcpy(RT.data, oldRT, 12 * sizeof(double));
			return;
		}
	}

	cv::Mat rotation_vector = cv::Mat_<double>(3, 1);
	SG::safeMemcpy(rotation_vector.data, xi, 3 * sizeof(double));
	cv::Mat R, R_;
	Rodrigues(rotation_vector, R, R_);
	memcpy(RT.data, R.data, 3 * sizeof(double)); memcpy(RT.data + 24, xi + 3, sizeof(double));
	memcpy(RT.data + 32, R.data + 24, 3 * sizeof(double)); memcpy(RT.data + 56, xi + 4, sizeof(double));
	memcpy(RT.data + 64, R.data + 48, 3 * sizeof(double)); memcpy(RT.data + 88, xi + 5, sizeof(double));

	for (int i = 0; i < 3; i++) {
		memcpy(J.data + 96 * i, R_.data + 72 * i, 24);
		memcpy(J.data + 96 * i + 32, R_.data + 72 * i + 24, 24);
		memcpy(J.data + 96 * i + 64, R_.data + 72 * i + 48, 24);
	}
	J.at<double>(3, 3) = 1;
	J.at<double>(4, 7) = 1;
	J.at<double>(5, 11) = 1;

	if (enableCache) {
		memcpy(oldXi, xi, 6 * sizeof(double));
		memcpy(oldJ, J.data, 72 * sizeof(double));
		memcpy(oldRT, RT.data, 12 * sizeof(double));
	}
}

cv::Mat RT::toFullPrjMat(const cv::Mat& rotation_vector, const cv::Mat& translation_vector)
{
	cv::Mat rMat;
	cv::Rodrigues(rotation_vector, rMat);
	return (cv::Mat_<double>(4, 4) << rMat.at<double>(0, 0), rMat.at<double>(0, 1), rMat.at<double>(0, 2), translation_vector.at<double>(0),
		rMat.at<double>(1, 0), rMat.at<double>(1, 1), rMat.at<double>(1, 2), translation_vector.at<double>(1),
		rMat.at<double>(2, 0), rMat.at<double>(2, 1), rMat.at<double>(2, 2), translation_vector.at<double>(2),
		0, 0, 0, 1);
}

double RT::getScale(const float3Vec& src, const float3Vec& dst)
{
	//get center
	if (!SG::checkSameSize(src, dst))
	{
		LOG(ERROR) << "size don't fit" << std::endl;
		return -1;
	}
	int num = src.size();
	float num_1 = 1.0 / num;
	float3E src_center, dst_center;
	getCenter(src, src_center);
	getCenter(dst, dst_center);
	float3Vec src_shift, dst_shift;
	shiftCenter(src, -src_center, src_shift);
	shiftCenter(dst, -dst_center, dst_shift);
	double scale_sum = 0;
	for (int i = 0; i < num; i++)
	{
		scale_sum += safeDiv(dst_shift[i].norm(), src_shift[i].norm(), 0);
	}
	return scale_sum * num_1;
}

void RT::getScale3Dim(const float3Vec& src, const float3Vec& dst, float3E& res)
{
	res = float3E::Zero();
	//get bounding box for each src & dst
	doubleVec src_min, src_max, dst_min, dst_max;
	MeshTools::getBoundingBox(src, src_min, src_max);
	MeshTools::getBoundingBox(dst, dst_min, dst_max);

	for (int i = 0; i < 3; i++)
	{
		res[i] = safeDiv(dst_max[i] - dst_min[i], src_max[i] - src_min[i], 0);
	}
}

void  RT::getTranslate(const float3Vec& src, const intVec& src_set, const float3Vec& dst, const intVec& dst_set, float3E& trans)
{
	trans.setConstant(0);
	if (src_set.size() != dst_set.size() || src_set.empty())
	{
		LOG(ERROR) << "set size not fit." << std::endl;
		return;
	}
	int n_pair = src_set.size();
	for (int i = 0; i < n_pair; i++)
	{
		trans = trans + dst[dst_set[i]] - src[src_set[i]];
	}
	trans = 1.0 / n_pair * trans;
}

void RT::getTranslate(const float3Vec& src, const float3Vec& dst, float3E& trans)
{
	//directly calculate 
	if (!SG::checkSameSize(src, dst))
	{
		LOG(ERROR) << "size don't fit" << std::endl;
		return;
	}
	trans.setZero();
	int num = src.size();
	float num_1 = 1.0 / num;
	for (int i = 0; i < num; i++)
	{
		trans = trans + dst[i] - src[i];
	}
	trans = trans * num_1;
}

void RT::scaleInPlace(double scale, float3Vec& src)
{
	float3E src_center;
	getCenter(src, src_center);
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = scale* src[i] + (1-scale) * src_center;
	}
}

void RT::scaleInPlace(double scale, float3Vec& src, float3E& src_center)
{
	//float3E src_center;
	getCenter(src, src_center);
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = scale * src[i] + (1 - scale) * src_center;
	}
}

void RT::scaleInPlace(const float3E& scale, float3Vec& src)
{
	float3E src_center;
	getCenter(src, src_center);
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			src[i][iter_dim] = scale[iter_dim] * src[i][iter_dim] + (1 - scale[iter_dim]) * src_center[iter_dim];
		}
	}
}


void RT::scaleInPlace(const float3E& scale, const intVec& roi, float3Vec& src)
{
	float3Vec roi_slice;
	MeshTools::getSlice(src, roi, roi_slice);
	RT::scaleInPlace(scale, roi_slice);
	for (int i = 0; i< roi.size(); i++)
	{
		src[roi[i]] = roi_slice[i];
	}
}

void RT::scaleInPlaceNoShift(double scale, float3Vec& src)
{
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = scale * src[i];
	}
}

void RT::transformInPlace(const mat4f& t, float3Vec& src)
{
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		float4E expand(src[i][0], src[i][1], src[i][2], 1);
		float4E res = t * expand;
		src[i] = res.head<3>();
	}
}

void RT::translateInPlace(const float3E& t, float3Vec& src)
{
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = src[i] + t;
	}
}

void RT::translateRoiCenterInPlace(const float3E& dst_center, const intVec& roi, float3Vec& src)
{
	float3Vec roi_slice;
	float3E roi_center;
	MeshTools::getSlice(src, roi, roi_slice);
	RT::getCenter(roi_slice, roi_center);
	float3E translate = dst_center - roi_center;
	RT::translateInPlace(translate, roi_slice);
	for (int i = 0; i < roi.size(); i++)
	{
		src[roi[i]] = roi_slice[i];
	}
}

void RT::translateAndScaleInPlace(const float3E& t, double scale, float3Vec& src)
{
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = (src[i] + t)*scale;
	}
}

void RT::scaleAndTranslateInPlace(double scale, const float3E& t, float3Vec& src)
{
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = (src[i] + t)*scale;
	}
}

void RT::scaleAndTranslateInPlace(double scale, const float3E& scale_center, const float3E& t, float3Vec& src)
{
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		src[i] = (src[i] - scale_center)*scale + scale_center + t;
	}
}

void RT::scaleInCenterAndTranslateInPlace(double scale, const float3E& t, float3Vec& src)
{
	RT::scaleInPlace(scale, src);
	RT::translateInPlace(t, src);
}

void RT::getCenter(const float3Vec& points, float3E& center)
{
	center.setZero();
	for (int i = 0; i < points.size(); i++)
	{
		center = center + points[i];
	}
	float scale_1 = 1.0 / points.size();
	center = scale_1 * center;
}

void RT::shiftCenter(const float3Vec& src, const float3E& shift, float3Vec& dst)
{
	dst = src;
#pragma omp parallel for
	for (int i = 0; i < src.size(); i++)
	{
		dst[i] = src[i] + shift;
	}
}

void RT::getRotationInPlace(const mat3f& rot, float3Vec& pos)
{
#pragma omp parallel for
	for (int i = 0; i < pos.size(); i++)
	{
		pos[i] = rot * pos[i];
	}
}

