#ifndef IMAGE_PROJECTION_H
#define IMAGE_PROJECTION_H
#include "../Basic/CGPBaseHeader.h"
#include "../Basic/MeshHeader.h"
namespace CGP
{
	class Projection
	{
	public:
		Projection(float fx, float fy, float cx, float cy);
		cvMatD intrisic_;
		float fx_;
		float fy_;
		float cx_;
		float cy_;

		void getMeshToImageRT(const MeshCompress& src, const intVec& mesh_idx,
			const intVec& used_vec, const vecF& landmark_xy, cvMatD& rvec, cvMatD& tvec);
	};

}

#endif
