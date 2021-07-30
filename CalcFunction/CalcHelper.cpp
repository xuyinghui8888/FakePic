#include "CalcHelper.h"
#include "../Basic/SafeGuard.h"
using namespace CGP;

void CalcHelper::keepRatioStrechX(const floatVec& src, const intVec& roi, int left_idx, int right_idx,
	float scale, floatVec& res)
{
	//src xyxy order
	float src_x = src[right_idx * 2] - src[left_idx * 2];
	float dst_x = src_x * scale;
	floatVec ratio_x;
	for (int i = 0; i < roi.size(); i++)
	{
		float dist_i_left = src[roi[i]*2] - src[left_idx * 2];
		ratio_x.push_back(dist_i_left / (1.0*src_x));
	}

	float dst_x_left = res[left_idx * 2];
	//get new pos
	for (int i = 0; i < roi.size(); i++)
	{
		res[roi[i] * 2] = dst_x * ratio_x[i] + dst_x_left;
	}
	//end res
}

double CalcHelper::getEigenVectorDis(const MeshCompress& base, intVec& src_idx, const intVec& dst_idx)
{
	float3Vec src, dst;
	MeshTools::getSlice(base.pos_, src_idx, src);
	MeshTools::getSlice(base.pos_, dst_idx, dst);
	return getEigenVectorDis(src, dst);
}

