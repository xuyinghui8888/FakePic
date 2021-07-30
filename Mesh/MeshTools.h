#ifndef __MESHTOOLS_H__
#define __MESHTOOLS_H__
#include "../Basic/CGPBaseHeader.h"
#include "MeshCompress.h"
namespace CGP
{
	//only due with triangle mesh
	namespace MeshTools
	{
		intVec getSrcToDstMatch(const MeshCompress& mesh, const intVec& src, const intVec& dst, double match_thres);
		intVec getSrcToDstMatchKeepSign(const MeshCompress& mesh, const intVec& src, const intVec& dst, double match_thres, bool match_same = false);
		intVec getSrcToDstMatchKeepSignTopN(const MeshCompress& mesh, const intVec& src, int top_n, const intVec& dst, double match_thres, bool match_same = false);
		intVec getMoveIdx(const MeshCompress& src, const MeshCompress& dst, double thres);
		intVec getFixIdx(const MeshCompress& src, const MeshCompress& dst, double thres);
		//from unity(src) to maya(dst) mapping
		intVec getMatchBasedOnUVAndPos(const MeshCompress& src, const MeshCompress& dst, 
			double pos_thres, double uv_thres);
		intVec getMatchBasedOnUV(const MeshCompress& src, const MeshCompress& dst, double uv_thres);
		//from all to part
		intVec getMatchBasedOnPosDstToSrc(const MeshCompress& src, const MeshCompress& dst,
			double pos_thres, bool hard_thres = false);
		intVec getMatchBasedOnPosSrcToDst(const MeshCompress& src, const MeshCompress& dst,
			double pos_thres, bool hard_thres = false);
		intVec forceXMatch(const MeshCompress& src, const MeshCompress& dst, double pos_thres);
		intVec getMatchBasedOnPosRoi(const MeshCompress& src, const intVec& src_roi, const intVec& dst_roi, double pos_thres, bool hard_thres = false);
		intVec getMatchBasedOnPosRoiSimX(const MeshCompress& src, const intVec& src_roi, const intVec& dst_roi, double pos_thres, bool hard_thres = false);
		intVec getMatchAnyUV(const MeshCompress& src, const MeshCompress& dst, double uv_thres, bool hard_thres = false);
		//double match, match src--->dst, dst---->src must match
		intVec getMatchRawUV(const MeshCompress& src, const MeshCompress& dst, double thres);
		intVec getDoubleMatchBasedOnPos(const MeshCompress& src, const MeshCompress& dst,
			double pos_thres, bool hard_thres = false);
		void getNormal(const float3Vec& pos, vecD& normal);
		void getBoundingBox(const float3Vec& xyz, doubleVec& xyz_min, doubleVec& xyz_max);
		void getBoundingBox(const std::vector<float3Vec>& xyz, doubleVec& xyz_min, doubleVec& xyz_max);
		void getCenter(const float3Vec& xyz, float3E& center);
		void getCenter(const float3Vec& xyz, const intVec& roi, float3E& center);

		void putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
			const intVec& dst_idx, MeshCompress& res, const float3E& dir = float3E(1,1,1));

		void putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst, 
			const intVec& dst_idx, MeshCompress& res, double& scale, float3E& translate);

		void putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
			const intVec& dst_idx, const float3E& dir, MeshCompress& res, double& scale, float3E& scale_center, float3E& translate);

		void putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
			const intVec& dst_idx, MeshCompress& res, double& scale, float3E& scale_center, float3E& translate);
		
		void putSrcToDstFixScale(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
			const intVec& dst_idx, MeshCompress& res, const double& scale);
		
		void putSrcWithScaleTranslate(const MeshCompress& src, const double& scale, const float3E& translate, 
			MeshCompress& res);

		void getScale3Dim(const float3Vec& src, const float3Vec& dst, const intVec& roi, float3E& scale);
		void replaceVertexInPlace(MeshCompress& src, const MeshCompress& dst, const intVec& roi);
		void scaleBSInPlace(const MeshCompress&src, double scale, MeshCompress& dst);
		//put roi indexs and pass mapping
		template<class T>
		void getSlice(const T& src, const intVec& roi, T& res)
		{
			int max_pos = src.size();
			res.clear();
			for (auto i : roi)
			{
				if (i >= max_pos || i < 0)
				{
					LOG(ERROR) << "roi failed, out of range£¬ put 0 instead." << std::endl;
				}
				else
				{
					res.push_back(src[i]);
				}
			}
		}		
	};
}
#endif