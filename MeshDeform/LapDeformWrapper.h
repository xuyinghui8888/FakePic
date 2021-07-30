#ifndef __LAPLACIANDEFORM_WRAPPER_H__
#define __LAPLACIANDEFORM_WRAPPER_H__
#include "../Mesh/MeshCompress.h"
#include "LaplacianDeformation.h"
namespace CGP
{
	namespace SIMDEFORM
	{
		void moveHandle(const MeshCompress& src, const intVec& src_idx,
			const MeshCompress& dst, const intVec& dst_idx, MeshCompress& res);

		void moveHandle(const MeshCompress& src, const intVec& src_idx,
			const float3Vec& src_pos, const intVec& fix_idx, const intVec& src_fix_pair, MeshCompress& res);

		void moveHandle(const MeshCompress& src, const intVec& src_idx,
			const MeshCompress& dst, const intVec& dst_idx, 
			const intVec& fix_idx, const intVec& src_fix_pair, MeshCompress& res);

		void moveHandle(const MeshCompress& src, const MeshCompress& dst, 
			const intVec& src_to_dst, bool is_src_to_dst, MeshCompress& res);

		void replaceHandle(const MeshCompress& src, const intVec& src_idx,
			const MeshCompress& dst, const intVec& dst_idx, MeshCompress& res);
	}
}
#endif