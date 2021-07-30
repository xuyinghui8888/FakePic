#include "../Basic/CGPBaseHeader.h"
#include "../Mapping/Mapping.h"
#include "LapDeformWrapper.h"
#include "LaplacianDeformation.h"

using namespace CGP;
void SIMDEFORM::moveHandle(const MeshCompress& src, const intVec& src_idx,
	const MeshCompress& dst, const intVec& dst_idx, MeshCompress& res)
{
	float3Vec move_to_pos;
	for (int i : dst_idx)
	{
		move_to_pos.push_back(dst.pos_[i]);
	}
	res = src;
	LaplacianDeform move_src;
	move_src.init(res, src_idx, {});
	move_src.deform(move_to_pos, res.pos_);	
}

void SIMDEFORM::replaceHandle(const MeshCompress& src, const intVec& src_idx,
	const MeshCompress& dst, const intVec& dst_idx, MeshCompress& res)
{
	if (src_idx.size() != dst_idx.size())
	{
		LOG(ERROR) << "direct replace error in replaceHandle." << std::endl;
		return;
	}
	res = src;
	int n_size = src_idx.size();
	for (size_t i = 0; i < n_size; i++)
	{
		res.pos_[src_idx[i]] = dst.pos_[dst_idx[i]];
	}
}

void SIMDEFORM::moveHandle(const MeshCompress& src, const intVec& src_idx,
	const float3Vec& src_pos, const intVec& fix_idx, const intVec& src_fix_pair, MeshCompress& res)
{
	//safecheck
	auto min_max_src_idx = std::minmax_element(src_idx.begin(), src_idx.end());
	auto min_max_fix_idx = std::minmax_element(fix_idx.begin(), fix_idx.end());

	if (*(min_max_src_idx.second) >= src.n_vertex_ || *(min_max_src_idx.first) < 0)
	{
		LOG(ERROR) << "src check failed." << std::endl;
	}

	if (*(min_max_fix_idx.second) >= src.n_vertex_ || *(min_max_fix_idx.first) < 0)
	{
		LOG(ERROR) << "fix check failed." << std::endl;
	}

	float3Vec move_to_pos = src_pos;
	res = src;
	LaplacianDeform move_src;
	move_src.init(res, src_idx, fix_idx, src_fix_pair);
	move_src.deform(move_to_pos, res.pos_);
}

void SIMDEFORM::moveHandle(const MeshCompress& src, const intVec& src_idx,
	const MeshCompress& dst, const intVec& dst_idx, 
	const intVec& fix_idx, const intVec& src_fix_pair, MeshCompress& res)
{
	//safecheck
	auto min_max_src_idx = std::minmax_element(src_idx.begin(), src_idx.end());
	auto min_max_dst_idx = std::minmax_element(dst_idx.begin(), dst_idx.end());
	auto min_max_fix_idx = std::minmax_element(fix_idx.begin(), fix_idx.end());
	
	if (*(min_max_src_idx.second) >= src.n_vertex_ || *(min_max_src_idx.first) < 0)
	{
		LOG(ERROR) << "src check failed." << std::endl;
	}

	if (!fix_idx.empty() && (*(min_max_fix_idx.second) >= src.n_vertex_ || *(min_max_fix_idx.first) < 0))
	{
		LOG(ERROR) << "fix check failed." << std::endl;
	}

	if (*(min_max_dst_idx.second) >= dst.n_vertex_ || *(min_max_dst_idx.first) < 0)
	{
		LOG(ERROR) << "src check failed." << std::endl;
	}

	if (src_idx.size() != dst_idx.size())
	{
		LOG(ERROR) << "src_idx != dst_idx" << std::endl;
	}
		
	
	float3Vec move_to_pos;
	for (int i : dst_idx)
	{
		move_to_pos.push_back(dst.pos_[i]);
	}
	res = src;
	LaplacianDeform move_src;
	move_src.init(res, src_idx, fix_idx, src_fix_pair);
	move_src.deform(move_to_pos, res.pos_);
}

void SIMDEFORM::moveHandle(const MeshCompress& src, const MeshCompress& dst,
	const intVec& src_to_dst, bool is_src_to_dst, MeshCompress& res)
{
	intX2Vec double_src_to_dst_mapping;
	double_src_to_dst_mapping = MAP::singleToDoubleMap(src_to_dst);
	if (is_src_to_dst == false)
	{
		std::swap(double_src_to_dst_mapping[0], double_src_to_dst_mapping[1]);
	}
	
	moveHandle(src, double_src_to_dst_mapping[0], dst, double_src_to_dst_mapping[1], res);
}

