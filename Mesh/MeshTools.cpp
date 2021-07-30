#include "MeshTools.h"
#include "../FileIO/FileIO.h"
#include "../RT/RT.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Mapping/Mapping.h"
using namespace CGP;
intVec MeshTools::getSrcToDstMatch(const MeshCompress& mesh, const intVec& src, const intVec& dst, double match_thres)
{
	int n_src = src.size();
	int n_dst = dst.size();
	intVec map_src_dst(2*n_src, -1);
	intVec dst_visited(n_dst, 0);

	for (size_t i = 0; i < n_src; i++)
	{
		int src_id = src[i];
		//matching pos is not valid
		floatVec match_dis(n_dst, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_dst; j++)
		{
			int dst_id = dst[j];
			if (dst_visited[dst_id])
			{

			}
			else
			{
				match_dis[j] = (mesh.pos_[dst_id] - mesh.pos_[src_id]).norm();
			}
		}
		auto res = std::minmax_element(match_dis.begin(), match_dis.end());
		int match_id = res.first - match_dis.begin();
		int dst_id = dst[match_id];
		if (match_dis[match_id] < match_thres)
		{
			dst_visited[match_id] = 1;
			map_src_dst[2 * i] = src_id;
			map_src_dst[2 * i + 1] = dst_id;
		}		
	}
	return map_src_dst;
}

intVec MeshTools::getSrcToDstMatchKeepSign(const MeshCompress& mesh, const intVec& src, const intVec& dst, double match_thres, bool match_same)
{
	int n_src = src.size();
	int n_dst = dst.size();
	intVec map_src_dst(2 * n_src, -1);
	intVec dst_visited(n_dst, 0);

	for (size_t i = 0; i < n_src; i++)
	{
		int src_id = src[i];
		//matching pos is not valid
		floatVec match_dis(n_dst, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_dst; j++)
		{
			int dst_id = dst[j];
			if (dst_visited[j])
			{

			}
			else
			{
				if ((mesh.pos_[src_id] - mesh.pos_[dst_id]).y() > 0)
				{
					match_dis[j] = (mesh.pos_[dst_id] - mesh.pos_[src_id]).norm();
				}
				else
				{
					match_dis[j] = (mesh.pos_[dst_id] - mesh.pos_[src_id]).norm()+10;
				}
			}
		}
		auto res = std::minmax_element(match_dis.begin(), match_dis.end());
		int match_id = res.first - match_dis.begin();
		int dst_id = dst[match_id];
		if (match_dis[match_id] < match_thres)
		{
			if (match_same == false)
			{
				dst_visited[match_id] = 1;
			}
			map_src_dst[2 * i] = src_id;
			map_src_dst[2 * i + 1] = dst_id;
		}
	}
	return map_src_dst;
}

intVec MeshTools::getSrcToDstMatchKeepSignTopN(const MeshCompress& mesh, const intVec& src, int top_n, const intVec& dst, double match_thres, bool match_same)
{
	int n_src = src.size();
	int n_dst = dst.size();
	intVec map_src_dst;
	intVec dst_visited(n_dst, 0);

	for (size_t i = 0; i < n_src; i++)
	{
		int src_id = src[i];
		//matching pos is not valid
		floatVec match_dis(n_dst, INT_MAX);
		for (int j = 0; j < n_dst; j++)
		{
			int dst_id = dst[j];
			if (dst_visited[j])
			{

			}
			else
			{
				if ((mesh.pos_[src_id] - mesh.pos_[dst_id]).y() > 0)
				{
					match_dis[j] = (mesh.pos_[dst_id] - mesh.pos_[src_id]).norm();
				}
				else
				{
					match_dis[j] = (mesh.pos_[dst_id] - mesh.pos_[src_id]).norm() + 10;
				}
			}
		}
		for (int iter_top = 0; iter_top < top_n; iter_top++)
		{
			auto res = std::minmax_element(match_dis.begin(), match_dis.end());
			int match_id = res.first - match_dis.begin();
			int dst_id = dst[match_id];
			if (match_dis[match_id] < match_thres)
			{
				if (match_same == false)
				{
					dst_visited[match_id] = 1;
				}
				map_src_dst.push_back(src_id);
				map_src_dst.push_back(dst_id);
			}
			match_dis[match_id] = INTMAX_MAX;
		}		
	}
	return map_src_dst;
}

intVec MeshTools::getMoveIdx(const MeshCompress& src, const MeshCompress& dst, double thres)
{
	floatVec dis(src.pos_.size(), -1);
#pragma omp parallel for
	for (int i = 0; i < src.pos_.size(); i++)
	{
		dis[i] = (src.pos_[i] - dst.pos_[i]).norm();
	}
	intVec move;
	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (dis[i] > thres)
		{
			move.push_back(i);
		}
	}
	return move;
}

intVec MeshTools::getFixIdx(const MeshCompress& src, const MeshCompress& dst, double thres)
{
	floatVec dis(src.pos_.size(), -1);
#pragma omp parallel for
	for (int i = 0; i < src.pos_.size(); i++)
	{
		dis[i] = (src.pos_[i] - dst.pos_[i]).norm();
	}
	intVec fix;
	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (dis[i] < thres)
		{
			fix.push_back(i);
		}
	}
	return fix;
}

void MeshTools::getNormal(const float3Vec& pos, vecD& normal)
{
	if (pos.empty())
	{
		LOG(INFO) << "pos is empty." << std::endl;
		return;
	}
	int n_pos = pos.size();
	matD A(n_pos*(n_pos+1)/2 + 1, 3);
	A.setConstant(0);
	int count = 0;
	for (int i = 0; i < n_pos; i++)
	{
		for (int j = i; j < n_pos; j++)
		{
			float3E line = pos[i] - pos[j];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(count, iter_dim) = line(iter_dim);
			}
			count++;
		}		
	}
	double weight = 1e6;
	vecD B(count+1);
	B.setConstant(0);
	A(count, 2) = weight;
	B(count) = weight;

	DenseSolver dense_solver;
	dense_solver.compute(A);
	normal = dense_solver.solve(B);
}

void MeshTools::getBoundingBox(const std::vector<float3Vec>& xyz, doubleVec& xyz_min, doubleVec& xyz_max)
{
	if (xyz.empty())
	{
		LOG(WARNING) << "xyz empty." << std::endl;
		xyz_min = doubleVec{ 0.0,0.0,0.0 };
		xyz_max = doubleVec{ 0.0,0.0,0.0 };
	}
	else
	{
		getBoundingBox(xyz[0], xyz_min, xyz_max);
		for (int i = 1; i < xyz.size(); i++)
		{
			doubleVec xyz_min_iter, xyz_max_iter;
			getBoundingBox(xyz[i], xyz_min_iter, xyz_max_iter);
			xyz_min = CalcHelper::multiMin(xyz_min, xyz_min_iter);
			xyz_max = CalcHelper::multiMax(xyz_max, xyz_max_iter);
		}
	}
}

void MeshTools::getBoundingBox(const float3Vec& xyz, doubleVec& xyz_min, doubleVec& xyz_max)
{
	xyz_min = { 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX };
	xyz_max = { -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX };
#pragma omp parallel for 
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		for (int iter_idx = 0; iter_idx < xyz.size(); iter_idx++)
		{
			float3E pos_i = xyz[iter_idx];
			xyz_min[iter_dim] = DMIN(xyz_min[iter_dim], pos_i(iter_dim));
			xyz_max[iter_dim] = DMAX(xyz_max[iter_dim], pos_i(iter_dim));
		}
	}
}

intVec MeshTools::getMatchBasedOnUVAndPos(const MeshCompress& src, const MeshCompress& dst, 
	double pos_thres, double uv_thres)
{

	if (src.n_vertex_ < dst.n_vertex_)
	{
		LOG(ERROR) << "src && dst is switched." << std::endl;
	}


	if (src.n_uv_ < src.n_vertex_ || dst.n_uv_ < dst.n_vertex_)
	{
		LOG(ERROR) << "must have uv info." << std::endl;
	}


	if (src.tri_.size() != src.tri_uv_.size() || dst.tri_.size() != dst.tri_uv_.size())
	{
		LOG(ERROR) << "triangle is not the same with triangle based uv" << std::endl;
	}

	//get vertex to uv index
	intSetVec src_uv_idx, dst_uv_idx;
	src_uv_idx.resize(src.n_vertex_);
	dst_uv_idx.resize(dst.n_vertex_);
	for (int iter_vertex = 0; iter_vertex < src.n_tri_*3 ; iter_vertex++)
	{
		int vertex_idx = src.tri_[iter_vertex];
		int uv_idx = src.tri_uv_[iter_vertex];
		src_uv_idx[vertex_idx].insert(uv_idx);
	}

	for (int iter_vertex = 0; iter_vertex < dst.n_tri_*3; iter_vertex++)
	{
		int vertex_idx = dst.tri_[iter_vertex];
		int uv_idx = dst.tri_uv_[iter_vertex];
		dst_uv_idx[vertex_idx].insert(uv_idx);
	}

	//matching
	int n_src = src.n_vertex_;
	int n_dst = dst.n_vertex_;
	intVec match_src_dst(n_src, -1);
	for (int i = 0; i < n_src; i++)
	{
		int src_id = i;
		//matching pos is not valid
		floatVec match_pos(n_dst, INT_MAX);
		//save for match_uv minimal
		floatVec match_uv(n_dst, INT_MAX);
		floatVec match_dis(n_dst, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_dst; j++)
		{
			int dst_id = j;			
			match_pos[j] = (src.pos_[src_id] - dst.pos_[dst_id]).norm();
			double min_uv_dis = INT_MAX;
			for (int iter_i : src_uv_idx[i])
			{
				for (int iter_j : dst_uv_idx[j])
				{
					min_uv_dis = DMIN(min_uv_dis, (src.tex_cor_[iter_i] - dst.tex_cor_[iter_j]).norm());
				}
			}
			match_uv[j] = min_uv_dis;
			match_dis[j] = match_uv[j] + match_pos[j];
		}
		auto res = std::minmax_element(match_dis.begin(), match_dis.end());
		int match_id = res.first - match_dis.begin();
		match_src_dst[i] = match_id;
		if (*res.first > pos_thres * uv_thres)
		{
			LOG(INFO) << "*res.second: " << *res.first << std::endl;
			LOG(ERROR) << "res dis larger than threshold." << std::endl;
		}
	}
	return match_src_dst;
}

void MeshTools::getCenter(const float3Vec& xyz, float3E& center)
{
	center = float3E(0, 0, 0);
	if (xyz.empty())
	{
		return;
	}
	for (int i = 0; i < xyz.size(); i++)
	{
		center = center + xyz[i];
	}
	center = center * 1.0 / xyz.size();
}

void MeshTools::getCenter(const float3Vec& xyz, const intVec& roi, float3E& center)
{
	center = float3E(0, 0, 0);
	if (roi.empty())
	{
		return;
	}
	for (int i : roi)
	{
		center = center + xyz[i];
	}
	center = center * 1.0 / roi.size();
}

void MeshTools::putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
	const intVec& dst_idx, MeshCompress& res, const float3E& dir)
{
	res = src;	
	float3Vec dst_pos, src_pos;
	src.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	float scale = RT::getScale(src_pos, dst_pos);
	//LOG(INFO) << "scale: " << scale << std::endl;
	RT::scaleInPlace(scale, res.pos_);
	//get slice for scaled pos
	res.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	float3E translate;
	RT::getTranslate(src_pos, dst_pos, translate);
	translate = translate.cwiseProduct(dir);
	//LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, res.pos_);
}

void MeshTools::putSrcWithScaleTranslate(const MeshCompress& src, const double& scale, const float3E& translate, MeshCompress& res)
{
	res = src;	
	RT::scaleInPlace(scale, res.pos_);
	RT::translateInPlace(translate, res.pos_);
}

void MeshTools::putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
	const intVec& dst_idx, MeshCompress& res, double& scale, float3E& translate)
{
	res = src;
	float3Vec dst_pos, src_pos;
	src.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	scale = RT::getScale(src_pos, dst_pos);
	//LOG(INFO) << "scale: " << scale << std::endl;
	RT::scaleInPlace(scale, res.pos_);
	//get slice for scaled pos
	res.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	RT::getTranslate(src_pos, dst_pos, translate);
	//LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, res.pos_);
}

void MeshTools::putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
	const intVec& dst_idx, const float3E& dir, MeshCompress& res, double& scale, float3E& scale_center, float3E& translate)
{
	res = src;
	float3Vec dst_pos, src_pos;
	src.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	scale = RT::getScale(src_pos, dst_pos);
	//LOG(INFO) << "scale: " << scale << std::endl;
	RT::scaleInPlace(scale, res.pos_, scale_center);
	//get slice for scaled pos
	res.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	RT::getTranslate(src_pos, dst_pos, translate);
	translate = translate.cwiseProduct(dir);
	//LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, res.pos_);
}

void MeshTools::putSrcToDst(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
	const intVec& dst_idx, MeshCompress& res, double& scale, float3E& scale_center, float3E& translate)
{
	res = src;
	float3Vec dst_pos, src_pos;
	src.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	scale = RT::getScale(src_pos, dst_pos);
	//LOG(INFO) << "scale: " << scale << std::endl;
	RT::scaleInPlace(scale, res.pos_, scale_center);
	//get slice for scaled pos
	res.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	RT::getTranslate(src_pos, dst_pos, translate);
	//LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, res.pos_);
}

void MeshTools::putSrcToDstFixScale(const MeshCompress& src, const intVec& src_idx, const MeshCompress& dst,
	const intVec& dst_idx, MeshCompress& res, const double& scale)
{
	res = src;
	float3Vec dst_pos, src_pos;
	src.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	//LOG(INFO) << "scale: " << scale << std::endl;
	RT::scaleInPlace(scale, res.pos_);
	//get slice for scaled pos
	res.getSlice(src_idx, src_pos);
	dst.getSlice(dst_idx, dst_pos);
	float3E translate;
	RT::getTranslate(src_pos, dst_pos, translate);
	//LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, res.pos_);
}

void MeshTools::getScale3Dim(const float3Vec& src, const float3Vec& dst, const intVec& roi, float3E& scale)
{
	float3Vec src_roi, dst_roi;
	MeshTools::getSlice(src, roi, src_roi);
	MeshTools::getSlice(dst, roi, dst_roi);
	RT::getScale3Dim(src_roi, dst_roi, scale);
}

void MeshTools::replaceVertexInPlace(MeshCompress& src, const MeshCompress& dst, const intVec& roi)
{
#pragma omp parallel for
	for (int i = 0; i < roi.size(); i++)
	{
		src.pos_[roi[i]] = dst.pos_[roi[i]];
	}
}

intVec MeshTools::getMatchBasedOnPosDstToSrc(const MeshCompress& src, const MeshCompress& dst, double pos_thres,
	bool hard_thres)
{

	if (src.n_vertex_ < dst.n_vertex_)
	{
		LOG(ERROR) << "src && dst is switched." << std::endl;
	}
	//matching
	int n_src = src.n_vertex_;
	int n_dst = dst.n_vertex_;
	intVec match_dst_src(n_dst, -1);
	for (int i = 0; i < n_dst; i++)
	{
		//matching pos is not valid
		floatVec match_pos(n_src, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_src; j++)
		{
			match_pos[j] = (dst.pos_[i] - src.pos_[j]).norm();
		}
		auto res = std::minmax_element(match_pos.begin(), match_pos.end());
		int match_id = res.first - match_pos.begin();
		if (!hard_thres)
		{
			match_dst_src[i] = match_id;
			if (*res.first > pos_thres)
			{
				LOG(INFO) << "*res.second: " << *res.first << std::endl;
				LOG(ERROR) << "res dis larger than threshold." << std::endl;
			}
		}
		else
		{
			if (*res.first < pos_thres)
			{
				match_dst_src[i] = match_id;
			}
		}		
	}
	return match_dst_src;
}

intVec MeshTools::getMatchBasedOnPosSrcToDst(const MeshCompress& src, const MeshCompress& dst, double pos_thres,
	bool hard_thres)
{
	if (src.n_vertex_ > dst.n_vertex_)
	{
		intVec res = getMatchBasedOnPosDstToSrc(src, dst, pos_thres, hard_thres);
		intVec inv_res = MAP::getReverseMapping(res);
		return inv_res;

	}
	else
	{
		intVec res = getMatchBasedOnPosDstToSrc(dst, src, pos_thres, hard_thres);		
		return res;
	}
}

intVec MeshTools::forceXMatch(const MeshCompress& src, const MeshCompress& dst, double pos_thres)
{
	if (src.pos_.size() != dst.pos_.size())
	{
		LOG(ERROR) << "forceXMatch must have same size" << std::endl;
	}
	int n_size = src.n_vertex_;
	doubleVec src_x(n_size, 0), dst_x(n_size, 0);
	for (int i = 0; i < n_size; i++)
	{
		src_x[i] = src.pos_[i].x();
		dst_x[i] = dst.pos_[i].x();
	}
	intVec src_order(n_size, -1), dst_order(n_size, -1);
	CalcHelper::pairSort(src_x, src_order);
	CalcHelper::pairSort(dst_x, dst_order);

	intVec inv_src_order = MAP::getReverseMapping(src_order);
	intVec inv_dst_order = MAP::getReverseMapping(dst_order);


	intVec src_dst_match(n_size, 0);
	for (int i = 0; i < n_size; i++)
	{
		src_dst_match[inv_src_order[i]] = inv_dst_order[i];
	}
	return src_dst_match;
}



intVec MeshTools::getMatchBasedOnPosRoi(const MeshCompress& src, const intVec& src_roi, const intVec& dst_roi, double pos_thres, bool hard_thres)
{
	MeshCompress src_part = src;
	intVec src_all_to_roi = src_part.keepRoi(src_roi);
	MeshCompress dst_part = src;
	intVec dst_all_to_roi = dst_part.keepRoi(dst_roi);
	//intVec res = getMatchBasedOnPosSrcToDst(src_part, dst_part, pos_thres, hard_thres);
	intVec res = forceXMatch(src_part, dst_part, pos_thres);
	intVec match_src_dst(res.size()*2, 0);

	intVec inv_src_all_to_roi = MAP::getReverseMapping(src_all_to_roi);
	intVec inv_dst_all_to_roi = MAP::getReverseMapping(dst_all_to_roi);

	int n_size = res.size();
	for (int i = 0; i < n_size; i++)
	{
		int inv_map_src_idx = inv_src_all_to_roi[i];
		int inv_map_dst_idx = inv_dst_all_to_roi[res[i]];
		
		if (inv_map_src_idx<0 || inv_map_src_idx>src.n_vertex_ || inv_map_dst_idx<0 || inv_map_dst_idx>src.n_vertex_)
		{
			LOG(ERROR) << "safe check failed." << std::endl;
		}
		else
		{
			match_src_dst[2 * i + 0] = inv_map_src_idx;
			match_src_dst[2 * i + 1] = inv_map_dst_idx;
		}
	}
	return match_src_dst;

}

intVec MeshTools::getMatchBasedOnPosRoiSimX(const MeshCompress& src, const intVec& src_roi, const intVec& dst_roi, double pos_thres, bool hard_thres)
{
	MeshCompress src_part = src;
	intVec src_all_to_roi = src_part.keepRoi(src_roi);
	MeshCompress dst_part = src;
	intVec dst_all_to_roi = dst_part.keepRoi(dst_roi);
	intVec res = getMatchBasedOnPosSrcToDst(src_part, dst_part, pos_thres, hard_thres);
	intVec match_src_dst(res.size() * 2, 0);

	intVec inv_src_all_to_roi = MAP::getReverseMapping(src_all_to_roi);
	intVec inv_dst_all_to_roi = MAP::getReverseMapping(dst_all_to_roi);

	int n_size = res.size();
	for (int i = 0; i < n_size; i++)
	{
		int inv_map_src_idx = inv_src_all_to_roi[i];
		int inv_map_dst_idx = inv_dst_all_to_roi[res[i]];

		if (inv_map_src_idx<0 || inv_map_src_idx>src.n_vertex_ || inv_map_dst_idx<0 || inv_map_dst_idx>src.n_vertex_)
		{
			LOG(ERROR) << "safe check failed." << std::endl;
		}
		else
		{
			match_src_dst[2 * i + 0] = inv_map_src_idx;
			match_src_dst[2 * i + 1] = inv_map_dst_idx;
		}
	}
	return match_src_dst;

}

intVec MeshTools::getDoubleMatchBasedOnPos(const MeshCompress& src, const MeshCompress& dst, double pos_thres,
	bool hard_thres)
{
	if (src.pos_.empty() || dst.pos_.empty())
	{
		LOG(ERROR) << "src && dst is switched." << std::endl;
	}

	//matching
	int n_src = src.n_vertex_;
	int n_dst = dst.n_vertex_;
	intVec match_dst_src(n_dst, -1);
	intVec match_src_dst(n_src, -1);
	for (int i = 0; i < n_dst; i++)
	{
		//matching pos is not valid
		floatVec match_pos(n_src, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_src; j++)
		{
			match_pos[j] = (dst.pos_[i] - src.pos_[j]).norm();
		}
		auto res = std::minmax_element(match_pos.begin(), match_pos.end());
		int match_id = res.first - match_pos.begin();
		if (match_id < 0 || match_id >= n_src)
		{
			LOG(ERROR) << "match error" << std::endl;
		}
		if (!hard_thres)
		{
			match_dst_src[i] = match_id;
			if (*res.first > pos_thres)
			{
				LOG(INFO) << "*res.second: " << *res.first << std::endl;
				LOG(ERROR) << "res dis larger than threshold." << std::endl;
			}
		}
		else
		{
			if (*res.first < pos_thres)
			{
				match_dst_src[i] = match_id;
			}
		}
	}

	//map reverse
	for (int i = 0; i < n_src; i++)
	{
		//matching pos is not valid
		floatVec match_pos(n_dst, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_dst; j++)
		{
			match_pos[j] = (src.pos_[i] - dst.pos_[j]).norm();
		}
		auto res = std::minmax_element(match_pos.begin(), match_pos.end());
		int match_id = res.first - match_pos.begin();
		if (match_id < 0 || match_id >= n_dst)
		{
			LOG(ERROR) << "match error" << std::endl;
		}
		if (!hard_thres)
		{
			match_src_dst[i] = match_id;
			if (*res.first > pos_thres)
			{
				LOG(INFO) << "*res.second: " << *res.first << std::endl;
				LOG(ERROR) << "res dis larger than threshold." << std::endl;
			}
		}
		else
		{
			if (*res.first < pos_thres)
			{
				match_src_dst[i] = match_id;
			}
		}
	}	

	//filter
	int match_count = 0;
	intVec src_to_dst(n_src, -1);
	for (int i = 0; i < n_src; i++)
	{
		int idx_dst = match_src_dst[i];
		int idx_src = -1;
		if (idx_dst >= 0 && idx_dst < n_dst)
		{
			idx_src = match_dst_src[idx_dst];
			if (i == idx_src)
			{
				src_to_dst[i] = idx_dst;
				match_count++;
			}
		}		
	}

	LOG(INFO) << "double map: " << match_count << std::endl;

	return src_to_dst;
}

intVec MeshTools::getMatchBasedOnUV(const MeshCompress& src, const MeshCompress& dst, double uv_thres)
{
	if (src.n_vertex_ <= 0 || dst.n_vertex_ <= 0)
	{
		LOG(ERROR) << "vertex empty" << std::endl;
	}

	if (src.n_uv_ < src.n_vertex_ || dst.n_uv_ < dst.n_vertex_)
	{
		LOG(ERROR) << "must have uv info." << std::endl;
	}

	if (src.tri_.size() != src.tri_uv_.size() || dst.tri_.size() != dst.tri_uv_.size())
	{
		LOG(ERROR) << "triangle is not the same with triangle based uv" << std::endl;
	}

	//get vertex to uv index
	intSetVec src_uv_idx, dst_uv_idx;
	src_uv_idx.resize(src.n_vertex_);
	dst_uv_idx.resize(dst.n_vertex_);
	for (int iter_vertex = 0; iter_vertex < src.n_tri_ * 3; iter_vertex++)
	{
		int vertex_idx = src.tri_[iter_vertex];
		int uv_idx = src.tri_uv_[iter_vertex];
		src_uv_idx[vertex_idx].insert(uv_idx);
	}

	for (int iter_vertex = 0; iter_vertex < dst.n_tri_ * 3; iter_vertex++)
	{
		int vertex_idx = dst.tri_[iter_vertex];
		int uv_idx = dst.tri_uv_[iter_vertex];
		dst_uv_idx[vertex_idx].insert(uv_idx);
	}

	//matching
	int n_src = src.n_vertex_;
	int n_dst = dst.n_vertex_;
	intVec match_src_dst(n_src, -1);
	for (int i = 0; i < n_src; i++)
	{
		int src_id = i;
		//matching pos is not valid
		//save for match_uv minimal
		floatVec match_uv(n_dst, INT_MAX);
		floatVec match_dis(n_dst, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_dst; j++)
		{
			int dst_id = j;
			double min_uv_dis = INT_MAX;
			for (int iter_i : src_uv_idx[i])
			{
				for (int iter_j : dst_uv_idx[j])
				{
					min_uv_dis = DMIN(min_uv_dis, (src.tex_cor_[iter_i] - dst.tex_cor_[iter_j]).norm());
				}
			}
			match_uv[j] = min_uv_dis;
			match_dis[j] = match_uv[j];
		}
		auto res = std::minmax_element(match_dis.begin(), match_dis.end());
		int match_id = res.first - match_dis.begin();
		match_src_dst[i] = match_id;
		if (*res.first > uv_thres)
		{
			LOG(INFO) << "*res.second: " << *res.first << std::endl;
			LOG(ERROR) << "res dis larger than threshold." << std::endl;
		}
	}
	return match_src_dst;
}

intVec MeshTools::getMatchRawUV(const MeshCompress& src, const MeshCompress& dst, double thres)
{
	if (src.pos_.empty() || dst.pos_.empty() || src.tri_uv_.empty() || dst.tri_uv_.empty())
	{
		LOG(INFO) << "src.n_vertex_ :" << src.n_vertex_ << std::endl;
		LOG(INFO) << "dst.n_vertex_ :" << dst.n_vertex_ << std::endl;
		LOG(INFO) << "src.n_tri_uv_ :" << src.n_uv_ << std::endl;
		LOG(INFO) << "dst.n_tri_uv_ :" << dst.n_uv_ << std::endl;
		LOG(ERROR) << "data check failed." << std::endl;
		return {};
	}

	//get vertex to uv index
	intSetVec src_uv_idx, dst_uv_idx;
	src_uv_idx.resize(src.n_vertex_);
	dst_uv_idx.resize(dst.n_vertex_);
	for (int iter_vertex = 0; iter_vertex < src.n_tri_ * 3; iter_vertex++)
	{
		int vertex_idx = src.tri_[iter_vertex];
		int uv_idx = src.tri_uv_[iter_vertex];
		src_uv_idx[vertex_idx].insert(uv_idx);
	}

	for (int iter_vertex = 0; iter_vertex < dst.n_tri_ * 3; iter_vertex++)
	{
		int vertex_idx = dst.tri_[iter_vertex];
		int uv_idx = dst.tri_uv_[iter_vertex];
		dst_uv_idx[vertex_idx].insert(uv_idx);
	}

	//matching
	int n_src = src.n_vertex_;
	int n_dst = dst.n_vertex_;
	intVec match_src_dst(n_src, -1);
	for (int i = 0; i < n_src; i++)
	{
		int src_id = i;
		//matching pos is not valid
		//save for match_uv minimal
		floatVec match_uv(n_dst, INT_MAX);
		floatVec match_dis(n_dst, INT_MAX);
#pragma omp parallel for
		for (int j = 0; j < n_dst; j++)
		{
			int dst_id = j;
			double min_uv_dis = INT_MAX;
			for (int iter_i : src_uv_idx[i])
			{
				for (int iter_j : dst_uv_idx[j])
				{
					//todo use same side
					if (src.pos_[i].x()*dst.pos_[j].x() > -1e-5)
					{
						min_uv_dis = DMIN(min_uv_dis, (src.tex_cor_[iter_i] - dst.tex_cor_[iter_j]).norm());
					}
				}
			}
			match_uv[j] = min_uv_dis;
			match_dis[j] = match_uv[j];
		}
		auto res = std::minmax_element(match_dis.begin(), match_dis.end());
		int match_id = res.first - match_dis.begin();
		match_src_dst[i] = match_id;
		if (*res.first > thres)
		{
			//LOG(INFO) << "*res.second: " << *res.first << std::endl;
			//(ERROR) << "res dis larger than threshold." << std::endl;
			match_src_dst[i] = -1;
		}
	}
	return match_src_dst;
}

intVec MeshTools::getMatchAnyUV(const MeshCompress& src, const MeshCompress& dst, double pos_thres,
	bool hard_thres)
{
	if (src.pos_.empty() || dst.pos_.empty() || src.tri_uv_.empty() || dst.tri_uv_.empty())
	{
		LOG(INFO) << "src.n_vertex_ :" << src.n_vertex_ << std::endl;
		LOG(INFO) << "dst.n_vertex_ :" << dst.n_vertex_ << std::endl;
		LOG(INFO) << "src.n_tri_uv_ :" << src.n_uv_ << std::endl;
		LOG(INFO) << "dst.n_tri_uv_ :" << dst.n_uv_ << std::endl;
		LOG(ERROR) << "data check failed." << std::endl;
		return {};
	}

	//matching
	int n_src = src.n_vertex_;
	int n_dst = dst.n_vertex_;
	
	intVec map_src_to_dst = getMatchRawUV(src, dst, pos_thres);
	intVec map_dst_to_src = getMatchRawUV(dst, src, pos_thres);

	intVec match_src_to_dst_bi(n_src, -1);
	intSet discard_multi_matching;
	//#pragma omp parallel for
	for (int i = 0; i < n_src; i++)
	{
		int src_idx = i;
		int dst_idx = map_src_to_dst[i];
		if (dst_idx > -1 && dst_idx < n_dst && map_dst_to_src[dst_idx] == src_idx)
		{
			match_src_to_dst_bi[src_idx] = dst_idx;
		}
		else
		{
			discard_multi_matching.insert(src_idx);
		}
	}

	LOG(INFO) << "discard size: " << discard_multi_matching.size() << std::endl;
	return map_src_to_dst;
	return match_src_to_dst_bi;
}

void MeshTools::scaleBSInPlace(const MeshCompress& src, double scale, MeshCompress& dst)
{
	if (src.pos_.size() != dst.pos_.size())
	{
		LOG(ERROR) << "src && dst size not fit." << std::endl;
		return;
	}
	int n_vertex = src.pos_.size();
#pragma omp parallel for
	for (int i = 0; i < n_vertex; i++)
	{
		dst.pos_[i] = (dst.pos_[i] - src.pos_[i])*scale + src.pos_[i];
	}
}