#include "RigidBsGenerate.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Metric/Metric.h"
using namespace CGP;

void RIGIDBS::getSysDirectMappingFromMesh(const MeshCompress& part_base, const cstr& reference_root,
	const cstr& result_root, const intVec& part_all_map, const json& mesh_mapping, bool is_left, const float3E& t_shift)
{
	SG::needPath(result_root);
	intVec select_idx = divideIdx(part_base, part_all_map, is_left);
	cstrVec part_vec, all_vec;
	for (auto& el : mesh_mapping.items())
	{
		cstr result_name = el.key();
		cstr reference_name = el.value().get<cstr>();
		part_vec.push_back(result_name);
		all_vec.push_back(reference_name);
	}

	for (int i = 0; i<part_vec.size(); i++)
	{
		cstr result_name = part_vec[i];
		cstr reference_name = all_vec[i];
		MeshCompress reference(reference_root + reference_name + ".obj");
		MeshCompress result = part_base;
		for (int i = 0; i < select_idx.size(); i++)
		{
			int result_idx = i;
			int reference_idx = select_idx[i];
			if (reference_idx < 0)
			{
				//skip idx
			}
			else
			{
				result.pos_[result_idx] = reference.pos_[reference_idx] + t_shift;
			}
		}
		result.saveObj(result_root + result_name + ".obj");
	}
}

void RIGIDBS::getUsingZeroShift(const MeshCompress& ori, const cstrVec& names, const cstr& result_dir)
{
	for (auto i : names)
	{
		ori.saveObj(result_dir + i + ".obj");
	}
}

intVec RIGIDBS::divideIdx(const MeshCompress& part_base, const intVec& roi, bool is_left)
{
	intVec res = roi;
	for (int i  = 0; i< res.size(); i++)
	{
		//reverse set to -1
		if (is_left && part_base.pos_[i].x() < 0)
		{
			res[i] = -1;
		}

		if (!is_left && part_base.pos_[i].x() > 0)
		{
			res[i] = -1;
		}
	}
	return res;
}

void RIGIDBS::getJsonExp(const MeshCompress& part_base, const cstr& result_dir, json& res)
{
	floatVec B_pos(part_base.n_vertex_ * 3, 0);
	SG::safeMemcpy(B_pos.data(), part_base.pos_[0].data(), sizeof(float)*part_base.n_vertex_ * 3);
	cstrVec file_obj = FILEIO::getFolderFiles(result_dir, ".obj");
	
	for (int i = 0; i < file_obj.size(); i++)
	{
		MeshCompress iter_obj(result_dir + file_obj[i]);
		floatVec in_bs_temp(part_base.n_vertex_ * 3);
		SG::safeMemcpy(in_bs_temp.data(), iter_obj.pos_[0].data(), sizeof(float)*iter_obj.n_vertex_ * 3);
		std::transform(in_bs_temp.begin(), in_bs_temp.end(),
			B_pos.begin(), in_bs_temp.begin(), std::minus<float>());
		cstr raw_name = FILEIO::getFileNameWithoutExt(file_obj[i]);
		res[raw_name] = in_bs_temp;
	}
}

void RIGIDBS::getSysDivideAndConquer(const MeshCompress& src, const MeshCompress& ref, 
	const cstr& ref_root, const cstr& res_root, const cstr& temp_root)
{
	SG::needPath(res_root);
	cstrVec folder_items = FILEIO::getFolderFiles(ref_root, ".obj");
	intVec all_idx(src.n_vertex_, 0);
	std::iota(all_idx.begin(), all_idx.end(), 0);
	intVec left_idx = divideIdx(src, all_idx, true);
	intVec left_idx_rec = CalcHelper::keepValueBiggerThan(left_idx, -0.5);
	intVec right_idx = divideIdx(src, all_idx, false);
	intVec right_idx_rec = CalcHelper::keepValueBiggerThan(right_idx, -0.5);
	double scale_left, scale_right;
	float3E ref_to_src_left, ref_to_src_right;
	MeshCompress move_ref_left;
	MeshCompress move_ref_right;
	MeshTools::putSrcToDst(ref, left_idx_rec, src, left_idx_rec, move_ref_left, scale_left, ref_to_src_left);
	MeshTools::putSrcToDst(ref, right_idx_rec, src, right_idx_rec, move_ref_right, scale_right, ref_to_src_right);
	move_ref_left.saveObj(temp_root + "move_guijie_v1_left.obj");
	move_ref_right.saveObj(temp_root + "move_guijie_v1_right.obj");
	MeshCompress move_ref = move_ref_left;
	MeshTools::replaceVertexInPlace(move_ref, move_ref_right, right_idx_rec);
	//移动之后基本匹配
	MeshCompress mesh_error = Metric::getError(move_ref, src);
	mesh_error.saveVertexColor(temp_root + "move_error.obj");
	cstrVec folder_files = FILEIO::getFolderFiles(ref_root, ".obj");
	for(auto i: folder_files)
	{
		MeshCompress iter_mesh_left(ref_root + i);
		MeshCompress iter_mesh_right = iter_mesh_left;
		RT::scaleInCenterAndTranslateInPlace(scale_left, ref_to_src_left, iter_mesh_left.pos_);
		RT::scaleInCenterAndTranslateInPlace(scale_right, ref_to_src_right, iter_mesh_right.pos_);
		MeshTools::replaceVertexInPlace(iter_mesh_left, iter_mesh_right, right_idx_rec);
		iter_mesh_left.saveObj(res_root + i);
	}
}

void RIGIDBS::getSingleBS(const MeshCompress& src, const MeshCompress& ref,
	const cstr& ref_root, const cstr& res_root, const cstr& temp_root)
{
	SG::needPath(res_root);
	cstrVec folder_items = FILEIO::getFolderFiles(ref_root, ".obj");
	intVec all_idx(src.n_vertex_, 0);	
	std::iota(all_idx.begin(), all_idx.end(), 0);
	double scale;
	float3E ref_to_src;
	MeshCompress move_ref;
	MeshTools::putSrcToDst(ref, all_idx, src, all_idx, move_ref, scale, ref_to_src);
	move_ref.saveObj(temp_root + "move_guijie_v1.obj");
	//移动之后基本匹配
	MeshCompress mesh_error = Metric::getError(move_ref, src);
	mesh_error.saveVertexColor(temp_root + "move_error.obj");
	cstrVec folder_files = FILEIO::getFolderFiles(ref_root, ".obj");
	for (auto i : folder_files)
	{
		MeshCompress iter_mesh(ref_root + i);
		RT::scaleInPlace(scale, iter_mesh.pos_);
		RT::translateInPlace(ref_to_src, iter_mesh.pos_);
		iter_mesh.saveObj(res_root + i);
	}
}