#include "TopTransfer.h"
#include "../FileIO/FileIO.h"
#include "../Mesh/ObjLoader.h"
#include "../VisHelper/VisHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../MeshDeform/LaplacianDeformation.h"
#include "../NRICP/register.h"
#include "../NRICP/demo.h"
#include "../RT/RT.h"
#include "../RigidAlign/icp.h"
#include "../Config/Tensor.h"
#include "../Config/TensorHelper.h"
#include "../Config/JsonHelper.h"
#include "../Sysmetric/Sysmetric.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Metric/Metric.h"
#include "../MNN/MnnModel.h"
#include "../MNN/FeatureExtract.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../Mesh/MeshTools.h"
#include "../Test/TinyTool.h"
#include "../ImageSim/ImageUtils.h"
#include "../Test/Prepare.h"
#include "../MNN/KeypointDet.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../FileIO/FileIO.h"
#include "../Test/TinyTool.h"
#include "../Eyebrow/EyebrowType.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Basic/ToDst.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../Debug/DebugTools.h"
#include "../RigidBS/RigidBsGenerate.h"
#include "../ExpGen/ExpGen.h"
#include "../ExpGen/BsGenerate.h"
#include "../Mapping/Mapping.h"
#include "../MeshDeform/LapDeformWrapper.h"
#include "../Postprocess/EasyGuijie.h"
#include "../Lighting/Lighting.h"
#include "../OptV2/OptTypeV3.h"

using namespace CGP;

void TOPTRANSFER::cutFWHToHalf(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	auto& fwh_sys = const_var->ptr_data->fwh_sys_finder_;
	MeshCompress fwh_template("D:/data/0624_01/3dmm_fwh.obj");
	intVec fwh_id = FILEIO::loadIntDynamic("D:/data/0624_01/fwh_id.txt");
	intVec bfw_id = FILEIO::loadIntDynamic("D:/data/0624_01/bfm_id.txt");
	intVec keep_whole = FILEIO::loadIntDynamic("D:/data/0624_01/fwh_3.txt");
	intVec left_eye = FILEIO::loadIntDynamic("D:/data/0624_01/left_skip.txt");
	intVec right_eye = FILEIO::loadIntDynamic("D:/data/0624_01/right_skip.txt");
	intVec mouth = FILEIO::loadIntDynamic("D:/data/0624_01/mouth_skip.txt");
	intVec map_whole_part = fwh_template.discard(keep_whole, { left_eye , right_eye , mouth });
	intVec map_part_whole(fwh_template.pos_.size());
	for (int i = 0; i < map_whole_part.size(); i++)
	{
		if (map_whole_part[i] > -0.5)
		{
			map_part_whole.at(map_whole_part[i]) = i;
		}
	}
	//test for id
	std::ofstream out_fwh_id("D:/data/0624_01/part_bfw_id.txt");
	intVec half_fwh_id(fwh_id.size());
	for (int i = 0; i < fwh_id.size(); i++)
	{
		if (map_whole_part[fwh_id[i]] < 0)
		{
			LOG(INFO) << "point idx " << fwh_id[i] << ", missed" << std::endl;
		}
		else
		{
			half_fwh_id[i] = map_whole_part[fwh_id[i]];
			out_fwh_id << half_fwh_id[i] << ",";
		}
	}
	out_fwh_id.close();
	fwh_template.saveObj("D:/data/0624_01/3dmm_part_fwh.obj");
	std::ofstream out_cons("D:/data/0624_01/part_bfm.cons");
	out_cons << fwh_id.size() << std::endl;
	for (int i = 0; i < fwh_id.size(); i++)
	{
		out_cons << half_fwh_id[i] << ", " << bfw_id[i] << std::endl;
	}
	out_cons.close();
	//find closesed
	MeshCompress fwh_deform("D:/data/0624_01/3dmm_fwh.obj");
	MeshCompress deep3d_res("D:/data/0624_01/3dmm.obj");
	//find mapping
	intVec fwh_bfm(fwh_deform.pos_.size());
	//load right
	intVec fwh_bfm_mapping = FILEIO::loadIntDynamic("D:/data/0624_01/corre_26.000000.txt");
	intVec fwh_whole_bfm_mapping = fwh_bfm_mapping;
	for (int i = 0; i < fwh_whole_bfm_mapping.size() / 2; i++)
	{
		int idx_part_fwh = fwh_bfm_mapping[2 * i];
		int idx_whole_fwh = map_part_whole[idx_part_fwh];
		fwh_whole_bfm_mapping[2 * i] = idx_whole_fwh;
	}
	FILEIO::saveDynamic("D:/data/0624_01/whole_bfm.txt", fwh_whole_bfm_mapping, ",");
	intVec fwh_s0_raw = FILEIO::loadIntDynamic("D:/data/0624_01/fwh_s0.txt");
	//fwh_sys.getSysIdsInPlace(fwh_s0_raw);
	intVec fwh_s0;
	intVec right_exp = FILEIO::loadIntDynamic("D:/data/0624_01/right_expand.txt");
	intVec mouth_exp = FILEIO::loadIntDynamic("D:/data/0624_01/mouth_exp.txt");
	intVec cheek_exp = { 3864,6767 };
	for (int i = 0; i < fwh_s0_raw.size(); i++)
	{
		int idx = fwh_s0_raw[i];
		if (std::find(left_eye.begin(), left_eye.end(), idx) == left_eye.end()
			&& std::find(right_exp.begin(), right_exp.end(), idx) == right_exp.end()
			&& std::find(mouth_exp.begin(), mouth_exp.end(), idx) == mouth_exp.end()
			&& std::find(cheek_exp.begin(), cheek_exp.end(), idx) == cheek_exp.end())
		{
			if (i % 6 == 0)
			{
				fwh_s0.push_back(fwh_s0_raw[i]);
			}
		}
	}
	intSet fwh_roi(fwh_s0.begin(), fwh_s0.end());
	intSet fwh_all(fwh_s0_raw.begin(), fwh_s0_raw.end());

	LOG(INFO) << "fwh_sys " << const_var->ptr_data->fwh_sys_finder_.match_ids_[0] << std::endl;
	LOG(INFO) << "fwh_sys " << fwh_template.pos_[const_var->ptr_data->fwh_sys_finder_.match_ids_[0]] << std::endl;
	double shift_x = 0;
	int count_x = 0;
	double shift_z = 0;
	int count_z = 0;

	LaplacianDeform to_bfm;
	float3Vec deform_pos;
	MeshCompress fwh_3dmm("D:/data/0624_01/3dmm_fwh.obj");
	intVec fwh_fix = FILEIO::loadIntDynamic("D:/data/0624_01/fix_ear_neck.txt");
	const_var->ptr_data->fwh_sys_finder_.getSysIdsInPlace(fwh_fix);
	intVec fwh_s0_lap;
	intVec fwh_map_idx, bfm_map_idx;
	intSet fwh_fix_set(fwh_fix.begin(), fwh_fix.end());
	for (int i = 0; i < fwh_whole_bfm_mapping.size() / 2; i++)
	{
		int idx_fwh = fwh_whole_bfm_mapping[2 * i];
		int idx_bfm = fwh_whole_bfm_mapping[2 * i + 1];
		if (const_var->ptr_data->fwh_sys_finder_.right_ids_.count(idx_fwh) && fwh_roi.count(idx_fwh))
		{
			fwh_map_idx.push_back(idx_fwh);
			bfm_map_idx.push_back(idx_bfm);
		}
	}

	//save for trans idx();
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/fwh_id_trans.txt", fwh_map_idx, ",");
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/bfm_id_trans.txt", bfm_map_idx, ",");
	float3Vec fwh_slice, deep3d_slice;
	fwh_deform.getSlice(fwh_map_idx, fwh_slice);
	deep3d_res.getSlice(bfm_map_idx, deep3d_slice);
	float3E translate;
	RT::getTranslate(deep3d_slice, fwh_slice, translate);
	translate.x() = 0;
	LOG(INFO) << "translate: " << translate << std::endl;
	RT::translateInPlace(translate, deep3d_res.pos_);
	deep3d_res.saveObj("D:/data/0624_01/3dmm_trans.obj");
	intVec mid_fwh_select_vertex, mid_bfm_select_vertex;
	for (int i = 0; i < fwh_whole_bfm_mapping.size() / 2; i++)
	{
		int idx_fwh = fwh_whole_bfm_mapping[2 * i];
		int idx_bfm = fwh_whole_bfm_mapping[2 * i + 1];
		if (fwh_sys.mid_ids_.count(idx_fwh) && fwh_roi.count(idx_fwh))
		{
			//in mid line
			shift_x += deep3d_res.pos_[idx_bfm].x();
			count_x++;
			mid_fwh_select_vertex.push_back(idx_fwh);
			mid_bfm_select_vertex.push_back(idx_bfm);
		}
	}
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/bfm_mid.txt", mid_bfm_select_vertex, ",");
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/fwh_mid.txt", mid_fwh_select_vertex, ",");
	shift_x = shift_x / (1.0f*count_x);
	intVec deform_bfm;
	for (int i = 0; i < fwh_whole_bfm_mapping.size() / 2; i++)
	{
		int idx_fwh = fwh_whole_bfm_mapping[2 * i];
		int idx_bfm = fwh_whole_bfm_mapping[2 * i + 1];
		if (idx_bfm == 0)
		{
			//skip
		}
		else
		{
			if (fwh_sys.right_ids_.count(idx_fwh) && fwh_roi.count(idx_fwh) && !fwh_fix_set.count(idx_fwh))
			{
				auto pos_roi_right = std::find(fwh_s0.begin(), fwh_s0.end(), idx_fwh);
				float3E right = deep3d_res.pos_[idx_bfm];
				float3E left = right;
				left.x() = 2 * shift_x - right.x();
				if (right.x() > shift_x && shift_x > left.x() && left.x() < -0.05 && right.x() > 0.05)
				{
					deform_pos.push_back(deep3d_res.pos_[idx_bfm]);
					deform_bfm.push_back(idx_bfm);
					fwh_s0_lap.push_back(idx_fwh);
					int left_idx = fwh_sys.getSysId(idx_fwh);
					if (!fwh_fix_set.count(left_idx))
					{
						deform_pos.push_back(left);
						deform_bfm.push_back(-idx_bfm);
						fwh_s0_lap.push_back(left_idx);
					}
				}
			}
			if (fwh_sys.mid_ids_.count(idx_fwh) && fwh_roi.count(idx_fwh) && !fwh_fix_set.count(idx_fwh))
			{
				auto pos_mid = std::find(fwh_s0.begin(), fwh_s0.end(), idx_fwh);
				float3E mid = deep3d_res.pos_[idx_bfm];
				//mid.x() = 0;
				deform_pos.push_back(mid);
				deform_bfm.push_back(idx_bfm);
				fwh_s0_lap.push_back(idx_fwh);
			}
		}
	}
	//to_bfm.init(fwh_3dmm, fwh_s0_lap, fwh_fix, { fwh_id[41], fwh_id[38], fwh_id[35], fwh_id[34], fwh_id[40], fwh_id[39] });
	intVec mouth_close_vec = {
		8921, 8802, //mid
		8919, 8824, //right+4
		8917, 8859, //right+8
		10355,10312,//left-4
		10345,10334, //left-8
		3246, 10293,//left corner
		10343,10323,//left corner-left
		const_var->ptr_data->fwh_sys_finder_.getSysId(3246), const_var->ptr_data->fwh_sys_finder_.getSysId(10293),
		const_var->ptr_data->fwh_sys_finder_.getSysId(10343), const_var->ptr_data->fwh_sys_finder_.getSysId(10323),
	};
	to_bfm.init(fwh_3dmm, fwh_s0_lap, fwh_fix, mouth_close_vec);
	to_bfm.deform(deform_pos, fwh_3dmm.pos_);
	fwh_3dmm.saveObj("D:/data/0624_01/bfm_lap.obj");
	const_var->ptr_data->fwh_sys_finder_.getSysPosLapInPlace(fwh_3dmm.pos_);
	fwh_3dmm.saveObj("D:/data/0624_01/bfm_lap_sys.obj");

	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/fwh_fix.txt", fwh_fix, ",");
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/bfm_deform.txt", deform_bfm, ",");
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/fwh_deform.txt", fwh_s0_lap, ",");
	FILEIO::saveDynamic("D:/data/server_pack/post_3dmm/mouth_close.txt", mouth_close_vec, ",");
	LaplacianDeform post_mouth;
	post_mouth.init(fwh_3dmm, { fwh_id[41], fwh_id[35],  fwh_id[40] }, fwh_fix);
	float3Vec mouth_deform;
	mouth_deform.push_back(fwh_3dmm.pos_[fwh_id[38]]);
	mouth_deform.push_back(fwh_3dmm.pos_[fwh_id[34]]);
	mouth_deform.push_back(fwh_3dmm.pos_[fwh_id[39]]);
	post_mouth.deform(mouth_deform, fwh_3dmm.pos_);
	fwh_3dmm.saveObj("D:/data/0624_01/bfm_lap_mouth.obj");
	LOG(INFO) << "end of project process." << std::endl;
}

void TOPTRANSFER::getFWHToGuijieNoLash()
{
	cstr root = "D:/dota201104/1104_fwh_guijie/";
	//cstr root = "D:/dota210121/0126_fwh_guijie_move/";
	//guijie dump eyelash
	intVec guijie_discard = FILEIO::loadIntDynamic("D:/dota201104/1104_fwh_guijie/discard_vertex.txt");

	//move cto model to neutral scale
	MeshCompress fwh_obj(root + "bfm_lap_sys_adjust_uv.obj");
	MeshCompress guijie_obj(root + "0_opt.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");

	intVec all_part_map = guijie_obj.discard(guijie_discard);
	intVec re_guijie_68;
	MeshTools::getSlice(all_part_map, guijie_68_idx, re_guijie_68);

	FILEIO::saveDynamic(root + "mapping.txt", all_part_map, ",");
	FILEIO::saveDynamic(root + "re_guijie_68.txt", re_guijie_68, ",");
	guijie_obj.saveObj(root + "re_guijie.obj");

	MeshCompress guijie_move;
	MeshTools::putSrcToDst(guijie_obj, re_guijie_68, fwh_obj, fwh_68_idx, guijie_move);
	guijie_move.saveObj(root + "move_guijie.obj");
}

void TOPTRANSFER::getFWHToGuijieV1NoLash()
{
	cstr root = "D:/dota201201/1220_fwh_guijie/";
	//guijie dump eyelash
	intVec guijie_discard = FILEIO::loadIntDynamic(root + "discard_vertex.txt");

	//move cto model to neutral scale
	MeshCompress fwh_obj(root + "bfm_lap_sys.obj");
	MeshCompress guijie_obj(root + "Guijie_head.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");

	intVec all_part_map = guijie_obj.discard(guijie_discard);
	intVec re_guijie_68;
	MeshTools::getSlice(all_part_map, guijie_68_idx, re_guijie_68);

	FILEIO::saveDynamic(root + "mapping.txt", all_part_map, ",");
	FILEIO::saveDynamic(root + "re_guijie_68.txt", re_guijie_68, ",");
	guijie_obj.saveObj(root + "re_guijie.obj");

	MeshCompress guijie_move;
	MeshTools::putSrcToDst(guijie_obj, re_guijie_68, fwh_obj, fwh_68_idx, guijie_move);
	guijie_move.saveObj(root + "move_guijie.obj");
}

void TOPTRANSFER::getFWHToGuijieV1NoLashWithRing()
{
	cstr root = "D:/dota201201/1222_fwh_guijie/";
	//guijie dump eyelash
	intVec guijie_discard = FILEIO::loadIntDynamic(root + "discard_vertex.txt");

	//move cto model to neutral scale
	MeshCompress fwh_obj(root + "bfm_lap_sys.obj");
	MeshCompress guijie_obj(root + "Guijie_head.obj");

	intVec small_ring = FILEIO::loadIntDynamic(root + "fix_guijie.txt");
	intVec big_ring = FILEIO::loadIntDynamic(root + "fix_guijie_expand.txt");
	intVec fwh_index = FILEIO::loadIntDynamic(root + "fwh_ring.txt");
	intVec guijie_index = FILEIO::loadIntDynamic(root + "guijie_ring.txt");
	

	MeshCompress guijie_move = guijie_obj;
	MeshTools::putSrcToDst(guijie_obj, guijie_index, fwh_obj, fwh_index, guijie_move);
	guijie_move.saveObj(root + "move_guijie_all.obj");
	guijie_move.discard(guijie_discard);
	guijie_move.saveObj(root + "move_guijie_part.obj");
}

void TOPTRANSFER::transferUVFwhToGuijie()
{
	MeshCompress fwh_from("D:/dota201104/1104_fwh_guijie/bfm_lap_sys_adjust_uv.obj");
	MeshCompress guijie_to("D:/dota201104/1104_fwh_guijie/move_guijie.obj");
	MeshCompress fwh_deform("D:/dota201104/1104_fwh_guijie/deform.obj");
	MeshCompress guijie_move("D:/dota201104/1104_fwh_guijie/re_guijie.obj");
	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic("D:/dota201104/1104_fwh_guijie/corre_26.000000.txt");
	guijie_to.material_ = fwh_from.material_;
	intX2Vec map_guijie_to_fwh(guijie_to.n_vertex_);
	for (int i = 0; i < map_uv_fwh_to_guijie.size() / 2; i++)
	{
		int fwh_from = map_uv_fwh_to_guijie[2 * i];
		int guijie_to = map_uv_fwh_to_guijie[2 * i + 1];
		map_guijie_to_fwh[guijie_to].push_back(fwh_from);
	}
	guijie_to.tex_cor_.resize(guijie_to.n_vertex_);
	guijie_to.n_uv_ = guijie_to.n_vertex_;
	guijie_to.tri_uv_.clear();
	guijie_to.tri_uv_ = guijie_to.tri_;

#pragma omp parallel for
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		int min_index = -1;
		float min_dis = 1e8;
		for (int iter = 0; iter < map_guijie_to_fwh[i].size(); iter++)
		{
			int fwh_idx = map_guijie_to_fwh[i][iter];
			float cur_dis = (guijie_move.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
			min_index = min_dis > cur_dis ? fwh_idx : min_index;
			min_dis = DMIN(min_dis, cur_dis);
		}
		if (min_index < 0)
		{
			min_index = -1;
			min_dis = 1e8;
			for (int iter = 0; iter < fwh_deform.pos_.size(); iter++)
			{
				int fwh_idx = iter;
				float cur_dis = (guijie_move.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
				min_index = min_dis > cur_dis ? fwh_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
		}

		if (min_index < 0)
		{
			LOG(ERROR) << "match missed" << std::endl;
		}
		else
		{
			guijie_to.tex_cor_[i] = fwh_from.tex_cor_[min_index];
		}

	}

	guijie_to.saveObj("D:/dota201104/1104_fwh_guijie/guijie_transfer_uv.obj");
}

void TOPTRANSFER::fromIsvToBase()
{
	//get mapping from isv to base
	MeshCompress guijie = "D:/dota210419/0421_diff/guijie.obj";
	MeshCompress fwh = "D:/dota210419/0421_diff/fwh.obj";

	cstr ref_root = "D:/dota210419/0121_fwh_guijie/";
	cstr root = "D:/dota210419/0421_diff/";


	intVec discard_vertex = FILEIO::loadIntDynamic(ref_root + "discard_vertex.txt");

	guijie.discard(discard_vertex);
	guijie.saveObj(root + "guijie_df.obj");


}

void TOPTRANSFER::rawMatching()
{
	cstr root = "D:/dota210419/0421_diff/";
	MeshCompress guijie = root + "guijie_df.obj";
	MeshCompress fwh = root + "fwh.obj";

#if 0
	/*  dis   */
	double dis_thres = 0.05;
	intVec map_uv_fwh_to_guijie;
	for (int i_guijie = 0; i_guijie < guijie.n_vertex_; i_guijie++)
	{
		for (int i_fwh = 0; i_fwh < fwh.n_vertex_; i_fwh++)
		{
			float3E pq = guijie.pos_[i_guijie] - fwh.pos_[i_fwh];
			float3E q_norm = fwh.normal_[i_fwh];
			float cur_dist = std::abs(pq.dot(q_norm));
			float euler_dist = (guijie.pos_[i_guijie] - fwh.pos_[i_fwh]).norm();
			if (cur_dist < dis_thres && euler_dist< dis_thres)
			{
				map_uv_fwh_to_guijie.push_back(i_fwh);
				map_uv_fwh_to_guijie.push_back(i_guijie);
				//std::cout << i_fwh << ",";
			}

		}
	}
	FILEIO::saveDynamic(root + "corre_26.000000.txt", map_uv_fwh_to_guijie, ",");
#else
	/*
	处理matching
	*/
	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic(root + "corre_26.000000.txt");
	intVec loading_ori_mapping = FILEIO::loadIntDynamic(root + "mapping.txt");
	intVec discard_mapping_fix;
	intVec discard_mapping_fix_expand;
	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/multiPack/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/multiPack/1220_fwh_guijie/fix_guijie_expand.txt");

	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}

	for (int i : loading_fix_part_expand)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix_expand.push_back(loading_ori_mapping[i]);
		}
	}


	MeshCompress guijie_to = guijie;
	MeshCompress fwh_deform = fwh;

	intVec loading_part_to_whole(guijie_to.n_vertex_);
	for (size_t i = 0; i < loading_ori_mapping.size(); i++)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			loading_part_to_whole[loading_ori_mapping[i]] = i;
		}
	}




	intX2Vec map_guijie_to_fwh(guijie.n_vertex_);
	for (int i = 0; i < map_uv_fwh_to_guijie.size() / 2; i++)
	{
		int fwh_from = map_uv_fwh_to_guijie[2 * i];
		int guijie_to = map_uv_fwh_to_guijie[2 * i + 1];
		map_guijie_to_fwh[guijie_to].push_back(fwh_from);
	}


	intVec fwh_cheek = FILEIO::loadIntDynamic(root + "fwh_skip_cheek.txt");
	intSet fwh_cheek_set(fwh_cheek.begin(), fwh_cheek.end());


	guijie_to.tex_cor_.resize(guijie_to.n_vertex_);
	guijie_to.n_uv_ = guijie_to.n_vertex_;
	guijie_to.tri_uv_.clear();
	guijie_to.tri_uv_ = guijie_to.tri_;

	intX2Vec guijie_to_fwh(guijie_to.n_vertex_);
	intX2Vec fwh_to_guijie(fwh_deform.n_vertex_);
	intSet discard_multi_matching;
	//#pragma omp parallel for
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		int min_index = -1;
		float min_dis = 1e8;
		for (int iter = 0; iter < map_guijie_to_fwh[i].size(); iter++)
		{
			int fwh_idx = map_guijie_to_fwh[i][iter];
			float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
			min_index = min_dis > cur_dis ? fwh_idx : min_index;
			min_dis = DMIN(min_dis, cur_dis);
		}
		if (min_index < 0)
		{
			min_index = -1;
			min_dis = 1e8;
			for (int iter = 0; iter < fwh_deform.pos_.size(); iter++)
			{
				int fwh_idx = iter;
				float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
				min_index = min_dis > cur_dis ? fwh_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
		}

		if (min_index < 0)
		{
			LOG(ERROR) << "match missed" << std::endl;
		}
		else
		{

			if (guijie_to_fwh[i].size() > 0 || fwh_to_guijie[min_index].size() > 0)
			{
				LOG(INFO) << "discard here comes multiple matching" << std::endl;
				LOG(INFO) << "fwh: " << min_index << std::endl;
				LOG(INFO) << "guijie: " << i << std::endl;
				discard_multi_matching.insert(i);
			}
			else if (!fwh_cheek_set.count(min_index))
			{
				guijie_to_fwh[i].push_back(min_index);
				fwh_to_guijie[min_index].push_back(i);
			}
		}
	}
	LOG(INFO) << "discard size: " << discard_multi_matching.size() << std::endl;


	intVec move_idx;
	intVec move_idx_all;
	float3Vec move_pos;
	intSet fix_idx(discard_mapping_fix.begin(), discard_mapping_fix.end());
	intSet fix_idx_expand(discard_mapping_fix_expand.begin(), discard_mapping_fix_expand.end());


	intVec fwh_move_idx, guijie_move_idx_all, guijie_move_idx_part;


	for (int i = 0; i < guijie_to_fwh.size(); i++)
	{
		if (discard_multi_matching.count(i) || fix_idx_expand.count(i) || guijie_to_fwh[i].empty())
		{

		}
		else
		{
			move_idx.push_back(i);
			move_idx_all.push_back(loading_part_to_whole[i]);
			move_pos.push_back(fwh.pos_[guijie_to_fwh[i][0]]);
			fwh_move_idx.push_back(guijie_to_fwh[i][0]);
			guijie_move_idx_all.push_back(loading_part_to_whole[i]);
			guijie_move_idx_part.push_back(i);
		}
	}

	intVec guijie_ring;
	intVec guijie_ring_part;
	intVec fwh_ring;
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		if (guijie_to_fwh[i].size() > 0 && fix_idx_expand.count(i) && !fix_idx.count(i))
		{
			guijie_ring.push_back(loading_part_to_whole[i]);
			guijie_ring_part.push_back(i);
			fwh_ring.push_back(guijie_to_fwh[i][0]);
		}
	}


	FILEIO::saveDynamic(root + "guijie_ring.txt", guijie_ring, ",");
	FILEIO::saveDynamic(root + "fwh_ring.txt", fwh_ring, ",");
	FILEIO::saveDynamic(root + "fwh_move_idx.txt", fwh_move_idx, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_all.txt", guijie_move_idx_all, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_part.txt", guijie_move_idx_part, ",");

	//testing

	MeshCompress fwh_move = fwh;
	MeshCompress guijie_template = guijie_to;
	SIMDEFORM::moveHandle(guijie_to, guijie_move_idx_part, fwh_move, fwh_move_idx, discard_mapping_fix, {}, guijie_template);
	guijie_template.saveObj(root + "guijie_map_back_ring.obj");

	
	guijie_template = root + "guijie.obj";
	LaplacianDeform to_fwh;
	to_fwh.init(guijie_template, move_idx_all, discard_mapping_fix);
	to_fwh.deform(move_pos, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_map_back.obj");

#endif
}

void TOPTRANSFER::getMatchFromDF()
{
	//origin path
	//cstr root = "D:/dota201201/1220_fwh_guijie/";
	cstr root = "D:/dota210121/0121_fwh_guijie/";
	MeshCompress guijie_to(root + "move_guijie.obj");
	MeshCompress fwh_deform(root + "deform.obj");
	MeshCompress fwh_ori(root + "bfm_lap_sys.obj");
	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic(root + "corre_26.000000.txt");
	intX2Vec map_guijie_to_fwh(guijie_to.n_vertex_);
	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	intVec loading_ori_mapping = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/mapping.txt");
	intVec loading_part_to_whole(guijie_to.n_vertex_);
	for (size_t i = 0; i < loading_ori_mapping.size(); i++)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			loading_part_to_whole[loading_ori_mapping[i]] = i;
		}
	}

	intVec discard_mapping_fix;
	intVec discard_mapping_fix_expand;
	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}

	for (int i : loading_fix_part_expand)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix_expand.push_back(loading_ori_mapping[i]);
		}
	}


	for (int i = 0; i < map_uv_fwh_to_guijie.size() / 2; i++)
	{
		int fwh_from = map_uv_fwh_to_guijie[2 * i];
		int guijie_to = map_uv_fwh_to_guijie[2 * i + 1];
		map_guijie_to_fwh[guijie_to].push_back(fwh_from);
	}
	guijie_to.tex_cor_.resize(guijie_to.n_vertex_);
	guijie_to.n_uv_ = guijie_to.n_vertex_;
	guijie_to.tri_uv_.clear();
	guijie_to.tri_uv_ = guijie_to.tri_;

	intX2Vec guijie_to_fwh(guijie_to.n_vertex_);
	intX2Vec fwh_to_guijie(fwh_deform.n_vertex_);
	intSet discard_multi_matching;
//#pragma omp parallel for
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		int min_index = -1;
		float min_dis = 1e8;
		for (int iter = 0; iter < map_guijie_to_fwh[i].size(); iter++)
		{
			int fwh_idx = map_guijie_to_fwh[i][iter];
			float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
			min_index = min_dis > cur_dis ? fwh_idx : min_index;
			min_dis = DMIN(min_dis, cur_dis);
		}
		if (min_index < 0)
		{
			min_index = -1;
			min_dis = 1e8;
			for (int iter = 0; iter < fwh_deform.pos_.size(); iter++)
			{
				int fwh_idx = iter;
				float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
				min_index = min_dis > cur_dis ? fwh_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
		}

		if (min_index < 0)
		{
			LOG(ERROR) << "match missed" << std::endl;
		}
		else
		{

			if (guijie_to_fwh[i].size() > 0 || fwh_to_guijie[min_index].size() > 0)
			{
				LOG(INFO) << "discard here comes multiple matching" << std::endl;
				LOG(INFO) << "fwh: " << min_index << std::endl;
				LOG(INFO) << "guijie: " << i << std::endl;
				discard_multi_matching.insert(i);
			}
			else 
			{
				guijie_to_fwh[i].push_back(min_index);
				fwh_to_guijie[min_index].push_back(i);
			}
		}
	}
	LOG(INFO) << "discard size: " << discard_multi_matching.size() << std::endl;

	intVec move_idx;
	float3Vec move_pos;
	intSet fix_idx(discard_mapping_fix.begin(), discard_mapping_fix.end());
	intSet fix_idx_expand(discard_mapping_fix_expand.begin(), discard_mapping_fix_expand.end());


	intVec fwh_move_idx, guijie_move_idx_all, guijie_move_idx_part;


	for (int i = 0; i < guijie_to_fwh.size(); i++)
	{
		if (discard_multi_matching.count(i) || fix_idx_expand.count(i) || guijie_to_fwh[i].empty())
		{

		}
		else
		{
			move_idx.push_back(i);
			move_pos.push_back(fwh_ori.pos_[guijie_to_fwh[i][0]]);

			fwh_move_idx.push_back(guijie_to_fwh[i][0]);
			guijie_move_idx_all.push_back(loading_part_to_whole[i]);
			guijie_move_idx_part.push_back(i);
		}
	}

	intVec guijie_ring;
	intVec fwh_ring;
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		if (guijie_to_fwh[i].size() > 0 && fix_idx_expand.count(i) && !fix_idx.count(i))
		{
			guijie_ring.push_back(loading_part_to_whole[i]);
			fwh_ring.push_back(guijie_to_fwh[i][0]);
		}
	}

	FILEIO::saveDynamic(root + "guijie_ring.txt", guijie_ring, ",");
	FILEIO::saveDynamic(root + "fwh_ring.txt", fwh_ring, ",");

	FILEIO::saveDynamic(root + "fwh_move_idx.txt", fwh_move_idx, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_all.txt", guijie_move_idx_all, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_part.txt", guijie_move_idx_part, ",");
	
	MeshCompress guijie_template = guijie_to;
	LaplacianDeform to_fwh;
	to_fwh.init(guijie_template, move_idx, discard_mapping_fix);
	to_fwh.deform(move_pos, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_map_back.obj");

}

void TOPTRANSFER::getMatchFromDFRefine()
{
	//origin path
	//cstr root = "D:/dota201201/1220_fwh_guijie/";
	//cstr root = "D:/dota210121/0121_fwh_guijie/";
	//getFWHToGuijieNoLash();
	cstr root = "D:/dota210121/0127_fwh_guijie/";
	MeshCompress fwh_ori(root + "bfm_lap_sys.obj");
	MeshCompress guijie_ori(root + "head.obj");

	intVec discard_vertex = FILEIO::loadIntDynamic(root + "discard_vertex.txt");
	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");
	
	intVec all_part_map = guijie_ori.discard(discard_vertex);
	intVec re_guijie_68;
	MeshTools::getSlice(all_part_map, guijie_68_idx, re_guijie_68);

	FILEIO::saveDynamic(root + "mapping.txt", all_part_map, ",");
	FILEIO::saveDynamic(root + "re_guijie_68.txt", re_guijie_68, ",");
	guijie_ori.saveObj(root + "re_guijie.obj");

	MeshCompress guijie_move;
	double guijie_to_fwh_scale;
	float3E guijie_to_fwh_translate;
	MeshTools::putSrcToDst(guijie_ori, re_guijie_68, fwh_ori, fwh_68_idx, guijie_move, guijie_to_fwh_scale, guijie_to_fwh_translate);
	guijie_move.saveObj(root + "move_guijie.obj");

	MeshCompress guijie_to(root + "move_guijie.obj");
	MeshCompress fwh_deform(root + "deform.obj");

	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic(root + "corre_26.000000.txt");
	intX2Vec map_guijie_to_fwh(guijie_to.n_vertex_);
	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	intVec loading_ori_mapping = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/mapping.txt");
	intVec fwh_cheek = FILEIO::loadIntDynamic("D:/dota210121/0121_fwh_guijie/fwh_skip_cheek.txt");
	intSet fwh_cheek_set(fwh_cheek.begin(), fwh_cheek.end());
	intVec loading_part_to_whole(guijie_to.n_vertex_);
	for (size_t i = 0; i < loading_ori_mapping.size(); i++)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			loading_part_to_whole[loading_ori_mapping[i]] = i;
		}
	}

	intVec discard_mapping_fix;
	intVec discard_mapping_fix_expand;
	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}

	for (int i : loading_fix_part_expand)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix_expand.push_back(loading_ori_mapping[i]);
		}
	}


	for (int i = 0; i < map_uv_fwh_to_guijie.size() / 2; i++)
	{
		int fwh_from = map_uv_fwh_to_guijie[2 * i];
		int guijie_to = map_uv_fwh_to_guijie[2 * i + 1];
		map_guijie_to_fwh[guijie_to].push_back(fwh_from);
	}
	guijie_to.tex_cor_.resize(guijie_to.n_vertex_);
	guijie_to.n_uv_ = guijie_to.n_vertex_;
	guijie_to.tri_uv_.clear();
	guijie_to.tri_uv_ = guijie_to.tri_;

	intX2Vec guijie_to_fwh(guijie_to.n_vertex_);
	intX2Vec fwh_to_guijie(fwh_deform.n_vertex_);
	intSet discard_multi_matching;
	//#pragma omp parallel for
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		int min_index = -1;
		float min_dis = 1e8;
		for (int iter = 0; iter < map_guijie_to_fwh[i].size(); iter++)
		{
			int fwh_idx = map_guijie_to_fwh[i][iter];
			float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
			min_index = min_dis > cur_dis ? fwh_idx : min_index;
			min_dis = DMIN(min_dis, cur_dis);
		}
		if (min_index < 0)
		{
			min_index = -1;
			min_dis = 1e8;
			for (int iter = 0; iter < fwh_deform.pos_.size(); iter++)
			{
				int fwh_idx = iter;
				float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
				min_index = min_dis > cur_dis ? fwh_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
		}

		if (min_index < 0)
		{
			LOG(ERROR) << "match missed" << std::endl;
		}
		else
		{

			if (guijie_to_fwh[i].size() > 0 || fwh_to_guijie[min_index].size() > 0)
			{
				LOG(INFO) << "discard here comes multiple matching" << std::endl;
				LOG(INFO) << "fwh: " << min_index << std::endl;
				LOG(INFO) << "guijie: " << i << std::endl;
				discard_multi_matching.insert(i);
			}
			else if(!fwh_cheek_set.count(min_index))
			{
				guijie_to_fwh[i].push_back(min_index);
				fwh_to_guijie[min_index].push_back(i);
			}
		}
	}
	LOG(INFO) << "discard size: " << discard_multi_matching.size() << std::endl;

	intSet fix_idx(discard_mapping_fix.begin(), discard_mapping_fix.end());
	intSet fix_idx_expand(discard_mapping_fix_expand.begin(), discard_mapping_fix_expand.end());


	intVec fwh_move_idx, guijie_move_idx_all, guijie_move_idx_part;


	for (int i = 0; i < guijie_to_fwh.size(); i++)
	{
		if (discard_multi_matching.count(i) || fix_idx_expand.count(i) || guijie_to_fwh[i].empty())
		{

		}
		else
		{
			fwh_move_idx.push_back(guijie_to_fwh[i][0]);
			guijie_move_idx_all.push_back(loading_part_to_whole[i]);
			guijie_move_idx_part.push_back(i);
		}
	}

	intVec guijie_ring;
	intVec guijie_ring_part;
	intVec fwh_ring;
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		if (guijie_to_fwh[i].size() > 0 && fix_idx_expand.count(i) && !fix_idx.count(i))
		{
			guijie_ring.push_back(loading_part_to_whole[i]);
			guijie_ring_part.push_back(i);
			fwh_ring.push_back(guijie_to_fwh[i][0]);
		}
	}

	FILEIO::saveDynamic(root + "guijie_ring.txt", guijie_ring, ",");
	FILEIO::saveDynamic(root + "fwh_ring.txt", fwh_ring, ",");

	FILEIO::saveDynamic(root + "fwh_move_idx.txt", fwh_move_idx, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_all.txt", guijie_move_idx_all, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_part.txt", guijie_move_idx_part, ",");

	//move
	MeshCompress fwh_move = fwh_ori;
	double scale;
	float3E translate;
	MeshTools::putSrcToDst(fwh_ori, fwh_ring, guijie_to, guijie_ring_part, fwh_move, scale, translate);

	fwh_move.saveObj(root + "fwh_move.obj");
#if 1
	MeshCompress guijie_template = guijie_to;
	SIMDEFORM::moveHandle(guijie_to, guijie_move_idx_part, fwh_move, fwh_move_idx, discard_mapping_fix, {}, guijie_template);
	guijie_template.saveObj(root + "guijie_map_back_ring.obj");
#endif

	RT::scaleInCenterAndTranslateInPlace(1.0 / guijie_to_fwh_scale, -guijie_to_fwh_translate, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_part_pca_space.obj");
	//put back
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
	
	MeshCompress pca_base = "D:/dota210121/0127_fwh_guijie/head.obj";
	intVec face_fix = FILEIO::loadIntDynamic("D:/dota210121/0127_fwh_guijie/fix_guijie_expand.txt");
	SIMDEFORM::moveHandle(pca_base, guijie_move_idx_all, guijie_template, guijie_move_idx_part, face_fix, guijie_eyelash, pca_base);
	pca_base.saveObj("D:/dota210121/0127_fwh_guijie/head_deform.obj");

}

void TOPTRANSFER::onlyLandmark()
{	
	cstr root = "D:/dota201201/1222_only_landmrk/";
	MeshCompress guijie_to(root + "guijie_back_v2.obj");
	MeshCompress fwh_ori(root + "bfm_lap_sys.obj");

	//move cto model to neutral scale
	MeshCompress fwh_obj(root + "bfm_lap_sys.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");

	intVec loading_ori_mapping = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/mapping.txt");
	intVec part_guijie_68;
	for (int i : guijie_68_idx)
	{
		part_guijie_68.push_back(loading_ori_mapping[i]);
	}

	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");

	intVec loading_part_to_whole(guijie_to.n_vertex_);
	for (size_t i = 0; i < loading_ori_mapping.size(); i++)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			loading_part_to_whole[loading_ori_mapping[i]] = i;
		}
	}

	intVec discard_mapping_fix;
	intVec discard_mapping_fix_expand;
	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}
	intVec move_idx;
	float3Vec move_pos;
	intSet fix_idx(discard_mapping_fix.begin(), discard_mapping_fix.end());
	intSet fix_idx_expand(discard_mapping_fix_expand.begin(), discard_mapping_fix_expand.end());
	
	
	for (int i = 0; i < part_guijie_68.size(); i++)
	{		
		move_idx.push_back(part_guijie_68[i]);
		move_pos.push_back(fwh_ori.pos_[fwh_68_idx[i]]);
	}

	LOG(INFO) << "sim mapping size: " << move_idx.size() << std::endl;
	

	MeshCompress guijie_template = guijie_to;
	LaplacianDeform to_fwh;
	to_fwh.init(guijie_template, move_idx, discard_mapping_fix);
	to_fwh.deform(move_pos, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_68.obj");
	intVec skip_eye_list = FILEIO::loadIntDynamic(root + "error.txt");
	intVec mapping_dump_error = guijie_template.discard(skip_eye_list);
	guijie_template.saveObj(root + "guijie_map_back_dump_error.obj");
	//2613 right_guijie_eye 756
	//1083 left_guijie_eye 7257

	intVec reverse_mapping_dump_error(guijie_template.n_vertex_);
	for (int i = 0; i<mapping_dump_error.size(); i++)
	{
		if (mapping_dump_error[i] >= 0)
		{
			reverse_mapping_dump_error[mapping_dump_error[i]] = i;
		}
	}
	move_idx.push_back(reverse_mapping_dump_error[2613]);
	move_pos.push_back(fwh_ori.pos_[756]);
	move_idx.push_back(reverse_mapping_dump_error[1083]);
	move_pos.push_back(fwh_ori.pos_[7257]);

	MeshCompress guijie_template_v2 = guijie_to;
	LaplacianDeform to_fwh_v2;
	to_fwh_v2.init(guijie_template_v2, move_idx, discard_mapping_fix);
	to_fwh_v2.deform(move_pos, guijie_template_v2.pos_);
	guijie_template_v2.saveObj(root + "guijie_68_v2.obj");
	guijie_template_v2.discard(skip_eye_list);
	guijie_template_v2.saveObj(root + "guijie_v2_map_back_dump_error.obj");
	//get mapping from ori
	intVec used_vertex;
	for (int i = 0; i < loading_ori_mapping.size(); i++)
	{
		int part_idx = loading_ori_mapping[i];
		if (part_idx >= 0)
		{
			int error_idx = mapping_dump_error[part_idx];
			if (error_idx >= 0)
			{
				used_vertex.push_back(i);
			}
		}
	}
	FILEIO::saveDynamic(root + "used_vertex.txt", used_vertex, ",");
}

void TOPTRANSFER::onlyLandmarkDrag()
{
	cstr root = "D:/dota201201/1223_test_infer/";
	
	MeshCompress guijie_to(root + "move_guijie.obj");
	MeshCompress fwh_ori(root + "bfm_lap_sys.obj");

	//move cto model to neutral scale
	MeshCompress fwh_obj(root + "bfm_lap_sys.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");

	intVec loading_ori_mapping = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/mapping.txt");
	intVec part_guijie_68;
	for (int i : guijie_68_idx)
	{
		part_guijie_68.push_back(loading_ori_mapping[i]);
	}

	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");

	intVec loading_part_to_whole(guijie_to.n_vertex_);
	for (size_t i = 0; i < loading_ori_mapping.size(); i++)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			loading_part_to_whole[loading_ori_mapping[i]] = i;
		}
	}

	intVec discard_mapping_fix;
	intVec discard_mapping_fix_expand;
	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}
	intVec move_idx;
	float3Vec move_pos;
	intSet fix_idx(discard_mapping_fix.begin(), discard_mapping_fix.end());
	intSet fix_idx_expand(discard_mapping_fix_expand.begin(), discard_mapping_fix_expand.end());


	for (int i = 0; i < part_guijie_68.size(); i++)
	{
		move_idx.push_back(part_guijie_68[i]);
		move_pos.push_back(fwh_ori.pos_[fwh_68_idx[i]]);
	}

	LOG(INFO) << "sim mapping size: " << move_idx.size() << std::endl;


	MeshCompress guijie_template = guijie_to;
	LaplacianDeform to_fwh;
	to_fwh.init(guijie_template, move_idx, discard_mapping_fix);
	to_fwh.deform(move_pos, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_68.obj");
	intVec skip_eye_list = FILEIO::loadIntDynamic(root + "error.txt");
	intVec mapping_dump_error = guijie_template.discard(skip_eye_list);
	guijie_template.saveObj(root + "guijie_map_back_dump_error.obj");
	//2613 right_guijie_eye 756
	//1083 left_guijie_eye 7257

	intVec reverse_mapping_dump_error(guijie_template.n_vertex_);
	for (int i = 0; i < mapping_dump_error.size(); i++)
	{
		if (mapping_dump_error[i] >= 0)
		{
			reverse_mapping_dump_error[mapping_dump_error[i]] = i;
		}
	}
	move_idx.push_back(reverse_mapping_dump_error[2613]);
	move_pos.push_back(fwh_ori.pos_[756]);
	move_idx.push_back(reverse_mapping_dump_error[1083]);
	move_pos.push_back(fwh_ori.pos_[7257]);

	MeshCompress guijie_template_v2 = guijie_to;
	LaplacianDeform to_fwh_v2;
	to_fwh_v2.init(guijie_template_v2, move_idx, discard_mapping_fix);
	to_fwh_v2.deform(move_pos, guijie_template_v2.pos_);
	guijie_template_v2.saveObj(root + "guijie_68_v2.obj");
	guijie_template_v2.discard(skip_eye_list);
	guijie_template_v2.saveObj(root + "guijie_v2_map_back_dump_error.obj");
	//get mapping from ori
	intVec used_vertex;
	intVec whole_part_mapping;
	for (int i = 0; i < loading_ori_mapping.size(); i++)
	{
		int part_idx = loading_ori_mapping[i];
		if (part_idx >= 0)
		{
			int error_idx = mapping_dump_error[part_idx];
			if (error_idx >= 0)
			{
				used_vertex.push_back(i);
			}
		}
	}
	FILEIO::saveDynamic(root + "used_vertex.txt", used_vertex, ",");
}

void TOPTRANSFER::putEyelashBack()
{
	//eyelash data
	intVec left_down_lash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	intVec left_up_lash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt");
	intVec right_down_lash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt");
	intVec right_up_lash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt");

	intVec left_down_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt");
	intVec left_up_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	intVec right_down_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt");
	intVec right_up_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt");

	intVec discard_all_part = FILEIO::loadIntDynamic("D:/dota201201/1222_fwh_to_guijie/mapping.txt");
	intVec discard_part_all = MAP::getReverseMapping(discard_all_part);

	//testing
	cstr root = "D:/dota201201/1222_fwh_to_guijie/";
	MeshCompress guijie_all("D:/dota201201/1222_fwh_to_guijie/move_guijie_all.obj");
	MeshCompress guijie_part("D:/dota201201/1222_fwh_to_guijie/guijie_back_v2.obj");

	//deform test
	MeshCompress guijie_all_deform = guijie_all;
	MeshCompress guijie_all_move = guijie_all_deform;
	SIMDEFORM::moveHandle(guijie_all_deform, guijie_part, discard_part_all, false, guijie_all_move);
	guijie_all_move.saveObj(root + "move_all_pos.obj");

	moveAndDeformEyelash(left_down_lash, left_down_match, guijie_all_move);
	moveAndDeformEyelash(right_down_lash, right_down_match, guijie_all_move);
	moveAndDeformEyelash(left_up_lash, left_up_match, guijie_all_move);
	moveAndDeformEyelash(right_up_lash, right_up_match, guijie_all_move);
	guijie_all_move.saveObj(root + "A_move_left_down.obj");
}

void TOPTRANSFER::moveAndDeformEyelash(const intVec& lash_idx, const intVec& match_idx, MeshCompress& guijie_all)
{
	MeshCompress eyelash_mesh = guijie_all;
	intVec lash_mapping = eyelash_mesh.keepRoi(lash_idx);
	intX2Vec lash_eye_match = MAP::splitMapping(match_idx, true, 2);
	intVec eyelash_self_idx = MAP::passIdxThroughMapping(lash_eye_match[0], { lash_mapping });
	intVec lash_idx_match_head = MAP::getReverseMapping(lash_mapping);

	moveAndDeformEyelash(lash_idx, eyelash_self_idx, lash_eye_match[1],
		lash_idx_match_head, guijie_all);
	//guijie_all_move.saveObj(root + "move_left_down.obj");
}

void TOPTRANSFER::moveAndDeformEyelash(const MeshCompress& lash_src, const intVec& match_eyelash_self_idx,
	const intVec& match_eye_idx, const intVec& lash_mapping_reverse, MeshCompress& guijie_all)
{
	float3Vec handle_to_pos;
	for (int i : match_eye_idx)
	{
		handle_to_pos.push_back(guijie_all.pos_[i]);
	}
	MeshCompress move_eyelash = lash_src;
	LaplacianDeform move_eyebrow;
	move_eyebrow.init(move_eyelash, match_eyelash_self_idx, {});
	move_eyebrow.deform(handle_to_pos, move_eyelash.pos_);
	//move_eyelash.saveObj(root + "left_down_lash_move_deform.obj");
	//mapping pos back
	for (int i = 0; i < lash_mapping_reverse.size(); i++)
	{
		guijie_all.pos_[lash_mapping_reverse[i]] = move_eyelash.pos_[i];
	}
	//guijie_all.saveObj(root + "A_left_down_lash_move.obj");
}

void TOPTRANSFER::moveAndDeformEyelash(const intVec& lash_idx, const intVec& match_eyelash_self_idx,
	const intVec& match_eye_idx, const intVec& lash_mapping_reverse, MeshCompress& guijie_all)
{
	MeshCompress eye_lash = guijie_all;
	eye_lash.keepRoi(lash_idx);
	MeshCompress lash_src = eye_lash;
	MeshTools::putSrcToDst(eye_lash, match_eyelash_self_idx, guijie_all, match_eye_idx, lash_src);
	moveAndDeformEyelash(lash_src, match_eyelash_self_idx,
		match_eye_idx, lash_mapping_reverse, guijie_all);
	
}

void TOPTRANSFER::transferSimDiff(const cstr& root, const cstr& obj)
{
	cstr data_root = "D:/multiPack/";
	//MeshCompress A_deform = "D:/dota201201/1223_test_routine/res/nricp.obj";
	MeshCompress A_deform = root + obj;
	MeshCompress A = data_root + "1223_test_infer/nricp.obj";
	//MeshCompress B = "D:/dota201201/1223_test_infer/move_guijie_all.obj";
	MeshCompress B = data_root + "0127_sl_head/face_interBS_origin_pos.obj";

	//cstr result_root = "D:/dota201201/1223_df_result/";
	cstr result_root = root;
	SG::needPath(result_root);

	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";

	bool is_compress = true;
	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));

	MeshCompress B_nolash = B;
	exp_ptr->dumpEyelash(B_nolash);
	B_nolash.saveObj(result_root + "B_nolash.obj");

	exp_ptr->fixEyelash(B);
	MeshCompress B_res = B;
	exp_ptr->getExpGuided(B, A, A_deform, B_res);
	nlohmann::json vertex_value;
	exp_ptr->getResultJson(vertex_value);
	//win need
	B_res.saveObj(result_root + "B_deform.obj");
	//exp_ptr->dumpEyelash(B_res);
	//B_res.saveObj(head_res_root + json_raw_name + "_no_lash.obj");
}

void TOPTRANSFER::transferSimDiff(const cstr& root, const cstr& obj, const MeshCompress& input_B)
{
	cstr data_root = "D:/multiPack/";
	//MeshCompress A_deform = "D:/dota201201/1223_test_routine/res/nricp.obj";
	MeshCompress A_deform = root + obj;
	MeshCompress A = data_root + "1223_test_infer/nricp.obj";
	//MeshCompress B = "D:/dota201201/1223_test_infer/move_guijie_all.obj";
	MeshCompress B = input_B;

	//cstr result_root = "D:/dota201201/1223_df_result/";
	cstr result_root = root;
	SG::needPath(result_root);

	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";

	bool is_compress = true;
	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));

	MeshCompress B_nolash = B;
	exp_ptr->dumpEyelash(B_nolash);
	B_nolash.saveObj(result_root + "B_nolash.obj");

	exp_ptr->fixEyelash(B);
	MeshCompress B_res = B;
	exp_ptr->getExpGuided(B, A, A_deform, B_res);
	nlohmann::json vertex_value;
	exp_ptr->getResultJson(vertex_value);
	//win need
	B_res.saveObj(result_root + "B_deform.obj");
	//exp_ptr->dumpEyelash(B_res);
	//B_res.saveObj(head_res_root + json_raw_name + "_no_lash.obj");
}

void TOPTRANSFER::transferSimDiff(const cstr& root, const cstr& obj, const MeshCompress& input_B, const std::shared_ptr<ExpGen> exp_ptr, std::shared_ptr<ResVar> res_ptr)
{
	cstr data_root = "D:/avatar/nl_linux/";
	MeshCompress A_deform = root + obj;

	cstr A_date_path = "";
	if (res_ptr->model_3dmm_type_ == Type3dmm::MS)
	{
		LOG(INFO) << "using Type3dmm::MS" << std::endl;
		A_date_path = data_root + "1223_test_infer/nricp.obj";
	}
	else if (res_ptr->model_3dmm_type_ == Type3dmm::NR || res_ptr->model_3dmm_type_ == Type3dmm::NR_RAW)
	{
		LOG(INFO) << "using Type3dmm::NR or NR_RAW" << std::endl;
		A_date_path = data_root + "nr_base/inter_test.obj";
	}
	else if (res_ptr->model_3dmm_type_ == Type3dmm::NR_CPP)
	{
		LOG(INFO) << "using Type3dmm::NR_CPP" << std::endl;
		A_date_path = data_root + "nr_base/inter_test.obj";
	}
	else
	{
		LOG(ERROR) << "undefined data in transferSimDiff functions" << std::endl;
	}
	
	MeshCompress A = A_date_path;

	MeshCompress B = input_B;

	cstr result_root = root;
	SG::needPath(result_root);

	bool is_compress = true;
	
	MeshCompress B_nolash = B;
	exp_ptr->dumpEyelash(B_nolash);
	B_nolash.saveObj(result_root + "B_nolash.obj");

	exp_ptr->fixEyelash(B);
	MeshCompress B_res = B;
	exp_ptr->getExpGuided(B, A, A_deform, B_res);
	nlohmann::json vertex_value;
	exp_ptr->getResultJson(vertex_value);

	//win need
	B_res.saveObj(result_root + "B_deform.obj");
}

void TOPTRANSFER::transferSimDiffTesting(const cstr& root, const cstr& obj, const MeshCompress& input_B, const std::shared_ptr<ExpGen> exp_ptr)
{
	cstr data_root = "D:/multiPack/";
	MeshCompress A_deform = root + obj;
	MeshCompress A = data_root + "0422_test_infer/nricp.obj";
	MeshCompress B = input_B;

	cstr result_root = root;
	SG::needPath(result_root);

	bool is_compress = true;

	MeshCompress B_nolash = B;
	exp_ptr->dumpEyelash(B_nolash);
	B_nolash.saveObj(result_root + "B_nolash.obj");

	exp_ptr->fixEyelash(B);
	MeshCompress B_res = B;
	exp_ptr->getExpGuided(B, A, A_deform, B_res);
	nlohmann::json vertex_value;
	exp_ptr->getResultJson(vertex_value);

	//win need
	B_res.saveObj(result_root + "B_deform.obj");
}

void TOPTRANSFER::localDeform()
{
	//test
	//compare root 	
	//cstr root = "D:/avatar200923/1019_00/";
	cstr root = "D:/dota201104/1105_guijie/";
	cstrVec file_id = {
		"0.jpg",
		"1.jpg",
		"2.jpg",
		"3.jpg",
		"4.jpg",
		"5.jpg",
		"6.jpg",
		"7.jpg",
	};

	//test for avatar generation v2
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt3_data/";

	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(exp_config));

	MeshCompress bfm_template = "D:/dota201201/1218_3dmm_test/res/3dmm.obj";
	MeshCompress bfm_input = "D:/dota210104/0120_color_fit_00_expand_eyes/3dmm.obj";
	MeshCompress guijie_template = "D:/dota210104/0120_color_fit_00_expand_eyes/B_deform.obj";
	MeshCompress guijie_input = "D:/dota210104/0120_color_fit_00_expand_eyes/B_deform.obj";
	MeshCompress guijie_output = guijie_input;

	optV3_ptr->optBasedOn3dmm(bfm_input, bfm_template, guijie_input, guijie_template, guijie_output);
}

void TOPTRANSFER::localDeform(const cstr& input_root)
{
	cstr data_root = "D:/multiPack/";
	//test for avatar generation v2
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt3_data/";

	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(exp_config));

	MeshCompress bfm_template = data_root + "1218_3dmm_test/res/3dmm.obj";
	MeshCompress bfm_input = input_root + "3dmm.obj";
	//MeshCompress guijie_template = "D:/dota201201/1222_fwh_guijie/move_guijie_all.obj";
	MeshCompress guijie_template = data_root + "0127_sl_head/face_interBS_origin_pos.obj";
	MeshCompress guijie_input = input_root + "B_deform.obj";
	MeshCompress guijie_output = guijie_input;

	optV3_ptr->optBasedOn3dmm(bfm_input, bfm_template, guijie_input, guijie_template, guijie_output);
	guijie_output.saveObj(input_root + "local_deform.obj");
}

void TOPTRANSFER::localDeform(const cstr& input_root, const MeshCompress& input_B)
{
	cstr data_root = "D:/multiPack/";
	//test for avatar generation v2
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt3_data/";

	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(exp_config));

	MeshCompress bfm_template = data_root + "1218_3dmm_test/res/3dmm.obj";
	MeshCompress bfm_input = input_root + "3dmm.obj";
	//MeshCompress guijie_template = "D:/dota201201/1222_fwh_guijie/move_guijie_all.obj";
	MeshCompress guijie_template = input_B;
	MeshCompress guijie_input = input_root + "B_deform.obj";
	MeshCompress guijie_output = guijie_input;

	optV3_ptr->optBasedOn3dmm(bfm_input, bfm_template, guijie_input, guijie_template, guijie_output);
	guijie_output.saveObj(input_root + "local_deform.obj");
}

void TOPTRANSFER::localDeform(const cstr& input_root, const MeshCompress& input_B, const std::shared_ptr<OptV3Gen> optV3_ptr)
{
	cstr data_root = "D:/avatar/nl_linux/";
	//test for avatar generation v2

	MeshCompress bfm_template = data_root + "1218_3dmm_test/res/3dmm.obj";
	MeshCompress bfm_input = input_root + "3dmm.obj";
	//MeshCompress guijie_template = "D:/dota201201/1222_fwh_guijie/move_guijie_all.obj";
	MeshCompress guijie_template = input_B;
	MeshCompress guijie_input = input_root + "B_deform.obj";
	MeshCompress guijie_output = guijie_input;

	optV3_ptr->optBasedOn3dmm(bfm_input, bfm_template, guijie_input, guijie_template, guijie_output);
	guijie_output.saveObj(input_root + "local_deform.obj");
}

void TOPTRANSFER::postProcessForGuijieTex()
{
	json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	cstr output_dir_ = config["output_dir_"].get<cstr>();
	cstrVec input = {
		"D:/dota201224/1224_demo/00_cartoon_pair/",
		"D:/dota201224/1224_demo/01_cartoon_pair/",
		"D:/dota201224/1224_demo/03_cartoon_pair/",
		"D:/dota201224/1224_demo/05_cartoon_pair/",
		"D:/dota201224/1224_demo/07_cartoon_pair/",
		"D:/dota201224/1224_demo/08_cartoon_pair/",
		"D:/dota201224/1224_demo/11_cartoon_pair/",
		"D:/dota201224/1224_demo/15_cartoon_pair/",
		"D:/dota201224/1224_demo/17_cartoon_pair/",
		"D:/dota201224/1224_demo/18_cartoon_pair/",
		"D:/dota201224/1224_demo/26_cartoon_pair/",
		"D:/dota201224/1224_demo/28_cartoon_pair/",
		"D:/dota201224/1224_demo/30_cartoon_pair/",
	};
	//input = { output_dir_ };
	for (auto i : input)
	{
		//guijieToFWHInstance(i);
		//transferSimDiff(i, "nricp.obj");
		EASYGUIJIE::transformEyesToMesh(i, "local_deform.obj");
		TOPTRANSFER::localDeform(i);
		//EASYGUIJIE::getEyebrow(i);
	}
}

void TOPTRANSFER::postProcessForGuijieTexBatch(const std::shared_ptr<ExpGen> exp_ptr, const std::shared_ptr<OptV3Gen> optV3_ptr,
	const std::shared_ptr<ConstVar> ptr_const_var, std::shared_ptr<ResVar> ptr_res_var)
{
	json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	cstr root = config["nl_root"].get<cstr>();
	cstr output_dir_ = config["output_dir_"].get<cstr>();
	cstrVec input = 
	{
		output_dir_,
	};

	//MeshCompress B = data_root + "0127_sl_head/face_interBS_origin_pos.obj";
	MeshCompress B = root + config["B"].get<cstr>();
	MeshCompress eyes = root + config["eyes"].get<cstr>();

	input = { output_dir_ };
	for (auto i : input)
	{
#if 1

		if (ptr_res_var->model_3dmm_type_ == Type3dmm::MS)
		{
			LOG(INFO) << "using ms version" << std::endl;
			guijieToFWHInstance(i);
			transferSimDiff(i, "nricp.obj", B, exp_ptr, ptr_res_var);
			TOPTRANSFER::localDeform(i, B, optV3_ptr);
		}
		else if(ptr_res_var->model_3dmm_type_ == Type3dmm::NR)
		{
			MeshCompress template_mesh = ptr_const_var->ptr_data->nr_tensor_.template_obj_;
			ptr_const_var->ptr_data->nr_tensor_.interpretIDFloat(template_mesh.pos_.data(), ptr_res_var->coef_3dmm_);
			template_mesh.saveObj(i + "nricp.obj");
			transferSimDiff(i, "nricp.obj", B, exp_ptr, ptr_res_var);
			TOPTRANSFER::localDeform(i, B, optV3_ptr);
		}
		else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_CPP)
		{
			MeshCompress template_mesh = ptr_const_var->ptr_data->nr_tensor_.template_obj_;
			ptr_const_var->ptr_data->nr_tensor_.interpretIDFloat(template_mesh.pos_.data(), ptr_res_var->coef_3dmm_);
			template_mesh.saveObj(i + "nricp.obj");
			transferSimDiff(i, "nricp.obj", B, exp_ptr, ptr_res_var);
			TOPTRANSFER::localDeform(i, B, optV3_ptr);
		}
		else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_CPP_RAW)
		{
			MeshCompress template_mesh = ptr_const_var->ptr_data->nr_tensor_.template_obj_;
			ptr_const_var->ptr_data->nr_tensor_.interpretIDFloat(template_mesh.pos_.data(), ptr_res_var->coef_3dmm_);
			template_mesh.saveObj(i + "local_deform.obj");
			//transferSimDiff(i, "nricp.obj", B, exp_ptr, ptr_res_var);
		}
		else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_RAW)
		{
			MeshCompress template_mesh = ptr_const_var->ptr_data->nr_tensor_.template_obj_;
			ptr_const_var->ptr_data->nr_tensor_.interpretIDFloat(template_mesh.pos_.data(), ptr_res_var->coef_3dmm_);
			template_mesh.saveObj(i + "local_deform.obj");
		}
		else
		{
			LOG(ERROR) << "undefined occur." << std::endl;
		}

#else
		guijieToFWHInstanceTesting(i);
		transferSimDiffTesting(i, "nricp.obj", B, exp_ptr);
#endif

		//EASYGUIJIE::transformEyesToMesh(i, "local_deform.obj");	
		//EASYGUIJIE::getEyebrow(i);
	}
}

void TOPTRANSFER::generateEyesBrow()
{
	json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	cstr output_dir_ = config["output_dir_"].get<cstr>();
	cstrVec input =
	{
		output_dir_,
	};	
	cstr root = config["nl_root"].get<cstr>();
	cstr head = root + config["B"].get<cstr>();
	cstr eyes = root + config["eyes"].get<cstr>();
	float scale = config["eyes_scale"].get<float>();
	float shift = config["eyes_shift"].get<float>();
	for (auto i : input)
	{		
		EASYGUIJIE::transformEyesToMesh(i, "local_deform.obj", head, eyes, scale, shift);
		EASYGUIJIE::getEyebrow(i, root);
	}
}

void TOPTRANSFER::guijieToFWHInstance(const cstr& obj_root)
{
	//cstr data_root = "D:/multiPack/";
	cstr data_root = "D:/avatar/nl_linux/";
	//68 drag	
	intVec loading_fix_part = FILEIO::loadIntDynamic(data_root + "1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic(data_root + "1220_fwh_guijie/fix_guijie_expand.txt");

	MeshCompress fwh_obj(obj_root + "bfm_lap_sys.obj");
	MeshCompress guijie_infer_base_obj(data_root + "1223_test_infer/guijie_infer_base.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic(data_root + "fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic(data_root + "guijie_68_sys.txt");

	intVec lash_eye_pair = FILEIO::loadIntDynamic(data_root + "left_down_match.txt");
	FILEIO::loadIntDynamic(data_root + "left_up_match.txt", lash_eye_pair);
	FILEIO::loadIntDynamic(data_root + "right_down_match.txt", lash_eye_pair);
	FILEIO::loadIntDynamic(data_root + "right_up_match.txt", lash_eye_pair);

	MeshCompress guijie_drag_68 = guijie_infer_base_obj;
	SIMDEFORM::moveHandle(guijie_infer_base_obj, guijie_68_idx, fwh_obj, fwh_68_idx, loading_fix_part,
		lash_eye_pair, guijie_drag_68);

	guijie_drag_68.saveObj(obj_root + "drag68.obj");

	intVec eye_skip_nricp = FILEIO::loadIntDynamic(data_root + "1223_test_infer/eye_skip_nricp.txt");
	intVec guijie_move_idx = FILEIO::loadIntDynamic(data_root + "1223_test_infer/guijie_move_idx_all.txt");
	intVec fwh_move_idx = FILEIO::loadIntDynamic(data_root + "1223_test_infer/fwh_move_idx.txt");

	intX2Vec refine_move_idx = MAP::discardRoiKeepOrder({ guijie_move_idx, fwh_move_idx }, guijie_move_idx, eye_skip_nricp);
	guijie_move_idx = refine_move_idx[0];
	fwh_move_idx = refine_move_idx[1];


	FILEIO::saveDynamic(data_root + "1223_test_infer/guijie_move_idx_skip_eyes.txt", guijie_move_idx, ",");
	FILEIO::saveDynamic(data_root + "1223_test_infer/fwh_move_idx_skip_eyes.txt", fwh_move_idx, ",");


	loading_fix_part.insert(loading_fix_part.end(), eye_skip_nricp.begin(), eye_skip_nricp.end());
	MeshCompress guijie_nricp = guijie_drag_68;
	SIMDEFORM::moveHandle(guijie_drag_68, guijie_move_idx, fwh_obj, fwh_move_idx, loading_fix_part,
		lash_eye_pair, guijie_nricp);
	guijie_nricp.saveObj(obj_root + "nricp.obj");
}

void TOPTRANSFER::guijieToFWHInstanceTesting(const cstr& obj_root)
{
	cstr data_root = "D:/multiPack/";
	cstr data_folder = "0422_test_infer/";
	//68 drag	
	intVec loading_fix_part = FILEIO::loadIntDynamic(data_root + data_folder + "fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic(data_root + data_folder + "fix_guijie_expand.txt");

	MeshCompress fwh_obj(obj_root + "bfm_lap_sys.obj");
	MeshCompress guijie_infer_base_obj(data_root + data_folder + "guijie_infer_base.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic(data_root + "fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic(data_root + "guijie_68_sys.txt");

	intVec lash_eye_pair = FILEIO::loadIntDynamic(data_root + "left_down_match.txt");
	FILEIO::loadIntDynamic(data_root + "left_up_match.txt", lash_eye_pair);
	FILEIO::loadIntDynamic(data_root + "right_down_match.txt", lash_eye_pair);
	FILEIO::loadIntDynamic(data_root + "right_up_match.txt", lash_eye_pair);

	MeshCompress guijie_drag_68 = guijie_infer_base_obj;
	//skip drag
	//SIMDEFORM::moveHandle(guijie_infer_base_obj, guijie_68_idx, fwh_obj, fwh_68_idx, loading_fix_part,
	//	lash_eye_pair, guijie_drag_68);

	guijie_drag_68.saveObj(obj_root + "drag68.obj");

	intVec eye_skip_nricp = FILEIO::loadIntDynamic(data_root + data_folder + "eye_skip_nricp.txt");
	intVec guijie_move_idx = FILEIO::loadIntDynamic(data_root + data_folder + "guijie_move_idx_all.txt");
	intVec fwh_move_idx = FILEIO::loadIntDynamic(data_root + data_folder + "fwh_move_idx.txt");

	intX2Vec refine_move_idx = MAP::discardRoiKeepOrder({ guijie_move_idx, fwh_move_idx }, guijie_move_idx, eye_skip_nricp);
	guijie_move_idx = refine_move_idx[0];
	fwh_move_idx = refine_move_idx[1];


	FILEIO::saveDynamic(data_root + data_folder + "guijie_move_idx_skip_eyes.txt", guijie_move_idx, ",");
	FILEIO::saveDynamic(data_root + data_folder + "fwh_move_idx_skip_eyes.txt", fwh_move_idx, ",");


	loading_fix_part.insert(loading_fix_part.end(), eye_skip_nricp.begin(), eye_skip_nricp.end());
	MeshCompress guijie_nricp = guijie_drag_68;
	SIMDEFORM::moveHandle(guijie_drag_68, guijie_move_idx, fwh_obj, fwh_move_idx, loading_fix_part,
		lash_eye_pair, guijie_nricp);
	guijie_nricp.saveObj(obj_root + "nricp.obj");
}

void TOPTRANSFER::guijieToFWHInferExample()
{
	cstr root = "D:/dota201201/1223_test_infer/";
	//68 drag	
	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	
	MeshCompress fwh_obj(root + "bfm_lap_sys.obj");
	MeshCompress guijie_infer_base_obj(root + "guijie_infer_base.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");	

	intVec lash_eye_pair = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt", lash_eye_pair);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", lash_eye_pair);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", lash_eye_pair);

	MeshCompress guijie_drag_68 = guijie_infer_base_obj;
	SIMDEFORM::moveHandle(guijie_infer_base_obj, guijie_68_idx, fwh_obj, fwh_68_idx, loading_fix_part,
		lash_eye_pair, guijie_drag_68);

	guijie_drag_68.saveObj(root + "drag68.obj");

	intVec eye_skip_nricp = FILEIO::loadIntDynamic(root + "eye_skip_nricp.txt");
	intVec guijie_move_idx = FILEIO::loadIntDynamic(root + "guijie_move_idx_all.txt");
	intVec fwh_move_idx = FILEIO::loadIntDynamic(root + "fwh_move_idx.txt");

	intX2Vec refine_move_idx = MAP::discardRoiKeepOrder({ guijie_move_idx, fwh_move_idx}, guijie_move_idx, eye_skip_nricp);
	guijie_move_idx = refine_move_idx[0];
	fwh_move_idx = refine_move_idx[1];


	loading_fix_part.insert(loading_fix_part.end(), eye_skip_nricp.begin(), eye_skip_nricp.end());
	MeshCompress guijie_nricp = guijie_drag_68;
	SIMDEFORM::moveHandle(guijie_drag_68, guijie_move_idx, fwh_obj, fwh_move_idx, loading_fix_part,
		lash_eye_pair, guijie_nricp);
	guijie_nricp.saveObj(root + "nricp.obj");
}

void TOPTRANSFER::getMatchFromDFAdv()
{
	cstr root = "D:/dota201201/1222_iter_df/";
	MeshCompress guijie_to(root + "guijie_back_v2.obj");
	MeshCompress fwh_deform(root + "deform.obj");
	MeshCompress fwh_ori(root + "bfm_lap_sys.obj");
	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic(root + "corre_26.000000.txt");
	
	guijie_to.generateNormal();
	fwh_deform.generateNormal();
	fwh_ori.generateNormal();

	intX2Vec map_guijie_to_fwh(guijie_to.n_vertex_);

	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	intVec loading_ori_mapping = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/mapping.txt");

	intVec loading_part_to_whole(guijie_to.n_vertex_);
	for (size_t i = 0; i < loading_ori_mapping.size(); i++)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			loading_part_to_whole[loading_ori_mapping[i]] = i;
		}
	}

	intVec discard_mapping_fix;
	intVec discard_mapping_fix_expand;
	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}

	for (int i : loading_fix_part_expand)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix_expand.push_back(loading_ori_mapping[i]);
		}
	}


	for (int i = 0; i < map_uv_fwh_to_guijie.size() / 2; i++)
	{
		int fwh_from = map_uv_fwh_to_guijie[2 * i];
		int guijie_to = map_uv_fwh_to_guijie[2 * i + 1];
		map_guijie_to_fwh[guijie_to].push_back(fwh_from);
	}
	guijie_to.tex_cor_.resize(guijie_to.n_vertex_);
	guijie_to.n_uv_ = guijie_to.n_vertex_;
	guijie_to.tri_uv_.clear();
	guijie_to.tri_uv_ = guijie_to.tri_;

	intX2Vec guijie_to_fwh(guijie_to.n_vertex_);
	intX2Vec fwh_to_guijie(fwh_deform.n_vertex_);
	intSet discard_multi_matching;
	double thres = 2;
	intVec fwh_error = FILEIO::loadIntDynamic(root + "fwh_error.txt");
	intSet fwh_error_set(fwh_error.begin(), fwh_error.end());
	//#pragma omp parallel for
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		int min_index = -1;
		float min_dis = 1e8;
		for (int iter = 0; iter < map_guijie_to_fwh[i].size(); iter++)
		{
			int fwh_idx = map_guijie_to_fwh[i][iter];
			//float temp_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
			//float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm() + (guijie_to.normal_[i] - fwh_deform.normal_[fwh_idx]).norm();
			//if (temp_dis > thres)
			//{
			//	cur_dis = 1e10;
			//}			
			//cur_dis = cur_dis * cur_dis;
			float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
			min_index = min_dis > cur_dis ? fwh_idx : min_index;
			min_dis = DMIN(min_dis, cur_dis);
		}
		if (min_index < 0)
		{
#if 0
			min_index = -1;
			min_dis = 1e8;
			for (int iter = 0; iter < fwh_deform.pos_.size(); iter++)
			{
				int fwh_idx = iter;
				//float temp_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();
				//float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm() + (guijie_to.normal_[i] - fwh_deform.normal_[fwh_idx]).norm();
				//if (temp_dis > thres)
				//{
				//	cur_dis = 1e10;
				//}
				//cur_dis = cur_dis * cur_dis;
				float cur_dis = (guijie_to.pos_[i] - fwh_deform.pos_[fwh_idx]).norm();		
				min_index = min_dis > cur_dis ? fwh_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
#endif
		}

		if (min_index < 0)
		{
			//LOG(ERROR) << "match missed" << std::endl;
		}
		else
		{

			if (guijie_to_fwh[i].size() > 0 || fwh_to_guijie[min_index].size() > 0)
			{
				//LOG(INFO) << "discard here comes multiple matching" << std::endl;
				//LOG(INFO) << "fwh: " << min_index << std::endl;
				//LOG(INFO) << "guijie: " << i << std::endl;
				discard_multi_matching.insert(i);
			}
			else
			{
				guijie_to_fwh[i].push_back(min_index);
				fwh_to_guijie[min_index].push_back(i);
			}
		}
	}
	LOG(INFO) << "discard size: " << discard_multi_matching.size() << std::endl;

	intVec move_idx;
	float3Vec move_pos;
	intSet fix_idx(discard_mapping_fix.begin(), discard_mapping_fix.end());
	intSet fix_idx_expand(discard_mapping_fix_expand.begin(), discard_mapping_fix_expand.end());


	intVec fwh_move_idx, guijie_move_idx_all, guijie_move_idx_part;

	intVec skip_eye_list = FILEIO::loadIntDynamic(root + "error.txt");
	intSet skip_eye_set(skip_eye_list.begin(), skip_eye_list.end());
	for (int i = 0; i < guijie_to_fwh.size(); i++)
	{
		if (discard_multi_matching.count(i) || fix_idx_expand.count(i) || guijie_to_fwh[i].empty() || skip_eye_set.count(i)
			|| fwh_error_set.count(guijie_to_fwh[i][0]))
		{

		}
		else
		{
			move_idx.push_back(i);
			move_pos.push_back(fwh_ori.pos_[guijie_to_fwh[i][0]]);

			fwh_move_idx.push_back(guijie_to_fwh[i][0]);
			guijie_move_idx_all.push_back(loading_part_to_whole[i]);
			guijie_move_idx_part.push_back(i);
		}
	}

	LOG(INFO) << "sim mapping size: " << move_idx.size() << std::endl;

	intVec guijie_ring;
	intVec fwh_ring;
	for (int i = 0; i < guijie_to.n_vertex_; i++)
	{
		if (guijie_to_fwh[i].size() > 0 && fix_idx_expand.count(i) && !fix_idx.count(i))
		{
			guijie_ring.push_back(loading_part_to_whole[i]);
			fwh_ring.push_back(guijie_to_fwh[i][0]);
		}
	}

	FILEIO::saveDynamic(root + "guijie_ring.txt", guijie_ring, ",");
	FILEIO::saveDynamic(root + "fwh_ring.txt", fwh_ring, ",");

	FILEIO::saveDynamic(root + "fwh_move_idx.txt", fwh_move_idx, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_all.txt", guijie_move_idx_all, ",");
	FILEIO::saveDynamic(root + "guijie_move_idx_part.txt", guijie_move_idx_part, ",");

	MeshCompress guijie_template = guijie_to;
	LaplacianDeform to_fwh;
	to_fwh.init(guijie_template, move_idx, discard_mapping_fix);
	to_fwh.deform(move_pos, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_map_back.obj");
	guijie_template.discard(skip_eye_list);
	guijie_template.saveObj(root + "guijie_map_back_dump_error.obj");

}

void TOPTRANSFER::getMatchFromFile()
{
	cstr root = "D:/dota201201/1222_fwh_guijie/";
	MeshCompress guijie_to(root + "move_guijie_part.obj");
	MeshCompress fwh_deform(root + "deform.obj");
	MeshCompress fwh_ori(root + "bfm_lap_sys.obj");
	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic(root + "corre_26.000000.txt");
	intX2Vec map_guijie_to_fwh(guijie_to.n_vertex_);
	intVec loading_fix_part = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	intVec loading_ori_mapping = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/mapping.txt");
	intVec loading_part_to_whole(guijie_to.n_vertex_);
	

	intVec fwh_idx = FILEIO::loadIntDynamic("D:/dota201201/1222_fwh_guijie/fwh_move_idx.txt");
	intVec guijie_all_idx = FILEIO::loadIntDynamic("D:/dota201201/1222_fwh_guijie/guijie_move_idx_all.txt");
	intVec guijie_part_idx = FILEIO::loadIntDynamic("D:/dota201201/1222_fwh_guijie/guijie_move_idx_part.txt");
	
	float3Vec move_pos;
	for (int i = 0; i < guijie_part_idx.size(); i++)
	{
		move_pos.push_back(fwh_ori.pos_[fwh_idx[i]]);
	}

	MeshCompress guijie_template = guijie_to;
	LaplacianDeform to_fwh;
	intVec discard_mapping_fix;
	for (int i : loading_fix_part)
	{
		if (loading_ori_mapping[i] >= 0)
		{
			discard_mapping_fix.push_back(loading_ori_mapping[i]);
		}
	}
	to_fwh.init(guijie_template, guijie_part_idx, discard_mapping_fix);
	to_fwh.deform(move_pos, guijie_template.pos_);
	guijie_template.saveObj(root + "guijie_map_back.obj");

}

void TOPTRANSFER::fromGuijieToFWH()
{
	LOG(INFO) << "from guijie to fwh" << std::endl;
	cstr root = "D:/dota201201/1218_3dmm_test/reference/";
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);
	intVec guijie_68 = FILEIO::loadIntDynamic("C:/code/expgen_aquila/data/guijie_opt3_data/guijie_68_sys.txt");
	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	//obj in
	MeshCompress guijie_maya(root + "Guijie_head.obj");
	MeshCompress guijie_nolash = guijie_maya;
	intVec mapping = guijie_nolash.discard(guijie_eyelash);
	guijie_nolash.saveObj(root + "Guijie_nolash.obj");
	FILEIO::saveDynamic(root + "discard.txt", mapping, ",");
}

void TOPTRANSFER::dumpGuijieEyelash()
{
	cstr root = "D:/dota201116/1116_taobao/guijie/";
	//guijie dump eyelash
	intVec guijie_discard = FILEIO::loadIntDynamic("D:/dota201104/1104_fwh_guijie/discard_vertex.txt");

	MeshCompress guijie_obj(root + "0_opt.obj");

	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	intVec guijie_68_idx = FILEIO::loadIntDynamic("D:/avatar/guijie_opt2_data/guijie_68_sys.txt");

	intVec all_part_map = guijie_obj.discard(guijie_discard);
	intVec re_guijie_68;
	MeshTools::getSlice(all_part_map, guijie_68_idx, re_guijie_68);

	FILEIO::saveDynamic(root + "mapping.txt", all_part_map, ",");
	FILEIO::saveDynamic(root + "re_guijie_68.txt", re_guijie_68, ",");
	guijie_obj.saveObj(root + "re_guijie.obj");

}

void TOPTRANSFER::transferOnUV()
{
	MeshCompress base_eye("D:/dota201224/1227_maya/maya_eyes.obj");
	MeshCompress mofeng_eye("D:/dota201224/1227_maya/mofeng_eyes.obj");
	intVec res = MeshTools::getMatchBasedOnUV(base_eye, mofeng_eye, 0.01);
	//从右眼向左眼映射
	MeshCompress adjust_eye = base_eye;
	int n_size = adjust_eye.n_vertex_*0.5;
	for (int i = 0; i < n_size; i++)
	{
		adjust_eye.pos_[i] = mofeng_eye.pos_[i];
	}

	for (int i = n_size; i < 2*n_size; i++)
	{
		adjust_eye.pos_[i] = mofeng_eye.pos_[i];
	}
	adjust_eye.saveObj("D:/dota201224/1227_maya/adjust_eyes.obj");
}

void TOPTRANSFER::copyHeadBS(const json& config)
{
	cstr src_root = config["json_root"].get<cstr>() + "1/";
	cstr dst_root = config["headbs_result"].get<cstr>();
	SG::needPath(dst_root);
	cstrVec exp_name, sound_name;
	FILEIO::loadFixSize("D:/avatar/exp_server_config/ani_exp_pca/json_name.txt", exp_name);
	FILEIO::loadFixSize("D:/avatar/exp_server_config/ani_sound_pca/json_name.txt", sound_name);

	for (int i = 1; i < exp_name.size(); i++)
	{
		FILEIO::copyFile(src_root + exp_name[i] + ".obj", dst_root + exp_name[i] + ".obj");
	}

	for (int i = 1; i < sound_name.size(); i++)
	{
		FILEIO::copyFile(src_root + sound_name[i] + ".obj", dst_root + sound_name[i] + ".obj");
	}

	FILEIO::copyFile(src_root + "B_left_blink.obj", dst_root + "Ani_eyeBlinkLeft.obj");
	FILEIO::copyFile(src_root + "B_right_blink.obj", dst_root + "Ani_eyeBlinkRight.obj");


}

void TOPTRANSFER::serverGenExpFromMesh(const json& config)
{
	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";
	cstr root = config["json_root"].get<cstr>();
	cstr root_headbs = config["headbs_result"].get<cstr>();
	SG::needPath(root_headbs);
	bool is_compress = true;
	cstrVec json_vec = { "1.json" };
	std::shared_ptr<ExpGen> exp_ptr;
	std::shared_ptr<BsGenerate> bs_ptr;
	exp_ptr.reset(new ExpGen(exp_config));
	bs_ptr.reset(new BsGenerate(exp_config));

	for (int i = 0; i < json_vec.size(); i++)
	{
		cstr json_name = json_vec[i];
		cstr json_raw_name = FILEIO::getFileNameWithoutExt(json_name);
		SG::needPath(root + "/" + json_raw_name);
		json json_bs = FILEIO::loadJson(root + json_name);
		MeshCompress B(config["maya_head"].get<cstr>());
		//TinyTool::skeletonChange();
		//MeshCompress B(root + json_raw_name + ".obj");
		exp_ptr->fixEyelash(B);
		exp_ptr->setDebug(true);
		exp_ptr->setB(B);
		exp_ptr->getExpOMP(B, root + json_raw_name + "/");
		nlohmann::json vertex_value;
		exp_ptr->getResultJson(vertex_value);
		//win need
		SG::needPath(root + json_raw_name + "_res");
		FILEIO::saveJson(root + json_raw_name + "_res/" + json_raw_name + "_delta.json", vertex_value);
		FILEIO::saveJson(config["bs_delta"].get<cstr>() +"face_exp.json", vertex_value);
		FILEIO::saveJson(root + json_raw_name + "_res/" + json_raw_name + ".json", json_bs);
		B.saveObj(root + json_raw_name + ".obj");
		B.saveObj(root_headbs + json_raw_name + ".obj");
		exp_ptr->dumpEyelash(B);
		B.saveObj(root + json_raw_name + "_no_lash.obj");
	}
	
}

void TOPTRANSFER::getTemplateBS()
{
	json config;

#if 0
	//cstr root_path = "D:/dota210104/0111_generate_head/";
	//cstr data_path = "D:/dota210104/0111_generate_head/";
#else
	cstr root_path = "D:/dotaPTA/0319_yyx/";
	cstr data_path = "D:/dotaPTA/0319_yyx/";
#endif

	config["json_root"] = root_path + "json/";
	config["mapping_root"] = root_path + "mapping/";

	config["maya_head"] = root_path + "maya/head.obj";
	config["maya_eyes"] = root_path + "maya/eyes.obj";
	config["maya_eyebrow"] = root_path + "maya/eyebrow.obj";
	config["maya_tooth"] = root_path + "maya/tooth.obj";
	
	config["unity_head"] = root_path + "unity/Guijie_head.obj";
	config["unity_eyes"] = root_path + "unity/Guijie_eyes.obj";
	config["unity_eyebrow"] = root_path + "unity/Guijie_eyebrow.obj";
	config["unity_tooth"] = root_path + "unity/Guijie_tooth.obj";

	config["template_head"] = root_path + "template/head_guijie.obj";
	config["template_eyes"] = root_path + "template/eyes_guijie.obj";
	config["template_eyebrow"] = root_path + "template/eyebrow_guijie.obj";
	config["template_tooth"] = root_path + "template/tooth_guijie.obj";

	config["bs_delta"] = root_path + "bs_delta/";

	//head
	config["headbs_result"] = root_path + "debug/head_result/";
	// eyes
	config["eyebs_result"] = root_path + "debug/eyebs_result/";
	config["eyebs_temp"] = root_path + "debug/eyebs_temp/";
	// tooth
	config["toothbs_result"] = root_path + "debug/toothbs_result/";
	config["toothbs_temp"] = root_path + "debug/toothbs_temp/";
	// eyebrow
	config["eyebrowbs_result"] = root_path + "debug/eyebrowbs_result/";
	config["eyebrowbs_temp"] = root_path + "debug/eyebrowbs_temp/";
	// eyebrow config
	config["eyebrowbs_left_config"] = root_path + "bsTemplate/eyebrow_config/left_config.json";
	config["eyebrowbs_right_config"] = root_path + "bsTemplate/eyebrow_config/right_config.json";
	config["eyebrowbs_delete_config"] = root_path + "bsTemplate/eyebrow_config/delete_config.json";
	config["eyebrowbs_part_to_maya"] = root_path + "bsTemplate/eyebrow_config/part_to_maya.json";
	
	// head shapebs
	config["head_shapebs_result"] = root_path + "debug/head_shapebs_result/";
	
	// eye shapebs
	config["eye_shapebs_result"] = root_path + "debug/eye_shapebs_result/";
	config["eye_shapebs_temp"] = root_path + "debug/eye_shapebs_temp/";
	
	//eyebrow shapebs
	config["eyebrow_shapebs_result"] = root_path + "debug/eyebrow_shapebs_result/";

	// tooth shapebs
	config["tooth_shapebs_result"] = root_path + "debug/tooth_shapebs_result/";
	config["tooth_shapebs_temp"] = root_path + "debug/tooth_shapebs_temp/";
	
	// base data
	config["template_eyeAni"] = data_path + "bsTemplate/eye_ani/";
	config["template_toothAni"] = data_path + "bsTemplate/tooth_ani/";
	config["template_eyebrowAni"] = data_path + "json/1/";
	config["template_headShapeBs"] = data_path + "bsTemplate/head_shapebs/";
	config["template_eyeShapeBs"] = data_path + "bsTemplate/eye_shapebs/";
	config["template_toothShapeBs"] = data_path + "bsTemplate/tooth_shapebs/";
	
	// quad data
	config["to_fbx_headAni"] = data_path + "toFbx/head_ani/";
	config["to_fbx_toothAni"] = data_path + "toFbx/tooth_ani/";
	config["to_fbx_eyebrowAni"] = data_path + "toFbx/eyebrow_ani/";
	config["to_fbx_headShapeBs"] = data_path + "toFbx/head_shape/";
	config["to_fbx_eyeShapeBs"] = data_path + "toFbx/eye_shape/";
	config["to_fbx_toothShapeBs"] = data_path + "toFbx/tooth_shape/";

	bool b_maya_unity_mapping = false;
	bool b_gen_face_expression = true;
	bool b_gen_face_shape = true;
	bool b_gen_eye_expression = true;
	bool b_gen_eye_shape = true;
	bool b_gen_tooth_expression = true;
	bool b_gen_tooth_shape = true;
	bool b_gen_eyebrow_base = true;
	bool b_gen_eyebrow_expression = true;
	bool b_gen_eyebrow_shape = true;
	bool b_pack_bs = true;
	bool b_to_quad_name = false;

	if(b_maya_unity_mapping)
	{
		LOG(INFO) << "generate mapping from maya to unity. " << std::endl;
		getMappingFromMayaToUnity(config);
		LOG(INFO) << "end mapping from maya to unity. " << std::endl;
	}
	else
	{
		LOG(INFO) << "skip maya/unity mapping. " << std::endl;
	}

	if(b_gen_face_expression)
	{
		LOG(INFO) << "generate Face Expression. " << std::endl;
		serverGenExpFromMesh(config);
		copyHeadBS(config);
		LOG(INFO) << "end Face Expression. " << std::endl;
	}
	else
	{
		LOG(INFO) << "skip generate Face Expression. " << std::endl;
	}

	if(b_gen_face_shape)
	{
		LOG(INFO) << "generate head shape bs data." << std::endl;
		getShapeBS(config);
		LOG(INFO) << "end head shape bs data." << std::endl;
	}
	else
	{
		LOG(INFO) << "skip generate head shape bs. " << std::endl;
	}

	if(b_gen_eye_expression)
	{
		LOG(INFO) << "generate eye Expression. " << std::endl;
		generateEyeBS(config);
		LOG(INFO) << "end eye Expression. " << std::endl;
	}
	else
	{
		LOG(INFO) << "skip eye Expression. " << std::endl;
	}
	
	if(b_gen_eye_shape)
	{
		LOG(INFO) << "generate eye shape bs data." << std::endl;
		getShpaeEyeBS(config);
		LOG(INFO) << "end eye shape bs data." << std::endl;
	}
	else
	{
		LOG(INFO) << "skip eye shape bs data." << std::endl;
	}

	if (b_gen_tooth_expression)
	{
		LOG(INFO) << "generate tooth Expression. " << std::endl;
		generateToothBS(config);
		LOG(INFO) << "end tooth Expression. " << std::endl;
	}
	else
	{
		LOG(INFO) << "skip tooth Expression." << std::endl;
	}

	if (b_gen_tooth_shape)
	{
		LOG(INFO) << "generate tooth shape bs data." << std::endl;
		getToothShapeBS(config);
		LOG(INFO) << "end tooth shape bs data." << std::endl;
	}
	else
	{
		LOG(INFO) << "skip generate tooth shape." << std::endl;
	}

	if (b_gen_eyebrow_base)
	{
		LOG(INFO) << "adjust eyebrow shape." << std::endl;
		EASYGUIJIE::getGuijieEyebrow(config);
		LOG(INFO) << "end adjust eyebrow shape." << std::endl;
	}
	else
	{
		LOG(INFO) << "skip adjust eyebrow shape." << std::endl;
	}

	if (b_gen_eyebrow_expression)
	{
		LOG(INFO) << "generate eyebrow Expression. " << std::endl;
		generateEyebrowBS(config);
		LOG(INFO) << "end eyebrow Expression. " << std::endl;
	}
	else
	{
		LOG(INFO) << "skip eyebrow Expression " << std::endl;
	}

	if (b_gen_eyebrow_shape)
	{
		LOG(INFO) << "generate eyebrow shape bs data." << std::endl;
		getShpaeEyeBrowBS(config);
		LOG(INFO) << "end eyebrow shape bs data." << std::endl;
	}
	else
	{
		LOG(INFO) << "skip eyebrow shape bs data " << std::endl;
	}

	if(b_pack_bs)
	{
		LOG(INFO) << "packing express delta file." << std::endl;
		packDeltaForExp(config);
		LOG(INFO) << "end express delta file." << std::endl;
	}
	else
	{
		LOG(INFO) << "skip express delta file." << std::endl;
	}

	if(b_to_quad_name)
	{
		LOG(INFO) << "turn to quad && set name" << std::endl;
		addBS(config);
		LOG(INFO) << "end quad && set name" << std::endl;
	}
	else
	{

	}
}

void TOPTRANSFER::getShpaeEyeBS(const json& config)
{
	//get for eyebs, split and generate
	cstr ref_folder = config["template_eyeShapeBs"].get<cstr>();
	cstr result_folder = config["eye_shapebs_result"].get<cstr>();
	cstr temp_folder = config["eye_shapebs_temp"].get<cstr>();
	SG::needPath(result_folder);
	SG::needPath(temp_folder);

	MeshCompress guijie_src(config["maya_eyes"].get<cstr>());
	MeshCompress guijie_template(config["template_eyes"].get<cstr>());
	RIGIDBS::getSysDivideAndConquer(guijie_src, guijie_template, ref_folder, result_folder, temp_folder);

}

void TOPTRANSFER::getShapeBS(const json& config)
{
	//head
	//cstr head_root = "D:/dota201201/1216_nielian/bs_head/";
	//cstr head_res_root = "D:/dota201201/1216_nielian/bs_head_res/";
	cstr head_root = config["template_headShapeBs"].get<cstr>();
	cstr head_res_root = config["head_shapebs_result"].get<cstr>();
	
	SG::needPath(head_res_root);
	
	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";
	bool is_compress = true;
	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));
	cstrVec input = FILEIO::getFolderFiles(head_root, ".obj");

	MeshCompress B(config["maya_head"].get<cstr>());
	for (int i = 0; i < input.size(); i++)
	{
		MeshCompress A_deform(head_root + input[i]);
		exp_ptr->fixEyelash(B);
		MeshCompress B_res = B;
		exp_ptr->getExpGuided(B, A_deform, B_res);
		nlohmann::json vertex_value;
		exp_ptr->getResultJson(vertex_value);
		//win need
		cstr json_raw_name = FILEIO::getFileNameWithoutExt(input[i]);
		B_res.saveObj(head_res_root + input[i]);
		//exp_ptr->dumpEyelash(B_res);
		//B_res.saveObj(head_res_root + json_raw_name + "_no_lash.obj");
	}
}

void TOPTRANSFER::getShpaeEyeBrowBS(const json& config)
{
	cstr result = config["eyebrow_shapebs_result"].get<cstr>();
	SG::needPath(result);
#if 0
	{
		MeshCompress eyebrow_guijie("D:/dota201201/1214_eyebrow_mapping/eyebrow.obj");
		MeshCompress eyebrow_front("D:/dota201201/1214_eyebrow_mapping/eyebrow_Front.obj");
		float3E translate;
		RT::getTranslate(eyebrow_guijie.pos_, eyebrow_front.pos_, translate);
		LOG(INFO) << "translate: " << translate << std::endl;
		//0.92 
		//初始offset 0.08 + bs offset 0.92
	}
#endif
	MeshCompress base(config["maya_eyebrow"].get<cstr>());
	float3E translate(0, 0, 1.0);
	RT::translateInPlace(translate, base.pos_);
	base.saveObj(result + "eyebrow_Front.obj");
}

void TOPTRANSFER::generateEyebrowBS(const json& config)
{
	cstr eyebrow_ani_folder = config["template_eyebrowAni"].get<cstr>();
	cstr result_folder = config["eyebrowbs_result"].get<cstr>();
	cstr bs_delta = config["bs_delta"].get<cstr>();

	SG::needPath(bs_delta);
	SG::needPath(result_folder);
#if 0
	{
		MeshCompress eyebrow_guijie("D:/dota201201/1214_eyebrow_mapping/eyebrow.obj");
		MeshCompress eyebrow_front("D:/dota201201/1214_eyebrow_mapping/eyebrow_Front.obj");
		float3E translate;
		RT::getTranslate(eyebrow_guijie.pos_, eyebrow_front.pos_, translate);
		LOG(INFO) << "translate: " << translate << std::endl;
		//0.92 
		//初始offset 0.08 + bs offset 0.92
	}
#endif

	json left_config = FILEIO::loadJson(config["eyebrowbs_left_config"].get<cstr>());
	json right_config = FILEIO::loadJson(config["eyebrowbs_right_config"].get<cstr>());
	json delete_config = FILEIO::loadJson(config["eyebrowbs_delete_config"].get<cstr>());
	json mapping_part_all = FILEIO::loadJson(config["eyebrowbs_part_to_maya"].get<cstr>());
	//LOG(INFO) << "left_config: " << left_config << std::endl;
	//LOG(INFO) << "right_config: " << right_config << std::endl;
	//LOG(INFO) << "delete_config: " << delete_config << std::endl;
	MeshCompress base(config["maya_eyebrow"].get<cstr>());
	//DebugTools::printJson(left_config);
	intVec part_all_mapping = mapping_part_all["part_to_all_eyebrow"].get<intVec>();
	//force delete_config
	cstrVec delete_names = delete_config["set_zero"].get<cstrVec>();
	RIGIDBS::getUsingZeroShift(base, delete_names, result_folder);
	//此处经验值
	float3E translate = float3E(0, 0, 0.25);
	RIGIDBS::getSysDirectMappingFromMesh(base, eyebrow_ani_folder, result_folder, part_all_mapping, left_config,
		true, translate);
	RIGIDBS::getSysDirectMappingFromMesh(base, eyebrow_ani_folder, result_folder, part_all_mapping, right_config,
		false, translate);
	json exp_result;
	RIGIDBS::getJsonExp(base, result_folder, exp_result);
	exp_result["n_vertex"] = base.n_vertex_;
	exp_result["is_compress"] = false;
	FILEIO::saveJson(bs_delta + "eyebrow_exp.json", exp_result);
}

void TOPTRANSFER::generateEyeBS(const json& config)
{
	cstr eye_ani_folder = config["template_eyeAni"].get<cstr>();
	cstr result_folder = config["eyebs_result"].get<cstr>();
	cstr temp_folder = config["eyebs_temp"].get<cstr>();
	cstr bs_delta = config["bs_delta"].get<cstr>();

	SG::needPath(bs_delta);
	SG::needPath(result_folder);
	SG::needPath(temp_folder);

	MeshCompress guijie_src(config["maya_eyes"].get<cstr>());
	MeshCompress guijie_template(config["template_eyes"].get<cstr>());
	RIGIDBS::getSysDivideAndConquer(guijie_src, guijie_template, eye_ani_folder, result_folder, temp_folder);
	json exp_result;
	RIGIDBS::getJsonExp(guijie_src, result_folder, exp_result);
	exp_result["n_vertex"] = guijie_src.n_vertex_;
	exp_result["is_compress"] = false;
	FILEIO::saveJson(bs_delta + "eyes_exp.json", exp_result);
}

void TOPTRANSFER::generateToothBS(const json& config)
{
	cstr tooth_ani_folder = config["template_toothAni"].get<cstr>();
	cstr result_folder = config["toothbs_result"].get<cstr>();
	cstr temp_folder = config["toothbs_temp"].get<cstr>();
	cstr bs_delta = config["bs_delta"].get<cstr>();

	SG::needPath(bs_delta);
	SG::needPath(result_folder);
	SG::needPath(temp_folder);

	MeshCompress guijie_src(config["maya_tooth"].get<cstr>());
	MeshCompress guijie_template(config["template_tooth"].get<cstr>());

	RIGIDBS::getSingleBS(guijie_src, guijie_template, tooth_ani_folder, result_folder, temp_folder);
	json exp_result;
	RIGIDBS::getJsonExp(guijie_src, result_folder, exp_result);
	exp_result["n_vertex"] = guijie_src.n_vertex_;
	exp_result["is_compress"] = false;
	FILEIO::saveJson(bs_delta + "tooth_exp.json", exp_result);
}

void TOPTRANSFER::getToothShapeBS(const json& config)
{
	cstr ref_folder = config["template_toothShapeBs"].get<cstr>();
	cstr result_folder = config["tooth_shapebs_result"].get<cstr>();
	cstr temp_folder = config["tooth_shapebs_temp"].get<cstr>();

	SG::needPath(result_folder);
	SG::needPath(temp_folder);

	MeshCompress guijie_src = config["maya_tooth"].get<cstr>();
	MeshCompress guijie_template = config["template_tooth"].get<cstr>();
	RIGIDBS::getSingleBS(guijie_src, guijie_template, ref_folder, result_folder, temp_folder);
}

void TOPTRANSFER::addBS(const json& config)
{
	cstr root = "D:/dota201201/1217_changeEye_bs/";
	cstr ori_folder = root + "ori/";
	cstr res_folder = root + "res/";

	SG::needPath(res_folder);
	MeshCompress ori_base("D:/dota201201/1217_changeEye_bs/ori/Guijie_eyes.obj");
	MeshCompress rotate_base("D:/dota201201/1217_changeEye_bs/ori/Ani_eyesLookDown.obj");

	cstrVec all_obj = FILEIO::getFolderFiles(ori_folder, ".obj");
	for (int i = 0; i < all_obj.size(); i++)
	{
		MeshCompress obj_in(ori_folder + all_obj[i]);
		cstr raw_name = FILEIO::getFileNameWithoutExt(all_obj[i]);
		for (int j = 0; j < ori_base.pos_.size(); j++)
		{
			obj_in.pos_[j] = obj_in.pos_[j] + 0.25*(rotate_base.pos_[j] - ori_base.pos_[j]);
		}
		obj_in.setGOption(raw_name);
		obj_in.saveObj(res_folder + all_obj[i]);
	}
}

void TOPTRANSFER::packDeltaForExp(const json& config)
{
	cstr root = config["bs_delta"].get<cstr>();
	cstrVec delta_files
	{
		"head", root + "face_exp.json",
		"eyebrow",  root + "eyebrow_exp.json",
		"eyes", root + "eyes_exp.json",
		"tooth", root + "tooth_exp.json",
	};
	json res;
	cstrVec dump_key = { "eye_translate" };
	for (int i = 0; i < delta_files.size() / 2; i++)
	{
		cstr tag = delta_files[2 * i];
		cstr delta = delta_files[2 * i + 1];
		json delta_value = FILEIO::loadJson(delta);
		for (cstr iter_dump : dump_key)
		{
			auto pos = delta_value.find(iter_dump);
			if (pos != delta_value.end())
			{
				delta_value.erase(pos);
			}
		}
		res[tag] = delta_value;
	}
	FILEIO::saveJson(root + "zpack_local.json", res);

	json template_string = FILEIO::loadJson(config["json_root"].get<cstr>() + "template.json");
	json template_no_string = FILEIO::loadJson(config["json_root"].get<cstr>() + "template.json");
	template_string["ExpGenGuijie"]["Exp"] = res.dump();
	FILEIO::saveJson(root + "zpack_oss_string.json", template_string);
	template_no_string["ExpGenGuijie"]["Exp"] = res;
	FILEIO::saveJson(root + "zpack_oss_no_string.json", template_no_string);
	/*
	keep certain keys

	cstrVec keep_keys = { "eyes", "eyebrow" };
	json keep_json = JsonTools::keepKey(rep_data, keep_keys);
	template_base["ExpGenGuijie"]["Exp"] = keep_json.dump();
	*/	   
}

void TOPTRANSFER::getMappingFromMayaToUnity(const json& config)
{
	//cstr root = "D:/data_20July/0731_guijie/";
	cstr root = config["mapping_root"].get<cstr>();
	SG::needPath(root);
	
	cstrVec maya_fbx =
	{
		config["maya_head"].get<cstr>(), config["unity_head"].get<cstr>(),
		config["maya_eyes"].get<cstr>(), config["unity_eyes"].get<cstr>(),
		config["maya_eyebrow"].get<cstr>(), config["unity_eyebrow"].get<cstr>(),
		config["maya_tooth"].get<cstr>(), config["unity_tooth"].get<cstr>(),
	};

	cstrVec signal =
	{
		"head",
		"eyes",
		"eyebrow",
		"tooth",
	};

	float3E translate = float3E(1, 1, 1);
	int n_pair = maya_fbx.size() / 2;
	json match;

	for (int i = 0; i < signal.size(); i++)
	{
		json match_iter;
		MeshCompress maya(maya_fbx[2 * i]);
		MeshCompress unity(maya_fbx[2 * i + 1]);
		intVec res = MeshTools::getMatchBasedOnUVAndPos(unity, maya, 1e-2, 1e-2);

		match_iter["unity_vertex"] = unity.n_vertex_;
		match_iter["maya_vertex"] = maya.n_vertex_;
		match_iter["unity_to_maya"] = res;

		match[signal[i]] = match_iter;
		//RT::translateInPlace(translate, maya.pos_);
		//std::vector<float> in_bs_temp(maya.n_vertex_ * 3);
		//SG::safeMemcpy(in_bs_temp.data(), maya.pos_[0].data(), sizeof(float)*maya.n_vertex_ * 3);
		//match[signal[i]] = in_bs_temp;s
		//maya.saveObj(root + maya_fbx[2 * i] + "_transform.obj");
	}

	FILEIO::saveJson(root + "unity_to_maya.json", match);
}

void TOPTRANSFER::getMDS()
{
	cstr root_path = "D:/dota210104/0108_mds/";
	if (false)
	{
		MeshCompress get_mds_model(root_path + "meidusha.obj");
		intVec keep_vertex = FILEIO::loadIntDynamic(root_path + "keep_vertex.txt");
		intVec keep_mapping = get_mds_model.keepRoi(keep_vertex);
		FILEIO::saveDynamic(root_path + "keep_mapping.txt", keep_mapping, ",");
		get_mds_model.saveObj(root_path + "head.obj");
	}

	if (false)
	{
		MeshCompress mds_head(root_path + "head.obj");
		cstr sys_path = "D:/dota210104/0108_mds/sys/";
		SysFinder::findSysBasedOnPosOnly(mds_head, sys_path, 1e-5);
		//move head
		intVec mid_ind = FILEIO::loadIntDynamic("D:/dota210104/0108_mds/sys/mid.txt");
		float3E mid_pos;
		MeshTools::getCenter(mds_head.pos_, mid_ind, mid_pos);
		float3E move_mid_dir = -mid_pos;
		move_mid_dir.y() = 0;
		move_mid_dir.z() = 0;
		RT::translateInPlace(move_mid_dir, mds_head.pos_);
		mds_head.saveObj(root_path + "head_move.obj");
	}

	if (false)
	{
		MeshCompress mds_head(root_path + "head_move.obj");
		cstr sys_path = "D:/dota210104/0108_mds/sys/";
		SysFinder::findSysBasedOnPosOnly(mds_head, sys_path, 3*1e-4);
		TinyTool::getSysLandmarkPoint("D:/dota210104/0108_mds/sys/",
		"config.json", "D:/dota210104/0108_mds/land68/", "guijie_hand.txt",
		"sys_68.txt", "guijie_68_sys.txt");

		intVec delete_mouth = FILEIO::loadIntDynamic(root_path + "delete_mouth.txt");
		intVec mouth_mapping = mds_head.discard(delete_mouth);
		intVec all_68 = FILEIO::loadIntDynamic(root_path + "land68/guijie_68_sys.txt");
		intVec mapping_landmark = MAP::passIdxThroughMapping(all_68, { mouth_mapping });

		FILEIO::saveDynamic(root_path + "land68/skip_mouth_68.txt", mapping_landmark, ",");
		mds_head.saveObj(root_path + "mds_skip_mouth.obj");
		/*
		显示正常：
		skip_mouth_68.txt   ---> mds_skip_mouth.obj
		guijie_68_sys.txt   ---> head_move.obj
		*/
	}

	if (false)
	{
		root_path = "D:/dota210104/0111_mds/";
		MeshCompress guijie_head(root_path + "head_guijie.obj");
		intVec discard_vertex = FILEIO::loadIntDynamic(root_path + "guijie_discard_vertex_add_mouth.txt");
		intVec guijie_mapping = guijie_head.discard(discard_vertex);
		FILEIO::saveDynamic(root_path + "guijie_all_part_mapping.txt", guijie_mapping, ",");
		//guijie_df == reguijie
		guijie_head.saveObj(root_path + "guijie_df.obj");
		//put src to dst
		MeshCompress head_mds(root_path + "mds_skip_mouth.obj");
		MeshCompress head_mds_res = head_mds;		
		intVec mds_idx = FILEIO::loadIntDynamic(root_path + "skip_mouth_68.txt");
		intVec guijie_idx = FILEIO::loadIntDynamic(root_path + "re_guijie_68.txt");
		MeshTools::putSrcToDst(head_mds, mds_idx, guijie_head, guijie_idx, head_mds_res);
		head_mds_res.saveObj(root_path + "head_mds_res.obj");

		ofstream out_cons(root_path + "guijie_mds.cons");
		out_cons << "68" << std::endl;
		for (int i = 0; i < 68; i++)
		{
			out_cons<< guijie_idx[i] << "," << mds_idx[i] << std::endl;
		}
		out_cons.close();
	}

	if (true)
	{
		root_path = "D:/dota210104/0111_mds/";
		MeshCompress guijie_df(root_path + "guijie_df.obj");
		MeshCompress guijie_deform(root_path + "deform.obj");
		MeshCompress mds_to(root_path + "head_mds_res.obj");
		
		intVec map_df_guijie_mds = FILEIO::loadIntDynamic(root_path + "corre_26.000000.txt");
		intX2Vec map_mds_guijie(mds_to.n_vertex_);
		intX2Vec map_guijie_mds(guijie_deform.n_vertex_);

		intX2Vec mds_to_guijie(mds_to.n_vertex_);
		intX2Vec guijie_to_mds(guijie_deform.n_vertex_);
		intSet discard_multi_matching;

		intVec move_vec_idx = FILEIO::loadIntDynamic(root_path + "guijie_move_idx.txt");
		intSet move_set_idx(move_vec_idx.begin(), move_vec_idx.end());
		intVec lap_guijie_to_mds(guijie_df.n_vertex_, -1);
		for (int i = 0; i < map_mds_guijie.size() / 2; i++)
		{
			int guijie_from = map_df_guijie_mds[2 * i];
			int mds_to = map_df_guijie_mds[2 * i + 1];
			map_mds_guijie[mds_to].push_back(guijie_from);
			map_guijie_mds[guijie_from].push_back(mds_to);
		}

		for (int i = 0; i < mds_to.n_vertex_; i++)
		{
			int min_index = -1;
			float min_dis = 1e8;
			for (int iter = 0; iter < map_mds_guijie[i].size(); iter++)
			{
				int guijie_idx = map_mds_guijie[i][iter];
				float cur_dis = (guijie_deform.pos_[guijie_idx] - mds_to.pos_[i]).norm();
				min_index = min_dis > cur_dis ? guijie_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
			if (min_index < 0)
			{
				min_index = -1;
				min_dis = 1e8;
				for (int iter = 0; iter < guijie_deform.pos_.size(); iter++)
				{
					int guijie_idx = iter;
					float cur_dis = (mds_to.pos_[i] - guijie_deform.pos_[guijie_idx]).norm();
					min_index = min_dis > cur_dis ? guijie_idx : min_index;
					min_dis = DMIN(min_dis, cur_dis);
				}
			}

			if (min_index < 0)
			{
				LOG(ERROR) << "match missed" << std::endl;
			}
			else
			{

				if (mds_to_guijie[i].size() > 0 || guijie_to_mds[min_index].size() > 0)
				{
					LOG(INFO) << "discard here comes multiple matching" << std::endl;
					LOG(INFO) << "fwh: " << min_index << std::endl;
					LOG(INFO) << "guijie: " << i << std::endl;
					discard_multi_matching.insert(i);
				}
				else
				{
					mds_to_guijie[i].push_back(min_index);
					guijie_to_mds[min_index].push_back(i);
					lap_guijie_to_mds[min_index] = i;
				}
			}
		}
		LOG(INFO) << "discard size: " << discard_multi_matching.size() << std::endl;

		MeshCompress guijie_all_deform = guijie_df;
		MeshCompress guijie_all_move = guijie_df;
		intVec avg_lap_guijie_to_mds(guijie_df.n_vertex_, -1);
		for (int i = 0; i < lap_guijie_to_mds.size()/5; i++)
		{
			avg_lap_guijie_to_mds[5 * i] = lap_guijie_to_mds[5 * i];
		}
		SIMDEFORM::moveHandle(guijie_all_deform, mds_to, lap_guijie_to_mds, true, guijie_all_move);
		guijie_all_move.saveObj(root_path + "guijie_lap_res.obj");
	}
	//
}

void TOPTRANSFER::generateTexDst()
{
	cstr root = "D:/avatar/nl_linux/";
	json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	cstr root_res = config["output_dir_"].get<cstr>();

	intVec render_gv2_idx = FILEIO::loadIntDynamic(root + "1223_test_infer/gv2_render_idx.txt");
	intVec mapping = FILEIO::loadIntDynamic(root_res + "image_mapping.txt");
	floatVec fwh_normal = FILEIO::loadFloatDynamic(root_res + "fwh_normal_after_RT.txt");
	floatVec fwh_normal_raw = FILEIO::loadFloatDynamic(root_res + "fwh_normal_raw.txt");
	int n_vertex = mapping.size() / 3;
#if 0
	MeshCompress obj_base = root + "0118_template_color/head.obj";
	MeshCompress obj_uv = root + "0118_template_color/head.obj";
#else
	MeshCompress obj_base = root + "guijie_version/guijie_v4.obj";
	MeshCompress obj_uv = root + "guijie_version/guijie_v4.obj";
#endif
	obj_base.keepRoi(render_gv2_idx);
	obj_base.saveObj(root_res + "select_vertex.obj");

	cv::Mat in_image = cv::imread(root_res + "input_landmark68.png");

	MeshCompress obj_base_color = obj_base;
	MeshCompress obj_base_pos = obj_base;
	obj_base_color.vertex_color_ = obj_base_color.pos_;
	intSet render_gv2_set(render_gv2_idx.begin(), render_gv2_idx.end());

	floatVec light_coefs = FILEIO::loadFloatDynamic(root_res + "light_coefs.txt", ',');
	
	for (int i = 0; i < render_gv2_idx.size(); i++)
	{
		int idx = render_gv2_idx[i];
		int idx_map = i;
		while (idx != mapping[3 * idx_map])
		{
			idx_map++;
		}

		if (idx != mapping[3 * idx_map])
		{
			LOG(INFO) << "index not match" << std::endl;
		}
		int x = mapping[3 * idx_map + 1];
		int y = mapping[3 * idx_map + 2];
		//float3E input_normal(fwh_normal_raw[3 * idx_map + 0], fwh_normal_raw[3 * idx_map + 1], fwh_normal_raw[3 * idx_map + 2]);
		//input_normal.normalize();
		//float3E rgb_weight;
		//input_normal = float3E(0, 0, 1.0);
		//ShLighting::calcRGBCoeff(light_coefs, input_normal, rgb_weight);
		//LOG(INFO) << "rgb_weight: " << rgb_weight << std::endl;
		//LOG(INFO) << "input_normal: " << input_normal << std::endl;
		cv::Vec3b pix_color = GETU3(in_image, y, x);
		//float nx = std::abs(fwh_normal[3 * idx_map + 0]);
		//float ny = std::abs(fwh_normal[3 * idx_map + 1]);
		//float nz = std::abs(fwh_normal[3 * idx_map + 2]);
		//float nxyz = DMAX(DMAX(nx, ny), nz);
		//nz = DMAX(nz, 0.7);
		double r = DLIP3(safeDiv(double(pix_color[2]), 0.8, 0), 0, 255);
		double g = DLIP3(safeDiv(double(pix_color[1]), 0.8, 0), 0, 255);
		double b = DLIP3(safeDiv(double(pix_color[0]), 0.8, 0), 0, 255);
		obj_base_color.vertex_color_[i] = float3E(r, g, b);
		obj_base_pos.pos_[i] = float3E(b, g, r);
	}
	obj_base_pos.saveObj(root_res + "vis_pos_normal.obj");
	obj_base_color.saveVertexColor(root_res + "vis_color_normal.obj");
	obj_base_pos.saveObj(root_res + "dst_normal.obj");	

	fixTexture(root_res, 1.0);
}

void TOPTRANSFER::generateTextureTensor()
{
	cstr root = "D:/data/testPic/cartoonBase_renamed/";
	cstr root_res = "D:/dota210104/0118_obj_1e4/";
	SG::needPath(root_res);
	cstrVec image_files = FILEIO::getFolderFiles(root, { ".png" }, false);
	cv::Mat image_mean = cv::imread("D:/dota210104/0117_pca_mean/mean.png");
	//intVec fwh_idx = FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/fwh_move_idx_skip_eyes.txt");
	//intVec gv2_idx = FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/guijie_move_idx_skip_eyes.txt");
	intVec fwh_idx = FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/fwh_move_idx.txt");
	intVec gv2_idx = FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/guijie_move_idx_all.txt");
	//changeImageToObj

	MeshCompress obj_base = "D:/dota210104/0118_template_color/head.obj";
	MeshCompress obj_uv = "D:/dota210104/0118_template_color/head.obj";

	intSetVec src_uv_idx;
	src_uv_idx.resize(obj_base.n_vertex_);
	for (int iter_vertex = 0; iter_vertex < obj_base.n_tri_ * 3; iter_vertex++)
	{
		int vertex_idx = obj_base.tri_[iter_vertex];
		int uv_idx = obj_base.tri_uv_[iter_vertex];
		src_uv_idx[vertex_idx].insert(uv_idx);
	}

	intSet gujie_inner_mouth = TDST::vecToSet(FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/guijie_inner_mouth.txt"));
	//intVec mapping = FILEIO::loadIntDynamic("D:/dota210104/0118_color_fit/image_mapping.txt");
	//floatVec fwh_normal = FILEIO::loadFloatDynamic("D:/dota210104/0118_color_fit/fwh_normal.txt");
	intVec mapping = FILEIO::loadIntDynamic("D:/dota210104/0119_color_fit_00_expand_eyes/image_mapping.txt");
	floatVec fwh_normal = FILEIO::loadFloatDynamic("D:/dota210104/0119_color_fit_00_expand_eyes/fwh_normal_raw.txt");
	int n_vertex = mapping.size() / 3;

	//generate texture mapping, double check
	intVec render_gv2_idx;
	for (int i = 0; i < n_vertex; i++)
	{
		int idx = mapping[3 * i];
		float nx = std::abs(fwh_normal[3 * i + 0]);
		float ny = std::abs(fwh_normal[3 * i + 1]);
		float nz = std::abs(fwh_normal[3 * i + 2]);
		if (src_uv_idx[idx].size() > 1 && !gujie_inner_mouth.count(idx))
		{
			LOG(INFO) << "multiple mapping i " << i << std::endl;
			LOG(INFO) << "multiple mapping: " << idx << std::endl;
			for (int iter_uv : src_uv_idx[idx])
			{
				LOG(INFO) << "index " << iter_uv << ": " << obj_base.tex_cor_[iter_uv].transpose() << std::endl;
			}
			LOG(INFO) << "end multiple mapping: " << i << std::endl;
		}
		else if (src_uv_idx[idx].size() == 1 && !gujie_inner_mouth.count(idx))
		{
			render_gv2_idx.push_back(idx);
		}
	}

#if 1
	FILEIO::saveDynamic("D:/dota201201/1223_test_infer/gv2_render_idx.txt", render_gv2_idx, ",");
#endif	
	obj_base.keepRoi(render_gv2_idx);
	obj_base.saveObj(root_res + "select_vertex.obj");
	cv::Mat in_image = cv::imread("D:/dota210104/0118_color_fit/input_landmark68.png");
	MeshCompress obj_base_color = obj_base;
	MeshCompress obj_base_pos = obj_base;
	obj_base_color.vertex_color_ = obj_base_color.pos_;
	intSet render_gv2_set(render_gv2_idx.begin(), render_gv2_idx.end());
	for (int i = 0; i < render_gv2_idx.size(); i++)
	{
		int idx = render_gv2_idx[i];
		int idx_map = i;
		while (idx != mapping[3 * idx_map])
		{
			idx_map++;
		}

		if (idx != mapping[3 * idx_map])
		{
			LOG(INFO) << "index not match" << std::endl;
		}
		int x = mapping[3 * idx_map + 1];
		int y = mapping[3 * idx_map + 2];
		cv::Vec3b pix_color = GETU3(in_image, y, x);
		obj_base_color.vertex_color_[i] = float3E(pix_color[2], pix_color[1], pix_color[0]);
		obj_base_pos.pos_[i] = float3E(pix_color[0], pix_color[1], pix_color[2]);
	}
	obj_base_pos.saveObj(root_res + "vis_pos.obj");
	obj_base_color.saveVertexColor(root_res + "vis_color.obj");
	obj_base_pos.saveObj(root_res + "dst.obj");
	for (int i = 0; i < render_gv2_idx.size(); i++)
	{
		int idx = render_gv2_idx[i];
		int idx_map = i;
		while (idx != mapping[3 * idx_map])
		{
			idx_map++;
		}

		if (idx != mapping[3 * idx_map])
		{
			LOG(INFO) << "index not match" << std::endl;
		}
		int x = mapping[3 * idx_map + 1];
		int y = mapping[3 * idx_map + 2];
		cv::Vec3b pix_color = GETU3(in_image, y, x);
		float nx = std::abs(fwh_normal[3 * i + 0]);
		float ny = std::abs(fwh_normal[3 * i + 1]);
		float nz = std::abs(fwh_normal[3 * i + 2]);
		//float nxyz = DMAX(DMAX(nx, ny), nz);
		nz = DMAX(nz, 0.7);
		double r = DLIP3(safeDiv(double(pix_color[2]), 0.8, 0), 0, 255);
		double g = DLIP3(safeDiv(double(pix_color[1]), 0.8, 0), 0, 255);
		double b = DLIP3(safeDiv(double(pix_color[0]), 0.8, 0), 0, 255);
		obj_base_color.vertex_color_[i] = float3E(r, g, b);
		obj_base_pos.pos_[i] = float3E(b, g, r);
	}
	obj_base_pos.saveObj(root_res + "vis_pos_normal.obj");
	obj_base_color.saveVertexColor(root_res + "vis_color_normal.obj");
	obj_base_pos.saveObj(root_res + "dst_normal.obj");

	{
		//cv::Mat image_pca = cv::imread("D:/dota210104/0117_pca/pca_mean.png");
		cv::Mat image_pca = cv::imread("D:/data/testPic/cartoonBase_renamed/000010.png");
		for (int i = 0; i < render_gv2_idx.size(); i++)
		{
			int idx = render_gv2_idx[i];
			int idx_map = i;
			while (idx != mapping[3 * idx_map])
			{
				idx_map++;
			}

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			int x = mapping[3 * idx_map + 1];
			int y = mapping[3 * idx_map + 2];

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			for (auto idx_uv : src_uv_idx[idx])
			{
				double y = (1 - obj_uv.tex_cor_[idx_uv].y())*image_pca.cols;
				double x = obj_uv.tex_cor_[idx_uv].x()*image_pca.rows;
				cv::Vec3b pix_color = GETU3(image_pca, y, x);
				obj_base_color.vertex_color_[i] = float3E(pix_color[2], pix_color[1], pix_color[0]);
				obj_base_pos.pos_[i] = float3E(pix_color[0], pix_color[1], pix_color[2]);
			}
		}
		cstr raw_name = "mean";
		obj_base_pos.saveObj(root_res + "raw/" + raw_name + ".obj");
		obj_base_color.saveVertexColor(root_res + raw_name + ".obj");
	}


	//save for each diff pos
	for (int i = 0; i < image_files.size(); i++)
	{
		cv::Mat image_pca = cv::imread(root + image_files[i]);
		for (int i = 0; i < render_gv2_idx.size(); i++)
		{
			int idx = render_gv2_idx[i];
			int idx_map = i;
			while (idx != mapping[3 * idx_map])
			{
				idx_map++;
			}

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			int x = mapping[3 * idx_map + 1];
			int y = mapping[3 * idx_map + 2];

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			for (auto idx_uv : src_uv_idx[idx])
			{
				double y = (1 - obj_uv.tex_cor_[idx_uv].y())*image_pca.cols;
				double x = obj_uv.tex_cor_[idx_uv].x()*image_pca.rows;
				cv::Vec3b pix_color = GETU3(image_pca, y, x);
				obj_base_color.vertex_color_[i] = float3E(pix_color[2], pix_color[1], pix_color[0]);
				obj_base_pos.pos_[i] = float3E(pix_color[0], pix_color[1], pix_color[2]);
			}
		}
		cstr raw_name = FILEIO::getFileNameWithoutExt(image_files[i]);
		cstrVec split_file_name;
		FILEIO::splitString(raw_name, split_file_name, '.');
		raw_name = split_file_name[0];
		obj_base_pos.saveObj(root_res + "raw/" + raw_name + ".obj");
		obj_base_color.saveVertexColor(root_res + raw_name + ".obj");
	}

	TinyTool::getFileNamesToPCAJson("D:/dota210104/0118_obj_1e4/raw/");
	PREPARE::prepareBSTensor("D:/dota210104/0118_obj_1e4/raw/",
		"D:/dota210104/0118_obj_1e4/tensor/");
}

void TOPTRANSFER::generateTextureTensorBatch()
{
	cstr multi_pack = "D:/multiPack/";
	cstr root_res = multi_pack + "0118_obj_1e4/";
	SG::needPath(root_res);
	SG::needPath(root_res+"raw/");
	boost::filesystem::remove_all(root_res + "raw");
	SG::needPath(root_res + "raw/");
	cstrVec image_files = FILEIO::getFolderFiles(multi_pack + "cb_2101/", { ".png" }, false);
	//changeImageToObj

#if 0
	MeshCompress obj_base = multi_pack + "0118_template_color/head.obj";
	MeshCompress obj_uv = multi_pack + "0118_template_color/head.obj";
#else
	MeshCompress obj_base = multi_pack + "guijie_version/guijie_v4.obj";
	MeshCompress obj_uv = multi_pack + "guijie_version/guijie_v4.obj";
#endif

	intSetVec src_uv_idx;
	src_uv_idx.resize(obj_base.n_vertex_);
	for (int iter_vertex = 0; iter_vertex < obj_base.n_tri_ * 3; iter_vertex++)
	{
		int vertex_idx = obj_base.tri_[iter_vertex];
		int uv_idx = obj_base.tri_uv_[iter_vertex];
		src_uv_idx[vertex_idx].insert(uv_idx);
	}

	intVec mapping = FILEIO::loadIntDynamic(multi_pack + "0118_template_color/image_mapping.txt");
	int n_vertex = mapping.size() / 3;

	//generate texture mapping, double check
	intVec render_gv2_idx = FILEIO::loadIntDynamic(multi_pack + "1223_test_infer/gv2_render_idx.txt");
	obj_base.keepRoi(render_gv2_idx);
	obj_base.saveObj(root_res + "select_vertex.obj");
	cv::Mat in_image = cv::imread(multi_pack + "0118_template_color/input_landmark68.png");
	MeshCompress obj_base_color = obj_base;
	MeshCompress obj_base_pos = obj_base;
	obj_base_color.vertex_color_ = obj_base_color.pos_;
	intSet render_gv2_set(render_gv2_idx.begin(), render_gv2_idx.end());
	for (int i = 0; i < render_gv2_idx.size(); i++)
	{
		int idx = render_gv2_idx[i];
		int idx_map = i;
		while (idx != mapping[3 * idx_map])
		{
			idx_map++;
		}

		if (idx != mapping[3 * idx_map])
		{
			LOG(INFO) << "index not match" << std::endl;
		}
		int x = mapping[3 * idx_map + 1];
		int y = mapping[3 * idx_map + 2];
		cv::Vec3b pix_color = GETU3(in_image, y, x);
		obj_base_color.vertex_color_[i] = float3E(pix_color[2], pix_color[1], pix_color[0]);
		obj_base_pos.pos_[i] = float3E(pix_color[0], pix_color[1], pix_color[2]);
	}
	obj_base_pos.saveObj(root_res + "vis_pos.obj");
	obj_base_color.saveVertexColor(root_res + "vis_color.obj");
	obj_base_pos.saveObj(root_res + "dst.obj");
	for (int i = 0; i < render_gv2_idx.size(); i++)
	{
		int idx = render_gv2_idx[i];
		int idx_map = i;
		while (idx != mapping[3 * idx_map])
		{
			idx_map++;
		}

		if (idx != mapping[3 * idx_map])
		{
			LOG(INFO) << "index not match" << std::endl;
		}
		int x = mapping[3 * idx_map + 1];
		int y = mapping[3 * idx_map + 2];
		cv::Vec3b pix_color = GETU3(in_image, y, x);
		double r = DLIP3(safeDiv(double(pix_color[2]), 0.8, 0), 0, 255);
		double g = DLIP3(safeDiv(double(pix_color[1]), 0.8, 0), 0, 255);
		double b = DLIP3(safeDiv(double(pix_color[0]), 0.8, 0), 0, 255);
		obj_base_color.vertex_color_[i] = float3E(r, g, b);
		obj_base_pos.pos_[i] = float3E(b, g, r);
	}
	obj_base_pos.saveObj(root_res + "vis_pos_normal.obj");
	obj_base_color.saveVertexColor(root_res + "vis_color_normal.obj");
	obj_base_pos.saveObj(root_res + "dst_normal.obj");

	{
		//cv::Mat image_pca = cv::imread("D:/dota210104/0117_pca/pca_mean.png");
		cv::Mat image_pca = cv::imread(multi_pack + "cb_2101/000010.png");
		for (int i = 0; i < render_gv2_idx.size(); i++)
		{
			int idx = render_gv2_idx[i];
			int idx_map = i;
			while (idx != mapping[3 * idx_map])
			{
				idx_map++;
			}

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			int x = mapping[3 * idx_map + 1];
			int y = mapping[3 * idx_map + 2];

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			for (auto idx_uv : src_uv_idx[idx])
			{
				double y = (1 - obj_uv.tex_cor_[idx_uv].y())*image_pca.cols;
				double x = obj_uv.tex_cor_[idx_uv].x()*image_pca.rows;
				cv::Vec3b pix_color = GETU3(image_pca, y, x);
				obj_base_color.vertex_color_[i] = float3E(pix_color[2], pix_color[1], pix_color[0]);
				obj_base_pos.pos_[i] = float3E(pix_color[0], pix_color[1], pix_color[2]);
			}
		}
		cstr raw_name = "mean";
		obj_base_pos.saveObj(root_res + "raw/" + raw_name + ".obj");
		obj_base_color.saveVertexColor(root_res + raw_name + ".obj");
	}


	//save for each diff pos
	for (int i = 0; i < image_files.size(); i++)
	{
		cv::Mat image_pca = cv::imread(multi_pack + "cb_2101/" + image_files[i]);
		for (int i = 0; i < render_gv2_idx.size(); i++)
		{
			int idx = render_gv2_idx[i];
			int idx_map = i;
			while (idx != mapping[3 * idx_map])
			{
				idx_map++;
			}

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			int x = mapping[3 * idx_map + 1];
			int y = mapping[3 * idx_map + 2];

			if (idx != mapping[3 * idx_map])
			{
				LOG(INFO) << "index not match" << std::endl;
			}
			for (auto idx_uv : src_uv_idx[idx])
			{
				double y = (1 - obj_uv.tex_cor_[idx_uv].y())*image_pca.cols;
				double x = obj_uv.tex_cor_[idx_uv].x()*image_pca.rows;
				cv::Vec3b pix_color = GETU3(image_pca, y, x);
				obj_base_color.vertex_color_[i] = float3E(pix_color[2], pix_color[1], pix_color[0]);
				obj_base_pos.pos_[i] = float3E(pix_color[0], pix_color[1], pix_color[2]);
			}
		}
		cstr raw_name = FILEIO::getFileNameWithoutExt(image_files[i]);
		cstrVec split_file_name;
		FILEIO::splitString(raw_name, split_file_name, '.');
		raw_name = split_file_name[0];
		obj_base_pos.saveObj(root_res + "raw/" + raw_name + ".obj");
		obj_base_color.saveVertexColor(root_res + raw_name + ".obj");
	}

	TinyTool::getFileNamesToPCAJson(multi_pack + "0118_obj_1e4/raw/");
	PREPARE::prepareBSTensor(multi_pack + "0118_obj_1e4/raw/",
		multi_pack + "0118_obj_1e4/tensor/");
}

void TOPTRANSFER::fixTexture(cstr& root_res, double reg_value)
{
	//cstr data_root = "D:/multiPack/";
	cstr data_root = "D:/avatar/nl_linux/";
	Tensor tex_tensor;
	JsonHelper::initData(data_root + "0118_obj_1e4/tensor/", "config.json", tex_tensor);
	MeshCompress dst = root_res + "dst_normal.obj";
	floatVec reg(tex_tensor.n_id_, reg_value);
	vecD coef;
	tex_tensor.fitID(dst.pos_, reg, coef);
	LOG(INFO) << "coef: " << std::endl << coef << std::endl;
	//dst*
	//coef.setConstant(0);
	//cstr root = "D:/dota210104/0118_1e4/";
	cstr root = data_root + "cb_2101/";
	//cstr root_res = "D:/dota210104/0118_obj_1e4/";
	SG::needPath(root_res);
	cstrVec image_files = FILEIO::getFolderFiles(root, { ".png" }, false);
	//cv::Mat image_mean = cv::imread("D:/dota210104/0117_pca_mean/mean.png");
	//move origin mean 10 to 00
	cv::Mat image_mean = cv::imread(data_root + "cb_2101/000010.png");
	cv::Mat res_int8;
	cvMatD3 res(image_mean.cols, image_mean.rows);
#pragma omp parallel for
	for (int y = 0; y < image_mean.cols; y++)
	{
#pragma omp parallel for
		for (int x = 0; x < image_mean.rows; x++)
		{
			cv::Vec3d pixel_buff;
			pixel_buff[0] = float(GETU3(image_mean, y, x)(0));
			pixel_buff[1] = float(GETU3(image_mean, y, x)(1));
			pixel_buff[2] = float(GETU3(image_mean, y, x)(2));
			SETD3(res, y, x, pixel_buff);
		}
	}

	for (int i = 0; i < coef.size(); i++)
	{
		cv::Mat in_image = cv::imread(root + image_files[i]);
		//cv::resize(in_image, in_image, cv::Size(512, 512));
#pragma omp parallel for
		for (int y = 0; y < image_mean.cols; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < image_mean.rows; x++)
			{
				cv::Vec3d pixel_buff = GETD3(res, y, x);
				cv::Vec3d pixel_diff;
				pixel_diff[0] = float(GETU3(in_image, y, x)(0)) - float(GETU3(image_mean, y, x)(0));
				pixel_diff[1] = float(GETU3(in_image, y, x)(1)) - float(GETU3(image_mean, y, x)(1));
				pixel_diff[2] = float(GETU3(in_image, y, x)(2)) - float(GETU3(image_mean, y, x)(2));
				SETD3(res, y, x, pixel_buff + pixel_diff * coef[i]);
			}
		}
		//res.convertTo(res_int8, CV_8UC3);
		//cv::cvtColor(res, res_int8, CV_8UC3);
		//cv::imshow("res_int8_iter", res_int8);
		//cv::imwrite(root_res + std::to_string(i) + ".png", res_int8);
		//cv::waitKey(0);
	}
	res.convertTo(res_int8, CV_8UC3);
	//cv::resize(res_int8, res_int8, cv::Size(2048, 2048));
	cv::imwrite(root_res + "dst_fusion_" + std::to_string(reg_value) + "_normal.png", res_int8);
}

void TOPTRANSFER::correctEyelashPair()
{
	MeshCompress mesh_base = "D:/dota210121/0126_guijie_resource/maya/head.obj";
	cstr root_res = "D:/dota210121/0126_guijie_resource/correct_eyelash/";
	//eyelash pair
	intVec left_up_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	intVec left_down_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt");
	intVec right_up_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt");
	intVec right_down_match = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt");
	auto res = MAP::splitMapping(left_up_match);
	DebugTools::cgPrint(res[0]);
	DebugTools::cgPrint(res[1]);
	intVec head_roi = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_up_face_lash.txt");
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(mesh_base, res[0], res[1]) << std::endl;

	//change
	intVec match_res = MeshTools::getMatchBasedOnPosRoi(mesh_base, res[0], head_roi, 0.1, false);
	res = MAP::splitMapping(match_res);
	DebugTools::cgPrint(res[0]);
	DebugTools::cgPrint(res[1]);

	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(mesh_base, res[0], res[1]) << std::endl;

	//put src to dst
	SIMDEFORM::replaceHandle(mesh_base, res[0], mesh_base, res[1], mesh_base);
	mesh_base.saveObj("D:/dota210121/0127_change_eyelash/fix_eyelash.obj");

}

void TOPTRANSFER::fixTexture(double reg_value)
{
	Tensor tex_tensor;
	JsonHelper::initData("D:/dota210104/0118_obj_1e4/tensor/", "config.json", tex_tensor);
	MeshCompress dst = "D:/dota210104/0118_obj_1e4/dst_normal.obj";
	floatVec reg(tex_tensor.n_id_, reg_value);
	vecD coef;
	tex_tensor.fitID(dst.pos_, reg, coef);
	LOG(INFO) << "coef: " << std::endl << coef << std::endl;
	//dst*



	//coef.setConstant(0);
	//cstr root = "D:/dota210104/0118_1e4/";
	cstr root = "D:/data/testPic/cartoonBase_renamed/";
	cstr root_res = "D:/dota210104/0118_obj_1e4/";
	SG::needPath(root_res);
	cstrVec image_files = FILEIO::getFolderFiles(root, { ".png" }, false);	
	//cv::Mat image_mean = cv::imread("D:/dota210104/0117_pca_mean/mean.png");
	cv::Mat image_mean = cv::imread("D:/data/testPic/cartoonBase_renamed/000010.png");
	cv::Mat res_int8;
	cvMatD3 res(image_mean.cols, image_mean.rows);
#pragma omp parallel for
	for (int y = 0; y < image_mean.cols; y++)
	{
#pragma omp parallel for
		for (int x = 0; x < image_mean.rows; x++)
		{
			cv::Vec3d pixel_buff;
			pixel_buff[0] = float(GETU3(image_mean, y, x)(0));
			pixel_buff[1] = float(GETU3(image_mean, y, x)(1));
			pixel_buff[2] = float(GETU3(image_mean, y, x)(2));
			SETD3(res, y, x, pixel_buff);
		}
	}

	for (int i = 0; i < coef.size(); i++)
	{
		cv::Mat in_image = cv::imread(root + image_files[i]);
		//cv::resize(in_image, in_image, cv::Size(512, 512));
#pragma omp parallel for
		for (int y = 0; y < image_mean.cols; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < image_mean.rows; x++)
			{
				cv::Vec3d pixel_buff = GETD3(res, y, x);
				cv::Vec3d pixel_diff;
				pixel_diff[0] = float(GETU3(in_image, y, x)(0)) - float(GETU3(image_mean, y, x)(0));
				pixel_diff[1] = float(GETU3(in_image, y, x)(1)) - float(GETU3(image_mean, y, x)(1));
				pixel_diff[2] = float(GETU3(in_image, y, x)(2)) - float(GETU3(image_mean, y, x)(2));
				SETD3(res, y, x, pixel_buff + pixel_diff * coef[i]);
			}
		}
		//res.convertTo(res_int8, CV_8UC3);
		//cv::cvtColor(res, res_int8, CV_8UC3);
		//cv::imshow("res_int8_iter", res_int8);
		//cv::imwrite(root_res + std::to_string(i) + ".png", res_int8);
		//cv::waitKey(0);
	}
	res.convertTo(res_int8, CV_8UC3);
	//cv::resize(res_int8, res_int8, cv::Size(2048, 2048));
	cv::imwrite(root_res + "dst_fusion_"+std::to_string(reg_value) + "_normal.png", res_int8);
}

void TOPTRANSFER::fixTextureCeres()
{
	Tensor tex_tensor;
	JsonHelper::initData("D:/dota210104/0118_obj_1e4/tensor/", "config.json", tex_tensor);
	MeshCompress dst = "D:/dota210104/0118_obj_1e4/dst.obj";
	MeshCompress mean_obj = "D:/dota210104/0118_obj_1e4/raw/mean.obj";
	float3E translate_before;
	RT::getTranslate(mean_obj.pos_, dst.pos_, translate_before);
	LOG(INFO) << "translate: " << translate_before.transpose() << std::endl;

	floatVec reg(tex_tensor.n_id_, 0.1);
	vecD coef;
	tex_tensor.fitID(dst.pos_, reg, coef);
	LOG(INFO) << "coef: " << std::endl << coef << std::endl;

	doubleVec translate(3, 0), shape_coef(17, 0);
	shape_coef[0] = 1.0;
	{
		float weight = 0.1;
		ceres::Problem fitting_pca;
		doubleVec scale = { 1.0 };

		//int n_vertex = tex_tensor.template_obj_.n_vertex_;
		int n_vertex = 2;
		intVec roi_temp(n_vertex, 0);
		std::iota(roi_temp.begin(), roi_temp.end(), 0);
		ceres::CostFunction* cost_function =
			new ceres::NumericDiffCostFunction<PREPARE::PCAVertexCostRoi,
			ceres::CENTRAL,
			2 * 3, //vertex*3
			1, /* scale */
			3 /* traslate*/,
			17 /*pca value*/>
			(new PREPARE::PCAVertexCostRoi(tex_tensor.data_, dst.pos_, roi_temp, tex_tensor.n_id_, n_vertex));
		fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());

		shape_coef[0] = 1;

		fitting_pca.SetParameterUpperBound(&scale[0], 0, 1 + 1e+6); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&scale[0], 0, 1 - 1e-6); // t_z has to be negative

#if 0
		fitting_pca.SetParameterUpperBound(&translate[0], 0, translate_before[0] + 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&translate[0], 0, translate_before[0] - 5); // t_z has to be negative
		fitting_pca.SetParameterUpperBound(&translate[0], 1, translate_before[1] + 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&translate[0], 1, translate_before[1] - 5); // t_z has to be negative
		fitting_pca.SetParameterUpperBound(&translate[0], 2, translate_before[2] + 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&translate[0], 2, translate_before[2] - 5); // t_z has to be negative
#endif

#if 1
		fitting_pca.SetParameterUpperBound(&translate[0], 0, 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&translate[0], 0, - 5); // t_z has to be negative
		fitting_pca.SetParameterUpperBound(&translate[0], 1, 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&translate[0], 1,- 5); // t_z has to be negative
		fitting_pca.SetParameterUpperBound(&translate[0], 2, 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&translate[0], 2, - 5); // t_z has to be negative
#endif


		fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1 + 1e-6); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1 - 1e-6); // t_z has to be negative		

		for (int iter_coef = 1; iter_coef < 120; iter_coef++)
		{
			//fitting_pca.SetParameterUpperBound(&shape_coef[0], iter_coef, 1); // t_z has to be negative
			//fitting_pca.SetParameterLowerBound(&shape_coef[0], iter_coef, 0); // t_z has to be negative
		}
		ceres::Solver::Options solver_options;
		solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		//solver_options.num_threads = 8;
		solver_options.minimizer_progress_to_stdout = true;
		solver_options.max_num_iterations = 100;
		ceres::Solver::Summary solver_summary;
		Solve(solver_options, &fitting_pca, &solver_summary);
		std::cout << solver_summary.BriefReport() << "\n";
	}




	//coef.setConstant(0);
	//cstr root = "D:/dota210104/0118_1e4/";
	cstr root = "D:/data/testPic/cartoonBase_renamed/";
	cstr root_res = "D:/dota210104/0118_obj_1e4/";
	SG::needPath(root_res);
	cstrVec image_files = FILEIO::getFolderFiles(root, { ".png" }, false);
	//cv::Mat image_mean = cv::imread("D:/dota210104/0117_pca_mean/mean.png");
	cv::Mat image_mean = cv::imread("D:/data/testPic/cartoonBase_renamed/000010.png");
	cv::Mat res_int8;
	cvMatD3 res(image_mean.cols, image_mean.rows);
#pragma omp parallel for
	for (int y = 0; y < image_mean.cols; y++)
	{
#pragma omp parallel for
		for (int x = 0; x < image_mean.rows; x++)
		{
			cv::Vec3d pixel_buff;
			pixel_buff[0] = float(GETU3(image_mean, y, x)(0)) + translate[0];
			pixel_buff[1] = float(GETU3(image_mean, y, x)(1)) + translate[1];
			pixel_buff[2] = float(GETU3(image_mean, y, x)(2)) + translate[2];
			SETD3(res, y, x, pixel_buff);
		}
	}

	for (int i = 0; i < coef.size(); i++)
	{
		cv::Mat in_image = cv::imread(root + image_files[i]);
		//cv::resize(in_image, in_image, cv::Size(512, 512));
#pragma omp parallel for
		for (int y = 0; y < image_mean.cols; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < image_mean.rows; x++)
			{
				cv::Vec3d pixel_buff = GETD3(res, y, x);
				cv::Vec3d pixel_diff;
				pixel_diff[0] = float(GETU3(in_image, y, x)(0)) - float(GETU3(image_mean, y, x)(0));
				pixel_diff[1] = float(GETU3(in_image, y, x)(1)) - float(GETU3(image_mean, y, x)(1));
				pixel_diff[2] = float(GETU3(in_image, y, x)(2)) - float(GETU3(image_mean, y, x)(2));
				SETD3(res, y, x, pixel_buff + pixel_diff * shape_coef[i+1]);
			}
		}
		//res.convertTo(res_int8, CV_8UC3);
		//cv::cvtColor(res, res_int8, CV_8UC3);
		//cv::imshow("res_int8_iter", res_int8);
		//cv::imwrite(root_res + std::to_string(i) + ".png", res_int8);
		//cv::waitKey(0);
	}
	res.convertTo(res_int8, CV_8UC3);
	//cv::resize(res_int8, res_int8, cv::Size(2048, 2048));
	cv::imwrite(root_res + "dst_fusion_ceres.png", res_int8);
}

void TOPTRANSFER::changeBSScale()
{
	MeshCompress base = "D:/dota210104/0111_generate_head/maya/eyes.obj";
	cstr obj_in = "D:/dota210104/0111_generate_head/debug/eyebs_result/";
	cstr obj_out = "D:/dota210104/0111_generate_head/debug/eyebs_result_scale/";
	SG::needPath(obj_out);
	double scale = 0.75;
	cstrVec obj_names = FILEIO::getFolderFiles(obj_in, { ".obj" }, false);
	for (int i = 0; i < obj_names.size(); i++)
	{
		MeshCompress iter_mesh = obj_in + obj_names[i];
		MeshTools::scaleBSInPlace(base, scale, iter_mesh);
		iter_mesh.saveObj(obj_out + obj_names[i]);
	}
	TESTFUNCTION::triToQuadInPlace(obj_out);
}

void TOPTRANSFER::transferSimDiff(const cstr& path_A, const cstr& path_B, const cstr& root_A, const cstr& root_res)
{
	cstrVec file_items = FILEIO::getFolderFiles(root_A, FILEIO::FILE_TYPE::MESH);
	SG::needPath(root_res);
	MeshCompress A_raw = path_A;
	MeshCompress A = A_raw;
	MeshCompress B = path_B;
	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";
	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));
	
	//in case pos varies
	intVec guijie_68 = FILEIO::loadIntDynamic("D:/dota210202/0203_v4/neck_fix.txt");
	//put A to B
	double scale; 
	float3E translate, scale_center;
	MeshTools::putSrcToDst(A_raw, guijie_68, B, guijie_68, A, scale, scale_center, translate);
	//A.saveObj("D:/dota210202/0203_v4/A_resize.obj");
	exp_ptr->fixEyelash(A);
	for (int i = 0; i < file_items.size(); i++)
	{
		MeshCompress A_deform = root_A + file_items[i];
		RT::scaleAndTranslateInPlace(scale, scale_center, translate, A_deform.pos_);
		//A_deform.saveObj("D:/dota210202/0203_v4/A_deform_resize.obj");
		exp_ptr->fixEyelash(A_deform);
		MeshCompress B_res = B;
		exp_ptr->getExpGuided(B, A, A_deform, B_res);
		B_res.discardMaterial();
		B_res.saveObj(root_res + file_items[i], "");
	}	
}

void TOPTRANSFER::transferSimDiff(const std::shared_ptr<ExpGen> exp_ptr, const MeshCompress& A, const MeshCompress& A_deform,
	const MeshCompress& B_init, MeshCompress& B_deform)
{
	MeshCompress B = B_init;
	bool is_compress = true;
	MeshCompress B_nolash = B;
	exp_ptr->dumpEyelash(B_nolash);
	exp_ptr->fixEyelash(B);
	B_deform = B;
	exp_ptr->getExpGuided(B, A, A_deform, B_deform);
	//nlohmann::json vertex_value;
	//exp_ptr->getResultJson(vertex_value);
	//win need
	//B_res.saveObj(result_root + "B_deform.obj");
}

void TOPTRANSFER::tenetTest()
{
	//测试效果很一般，需要重新确定
	MeshCompress mesh_A = "D:/dota210317/0329_tenet/local_deform.obj";
	MeshCompress mesh_A_deform = "D:/dota210317/0329_tenet/isv.obj";
	MeshCompress mesh_B = "D:/dota210317/0329_tenet/guijie_v3.obj";
	MeshCompress mesh_B_deform = mesh_B;

	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";
	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));

	exp_ptr->fixEyelash(mesh_A);
	exp_ptr->fixEyelash(mesh_A_deform);
	exp_ptr->fixEyelash(mesh_B);
	exp_ptr->fixEyelash(mesh_B_deform);

	transferSimDiff(exp_ptr, mesh_A, mesh_A_deform, mesh_B, mesh_B_deform);
	mesh_B_deform.saveObj("D:/dota210317/0329_tenet/tenet.obj");
}