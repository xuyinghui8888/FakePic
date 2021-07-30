#include "EasyGuijie.h"
#include "../Basic/SelfBasic.h"

using namespace CGP;

void EASYGUIJIE::getGuijieEyebrow()
{	
	cstr result = "D:/dota201224/1224_guijie_v2/result/";
	SG::needPath(result);
	//获得eyebrow
	json mapping_part_all = FILEIO::loadJson("D:/dota201201/1214_generate_eyebrow_bs/from/part_to_maya.json");
	intVec part_all_mapping = mapping_part_all["part_to_all_eyebrow"].get<intVec>();
	MeshCompress base("D:/dota201201/1217_hf_quad/base/eyebrow.obj");
	MeshCompress guijie("D:/dota201224/1224_guijie_v2/mofeng/head.obj");
	float3E translate = float3E(0, 0, 0.25);

	for (int i = 0; i < base.n_vertex_; i++)
	{
		base.pos_[i] = guijie.pos_[part_all_mapping[i]] + translate;
	}
	
	base.saveObj(result + "adjust_eyebrow.obj");
}

void EASYGUIJIE::getGuijieEyebrow(const json& config)
{
	MeshCompress base(config["maya_eyebrow"].get<cstr>());
	MeshCompress guijie(config["maya_head"].get<cstr>());
	//获得eyebrow
	json mapping_part_all = FILEIO::loadJson("D:/multiPack/1214_generate_eyebrow_bs/from/part_to_maya.json");
	intVec part_all_mapping = mapping_part_all["part_to_all_eyebrow"].get<intVec>();

	float3E translate = float3E(0, 0, 0.25);

	for (int i = 0; i < base.n_vertex_; i++)
	{
		base.pos_[i] = guijie.pos_[part_all_mapping[i]] + translate;
	}

	base.saveObj(config["maya_eyebrow"].get<cstr>());
}

void EASYGUIJIE::getEyebrow(const cstr& root, const cstr& data_root)
{
	//cstr data_root = "D:/multiPack/";
	MeshCompress base = data_root + "0111_generate_head/maya/eyebrow.obj";
	MeshCompress guijie = root + "local_deform.obj";
	//获得eyebrow
	json mapping_part_all = FILEIO::loadJson(data_root + "1214_generate_eyebrow_bs/from/part_to_maya.json");
	intVec part_all_mapping = mapping_part_all["part_to_all_eyebrow"].get<intVec>();

	float3E translate = float3E(0, 0, 0.025);

	for (int i = 0; i < base.n_vertex_; i++)
	{
		base.pos_[i] = guijie.pos_[part_all_mapping[i]] + translate;
	}

	base.saveObj(root + "eyebrow.obj");
}

void EASYGUIJIE::getEyebrow(const cstr& root)
{
	cstr data_root = "D:/multiPack/";
	MeshCompress base = data_root + "0111_generate_head/maya/eyebrow.obj";
	MeshCompress guijie = root + "local_deform.obj";
	//获得eyebrow
	json mapping_part_all = FILEIO::loadJson(data_root + "1214_generate_eyebrow_bs/from/part_to_maya.json");
	intVec part_all_mapping = mapping_part_all["part_to_all_eyebrow"].get<intVec>();

	float3E translate = float3E(0, 0, 0.025);

	for (int i = 0; i < base.n_vertex_; i++)
	{
		base.pos_[i] = guijie.pos_[part_all_mapping[i]] + translate;
	}

	base.saveObj(root + "eyebrow.obj");
}

void EASYGUIJIE::replaceUV(const cstr& src, const cstr& dst)
{
	MeshCompress mesh_src, mesh_dst;
	mesh_src.loadOri(src);
	mesh_dst.loadOri(dst);

	if (src.npos != dst.npos)
	{
		LOG(ERROR) << "vertex size not fit." << std::endl;
	}

	mesh_dst.pos_ = mesh_src.pos_;
	mesh_dst.saveOri(src);
}

void EASYGUIJIE::transformEyesToMesh(const cstr& root, const cstr& obj)
{
	SG::needPath(root);
	//cstr root = "D:/dota201224/1224_eyes_head/";
	MeshCompress all("D:/multiPack/base_head.obj");
	MeshCompress part("D:/multiPack/base_eyes.obj");
	intVec part_to_all = MeshTools::getDoubleMatchBasedOnPos(part, all, 0.15, true);
	intX2Vec res = MAP::singleToDoubleMap(part_to_all);
	FILEIO::saveDynamic(root + "eye_idx.txt", res[0], ",");
	FILEIO::saveDynamic(root + "head_idx.txt", res[1], ",");
	intVec eye_left_idx, eye_right_idx, head_left_idx, head_right_idx;
	//split left/right
	
	for (int i = 0; i < res[0].size(); i++)
	{
		if (part.pos_[res[0][i]].x() < 0)
		{
			//right eye
			eye_right_idx.push_back(res[0][i]);
			head_right_idx.push_back(res[1][i]);
		}
		else
		{
			eye_left_idx.push_back(res[0][i]);
			head_left_idx.push_back(res[1][i]);
		}
	}
	
#if 0
	//test for data
	intVec all_idx(part.n_vertex_, 0);
	std::iota(all_idx.begin(), all_idx.end(), 0);
	intVec left_idx = RIGIDBS::divideIdx(src, all_idx, true);
	intVec left_idx_rec = CalcHelper::keepValueBiggerThan(left_idx, -0.5);
	intVec right_idx = divideIdx(src, all_idx, false);
	intVec right_idx_rec = CalcHelper::keepValueBiggerThan(right_idx, -0.5);	
#endif
	
	//keep ratio
	//MeshCompress target("D:/dota201201/1223_df_result/B_deform.obj");
	MeshCompress target(root + obj);
	MeshCompress eye_deform_right = part;
	MeshCompress eye_deform_left = part;
	double scale;
	float3E translate;
	MeshTools::putSrcToDst(part, eye_right_idx, target, head_right_idx, eye_deform_right, scale, translate);
	MeshTools::putSrcToDst(part, eye_left_idx, target, head_left_idx, eye_deform_left, scale, translate);
	MeshTools::putSrcToDstFixScale(part, eye_right_idx, target, head_right_idx, eye_deform_right, scale*1.0);
	MeshTools::putSrcToDstFixScale(part, eye_left_idx, target, head_left_idx, eye_deform_left, scale*1.0);
	//eye_deform.saveObj("D:/dota201224/1224_eyes_head/move_eyes.obj");
	eye_deform_right.saveObj(root + "move_eyes_right.obj");
	eye_deform_left.saveObj(root + "move_eyes_left.obj");
	
	//split eyes	
	intVec all_idx(part.n_vertex_, 0);
	std::iota(all_idx.begin(), all_idx.end(), 0);
	intVec left_idx = RIGIDBS::divideIdx(part, all_idx, true);
	intVec left_idx_rec = CalcHelper::keepValueBiggerThan(left_idx, -0.5);
	intVec right_idx = RIGIDBS::divideIdx(part, all_idx, false);
	intVec right_idx_rec = CalcHelper::keepValueBiggerThan(right_idx, -0.5);

	MeshTools::replaceVertexInPlace(eye_deform_left, eye_deform_right, right_idx_rec);
	eye_deform_left.saveObj(root + "move_eyes.obj");
}

void EASYGUIJIE::transformEyesToMesh(const cstr& root, const cstr& obj, const cstr& ref_head, 
	const cstr& ref_eyes, float eye_scale, float eyes_shift)
{
	//cstr root = "D:/dota201224/1224_eyes_head/";
	MeshCompress all(ref_head);
	MeshCompress part(ref_eyes);
	intVec part_to_all = MeshTools::getDoubleMatchBasedOnPos(part, all, 0.15, true);
	intX2Vec res = MAP::singleToDoubleMap(part_to_all);
	FILEIO::saveDynamic(root + "eye_idx.txt", res[0], ",");
	FILEIO::saveDynamic(root + "head_idx.txt", res[1], ",");
	intVec eye_left_idx, eye_right_idx, head_left_idx, head_right_idx;
	//split left/right

	for (int i = 0; i < res[0].size(); i++)
	{
		if (part.pos_[res[0][i]].x() < 0)
		{
			//right eye
			eye_right_idx.push_back(res[0][i]);
			head_right_idx.push_back(res[1][i]);
		}
		else
		{
			eye_left_idx.push_back(res[0][i]);
			head_left_idx.push_back(res[1][i]);
		}
	}


#if 0
	//test for data
	intVec all_idx(part.n_vertex_, 0);
	std::iota(all_idx.begin(), all_idx.end(), 0);
	intVec left_idx = RIGIDBS::divideIdx(src, all_idx, true);
	intVec left_idx_rec = CalcHelper::keepValueBiggerThan(left_idx, -0.5);
	intVec right_idx = divideIdx(src, all_idx, false);
	intVec right_idx_rec = CalcHelper::keepValueBiggerThan(right_idx, -0.5);
#endif

	//keep ratio
	//MeshCompress target("D:/dota201201/1223_df_result/B_deform.obj");
	MeshCompress target(root + obj);
	MeshCompress eye_deform_right = part;
	MeshCompress eye_deform_left = part;
	double scale;
	float3E translate;
	MeshTools::putSrcToDst(part, eye_right_idx, target, head_right_idx, eye_deform_right, scale, translate);
	MeshTools::putSrcToDst(part, eye_left_idx, target, head_left_idx, eye_deform_left, scale, translate);
	MeshTools::putSrcToDstFixScale(part, eye_right_idx, target, head_right_idx, eye_deform_right, scale*eye_scale);
	MeshTools::putSrcToDstFixScale(part, eye_left_idx, target, head_left_idx, eye_deform_left, scale*eye_scale);
	//eye_deform.saveObj("D:/dota201224/1224_eyes_head/move_eyes.obj");
	eye_deform_right.saveObj(root + "move_eyes_right.obj");
	eye_deform_left.saveObj(root + "move_eyes_left.obj");

	//split eyes	
	intVec all_idx(part.n_vertex_, 0);
	std::iota(all_idx.begin(), all_idx.end(), 0);
	intVec left_idx = RIGIDBS::divideIdx(part, all_idx, true);
	intVec left_idx_rec = CalcHelper::keepValueBiggerThan(left_idx, -0.5);
	intVec right_idx = RIGIDBS::divideIdx(part, all_idx, false);
	intVec right_idx_rec = CalcHelper::keepValueBiggerThan(right_idx, -0.5);

	MeshTools::replaceVertexInPlace(eye_deform_left, eye_deform_right, right_idx_rec);
	RT::translateInPlace(float3E(0, 0, eyes_shift), eye_deform_left.pos_);
	eye_deform_left.saveObj(root + "move_eyes.obj");
}


