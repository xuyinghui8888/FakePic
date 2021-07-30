#include "OptTypeV3.h"
#include "../Basic/MeshHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../Solver/BVLSSolver.h"
#include "../CalcFunction/CalcHelper.h"
#include "../MeshDeform/LapDeformWrapper.h"
using namespace CGP;

OptV3Gen::OptV3Gen(const json& config)
{
	if (is_init_ == false)
	{
		init(config);
	}
}

void OptV3Gen::init(const json& config)
{
	if (is_init_ == true)
	{
		LOG(INFO) << "already init for data." << std::endl;
		return;
	}
	LOG(INFO) << "init for OptV3Gen." << std::endl;
	data_root_ = config["root"].get<cstr>();
	default_A_.loadObj(data_root_ + config["default_A"].get<cstr>());

	FILEIO::loadIntDynamic(data_root_ + config["landmark_68_bfm"].get<cstr>(), landmark_68_bfm_);
	FILEIO::loadIntDynamic(data_root_ + config["landmark_68_guijie"].get<cstr>(), landmark_68_guijie_);

	eyelash_ = FILEIO::loadIntDynamic(data_root_ + config["left_down_lash"].get<cstr>());
	FILEIO::loadIntDynamic(data_root_ + config["left_up_lash"].get<cstr>(), eyelash_);
	FILEIO::loadIntDynamic(data_root_ + config["right_down_lash"].get<cstr>(), eyelash_);
	FILEIO::loadIntDynamic(data_root_ + config["right_up_lash"].get<cstr>(), eyelash_);

	JsonHelper::initData(data_root_ + config["shape_bs"].get<cstr>(), "config.json", shape_tensor_);
	FILEIO::loadFixSize(data_root_ + config["shape_bs"].get<cstr>() + "json_name.txt", ordered_map_);
	JsonHelper::initData(data_root_ + config["bfm_tensor"].get<cstr>(),  "config.json", bfm_tensor_);

	loadType(data_root_ + config["eye_type_"].get<cstr>(), eye_type_);
	loadType(data_root_ + config["face_type_"].get<cstr>(), face_type_);
	loadType(data_root_ + config["mouth_type_"].get<cstr>(), mouth_type_);
	loadType(data_root_ + config["nose_type_"].get<cstr>(), nose_type_);

	loadStarType(data_root_ + config["guijie_star"].get<cstr>(), star_set_);

	JsonHelper::initData(data_root_ + config["face_id_"].get<cstr>(), "config.json", star_set_.tensor_id_);

	bs_prefix_python_ = config["bs_prefix_python"].get<cstr>();

	//fixed based on 68 landmarks
	left_eye_region_ = { 42,43,44,45,46,47 };
	right_eye_region_ = { 36, 37,38,39,40,41 };
	nose_region_ = {27, 28,29,30,31,32,33,34,35};
	//mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59,60};
	//1019version
	//mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59 };
	mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59, 61,62,63,65,66,67 };
	face_region_ = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

	is_init_ = true;
	LOG(INFO) << "end of init OptV3Gen." << std::endl;
}

void OptV3Gen::loadType(const cstr& root, FaceTypeInfo& res)
{
	//ToDo: loading 3dmm coefs
	json config = FILEIO::loadJson(root + "config.json");
	res.n_type_ = config["n_type_"].get<int>();
	if (res.n_type_ <= 0)
	{
		LOG(WARNING) << "loading n_type_ :" << res.n_type_ << std::endl;
		return;
	}

	res.bs_.resize(res.n_type_);
	res.coef_3dmm_.resize(res.n_type_);
	res.mesh_.resize(res.n_type_);
	res.landmark_68_.resize(res.n_type_);

#pragma omp parallel for
	for (int i = 0; i < res.n_type_; i++)
	{
		FILEIO::loadEigenMat(root + std::to_string(i) + "_bs.txt", res.bs_[i]);
		FILEIO::loadEigenMat(root + std::to_string(i) + "_3dmm.txt", res.coef_3dmm_[i]);
		//res.mesh_[i].loadObj(root + std::to_string(i) + "_3dmm.obj");
		//faster loading
		getBFMMesh(res.coef_3dmm_[i], res.mesh_[i]);
		res.mesh_[i].getSlice(landmark_68_bfm_, res.landmark_68_[i]);
	}
}

void OptV3Gen::loadStarType(const cstr& root, FaceStarInfo& res)
{
	//ToDo: loading 3dmm coefs
	json config = FILEIO::loadJson(root + "config.json");
	res.n_type_ = config["n_type_"].get<int>();
	if (res.n_type_ <= 0)
	{
		LOG(WARNING) << "loading n_type_ :" << res.n_type_ << std::endl;
		return;
	}

	res.bs_.resize(res.n_type_);
	res.coef_3dmm_.resize(res.n_type_);
	res.faceid_.resize(res.n_type_);
	res.mesh_.resize(res.n_type_);
	res.landmark_68_.resize(res.n_type_);
	res.fix_json_.resize(res.n_type_);

#pragma omp parallel for
	for (int i = 0; i < res.n_type_; i++)
	{
		FILEIO::loadEigenMat(root + std::to_string(i) + "_bs.txt", res.bs_[i]);
		FILEIO::loadEigenMat(root + std::to_string(i) + "_3dmm.txt", 80, 1, res.coef_3dmm_[i]);
		FILEIO::loadEigenMat(root + std::to_string(i) + "_faceid.txt", 512, 1, res.faceid_[i]);
		res.fix_json_[i] = FILEIO::loadJson(root + std::to_string(i) + "_fix_bs.json");
		//res.mesh_[i].loadObj(root + std::to_string(i) + "_3dmm.obj");
		//faster loading
		getBFMMesh(res.coef_3dmm_[i], res.mesh_[i]);
		res.mesh_[i].getSlice(landmark_68_bfm_, res.landmark_68_[i]);
	}
}

void OptV3Gen::getBFMMesh(const vecD& coef_3dmm, MeshCompress& res)
{
	res = bfm_tensor_.template_obj_;
	if (coef_3dmm.size() < 80)
	{
		LOG(ERROR) << "coef_3dmm error， return default mesh." << std::endl;
		return;
	}
	floatVec raw_3dmm = bfm_tensor_.interpretID(coef_3dmm.head(80));
	SG::safeMemcpy(res.pos_.data(), raw_3dmm.data(), raw_3dmm.size() * sizeof(float));
}

void OptV3Gen::getGuijieMesh(vecD& coef_bs, MeshCompress& res)
{
	res = shape_tensor_.template_obj_;
	if (coef_bs.size() != 65)
	{
		LOG(ERROR) << "coef_bs error， return default mesh." << std::endl;
		return;
	}
	floatVec raw_bs = shape_tensor_.interpretID(coef_bs);
	SG::safeMemcpy(res.pos_.data(), raw_bs.data(), raw_bs.size() * sizeof(float));
}

void OptV3Gen::getRawBs(const FaceAttPack& in_att, vecD& res)
{
	vecD eye_bs, face_bs, mouth_bs, nose_bs;
	eye_type_.getBs(in_att.eye_type_, eye_bs);
	face_type_.getBs(in_att.face_type_, face_bs);
	mouth_type_.getBs(in_att.mouth_type_, mouth_bs);
	nose_type_.getBs(in_att.nose_type_, nose_bs);
	res = eye_bs + face_bs + mouth_bs + nose_bs;
	LOG(INFO) << "check res: " << OptTools::checkAndFixValue(0, 1, res) << std::endl;
}


void OptV3Gen::getGuijieFromType(json& json_in, MeshCompress& type_res, json& json_out, MeshCompress& opt_res)
{
	FaceAttPack in_att;
	json_out = json_in;
	in_att.eye_type_ = json_in["eye_type"].get<int>();
	in_att.face_type_ = json_in["face_type"].get<int>();
	in_att.mouth_type_ = json_in["mouth_type"].get<int>();
	in_att.nose_type_ = json_in["nose_type"].get<int>();
	floatVec coef_3dmm_vec = json_in["coef_3dmm"].get<floatVec>();
	floatVec faceid_vec = json_in["faceid"].get<floatVec>();
	CalcHelper::vectorToEigen(coef_3dmm_vec, in_att.coef_3dmm_);
	CalcHelper::vectorToEigen(faceid_vec, in_att.faceid_);

	star_set_.tensor_id_.getMatch(in_att.faceid_.cast<float>(), in_att.match_id_, in_att.match_dis_, true);
	vecD type_bs;
	star_set_.getBs(in_att.match_id_, type_bs);
	float blend_weight = getBlendWeight(in_att);
	in_att.blend_weight_ = blend_weight;
	getGuijieMesh(type_bs, type_res);
	
	//set default value
	opt_res = type_res;
	vecD opt_bs = type_bs;

	optBsBasedOn3dmm(in_att, type_bs, type_res, opt_bs, opt_res);
	getResultJson(opt_bs, json_out, 1e-4);
	addFixToJson(star_set_, in_att, json_out);
	att_cur_ = in_att;
}

void OptV3Gen::addFixToJson(const FaceStarInfo& init_data, const FaceAttPack& att_info, nlohmann::json& res)
{
	json fix_json = init_data.getFixJson(att_info.match_id_);
	for (json::iterator it = fix_json.begin(); it != fix_json.end(); ++it) 
	{
		res[it.key()] = it.value();
	}
	json debug_info;
	debug_info["match_id"] = att_info.match_id_;
	debug_info["Routine"] = "optV3";
	debug_info["eye_type_"] = att_info.eye_type_;
	debug_info["face_type_"] = att_info.face_type_;
	debug_info["mouth_type_"] = att_info.mouth_type_;
	debug_info["nose_type_"] = att_info.nose_type_;
	debug_info["blend_weight_"] = att_info.blend_weight_;
	res["DebugInfo"] = debug_info;
}

float OptV3Gen::getBlendWeight(const FaceAttPack& in_att)
{
	//safe check
	int n_size = in_att.match_dis_.size();
	if (in_att.match_id_ >= 0 && in_att.match_id_ < n_size)
	{
		float dis = in_att.match_dis_[in_att.match_id_];
		return DLIP3(dis, 0, 0.8);
	}
	else
	{
		LOG(ERROR) << "match res error." << std::endl;
		return 0;
	}
}

void OptV3Gen::move35Ratio(const float3Vec& template_bfm, const float3Vec& template_guijie, const float3Vec& att_bfm, float3Vec& att_guijie)
{
	//only move vertex pos

}

void  OptV3Gen::rotateEyes(const float3Vec& template_bfm, const intVec& roi_left, const intVec& roi_right, const float3Vec& template_guijie, const float3Vec& att_bfm,
	float3Vec& att_guijie)
{
	float3E bfm_att_36_39 = att_bfm[39] - att_bfm[36];
	float3E guijie_att_36_39 = att_guijie[39] - att_guijie[36];

	Eigen::Quaternionf out_guijie_bfm_right = Eigen::Quaternionf::FromTwoVectors(guijie_att_36_39, bfm_att_36_39);
	Eigen::Matrix3f Rx_guijie_bfm_right = out_guijie_bfm_right.toRotationMatrix();


	float3E bfm_att_45_42 = att_bfm[42] - att_bfm[45];
	float3E guijie_att_45_42 = att_guijie[42] - att_guijie[45];

	Eigen::Quaternionf out_guijie_bfm_left = Eigen::Quaternionf::FromTwoVectors(guijie_att_45_42, bfm_att_45_42);
	Eigen::Matrix3f Rx_guijie_bfm_left = out_guijie_bfm_left.toRotationMatrix();
	//xx

	//get center
	float3E left_center;
	float3E right_center;
	MeshTools::getCenter(att_guijie, roi_left, left_center);
	MeshTools::getCenter(att_guijie, roi_right, right_center);

	LOG(INFO) << "Rx_guijie_bfm_left: " << std::endl << Rx_guijie_bfm_left << std::endl;
	LOG(INFO) << "guijie_att_45_42: " << std::endl << guijie_att_45_42.transpose() << std::endl;
	LOG(INFO) << "bfm_att_45_42: " << std::endl << bfm_att_45_42.transpose() << std::endl;

	LOG(INFO) << "Rx_guijie_bfm_right: " << std::endl << Rx_guijie_bfm_right << std::endl;

	//rotate
	for (int i : roi_left)
	{
		float3E ori_dis = att_guijie[i] - left_center;
		float3E rotate_dis = Rx_guijie_bfm_left *ori_dis;
		att_guijie[i] = rotate_dis + left_center;
	}

	//right

	for (int i : roi_right)
	{
		float3E ori_dis = att_guijie[i] - right_center;
		float3E rotate_dis = Rx_guijie_bfm_right *ori_dis;
		att_guijie[i] = rotate_dis + right_center;
	}
	
	//test
	//float3E guijie_att_36_39_after = att_guijie[39] - att_guijie[36];
	//float3E guijie_att_45_42_after = att_guijie[42] - att_guijie[45];

}

void OptV3Gen::optBsBasedOn3dmm(const FaceAttPack& in_att, const vecD& type_bs, const MeshCompress& type_res,
	vecD& opt_bs, MeshCompress& opt_res)
{
	//get in_bfm_mesh
	MeshCompress in_att_bfm;
	getBFMMesh(in_att.coef_3dmm_, in_att_bfm);
	float3Vec att_bfm_68, template_guijie_68, att_guijie_68;
	in_att_bfm.getSlice(landmark_68_bfm_, att_bfm_68);
	type_res.getSlice(landmark_68_guijie_, template_guijie_68);
	att_guijie_68 = template_guijie_68;

	//get scale
	float3Vec template_bfm;
	star_set_.getBFM68(in_att.match_id_, template_bfm);

	getDriftLandmark(template_bfm, left_eye_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmark(template_bfm, right_eye_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmark(template_bfm, face_region_, template_guijie_68, att_bfm_68, 0.5, att_guijie_68, true);
	//getDriftLandmarkXYNoScale(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmarkXYMagScale(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true, {1.25,0.75,0.75});
	//getDriftLandmark(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmark(template_bfm, nose_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);

	//face simulation test
	blendTypeWithLandmark(face_type_, in_att.face_type_, face_region_, in_att.blend_weight_, template_guijie_68, att_guijie_68);

	//rotate eyes
	rotateEyes(template_bfm, left_eye_region_, right_eye_region_, template_guijie_68, att_bfm_68, att_guijie_68);
	
	//move ratio to 三庭五眼
	move35Ratio(template_bfm, template_guijie_68, att_bfm_68, att_guijie_68);


	//fit bs
	vecD weight(68);
	weight.setConstant(1.0);
	intVec skip_idx = {49, 53, 51, 61,62,63,64,65,66,67,60,64 };
	//for (auto i: skip_idx)
	//{
		//weight[i] = 0.05;
	//}
	//39 upper lip
	intVec fix_tensor = {9,25,30,33,42,51};
	getTensorCloseCoef(shape_tensor_, type_bs, fix_tensor, landmark_68_guijie_, att_guijie_68, weight, opt_bs);
	//LOG(INFO) << "project_bs: " <<std::endl<< project_bs << std::endl;
	//change default man_neck part

	getGuijieMesh(opt_bs, opt_res);
}

void OptV3Gen::getTensorCloseCoef(const Tensor& tensor, const vecD& type_bs, const intVec& fix_tensor, 
	const intVec& roi, const float3Vec& dst_roi, const vecD& weight, vecD& coef)
{
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3;	
	int n_var = tensor.n_id_ - 1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1);

	for (int i: fix_tensor)
	{
		lower[i] = DMAX(type_bs[i] - 1e-5, 0);
		upper[i] = DMIN(type_bs[i] + 1e-5, 1);
	}

	LOG(INFO) << upper.transpose() << std::endl;
	int shift = tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < roi.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = roi[i];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim]*weight(i);
				B(3 * i + iter_dim) = (dst_roi[i][iter_dim] - tensor.data_[vertex_id * 3 + iter_dim])*weight(i);
			}
		}
	}
	if (debug_)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;
}


void OptV3Gen::getDriftLandmark(const float3Vec& template_bfm, const intVec& roi, 
	const float3Vec& template_guijie, const float3Vec& att_bfm, float reduce, float3Vec& att_guijie, bool is_shift_center)
{

	float3Vec template_bfm_roi, template_guijie_roi, att_bfm_roi, att_guijie_roi;
	MeshTools::getSlice(template_bfm, roi, template_bfm_roi);
	MeshTools::getSlice(template_guijie, roi, template_guijie_roi);
	MeshTools::getSlice(att_bfm, roi, att_bfm_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//get scale
	doubleVec template_bfm_xyz_min, template_bfm_xyz_max, template_guijie_xyz_min, template_guijie_xyz_max, scale;
	MeshTools::getBoundingBox(template_bfm_roi, template_bfm_xyz_min, template_bfm_xyz_max);
	MeshTools::getBoundingBox(template_guijie_roi, template_guijie_xyz_min, template_guijie_xyz_max);
	float3E template_bfm_center, att_bfm_center;
	MeshTools::getCenter(template_bfm_roi, template_bfm_center);
	MeshTools::getCenter(att_bfm_roi, att_bfm_center);

	LOG(INFO) << "center template: " << template_bfm_center.transpose() << std::endl;
	LOG(INFO) << "center att: " << att_bfm_center.transpose() << std::endl;

	scale.resize(3);
	//scale B/A
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		scale[iter_dim] = safeDiv(template_guijie_xyz_max[iter_dim] - template_guijie_xyz_min[iter_dim], template_bfm_xyz_max[iter_dim] - template_bfm_xyz_min[iter_dim], 0);
	}
	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];
		float3E aim_dis = att_bfm[landmark_idx] - template_bfm[landmark_idx];
		if (is_shift_center)
		{
			aim_dis = (att_bfm[landmark_idx] - att_bfm_center) - (template_bfm[landmark_idx] - template_bfm_center);
		}
		float3E opt_dis = aim_dis;
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			opt_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim] * reduce;
		}
		att_guijie[landmark_idx] = template_guijie[landmark_idx] + opt_dis;
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}


void OptV3Gen::getDriftLandmark(const FaceTypeInfo& part, int type, const intVec& roi,
	const float3Vec& template_guijie, const float3Vec& att_bfm, float reduce, float3Vec& att_guijie, bool is_shift_center)
{
	//get scale
	float3Vec template_bfm;
	part.getBFM68(type, template_bfm);

	float3Vec template_bfm_roi, template_guijie_roi, att_bfm_roi, att_guijie_roi;
	MeshTools::getSlice(template_bfm, roi, template_bfm_roi);
	MeshTools::getSlice(template_guijie, roi, template_guijie_roi);
	MeshTools::getSlice(att_bfm, roi, att_bfm_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//get scale
	doubleVec template_bfm_xyz_min, template_bfm_xyz_max, template_guijie_xyz_min, template_guijie_xyz_max, scale;
	MeshTools::getBoundingBox(template_bfm_roi, template_bfm_xyz_min, template_bfm_xyz_max);
	MeshTools::getBoundingBox(template_guijie_roi, template_guijie_xyz_min, template_guijie_xyz_max);
	float3E template_bfm_center, att_bfm_center;
	MeshTools::getCenter(template_bfm_roi, template_bfm_center);
	MeshTools::getCenter(att_bfm_roi, att_bfm_center);

	LOG(INFO) << "center template: " << template_bfm_center.transpose() << std::endl;
	LOG(INFO) << "center att: " << att_bfm_center.transpose() << std::endl;

	scale.resize(3);
	//scale B/A
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		scale[iter_dim] = safeDiv(template_guijie_xyz_max[iter_dim] - template_guijie_xyz_min[iter_dim], template_bfm_xyz_max[iter_dim] - template_bfm_xyz_min[iter_dim], 0);
	}
	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];
		float3E aim_dis = att_bfm[landmark_idx] - template_bfm[landmark_idx];
		if (is_shift_center)
		{
			aim_dis = (att_bfm[landmark_idx] - att_bfm_center) - (template_bfm[landmark_idx] - template_bfm_center);
		}
		float3E opt_dis = aim_dis;
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			opt_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim] * reduce;
		}
		att_guijie[landmark_idx] = template_guijie[landmark_idx] + opt_dis;
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}

void OptV3Gen::blendTypeWithLandmark(const FaceTypeInfo& part, int type, const intVec& roi, float reduce, const float3Vec& template_guijie, float3Vec& att_guijie)
{
	vecD part_bs;
	float3Vec part_guijie_roi, att_guijie_roi, part_guijie_68;
	part.getBs(type, part_bs);
	MeshCompress guijie_type;
	getGuijieMesh(part_bs, guijie_type);
	guijie_type.getSlice(landmark_68_guijie_, part_guijie_68);

	MeshTools::getSlice(part_guijie_68, roi, part_guijie_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];		
		att_guijie[landmark_idx] = part_guijie_roi[landmark_idx]*reduce + att_guijie[landmark_idx]*(1-reduce);
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}

void OptV3Gen::getDriftLandmarkXY(const FaceTypeInfo& part, int type, const intVec& roi,
	const float3Vec& template_guijie, const float3Vec& att_bfm, float reduce, float3Vec& att_guijie, bool is_shift_center)
{
	//get scale
	float3Vec template_bfm;
	part.getBFM68(type, template_bfm);

	float3Vec template_bfm_roi, template_guijie_roi, att_bfm_roi, att_guijie_roi;
	MeshTools::getSlice(template_bfm, roi, template_bfm_roi);
	MeshTools::getSlice(template_guijie, roi, template_guijie_roi);
	MeshTools::getSlice(att_bfm, roi, att_bfm_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//get scale
	doubleVec template_bfm_xyz_min, template_bfm_xyz_max, template_guijie_xyz_min, template_guijie_xyz_max, scale;
	MeshTools::getBoundingBox(template_bfm_roi, template_bfm_xyz_min, template_bfm_xyz_max);
	MeshTools::getBoundingBox(template_guijie_roi, template_guijie_xyz_min, template_guijie_xyz_max);
	float3E template_bfm_center, att_bfm_center;
	MeshTools::getCenter(template_bfm_roi, template_bfm_center);
	MeshTools::getCenter(att_bfm_roi, att_bfm_center);

	LOG(INFO) << "center template: " << template_bfm_center.transpose() << std::endl;
	LOG(INFO) << "center att: " << att_bfm_center.transpose() << std::endl;

	scale.resize(3);
	//scale B/A
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		scale[iter_dim] = safeDiv(template_guijie_xyz_max[iter_dim] - template_guijie_xyz_min[iter_dim], template_bfm_xyz_max[iter_dim] - template_bfm_xyz_min[iter_dim], 0);
	}
	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];
		float3E aim_dis = att_bfm[landmark_idx] - template_bfm[landmark_idx];
		if (is_shift_center)
		{
			aim_dis = (att_bfm[landmark_idx] - att_bfm_center) - (template_bfm[landmark_idx] - template_bfm_center);
		}
		float3E opt_dis = aim_dis;
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			opt_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim] * reduce;
		}
		opt_dis.z() = 0;
		att_guijie[landmark_idx] = template_guijie[landmark_idx] + opt_dis;
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}

void OptV3Gen::getDriftLandmarkXYNoScale(const FaceTypeInfo& part, int type, const intVec& roi,
	const float3Vec& template_guijie, const float3Vec& att_bfm, float reduce, float3Vec& att_guijie, bool is_shift_center)
{
	//get scale
	float3Vec template_bfm;
	part.getBFM68(type, template_bfm);

	float3Vec template_bfm_roi, template_guijie_roi, att_bfm_roi, att_guijie_roi;
	MeshTools::getSlice(template_bfm, roi, template_bfm_roi);
	MeshTools::getSlice(template_guijie, roi, template_guijie_roi);
	MeshTools::getSlice(att_bfm, roi, att_bfm_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//get scale
	doubleVec template_bfm_xyz_min, template_bfm_xyz_max, template_guijie_xyz_min, template_guijie_xyz_max, scale;
	MeshTools::getBoundingBox(template_bfm_roi, template_bfm_xyz_min, template_bfm_xyz_max);
	MeshTools::getBoundingBox(template_guijie_roi, template_guijie_xyz_min, template_guijie_xyz_max);
	float3E template_bfm_center, att_bfm_center;
	MeshTools::getCenter(template_bfm_roi, template_bfm_center);
	MeshTools::getCenter(att_bfm_roi, att_bfm_center);

	LOG(INFO) << "center template: " << template_bfm_center.transpose() << std::endl;
	LOG(INFO) << "center att: " << att_bfm_center.transpose() << std::endl;

	scale.resize(3);
	//scale B/A
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		scale[iter_dim] = safeDiv(template_guijie_xyz_max[iter_dim] - template_guijie_xyz_min[iter_dim], template_bfm_xyz_max[iter_dim] - template_bfm_xyz_min[iter_dim], 0);
	}
	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];
		float3E aim_dis = att_bfm[landmark_idx] - template_bfm[landmark_idx];
		if (is_shift_center)
		{
			aim_dis = (att_bfm[landmark_idx] - att_bfm_center) - (template_bfm[landmark_idx] - template_bfm_center);
		}
		float3E opt_dis = aim_dis;
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			opt_dis[iter_dim] = aim_dis[iter_dim] * reduce;
		}
		opt_dis.z() = 0;
		att_guijie[landmark_idx] = template_guijie[landmark_idx] + opt_dis;
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}

void OptV3Gen::getDriftLandmarkXYNoScale(const float3Vec& template_bfm, const intVec& roi, const float3Vec& template_guijie, const float3Vec& att_bfm, float reduce, float3Vec& att_guijie, bool is_shift_center)
{
	float3Vec template_bfm_roi, template_guijie_roi, att_bfm_roi, att_guijie_roi;
	MeshTools::getSlice(template_bfm, roi, template_bfm_roi);
	MeshTools::getSlice(template_guijie, roi, template_guijie_roi);
	MeshTools::getSlice(att_bfm, roi, att_bfm_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//get scale
	doubleVec template_bfm_xyz_min, template_bfm_xyz_max, template_guijie_xyz_min, template_guijie_xyz_max, scale;
	MeshTools::getBoundingBox(template_bfm_roi, template_bfm_xyz_min, template_bfm_xyz_max);
	MeshTools::getBoundingBox(template_guijie_roi, template_guijie_xyz_min, template_guijie_xyz_max);
	float3E template_bfm_center, att_bfm_center;
	MeshTools::getCenter(template_bfm_roi, template_bfm_center);
	MeshTools::getCenter(att_bfm_roi, att_bfm_center);

	LOG(INFO) << "center template: " << template_bfm_center.transpose() << std::endl;
	LOG(INFO) << "center att: " << att_bfm_center.transpose() << std::endl;

	scale.resize(3);
	//scale B/A
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		scale[iter_dim] = safeDiv(template_guijie_xyz_max[iter_dim] - template_guijie_xyz_min[iter_dim], template_bfm_xyz_max[iter_dim] - template_bfm_xyz_min[iter_dim], 0);
	}
	//scale y = 1
	scale[1] = 1.0;
	scale[2] = 1.0;

	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];
		float3E aim_dis = att_bfm[landmark_idx] - template_bfm[landmark_idx];
		if (is_shift_center)
		{
			aim_dis = (att_bfm[landmark_idx] - att_bfm_center) - (template_bfm[landmark_idx] - template_bfm_center);
		}
		float3E opt_dis = aim_dis;
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			opt_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim]* reduce;
		}
		opt_dis.z() = 0;
		att_guijie[landmark_idx] = template_guijie[landmark_idx] + opt_dis;
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}

void OptV3Gen::getDriftLandmarkXYMagScale(const float3Vec& template_bfm, const intVec& roi, const float3Vec& template_guijie, 
	const float3Vec& att_bfm, float reduce, float3Vec& att_guijie, bool is_shift_center, const floatVec& shrink)
{
	float3Vec template_bfm_roi, template_guijie_roi, att_bfm_roi, att_guijie_roi;
	MeshTools::getSlice(template_bfm, roi, template_bfm_roi);
	MeshTools::getSlice(template_guijie, roi, template_guijie_roi);
	MeshTools::getSlice(att_bfm, roi, att_bfm_roi);
	MeshTools::getSlice(att_guijie, roi, att_guijie_roi);

	//get scale
	doubleVec template_bfm_xyz_min, template_bfm_xyz_max, template_guijie_xyz_min, template_guijie_xyz_max, scale;
	MeshTools::getBoundingBox(template_bfm_roi, template_bfm_xyz_min, template_bfm_xyz_max);
	MeshTools::getBoundingBox(template_guijie_roi, template_guijie_xyz_min, template_guijie_xyz_max);
	float3E template_bfm_center, att_bfm_center;
	MeshTools::getCenter(template_bfm_roi, template_bfm_center);
	MeshTools::getCenter(att_bfm_roi, att_bfm_center);

	LOG(INFO) << "center template: " << template_bfm_center.transpose() << std::endl;
	LOG(INFO) << "center att: " << att_bfm_center.transpose() << std::endl;

	scale.resize(3);
	//scale B/A
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		scale[iter_dim] = safeDiv(template_guijie_xyz_max[iter_dim] - template_guijie_xyz_min[iter_dim], template_bfm_xyz_max[iter_dim] - template_bfm_xyz_min[iter_dim], 0);
	}

	if (shrink.size() < 3)
	{
		LOG(ERROR) << "shrink size not fit." << std::endl;
	}
	else
	{
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			scale[iter_dim] *= shrink[iter_dim];
		}
	}

	//att_guijie = template_guijie;
	for (int i = 0; i < roi.size(); i++)
	{
		int landmark_idx = roi[i];
		float3E aim_dis = att_bfm[landmark_idx] - template_bfm[landmark_idx];
		if (is_shift_center)
		{
			aim_dis = (att_bfm[landmark_idx] - att_bfm_center) - (template_bfm[landmark_idx] - template_bfm_center);
		}
		float3E opt_dis = aim_dis;
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			opt_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim] * reduce;
		}
		opt_dis.z() = 0;
		att_guijie[landmark_idx] = template_guijie[landmark_idx] + opt_dis;
		//LOG(INFO) << "index: " << landmark_idx << std::endl;
		//LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
		//LOG(INFO) << "opt_dis: " << opt_dis.transpose() << std::endl;
		//LOG(INFO) << "template_guijie[landmark_idx]: " << template_guijie[landmark_idx].transpose() << std::endl;
		//LOG(INFO) << "att_guijie[landmark_idx]: " << att_guijie[landmark_idx].transpose() << std::endl;
	}

}


void OptV3Gen::setDebug(bool is_debug)
{
	debug_ = is_debug;
}

void OptV3Gen::getTensorCoef(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor, const intVec& up_down_match,
	const MeshCompress& dst, vecD& coef, const cstr& output_path)
{
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3 + up_down_match.size()*0.5*3 + 1 ;
	double weight_sum_1 = 1e5;
	double weight_pair = 1e-1;
	double sum_extra = 1.02;
	int n_var = tensor.n_id_ -1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1);
	int shift = tensor.template_obj_.n_vertex_*3;
	int start_row = 0;
	for (int i = 0; i < roi.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = roi[i];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim];
				B(3 * i + iter_dim) = dst.pos_[vertex_id][iter_dim] - tensor.data_[vertex_id * 3 + iter_dim];
			}
		}
	}
	//LOG(INFO) << "A:" << std::endl << A << std::endl;
	//LOG(INFO) << "B:" << std::endl << B << std::endl;
	start_row = 3 * roi.size();
	//for match
	for (int i = 0; i < up_down_match.size()*0.5; i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_src = up_down_match[2 * i + 0];
			int vertex_dst = up_down_match[2 * i + 1];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				double A_coef = close_tensor.data_[shift + shift * j + vertex_src * 3 + iter_dim] - close_tensor.data_[shift + shift * j + vertex_dst * 3 + iter_dim];
				double B_coef = -(dst.pos_[vertex_src][iter_dim] - dst.pos_[vertex_dst][iter_dim]);
				A(start_row + 3 * i + iter_dim, j) = weight_pair * A_coef;
				B(start_row + 3 * i + iter_dim) = weight_pair * B_coef;
			}
		}
	}
	//LOG(INFO) << "A:" << std::endl << A << std::endl;
	//LOG(INFO) << "A:" << std::endl << A.row(489) << std::endl;
	//LOG(INFO) << "B:" << std::endl << B << std::endl;
	start_row = roi.size() * 3 + up_down_match.size()*0.5 * 3;
	//last
	for (int j = 0; j < n_var; j++)
	{
		A(start_row, j) = weight_sum_1 *1.0;
		B(start_row) = weight_sum_1 * sum_extra;
	}
	if (debug_)
	{
		//FILEIO::saveEigenDynamic(output_path + "A.txt", A);
		//FILEIO::saveEigenDynamic(output_path + "B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;
}

void OptV3Gen::dumpEyelash(MeshCompress& src)
{
	src.discard(eyelash_);
}

FaceStarInfo OptV3Gen::getStarInfo()
{
	return star_set_;
}

void OptV3Gen::optBasedOn3dmm(const MeshCompress& in_bfm, const MeshCompress& base_bfm, const MeshCompress& in_guijie,
	const MeshCompress& base_guijie, MeshCompress& opt_res)
{
	float3Vec att_bfm_68, template_guijie_68, att_guijie_68;
	in_bfm.getSlice(landmark_68_bfm_, att_bfm_68);
	base_guijie.getSlice(landmark_68_guijie_, template_guijie_68);
	att_guijie_68 = template_guijie_68;

	//get scale
	float3Vec template_bfm;
	base_bfm.getSlice(landmark_68_bfm_, template_bfm);

	getDriftLandmark(template_bfm, left_eye_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false);
	getDriftLandmark(template_bfm, right_eye_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false);
	getDriftLandmark(template_bfm, face_region_, template_guijie_68, att_bfm_68, 0.5, att_guijie_68, false);
	//getDriftLandmarkXYNoScale(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmarkXYMagScale(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false, { 1.25,0.75,0.75 });
	//getDriftLandmark(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmark(template_bfm, nose_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false);

	//rotate eyes
	rotateEyes(template_bfm, left_eye_region_, right_eye_region_, template_guijie_68, att_bfm_68, att_guijie_68);

	//dst att_guijie_68 pos

	//cstr root = "D:/code/auto3dData/";
	cstr root = "D:/avatar/nl_linux/";
	//intVec fix = FILEIO::loadIntDynamic(root + "D:/code/auto3dData/guijie/face_fix_reduce.txt");
	intVec fix = FILEIO::loadIntDynamic(root + "guijie/face_fix_reduce.txt");
	opt_res = in_guijie;
	intVec mouth_close = FILEIO::loadIntDynamic(root + "guijie/mouth_close.txt");
#if 0
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);
#else
	//intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	//FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	//FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	//FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
	intVec guijie_eyelash = FILEIO::loadIntDynamic(root + "guijie/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic(root + "guijie/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic(root + "guijie/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic(root + "guijie/pair_info/right_down_match.txt", guijie_eyelash);
#endif

	//此处是缺省过程
	//SIMDEFORM::moveHandle(in_guijie, landmark_68_guijie_, att_guijie_68, fix, guijie_eyelash, opt_res);
	
	//opt_res.saveObj("D:/dota210104/0120_color_fit_00_expand_eyes/local_deform.obj");

	intVec fix_expand = fix;
	//MeshCompress mean = "D:/dota210104/0120_color_fit_00_expand_eyes/B_deform.obj";
	FILEIO::loadIntDynamic(root + "guijie/mouth_structure_line.txt", fix_expand);
	SIMDEFORM::moveHandle(opt_res, mouth_close, in_guijie, mouth_close, fix_expand, guijie_eyelash, opt_res);
	//opt_res.saveObj("D:/dota210104/0120_color_fit_00_expand_eyes/close_local_deform.obj");
}

void OptV3Gen::optBasedOn3dmm(const MeshCompress& in_bfm, const MeshCompress& base_bfm, const MeshCompress& in_guijie,
	const MeshCompress& base_guijie, const json& config, MeshCompress& opt_res)
{
	float3Vec att_bfm_68, template_guijie_68, att_guijie_68;
	in_bfm.getSlice(landmark_68_bfm_, att_bfm_68);
	base_guijie.getSlice(landmark_68_guijie_, template_guijie_68);
	att_guijie_68 = template_guijie_68;

	//get scale
	float3Vec template_bfm;
	base_bfm.getSlice(landmark_68_bfm_, template_bfm);

	getDriftLandmark(template_bfm, left_eye_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false);
	getDriftLandmark(template_bfm, right_eye_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false);
	getDriftLandmark(template_bfm, face_region_, template_guijie_68, att_bfm_68, 0.5, att_guijie_68, false);
	//getDriftLandmarkXYNoScale(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmarkXYMagScale(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false, { 1.25,0.75,0.75 });
	//getDriftLandmark(template_bfm, mouth_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, true);
	getDriftLandmark(template_bfm, nose_region_, template_guijie_68, att_bfm_68, 1.0, att_guijie_68, false);

	//rotate eyes
	rotateEyes(template_bfm, left_eye_region_, right_eye_region_, template_guijie_68, att_bfm_68, att_guijie_68);

	//dst att_guijie_68 pos

	//cstr root = "D:/code/auto3dData/";
	cstr root = config["nl_root"].get<cstr>();
	//intVec fix = FILEIO::loadIntDynamic(root + "D:/code/auto3dData/guijie/face_fix_reduce.txt");
	intVec fix = FILEIO::loadIntDynamic(root + "guijie/face_fix_reduce.txt");
	opt_res = in_guijie;
	intVec mouth_close = FILEIO::loadIntDynamic(root + "guijie/mouth_close.txt");

	intVec guijie_eyelash = FILEIO::loadIntDynamic(root + "guijie/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic(root + "guijie/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic(root + "guijie/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic(root + "guijie/pair_info/right_down_match.txt", guijie_eyelash);

	//此处是缺省过程
	//SIMDEFORM::moveHandle(in_guijie, landmark_68_guijie_, att_guijie_68, fix, guijie_eyelash, opt_res);

	//opt_res.saveObj("D:/dota210104/0120_color_fit_00_expand_eyes/local_deform.obj");

	intVec fix_expand = fix;
	//MeshCompress mean = "D:/dota210104/0120_color_fit_00_expand_eyes/B_deform.obj";
	FILEIO::loadIntDynamic(root + "guijie/mouth_structure_line.txt", fix_expand);
	SIMDEFORM::moveHandle(opt_res, mouth_close, in_guijie, mouth_close, fix_expand, guijie_eyelash, opt_res);
	//opt_res.saveObj("D:/dota210104/0120_color_fit_00_expand_eyes/close_local_deform.obj");
}



