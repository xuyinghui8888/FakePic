#include "ExpGen.h"
#include "../Basic/MeshHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../Solver/BVLSSolver.h"
using namespace CGP;

ExpGen::ExpGen(const json& config)
{
	if (is_init_ == false)
	{
		init(config);
	}
}

void ExpGen::init(const json& config)
{
	if (is_init_ == true)
	{
		LOG(INFO) << "already init for data." << std::endl;
		return;
	}
	LOG(INFO) << "init for ExpGen." << std::endl;
	data_root_ = config["root"].get<cstr>();
	//set for testing path
	//loading for eyelash
	eyelash_ = FILEIO::loadIntDynamic(data_root_ + config["left_down_lash"].get<cstr>());
	FILEIO::loadIntDynamic(data_root_ + config["left_up_lash"].get<cstr>(), eyelash_);
	FILEIO::loadIntDynamic(data_root_ + config["right_down_lash"].get<cstr>(), eyelash_);
	FILEIO::loadIntDynamic(data_root_ + config["right_up_lash"].get<cstr>(), eyelash_);

	left_eye_part_ = FILEIO::loadIntDynamic(data_root_ + config["part_left_eye"].get<cstr>());
	right_eye_part_ = FILEIO::loadIntDynamic(data_root_ + config["part_right_eye"].get<cstr>());
	mouth_part_ = FILEIO::loadIntDynamic(data_root_ + config["part_mouth"].get<cstr>());
	nose_part_ = FILEIO::loadIntDynamic(data_root_ + config["part_nose"].get<cstr>());

	eyelash_eye_pair_ = FILEIO::loadIntDynamic(data_root_ + config["pair_left_down"].get<cstr>());
	FILEIO::loadIntDynamic(data_root_ + config["pair_left_up"].get<cstr>(), eyelash_eye_pair_);
	FILEIO::loadIntDynamic(data_root_ + config["pair_right_down"].get<cstr>(), eyelash_eye_pair_);
	FILEIO::loadIntDynamic(data_root_ + config["pair_right_up"].get<cstr>(), eyelash_eye_pair_);

	left_eye_match_ = FILEIO::loadIntDynamic(data_root_ + config["match_left_eye"].get<cstr>());
	right_eye_match_ = FILEIO::loadIntDynamic(data_root_ + config["match_right_eye"].get<cstr>());
	left_eye_blvs_dis_ = FILEIO::loadIntDynamic(data_root_ + config["guijie_blvs_dis"].get<cstr>());
	dis_thres_ = 0.5;

	JsonHelper::initData(data_root_ + config["guijie_sys_tensor"].get<cstr>(), "config.json", guijie_);
	JsonHelper::initData(data_root_ + config["eye_shape_tensor"].get<cstr>(), "config.json", eye_shape_tensor_);
	JsonHelper::initData(data_root_ + config["eye_left_blink_tensor"].get<cstr>(), "config.json", left_eye_blink_tensor_);
	JsonHelper::initData(data_root_ + config["eye_right_blink_tensor"].get<cstr>(), "config.json", right_eye_blink_tensor_);
	
	default_A_.loadObj(data_root_ + config["default_A"].get<cstr>());
	loadExp(data_root_ + config["Ani_sound"].get<cstr>(), ani_sound_);
	loadExp(data_root_ + config["Ani_exp"].get<cstr>(), ani_exp_);
	loadExp(data_root_ + config["Ani_eye"].get<cstr>(), ani_eye_);
	loadExp(data_root_ + config["Ani_test"].get<cstr>(), ani_test_);


	dt_A_ = default_A_;
	dt_B_ = default_A_;
	dt_B_land_ = default_A_;
	dt_A_deform_ = default_A_;
	fixEyelash(default_A_);
	is_init_ = true;
	fix_thres_ = config["fix_thres"].get<float>();
	xy_rotate_ = intX2Vec{ {1931, 1969},{3780,3818} };

	dt_part_config_.face_part_ = 
	intX2Vec{
		left_eye_part_, right_eye_part_,
		mouth_part_, nose_part_,
	};

	dt_part_config_.opt_type_ =
	{
		LandGuidedType::XYZ_OPT, LandGuidedType::XYZ_OPT,
		LandGuidedType::XY_OPT_NO_SCALE, LandGuidedType::XY_OPT,
	};
	
	//init for part normal
	DTGuidedLandmark dt_land;	
	dt_land.getPartNormal(dt_part_config_.face_part_, dt_part_config_.opt_type_, default_A_, A_part_normal_);
	LOG(INFO) << "end of init ExpGen." << std::endl;
}

void ExpGen::dumpEyelash(MeshCompress& src)
{
	src.discard(eyelash_);
}


void ExpGen::getExpOMP(const MeshCompress& B, const cstr& output_path)
{
	//omp_set_num_threads(4);
	LOG(INFO) << "omp_thread: " << omp_get_thread_num() << std::endl;
#ifndef _WIN32
#pragma omp  sections nowait
	{
#pragma omp section
		{
			getEyeBlink(B, ani_eye_, output_path);
		}
#pragma omp section
		{
			getExpGuided(B, ani_exp_, output_path);
		}
#pragma omp section
		{
			getExpGuided(B, ani_sound_, output_path);
		}
	}
#else
	//memory not enough
	getEyeBlink(B, ani_eye_, output_path);
	getExpGuided(B, ani_exp_, output_path);
	getExpGuided(B, ani_sound_, output_path);
#endif
}

void ExpGen::loadExp(const cstr& root, FacePartInfo& res)
{
	FILEIO::loadFixSize(root + "json_name.txt", res.name_);
	//去掉mean
	cstrVec(res.name_.begin() + 1, res.name_.end()).swap(res.name_);
	Tensor temp;
	JsonHelper::initData(root, "config.json", temp);
	res.n_obj_ = res.name_.size();
	res.ani_.resize(res.n_obj_);
	res.omp_float_.resize(res.n_obj_);
	res.dt_init_E1_.resize(res.n_obj_);
	res.dt_init_E1_B_.resize(res.n_obj_);
	int n_vertex = temp.template_obj_.n_vertex_;
	FILEIO::loadEigenMat(root + "move_idx.txt", res.move_idx_);

#pragma omp parallel for
	for (int i = 0; i < res.n_obj_; i++)
	{
		res.ani_[i] = temp.template_obj_;
		SG::safeMemcpy(res.ani_[i].pos_.data(), &temp.data_[n_vertex * 3 * (i + 1)], n_vertex * 3 * sizeof(float));
		fixEyelash(res.ani_[i]);
	}

#pragma omp parallel for
	for (int i = 0; i < res.n_obj_; i++)
	{
		DTGuidedLandmark dt_template;
		dt_template.initDynamicFix(default_A_, res.ani_[i], default_A_);
		res.dt_init_E1_[i] = dt_template.E1_;
		res.dt_init_E1_B_[i] = dt_template.E1_B_;
	}
}

void ExpGen::generatePCA()
{
	//TODO change of adding differ position
	cstr obj_from = "D:/avatar/0722_eye_clip/test/";
	cstr obj_to = "D:/avatar/0728_project_00/pca_00/";
	SG::needPath(obj_to);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_from, ".obj");
	cstrVec ordered_map;
	FILEIO::loadFixSize("D:/avatar/0727_eye_clip/raw/json_name.txt", ordered_map);
	intVec left_eye = FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/left_eye.txt");
	FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/right_up_lash.txt", left_eye);
	FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/right_down_lash.txt", left_eye);
	intVec left_eye_no_lash = FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/left_eye.txt");
	intVec right_eye = FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/right_eye.txt");
	FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/left_up_lash.txt", right_eye);
	FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/left_down_lash.txt", right_eye);
	intVec right_eye_no_lash = FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/right_eye.txt");
	   	 
	bool is_left = true;
	intSet roi_set = is_left ? intSet(left_eye.begin(), left_eye.end()) : intSet(right_eye.begin(), right_eye.end());
	intSet roi_no_lash = is_left ? intSet(left_eye_no_lash.begin(), left_eye_no_lash.end()) : intSet(right_eye_no_lash.begin(), right_eye_no_lash.end());
	intVec eye_region = is_left ? left_eye : right_eye;

	for (int i = 0; i < ordered_map.size(); i++)
	{
		MeshCompress iter_mesh("D:/avatar/0727_eye_clip/raw/" + ordered_map[i] + ".obj");
		iter_mesh.keepRoi(eye_region);
		iter_mesh.saveObj(obj_to + ordered_map[i] + ".obj");
	}
}

void ExpGen::testExp()
{
	//using value from C:\avatar\0728_project_00\proj_1
	cstr cur_root = "D:/avatar/0728_project_00/proj_1/";
	cstr out_root = "D:/avatar/0728_project_00/expgen_2/";
	SG::needPath(out_root);
	json eye_blink_left = FILEIO::loadJson(cur_root + "Ani_eyeBlinkLeft.obj.json");
	json eye_squint_left = FILEIO::loadJson(cur_root + "Ani_eyeSquintLeft.obj.json");
	json eye_wide_left = FILEIO::loadJson(cur_root + "Ani_eyeWideLeft.obj.json");
	intVec res = FILEIO::loadIntDynamic("D:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_down_match.txt", res);
	//generate random eye
	cstrVec ordered_map;
	FILEIO::loadFixSize("D:/avatar/0727_eye_clip/raw/json_name.txt", ordered_map);
	int num = ordered_map.size();
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 0.5);
	vecF random_coef(num - 1);
	//blink squint wide
	std::vector<vecF> exp_dir;
	vecF coef_blink(num - 1);
	vecF coef_squint(num - 1);
	vecF coef_wide(num - 1);
	for (int i = 0; i < num - 1; i++)
	{
		cstr name = ordered_map[i + 1];
		coef_blink[i] = eye_blink_left[name];
		coef_squint[i] = eye_squint_left[name];
		coef_wide[i] = eye_wide_left[name];
	}
	for (int i = 0; i < num-1; i++)
	{
		if (i != 0 && i != 10 && i != 15)
		{
			double number = distribution(generator);
			random_coef[i] = number;
		}
		else
		{
			random_coef[i] = 0;
		}
	}
	random_coef = coef_squint * 0.5;
	LOG(INFO) << "ceres_coef: " << random_coef.transpose() << std::endl;
	LOG(INFO) << "coef_squint: " << coef_squint.transpose() << std::endl;
	//get random mesh
	Tensor taobao_tensor;
	JsonHelper::initData("D:/avatar/0727_eye_clip/pca/", "config.json", taobao_tensor);
	MeshCompress face_mean("D:/avatar/0722_eye_clip/raw/mean.obj");
	MeshCompress face_random = face_mean;
	floatVec proj_random = taobao_tensor.interpretID(random_coef);
	SG::safeMemcpy(face_random.pos_.data(), proj_random.data(), proj_random.size() * sizeof(float));
	face_random.saveObj(out_root +"face_random.obj");
	MeshCompress delta_src = face_random;

	proj_random = taobao_tensor.interpretID(coef_blink);
	SG::safeMemcpy(face_random.pos_.data(), proj_random.data(), proj_random.size() * sizeof(float));
	face_random.saveObj(out_root + "face_blink.obj");

	//get diff coef
	vecF diff_squint = coef_squint;
	for (int i = 0; i < num-1; i++)
	{
		if (coef_squint[i] > 1e-2)
		{
			diff_squint[i] = DMIN(coef_squint[i], random_coef[i]);
		}
	}

	proj_random = taobao_tensor.interpretID(diff_squint);
	SG::safeMemcpy(face_random.pos_.data(), proj_random.data(), proj_random.size() * sizeof(float));
	face_random.saveObj(out_root + "face_diff.obj");
	
	MeshCompress mean_face, mean_face_dst, delta_dst;
	mean_face.loadObj("D:/avatar/0728_test/mean.obj");
	mean_face_dst.loadObj("D:/avatar/0728_test/left/Ani_eyeSquintLeft.obj");
	//delta_src = face_random;
	MeshTransfer trans_exp;
	intVec fix = trans_exp.initDynamicFix(face_random, mean_face_dst, delta_src, res,  0.001);
	//intVec fix = trans_exp.initDynamicFix(mean_face, mean_face_dst, res, 0.001);
	//trans_exp.initDynamicFix(mean_face, mean_face_dst, res, 0.001);
	delta_dst = mean_face;
	trans_exp.transfer(delta_src.pos_, delta_dst.pos_);
	delta_dst.saveObj(out_root + "delta_dst_generate.obj");
}

void ExpGen::fixEyelash(MeshCompress& dst)
{
#pragma omp parallel for
	for (int iter_eyelash = 0; iter_eyelash < eyelash_eye_pair_.size()/2; iter_eyelash++)
	{
		int idx_lash = eyelash_eye_pair_[iter_eyelash * 2];
		int idx_eye = eyelash_eye_pair_[iter_eyelash * 2 + 1];
		dst.pos_[idx_lash] = dst.pos_[idx_eye];
	}
}

void ExpGen::getExpGuided(const MeshCompress& B, FacePartInfo& part, const cstr& output_path)
{
	SG::needPath(output_path);
	LOG(INFO) << "start getExpGuided. " << std::endl;
	std::vector<float> B_pos(B.n_vertex_ * 3);
	SG::safeMemcpy(B_pos.data(), B.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);

#pragma omp parallel for
	for (int i = 0; i < part.n_obj_; i++)
	{
		const MeshCompress& A = default_A_;
		const MeshCompress& A_deform = part.ani_[i];
		MeshCompress B_land = B;
		DTGuidedLandmark dt_land;

		dt_land.setPart(dt_part_config_.face_part_, dt_part_config_.opt_type_, A, A_deform, B, {}, A_part_normal_, B_part_normal_);
	
		dt_land.setPairType(PairType::PAIR_DIS);
		intVec pair = eyelash_eye_pair_;
		dt_land.fastSetupE1A(part.dt_init_E1_[i], part.dt_init_E1_B_[i]);
		dt_land.initDynamicFix(A, A_deform, B, pair, 0.00001);
		dt_land.transfer(B.pos_, B_land.pos_);
		if (debug_)
		{
			fixEyelash(B_land);
			B_land.saveObj(output_path + part.name_[i] + ".obj");
			auto B_land_eyelash = B_land;
			dumpEyelash(B_land_eyelash);
			B_land_eyelash.saveObj(output_path + part.name_[i] + "_no_lash.obj");
		}
		std::vector<float> in_bs_temp(B_land.n_vertex_ * 3);
		SG::safeMemcpy(in_bs_temp.data(), B_land.pos_[0].data(), sizeof(float)*B_land.n_vertex_ * 3);
		std::transform(in_bs_temp.begin(), in_bs_temp.end(),
			B_pos.begin(), in_bs_temp.begin(), std::minus<float>());
		//std::transform(in_bs_temp.begin(), in_bs_temp.end(),
			//A_pos.begin(), in_bs_temp.begin(), std::minus<float>());

		cstr item_name = part.name_[i];
		part.omp_float_[i] = in_bs_temp;		

	}
	LOG(INFO) << "end exp gen. " << std::endl;
	part.update_ = true;
}

void ExpGen::getExpGuided(const MeshCompress& B, const MeshCompress& A_deform_input, MeshCompress& B_res)
{
	LOG(INFO) << "getExpGuided from file. " << std::endl;
	std::vector<float> B_pos(B.n_vertex_ * 3);
	SG::safeMemcpy(B_pos.data(), B.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);

	const MeshCompress& A = default_A_;
	const MeshCompress& A_deform = A_deform_input;
	B_res = B;
	DTGuidedLandmark dt_land;

	dt_land.setPart(dt_part_config_.face_part_, dt_part_config_.opt_type_, A, A_deform, B, {}, A_part_normal_, B_part_normal_);

	dt_land.setPairType(PairType::PAIR_DIS);
	intVec pair = eyelash_eye_pair_;	
	dt_land.initDynamicFix(A, A_deform, B, pair, 0.00001);
	dt_land.transfer(B.pos_, B_res.pos_);
	//下面fixeyelash增加与否不影响；
	//fixEyelash(B_res);
	LOG(INFO) << "end getExpGuided from file. " << std::endl;
}

void ExpGen::getExpGuided(const MeshCompress& B, const MeshCompress& A_input, const MeshCompress& A_deform_input, MeshCompress& B_res)
{
	LOG(INFO) << "getExpGuided from file. " << std::endl;
	std::vector<float> B_pos(B.n_vertex_ * 3);
	SG::safeMemcpy(B_pos.data(), B.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);

	const MeshCompress& A = A_input;
	const MeshCompress& A_deform = A_deform_input;
	B_res = B;
	DTGuidedLandmark dt_land;

	dt_land.setPart(dt_part_config_.face_part_, dt_part_config_.opt_type_, A, A_deform, B, {}, A_part_normal_, B_part_normal_);

	dt_land.setPairType(PairType::PAIR_DIS);
	intVec pair = eyelash_eye_pair_;
	dt_land.initDynamicFix(A, A_deform, B, pair, 0.00001);
	dt_land.transfer(B.pos_, B_res.pos_);
	//下面fixeyelash增加与否不影响；
	//fixEyelash(B_res);
	LOG(INFO) << "end getExpGuided from file. " << std::endl;
}


void ExpGen::setJsonValueFromPart(const FacePartInfo& part, nlohmann::json& res, bool is_compress)
{
	if (part.update_ == true)
	{
		if (is_compress == false)
		{
			for (int i = 0; i < part.omp_float_.size(); i++)
			{
				const auto res_exp = std::minmax_element(part.omp_float_[i].begin(), part.omp_float_[i].end());
				if (*res_exp.first > -fix_thres_ && *res_exp.second < fix_thres_)
				{
					LOG(INFO) << "fix exp: " << part.name_[i] << std::endl;
				}
				else
				{
					res[part.name_[i]] = part.omp_float_[i];
				}
			}
		}
#if 0
		else
		{
			intX2Vec move_idx(part.omp_float_.size(), intVec{});
			floatX2Vec move_value(part.omp_float_.size(), floatVec{});
#pragma omp parallel for
			for (int iter_dim = 1; iter_dim < part.move_idx_.rows(); iter_dim++)
			{
				//带有一个shift
				for (int iter_vertex = 0; iter_vertex < part.move_idx_.cols(); iter_vertex++)
				{
					if (part.move_idx_(iter_dim, iter_vertex) == 1)
					{
						//move
						move_idx[iter_dim -1].push_back(iter_vertex);
						for (int k = 0; k < 3; k++)
						{
							move_value[iter_dim - 1].push_back(part.omp_float_[iter_dim - 1][iter_vertex * 3 + k]);
						}
					}
				}
			}
			for (int i = 0; i < part.omp_float_.size(); i++)
			{
				const auto res_exp = std::minmax_element(move_value[i].begin(), move_value[i].end());
				if (*res_exp.first > -fix_thres_ && *res_exp.second < fix_thres_)
				{
					LOG(INFO) << "fix exp: " << part.name_[i] << std::endl;
				}
				else
				{
					res[part.name_[i]] = move_value[i];
					res[part.name_[i] + "_idx"] = move_idx[i];
				}
			}
		}
#else
		else
		{
			intX2Vec move_idx(part.omp_float_.size(), intVec{});
			floatX2Vec move_value(part.omp_float_.size(), floatVec{});
#pragma omp parallel for
			for (int iter_dim = 0; iter_dim < part.omp_float_.size(); iter_dim++)
			{
				//带有一个shift
				for (int iter_vertex = 0; iter_vertex < part.move_idx_.cols(); iter_vertex++)
				{
					const float* vertex_ptr = &part.omp_float_[iter_dim][iter_vertex * 3];
					if (std::abs(vertex_ptr[0])< fix_thres_ && std::abs(vertex_ptr[1]) < fix_thres_ && std::abs(vertex_ptr[2]) < fix_thres_)
					{

					}
					else
					{
						//move
						move_idx[iter_dim].push_back(iter_vertex);
						for (int k = 0; k < 3; k++)
						{
							move_value[iter_dim].push_back(vertex_ptr[k]);
						}
					}
				}
			}
			for (int i = 0; i < part.omp_float_.size(); i++)
			{
				res[part.name_[i]] = move_value[i];
				res[part.name_[i] + "_idx"] = move_idx[i];
			}
		}
#endif
	}
}

void ExpGen::getResultJson(nlohmann::json& res, bool is_compress)
{
	LOG(INFO) << "start get result " << std::endl;
	//bool is_compress = true;
	res["n_vertex"] = dt_A_.n_vertex_;
	res["is_compress"] = is_compress;
	setJsonValueFromPart(ani_test_, res, is_compress);
	setJsonValueFromPart(ani_exp_, res, is_compress);
	setJsonValueFromPart(ani_sound_, res, is_compress);
	setJsonValueFromPart(ani_eye_, res, is_compress);
	res["eye_translate"] = eye_trans_;
	LOG(INFO) <<  "end get result " << std::endl;
}

void ExpGen::testExpGuided(const cstr& input_obj, const cstr& output_path, int part)
{
	cstrVec src_folder =
	{
		"D:/avatar/0818_guijie_bs/eye_ani/", //0
		"D:/avatar/0818_guijie_bs/brow_ani/", //1
		"D:/avatar/0818_guijie_bs/mouth_ani/",//2
		"D:/avatar/0818_guijie_bs/cheek_ani/",//3
		"D:/avatar/0818_guijie_bs/all_ani/",//4
		"D:/avatar/0822_guijie_bs/prob/",//5
	};

	cstr input_bs = src_folder[part];
	SG::needPath(output_path);
	
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(input_bs, ".obj");

	intVec res = FILEIO::loadIntDynamic("D:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_down_match.txt", res);

	MeshCompress A("D:/avatar/0818_guijie_bs/mean_0804.obj");
	MeshCompress B(input_obj);
	for (int iter_eyelash = 0; iter_eyelash < res.size()*0.5; iter_eyelash++)
	{
		int idx_lash = res[iter_eyelash * 2];
		int idx_eye = res[iter_eyelash * 2 + 1];
		A.pos_[idx_lash] = A.pos_[idx_eye];
		B.pos_[idx_lash] = B.pos_[idx_eye];
	}
	std::vector<float> B_pos(A.n_vertex_ * 3);
	SG::safeMemcpy(B_pos.data(), B.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);
	std::vector<float> A_pos = B_pos;
	SG::safeMemcpy(A_pos.data(), A.pos_[0].data(), sizeof(float)*A.n_vertex_ * 3);


	intVec left_eye_part = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_eye_3.txt");
	intVec right_eye_part = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_3.txt");
	intVec mouth_part = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/mouth.txt");
	intVec nose_part = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/nose.txt");
	   
	intVec left_eye_match = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_eye_match_top3.txt");
	intVec right_eye_match = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_match_top3.txt");

	intVec left_eye_deform = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_eye_deform.txt");
	intVec right_eye_deform = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_deform.txt");
	intVec left_eye_lash_up = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_up_lash.txt");
	intVec left_eye_lash_down = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_down_lash.txt");
	intVec right_eye_lash_up = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_up_lash.txt");
	intVec right_eye_lash_down = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_down_lash.txt");

	nlohmann::json vertex_value;

	intX2Vec xy_rotate = { {1931, 1969},{3780,3818} };
	double dis_thres = 0.5;
	floatX2Vec omp_float(folder_file.size());
	cstrVec omp_name(folder_file.size());
	LOG(INFO) << "start exp gen. " << std::endl;
#pragma omp parallel for
	for (int i = 0; i<folder_file.size(); i++)
	{
		MeshCompress A_deform(input_bs + folder_file[i]);
		MeshCompress B_land = B;
		DTGuidedLandmark dt_land;
		intX2Vec face_part = 
		{ 
			left_eye_part, right_eye_part, 
			mouth_part, nose_part,
		};

		//force eyelash to eye
		for (int iter_eyelash = 0;  iter_eyelash < res.size()*0.5; iter_eyelash ++)
		{
			int idx_lash = res[iter_eyelash * 2];
			int idx_eye = res[iter_eyelash * 2 + 1];
			A.pos_[idx_lash] = A.pos_[idx_eye];
			B.pos_[idx_lash] = B.pos_[idx_eye];
			A_deform.pos_[idx_lash] = A_deform.pos_[idx_eye];
		}

		std::vector<LandGuidedType> opt_type = 
		{
			LandGuidedType::XY_ADD_PLANE_ROTATE, LandGuidedType::XY_ADD_PLANE_ROTATE,
			LandGuidedType::XY_OPT_NO_SCALE, LandGuidedType::XY_OPT,
		};
		dt_land.setPart(face_part, opt_type,  A, A_deform, B, xy_rotate);
		dt_land.setPairType(PairType::PAIR_DIS);
		intVec pair = res;
		cstrVec split_file_name;
		FILEIO::splitString(folder_file[i], split_file_name, '.');
		if (split_file_name[0] == "Ani_eyeBlinkLeft")
		{
			intVec left_eye_match_refine;
			intSet left_eye_part_set(left_eye_part.begin(), left_eye_part.end());
			for (int iter_match = 0; iter_match < left_eye_match.size()/2; iter_match++)
			{
				int pair_src = left_eye_match[iter_match * 2];
				int pair_dst = left_eye_match[iter_match * 2 + 1];
				float3E pair_dis = A_deform.pos_[pair_src] - A_deform.pos_[pair_dst];
				//if (left_eye_part_set.count(left_eye_match[iter_match * 2]) && left_eye_part_set.count(left_eye_match[iter_match * 2 + 1]))
				if (pair_dis.norm()> dis_thres)
				{
					LOG(INFO) << "part && match conflict: " << left_eye_match[iter_match * 2] << ", " << left_eye_match[iter_match * 2 + 1] << std::endl;
				}
				else
				{
					left_eye_match_refine.push_back(left_eye_match[iter_match * 2]);
					left_eye_match_refine.push_back(left_eye_match[iter_match * 2+1]);
				}
			}
			
			pair.insert(pair.end(), left_eye_match_refine.begin(), left_eye_match_refine.end());
			//pair.insert(pair.end(), left_eye_match_refine.begin(), left_eye_match_refine.end());
			//ori 1e5 1e3
			//dt_land.Transfer_Weight_Land = 1e1;
			//dt_land.Transfer_Weight_Pair = 1e3;
		}

		if (split_file_name[0] == "Ani_eyeBlinkRight")
		{
			intVec right_eye_match_refine;
			intSet right_eye_part_set(right_eye_part.begin(), right_eye_part.end());

			for (int iter_match = 0; iter_match < right_eye_match.size() / 2; iter_match++)
			{
				int pair_src = right_eye_match[iter_match * 2];
				int pair_dst = right_eye_match[iter_match * 2 + 1];
				float3E pair_dis = A_deform.pos_[pair_src] - A_deform.pos_[pair_dst];
				//if (right_eye_part_set.count(right_eye_match[iter_match * 2]) && right_eye_part_set.count(right_eye_match[iter_match * 2 + 1]))
				if(pair_dis.norm()> dis_thres)
				{
					LOG(INFO) << "part && match conflict: " << right_eye_match[iter_match * 2] << ", " << right_eye_match[iter_match * 2 + 1] << std::endl;
				}
				else
				{
					right_eye_match_refine.push_back(right_eye_match[iter_match * 2]);
					right_eye_match_refine.push_back(right_eye_match[iter_match * 2 + 1]);
				}
			}
			pair.insert(pair.end(), right_eye_match_refine.begin(), right_eye_match_refine.end());
			//pair.insert(pair.end(), right_eye_match.begin(), right_eye_match.end());
			//dt_land.Transfer_Weight_Land = 1e1;
			//dt_land.Transfer_Weight_Pair = 1e3;
		}

		dt_land.initDynamicFix(A, A_deform, B, pair, 0.00001);
		dt_land.transfer(B.pos_, B_land.pos_);
#if 0
		if (split_file_name[0] == "Ani_eyeBlinkLeft")
		{
			LaplacianDeform eye_close;
			intVec roi;
			roi.insert(roi.end(), left_eye_deform.begin(), left_eye_deform.end());
			roi.insert(roi.end(), left_eye_lash_up.begin(), left_eye_lash_up.end());
			roi.insert(roi.end(), left_eye_lash_down.begin(), left_eye_lash_down.end());
			intVec fix = B_land.getReverseSelection(roi);
			pair.insert(pair.end(), left_eye_match.begin(), left_eye_match.end());
			eye_close.init(B_land, {}, fix, pair);
			eye_close.deform({}, B_land.pos_, LinkType::KEEP_ZERO);
		}

		if (split_file_name[0] == "Ani_eyeBlinkRight")
		{
			LaplacianDeform eye_close;
			intVec roi;
			roi.insert(roi.end(), right_eye_deform.begin(), right_eye_deform.end());
			roi.insert(roi.end(), right_eye_lash_up.begin(), right_eye_lash_up.end());
			roi.insert(roi.end(), right_eye_lash_down.begin(), right_eye_lash_down.end());
			intVec fix = B_land.getReverseSelection(roi);
			pair.insert(pair.end(), right_eye_match.begin(), right_eye_match.end());
			eye_close.init(B_land, {}, fix, pair);
			eye_close.deform({}, B_land.pos_, LinkType::KEEP_ZERO);
		}
#endif
		B_land.saveObj(output_path + folder_file[i]);
		std::vector<float> in_bs_temp(B_land.n_vertex_ * 3);
		SG::safeMemcpy(in_bs_temp.data(), B_land.pos_[0].data(), sizeof(float)*B_land.n_vertex_ * 3);

		std::transform(in_bs_temp.begin(), in_bs_temp.end(),
			B_pos.begin(), in_bs_temp.begin(), std::minus<float>());
		//std::transform(in_bs_temp.begin(), in_bs_temp.end(),
			//A_pos.begin(), in_bs_temp.begin(), std::minus<float>());

#if 0
		FILEIO::saveToBinary(output_path + i + ".txt", in_bs_temp);
		FILEIO::saveDynamic(output_path + i + "_diff.txt", in_bs_temp, ",");
		cstr str_res;
		FILEIO::saveToStringBinaryViaChar(in_bs_temp, str_res);
		cstr item_name = FILEIO::getFileNameWithoutExt(i);
		vertex_value[item_name] = str_res;
#else
		cstr item_name = FILEIO::getFileNameWithoutExt(folder_file[i]);
		omp_float[i] = in_bs_temp;
		omp_name[i] = item_name;
#endif
	}
	LOG(INFO) << "end exp gen. " << std::endl;
	vertex_value["n_vertex"] = B.n_vertex_;
	for (int i = 0; i < omp_float.size(); i++)
	{
		vertex_value[omp_name[i]] = omp_float[i];
	}
	FILEIO::saveJson(output_path + FILEIO::getFileNameWithoutExt(input_obj)+"_delta.json", vertex_value);
}

void ExpGen::setDebug(bool is_debug)
{
	debug_ = is_debug;
}

void ExpGen::setB(MeshCompress& B)
{
	B_pos_.resize(B.n_vertex_ * 3);
	SG::safeMemcpy(B_pos_.data(), B.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);
	DTGuidedLandmark dt_land;
	dt_land.getPartNormal(dt_part_config_.face_part_, dt_part_config_.opt_type_, B, B_part_normal_);
	ani_test_.update_ = false;
	ani_eye_.update_ = false;
	ani_exp_.update_ = false;
	ani_sound_.update_ = false;
}

void ExpGen::getEyeBlink(const MeshCompress& B, FacePartInfo& res, const cstr& output_path)
{
	//fixEyelash(B);
	LOG(INFO) << "start getEyeBlink. " << std::endl;
	vecD left_coef;
	getTensorCoef(eye_shape_tensor_, left_eye_part_, left_eye_blink_tensor_, left_eye_blvs_dis_, B, left_coef, output_path);
	getTensorCoefDouble(eye_shape_tensor_, left_eye_part_, left_eye_blink_tensor_, right_eye_blink_tensor_, left_eye_blvs_dis_, B, res, output_path);
	LOG(INFO) << "end getEyeBlink. " << std::endl;
}

void ExpGen::getCloseShape(const Tensor& shape_tensor, const Tensor& close_tensor, const vecD& coef, intVec& eye_match, 
	MeshCompress& B, MeshCompress& B_res)
{
	MeshCompress A_deform = shape_tensor.template_obj_;
	close_tensor.interpretIDFloat(A_deform.pos_.data(), coef);
	MeshCompress A = shape_tensor.template_obj_;
	shape_tensor.interpretIDFloat(A.pos_.data(), coef);
	B_res = B;
	DTGuidedLandmark dt_land;
	//force eyelash to eye
	for (int iter_eyelash = 0; iter_eyelash < eyelash_eye_pair_.size()*0.5; iter_eyelash++)
	{
		int idx_lash = eyelash_eye_pair_[iter_eyelash * 2];
		int idx_eye = eyelash_eye_pair_[iter_eyelash * 2 + 1];
		A.pos_[idx_lash] = A.pos_[idx_eye];
		B.pos_[idx_lash] = B.pos_[idx_eye];
		A_deform.pos_[idx_lash] = A_deform.pos_[idx_eye];
	}
	intX2Vec xy_rotate = { {1931, 1969},{3780,3818} };
	dt_land.Transfer_Weight_Pair = 1e4;
	dt_land.setPart(dt_part_config_.face_part_, dt_part_config_.opt_type_, A, A_deform, B);
	dt_land.setPairType(PairType::PAIR_DIS);
	intVec pair = eyelash_eye_pair_;

	intVec eye_match_refine;
	for (int iter_match = 0; iter_match < eye_match.size() / 2; iter_match++)
	{
		int pair_src = eye_match[iter_match * 2];
		int pair_dst = eye_match[iter_match * 2 + 1];
		float3E pair_dis = A_deform.pos_[pair_src] - A_deform.pos_[pair_dst];
		//if (left_eye_part_set.count(left_eye_match[iter_match * 2]) && left_eye_part_set.count(left_eye_match[iter_match * 2 + 1]))
		
		if (pair_dis.norm() > dis_thres_)
		{
			LOG(INFO) << "part && match conflict: " << eye_match[iter_match * 2] << ", " << eye_match[iter_match * 2 + 1] << std::endl;
		}
		else
		{
			eye_match_refine.push_back(eye_match[iter_match * 2]);
			eye_match_refine.push_back(eye_match[iter_match * 2 + 1]);
		}
	}

	pair.insert(pair.end(), eye_match_refine.begin(), eye_match_refine.end());
	dt_land.initDynamicFix(A, A_deform, B, pair, 0.00001);

	dt_land.transfer(B.pos_, B_res.pos_);
	//pair.insert(pair.end(), left_eye_match_refine.begin(), left_eye_match_refine.end());
	//ori 1e5 1e3
	//dt_land.Transfer_Weight_Land = 1e1;
	//dt_land.Transfer_Weight_Pair = 1e1;
	if (debug_)
	{
		A_deform.saveObj("D:/avatar/0823_test/inter.obj");
		A_deform.discard(eyelash_);
		A_deform.saveObj("D:/avatar/0823_test/inter_no_lash.obj");
		A.saveObj("D:/avatar/0823_test/ori.obj");
		A.discard(eyelash_);
		A.saveObj("D:/avatar/0823_test/ori_no_lash.obj");
	}
}

void ExpGen::getCloseShape(MeshCompress& A, MeshCompress& A_deform, intVec& eye_match,
	MeshCompress& B, MeshCompress& B_res)
{
	B_res = B;
	DTGuidedLandmark dt_land;

	//force eyelash to eye
	for (int iter_eyelash = 0; iter_eyelash < eyelash_eye_pair_.size()*0.5; iter_eyelash++)
	{
		int idx_lash = eyelash_eye_pair_[iter_eyelash * 2];
		int idx_eye = eyelash_eye_pair_[iter_eyelash * 2 + 1];
		A.pos_[idx_lash] = A.pos_[idx_eye];
		B.pos_[idx_lash] = B.pos_[idx_eye];
		A_deform.pos_[idx_lash] = A_deform.pos_[idx_eye];
	}

	dt_land.Transfer_Weight_Land = 1e4;
	dt_land.setPart(dt_part_config_.face_part_, dt_part_config_.opt_type_, A, A_deform, B);
	dt_land.setPairType(PairType::PAIR_DIS);
	intVec pair = eyelash_eye_pair_;

	intVec eye_match_refine;
	for (int iter_match = 0; iter_match < eye_match.size() / 2; iter_match++)
	{
		int pair_src = eye_match[iter_match * 2];
		int pair_dst = eye_match[iter_match * 2 + 1];
		float3E pair_dis = A_deform.pos_[pair_src] - A_deform.pos_[pair_dst];
		//if (left_eye_part_set.count(left_eye_match[iter_match * 2]) && left_eye_part_set.count(left_eye_match[iter_match * 2 + 1]))

		if (pair_dis.norm() > dis_thres_)
		{
			//LOG(INFO) << "part && match conflict: " << eye_match[iter_match * 2] << ", " << eye_match[iter_match * 2 + 1] << std::endl;
		}
		else
		{
			eye_match_refine.push_back(eye_match[iter_match * 2]);
			eye_match_refine.push_back(eye_match[iter_match * 2 + 1]);
		}
	}

	pair.insert(pair.end(), eye_match_refine.begin(), eye_match_refine.end());
	dt_land.initDynamicFix(A, A_deform, B, pair, 0.00001);
	//dt_land.Transfer_Weight_Pair = 1e4;
	//dt_land.Transfer_Weight_Land = 1e2;
	dt_land.transfer(B.pos_, B_res.pos_);
	//pair.insert(pair.end(), left_eye_match_refine.begin(), left_eye_match_refine.end());
	//ori 1e5 1e3
	//dt_land.Transfer_Weight_Land = 1e1;
	//dt_land.Transfer_Weight_Pair = 1e1;
	/*
	A_deform.saveObj("D:/avatar/0823_test/inter.obj");
	A_deform.discard(eyelash_);
	A_deform.saveObj("D:/avatar/0823_test/inter_no_lash.obj");
	A.saveObj("D:/avatar/0823_test/ori.obj");
	A.discard(eyelash_);
	A.saveObj("D:/avatar/0823_test/ori_no_lash.obj");
	*/
}

void ExpGen::getTensorCoef(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor, const intVec& up_down_match, 
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

void ExpGen::getTensorCloseCoef(const Tensor& tensor, const intVec& roi, const MeshCompress& src, const MeshCompress& dst, const vecD& weight, vecD& coef)
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
	upper = weight;
	for (int i = 0; i < weight.size(); i++)
	{
		upper(i) = upper(i) + 1e-6;
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
				A(3 * i + iter_dim, j) = tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim];
				B(3 * i + iter_dim) = dst.pos_[vertex_id][iter_dim] - src.pos_[vertex_id][iter_dim];
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

void ExpGen::getEyeTransform(const MeshCompress& left_blink, const MeshCompress& right_blink, floatVec& trans)
{
	//get eye transform
	trans = floatVec{ 0,0,0 };
	intVec eye_dis_idx = { 2290, 3810 };
	float3E v_2290 = float3E(-5.6993, 152.7200, 7.5391);
	float3E v_3810 = float3E(-5.4025, 149.7730, 7.5045);
	float eye_dis = (v_2290 - v_3810).norm() + 0.1;
	float eye_A = (default_A_.pos_[2290] - default_A_.pos_[3810]).norm();
	//-0.2
	float eye_cur_dis = (left_blink.pos_[2290] - left_blink.pos_[3810]).norm();
	trans[2] = -0.2 /(eye_A - eye_dis) * (eye_A - eye_cur_dis);
	trans[2] = DMIN(trans[2], 0);
}

void ExpGen::getTensorCoefDouble(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor_left,
	const Tensor& close_tensor_right, const intVec& up_down_match, 
	const MeshCompress& dst, FacePartInfo& part, const cstr& output_path)
{
	vecD shape_coef, exp_coef;
	getTensorCoef(tensor, roi, close_tensor_left, up_down_match, dst, shape_coef);
	MeshCompress shape_A = tensor.template_obj_;
	tensor.interpretIDFloat(shape_A.pos_.data(), shape_coef);

	getTensorCloseCoef(close_tensor_left, roi, shape_A, dst, shape_coef, exp_coef);
	close_tensor_left.interpretIDFloatAdd((float*)(&shape_A.pos_[0]), exp_coef);
	close_tensor_right.interpretIDFloatAdd((float*)(&shape_A.pos_[0]), exp_coef);
	MeshCompress shape_A_left = shape_A;
	MeshCompress shape_A_right = shape_A;
	close_tensor_left.interpretIDFloat(shape_A_left.pos_.data(), shape_coef);
	close_tensor_right.interpretIDFloat(shape_A_right.pos_.data(), shape_coef);

	guijie_.keepOneSideUnchanged(shape_A.pos_, shape_A_left.pos_, false);
	guijie_.keepOneSideUnchanged(shape_A.pos_, shape_A_right.pos_, true);
	MeshCompress B = dst;
	MeshCompress B_left_blink = B;
	MeshCompress B_right_blink = B;
	fixEyelash(shape_A);
	fixEyelash(shape_A_left);
	fixEyelash(shape_A_right);
	fixEyelash(B);
	getCloseShape(shape_A, shape_A_left, left_eye_match_, B, B_left_blink);
	getCloseShape(shape_A, shape_A_right, right_eye_match_, B, B_right_blink);
	fixEyelash(B_left_blink);
	fixEyelash(B_right_blink);
	std::vector<float> in_bs_temp(B.n_vertex_ * 3);
	for (int i = 0; i < part.n_obj_; i++)
	{
		if (part.name_[i] == "Ani_eyeBlinkLeft")
		{
			SG::safeMemcpy(in_bs_temp.data(), B_left_blink.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);
			std::transform(in_bs_temp.begin(), in_bs_temp.end(), 
				B_pos_.begin(), in_bs_temp.begin(), std::minus<float>());
			part.omp_float_[i] = in_bs_temp;
		}
		else if (part.name_[i] == "Ani_eyeBlinkRight")
		{
			SG::safeMemcpy(in_bs_temp.data(), B_right_blink.pos_[0].data(), sizeof(float)*B.n_vertex_ * 3);
			std::transform(in_bs_temp.begin(), in_bs_temp.end(),
				B_pos_.begin(), in_bs_temp.begin(), std::minus<float>());
			part.omp_float_[i] = in_bs_temp;
		}
	}
	getEyeTransform(B_left_blink, B_right_blink, eye_trans_);
	part.update_ = true;
	//if (debug_)
	if (true)
	{
		cstr root = output_path;
		fixEyelash(shape_A);
		shape_A.saveObj(root + "shape_A.obj");
		shape_A.discard(eyelash_);
		shape_A.saveObj(root + "shape_A_no_lash.obj");

		fixEyelash(shape_A_left);
		shape_A_left.discard(eyelash_);
		shape_A_left.saveObj(root + "shape_A_left_no_lash.obj");

		fixEyelash(shape_A_right);
		shape_A_right.saveObj(root + "shape_A_right.obj");
		shape_A_right.discard(eyelash_);
		shape_A_right.saveObj(root + "shape_A_right_no_lash.obj");

		fixEyelash(B);
		B.saveObj(root + "B.obj");
		B.discard(eyelash_);
		B.saveObj(root + "B_no_lash.obj");

		fixEyelash(B_left_blink);
		B_left_blink.saveObj(root + "B_left_blink.obj");
		B_left_blink.discard(eyelash_);
		B_left_blink.saveObj(root + "B_left_blink_no_lash.obj");


		fixEyelash(B_right_blink);
		B_right_blink.saveObj(root + "B_right_blink.obj");
		B_right_blink.discard(eyelash_);
		B_right_blink.saveObj(root + "B_right_blink_no_lash.obj");
	}
}
