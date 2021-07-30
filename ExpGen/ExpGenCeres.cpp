#include "ExpGen.h"
#include "../Basic/MeshHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../Solver/BVLSSolver.h"
using namespace CGP;

ExpGen::ExpGen(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	//TODO set for general path
}

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
	//set for testing path
	//loading for eyelash
	eyelash_ = FILEIO::loadIntDynamic("C:/avatar/guijie/left_up_lash.txt");
	FILEIO::loadIntDynamic("C:/avatar/guijie/left_down_lash.txt", eyelash_);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_up_lash.txt", eyelash_);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_down_lash.txt", eyelash_);

	left_eye_part_ = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_eye_3.txt");
	right_eye_part_ = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_3.txt");
	mouth_part_ = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/mouth.txt");
	nose_part_ = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/nose.txt");

	eyelash_eye_pair_ = FILEIO::loadIntDynamic("C:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("C:/avatar/guijie/left_down_match.txt", eyelash_eye_pair_);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_up_match.txt", eyelash_eye_pair_);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_down_match.txt", eyelash_eye_pair_);

	left_eye_match_ = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_eye_match_top3.txt");
	right_eye_match_ = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_match_top3.txt");

	dis_thres_ = 0.5;

	JsonHelper::initData("D:/data_20July/0717_guijie_sys_tensor/", "config.json", guijie_);

	is_init_ = true;
	LOG(INFO) << "init for ExpGen." << std::endl;
}

void ExpGen::projectGuijieToBsCoefVertex()
{
	//TODO change of adding differ position
	cstr obj_from = "C:/avatar/0722_eye_clip/test/";
	cstr obj_to = "C:/avatar/0728_project_00/proj_01/";
	SG::needPath(obj_to);
	Tensor taobao_tensor;
	JsonHelper::initData("C:/avatar/0727_eye_clip/pca/", "config.json", taobao_tensor);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_from, ".obj");
	cstrVec ordered_map;
	FILEIO::loadFixSize("C:/avatar/0727_eye_clip/raw/json_name.txt", ordered_map);
	intVec left_eye = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_eye.txt");
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_up_lash.txt", left_eye);
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_down_lash.txt", left_eye);
	intVec left_eye_no_lash = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_eye.txt");
	intVec right_eye = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_eye.txt");
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_up_lash.txt", right_eye);
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_down_lash.txt", right_eye);
	intVec right_eye_no_lash= FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_eye.txt");
	
	bool is_left = true;
	intSet roi_set = is_left ? intSet(left_eye.begin(), left_eye.end()) : intSet(right_eye.begin(), right_eye.end());
	intSet roi_no_lash = is_left ? intSet(left_eye_no_lash.begin(), left_eye_no_lash.end()) : intSet(right_eye_no_lash.begin(), right_eye_no_lash.end());
	for (int i = 0; i < folder_file.size(); i++)
	{
		MeshCompress face_mean("C:/avatar/0722_eye_clip/raw/mean.obj");
		cstr file_name = obj_from + folder_file[i];
		if (!SG::isExist(file_name))
		{
			break;
		}
		int num = ordered_map.size();
		MeshCompress dst_obj(file_name);
		MeshCompress dst_obj_back = dst_obj;
		ceres::Problem fitting_pca;
		doubleVec shape_coef(num, 0);
		shape_coef[0] = 1.0;
		int n_vertex = dst_obj.n_vertex_;
		//changable points
		intVec getMoveIdx = MeshTools::getMoveIdx(face_mean, dst_obj, 1e-6);
		for (int iter_point : getMoveIdx)
		{
			if (roi_no_lash.count(iter_point))
			{
				floatVec pca_weight(num * 3, 0);
				for (int iter_pca = 0; iter_pca < num; iter_pca++)
				{
					for (int iter_dim = 0; iter_dim < 3; iter_dim++)
					{
						pca_weight[iter_pca * 3 + iter_dim] = taobao_tensor.data_[iter_pca * n_vertex * 3 + iter_point * 3 + iter_dim];
					}
				}
				ceres::CostFunction* cost_function =
					new ceres::NumericDiffCostFunction<PCAVertexCost,
					ceres::CENTRAL,
					3 /* traslate*/,
					17 /*pca value*/>
					(new PCAVertexCost(pca_weight, dst_obj.pos_[iter_point], taobao_tensor.n_id_));
				fitting_pca.AddResidualBlock(cost_function, NULL, shape_coef.data());
			}
		}

		shape_coef[0] = 1;
		fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1 + 1e-6); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1 - 1e-6); // t_z has to be negative		
		for (int iter_coef = 1; iter_coef < num; iter_coef++)
		{
			fitting_pca.SetParameterUpperBound(&shape_coef[0], iter_coef, 1); // t_z has to be negative
			fitting_pca.SetParameterLowerBound(&shape_coef[0], iter_coef, 0); // t_z has to be negative
		}
		ceres::Solver::Options solver_options;
		solver_options.linear_solver_type = ceres::DENSE_SCHUR;
		//solver_options.num_threads = 8;
		solver_options.minimizer_progress_to_stdout = true;
		solver_options.max_num_iterations = 100;
		ceres::Solver::Summary solver_summary;
		Solve(solver_options, &fitting_pca, &solver_summary);
		std::cout << solver_summary.BriefReport() << "\n";

		vecF ceres_coef(num - 1);
		for (int iter_pca = 1; iter_pca < num; iter_pca++)
		{
			ceres_coef[iter_pca - 1] = shape_coef[iter_pca];
		}

		LOG(INFO) << "ceres pca: " << shape_coef[0] << std::endl << ceres_coef.transpose() << std::endl;
		floatVec proj_res_roi = taobao_tensor.interpretID(ceres_coef);
		SG::safeMemcpy(dst_obj.pos_.data(), proj_res_roi.data(), proj_res_roi.size() * sizeof(float));
		dst_obj.saveObj(obj_to + folder_file[i] + "_ceres.obj");
		//diff bs

		json json_coef;
		json_coef["mean_face"] = 1.0;
		MeshCompress diff = face_mean;
		MeshCompress roi = face_mean;
		double mag_c = 2.0;
#if 1
//#pragma omp parallel for
		for (int iter_vertex = 0; iter_vertex < getMoveIdx.size(); iter_vertex++)
		{
			int idx_vertex = getMoveIdx[iter_vertex];
			diff.pos_[idx_vertex] += mag_c * (dst_obj_back.pos_[idx_vertex] - dst_obj.pos_[idx_vertex]);
			if (roi_set.count(idx_vertex))
			{
				roi.pos_[idx_vertex] = dst_obj.pos_[idx_vertex];
			}
		}
#else
#pragma omp parallel for
		for (int iter_vertex = 0; iter_vertex < diff.pos_.size(); iter_vertex++)
		{
			diff.pos_[iter_vertex] += mag_c * (src_obj_back.pos_[iter_vertex] - src_obj.pos_[iter_vertex]);
		}
#endif
		diff.saveObj(obj_to + folder_file[i] + "_diff.obj");
		roi.saveObj(obj_to + folder_file[i] + "_roi.obj");
		for (int iter_json = 1; iter_json < ordered_map.size(); iter_json++)
		{
			json_coef[ordered_map[iter_json]] = ceres_coef[iter_json - 1];
		}
		std::ofstream out_json(obj_to + folder_file[i] + ".json");
		out_json << json_coef << std::endl;
		out_json.close();
	}
}

void ExpGen::generatePCA()
{
	//TODO change of adding differ position
	cstr obj_from = "C:/avatar/0722_eye_clip/test/";
	cstr obj_to = "C:/avatar/0728_project_00/pca_00/";
	SG::needPath(obj_to);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_from, ".obj");
	cstrVec ordered_map;
	FILEIO::loadFixSize("C:/avatar/0727_eye_clip/raw/json_name.txt", ordered_map);
	intVec left_eye = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_eye.txt");
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_up_lash.txt", left_eye);
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_down_lash.txt", left_eye);
	intVec left_eye_no_lash = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_eye.txt");
	intVec right_eye = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_eye.txt");
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_up_lash.txt", right_eye);
	FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/left_down_lash.txt", right_eye);
	intVec right_eye_no_lash = FILEIO::loadIntDynamic("C:/avatar/0727_eye_clip/roi/right_eye.txt");
	   	 
	bool is_left = true;
	intSet roi_set = is_left ? intSet(left_eye.begin(), left_eye.end()) : intSet(right_eye.begin(), right_eye.end());
	intSet roi_no_lash = is_left ? intSet(left_eye_no_lash.begin(), left_eye_no_lash.end()) : intSet(right_eye_no_lash.begin(), right_eye_no_lash.end());
	intVec eye_region = is_left ? left_eye : right_eye;

	for (int i = 0; i < ordered_map.size(); i++)
	{
		MeshCompress iter_mesh("C:/avatar/0727_eye_clip/raw/" + ordered_map[i] + ".obj");
		iter_mesh.keepRoi(eye_region);
		iter_mesh.saveObj(obj_to + ordered_map[i] + ".obj");
	}
}

void ExpGen::testExp()
{
	//using value from C:\avatar\0728_project_00\proj_1
	cstr cur_root = "C:/avatar/0728_project_00/proj_1/";
	cstr out_root = "C:/avatar/0728_project_00/expgen_2/";
	SG::needPath(out_root);
	json eye_blink_left = FILEIO::loadJson(cur_root + "Ani_eyeBlinkLeft.obj.json");
	json eye_squint_left = FILEIO::loadJson(cur_root + "Ani_eyeSquintLeft.obj.json");
	json eye_wide_left = FILEIO::loadJson(cur_root + "Ani_eyeWideLeft.obj.json");
	intVec res = FILEIO::loadIntDynamic("C:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("C:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_down_match.txt", res);
	//generate random eye
	cstrVec ordered_map;
	FILEIO::loadFixSize("C:/avatar/0727_eye_clip/raw/json_name.txt", ordered_map);
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
	JsonHelper::initData("C:/avatar/0727_eye_clip/pca/", "config.json", taobao_tensor);
	MeshCompress face_mean("C:/avatar/0722_eye_clip/raw/mean.obj");
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
	mean_face.loadObj("C:/avatar/0728_test/mean.obj");
	mean_face_dst.loadObj("C:/avatar/0728_test/left/Ani_eyeSquintLeft.obj");
	//delta_src = face_random;
	MeshTransfer trans_exp;
	intVec fix = trans_exp.initDynamicFix(face_random, mean_face_dst, delta_src, res,  0.001);
	//intVec fix = trans_exp.initDynamicFix(mean_face, mean_face_dst, res, 0.001);
	//trans_exp.initDynamicFix(mean_face, mean_face_dst, res, 0.001);
	delta_dst = mean_face;
	trans_exp.transfer(delta_src.pos_, delta_dst.pos_);
	delta_dst.saveObj(out_root + "delta_dst_generate.obj");
}

void ExpGen::testExpGuided(const cstr& input_obj, const cstr& output_path, int part)
{
	cstrVec src_folder =
	{
		"C:/avatar/0818_guijie_bs/eye_ani/", //0
		"C:/avatar/0818_guijie_bs/brow_ani/", //1
		"C:/avatar/0818_guijie_bs/mouth_ani/",//2
		"C:/avatar/0818_guijie_bs/cheek_ani/",//3
		"C:/avatar/0818_guijie_bs/all_ani/",//4
		"C:/avatar/0822_guijie_bs/prob/",//5
	};

	cstr input_bs = src_folder[part];
	SG::needPath(output_path);
	
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(input_bs, ".obj");

	intVec res = FILEIO::loadIntDynamic("C:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("C:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_down_match.txt", res);

	MeshCompress A("C:/avatar/0818_guijie_bs/mean_0804.obj");
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

void ExpGen::getEyeBlink(const cstr& input_obj, const cstr& output_path, int part)
{
	cstrVec src_folder =
	{
		"C:/avatar/0818_guijie_bs/eye_ani/", //0
		"C:/avatar/0818_guijie_bs/brow_ani/", //1
		"C:/avatar/0818_guijie_bs/mouth_ani/",//2
		"C:/avatar/0818_guijie_bs/cheek_ani/",//3
		"C:/avatar/0818_guijie_bs/all_ani/",//4
		"C:/avatar/0822_guijie_bs/prob/",//5
	};

	cstr input_bs = src_folder[part];
	SG::needPath(output_path);

	CGP::cstrVec folder_file = FILEIO::getFolderFiles(input_bs, ".obj");

	intVec res = FILEIO::loadIntDynamic("C:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("C:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("C:/avatar/guijie/right_down_match.txt", res);

	MeshCompress A("C:/avatar/0818_guijie_bs/mean_0804.obj");
	MeshCompress A_deform = A;
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

	double dis_thres = 0.5;
	floatX2Vec omp_float(folder_file.size());
	cstrVec omp_name(folder_file.size());
	LOG(INFO) << "start exp gen. " << std::endl;

	//loading for tensor
	Tensor eye_shape_tensor, left_eye_blink, right_eye_blink;
	JsonHelper::initData("C:/avatar/0822_close/eye_pca/", "config.json", eye_shape_tensor);
	JsonHelper::initData("C:/avatar/0822_close/left_pca/", "config.json", left_eye_blink);
	JsonHelper::initData("C:/avatar/0822_close/right_pca/", "config.json", right_eye_blink);

	//A可变 B不可以变
	vecD left_coef, right_coef;
	intVec left_eye_match_dis = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_eye_match_refine.txt");
	getTensorCoef(eye_shape_tensor, left_eye_part, left_eye_blink, left_eye_match_dis, B, left_coef);
	getTensorCoefDouble(eye_shape_tensor, left_eye_part, left_eye_blink, right_eye_blink, left_eye_match_dis, B);

	


	MeshCompress left_blink, right_blink;
	getCloseShape(eye_shape_tensor, left_eye_blink, left_coef, left_eye_match, B, left_blink);
	getCloseShape(eye_shape_tensor, right_eye_blink, left_coef, right_eye_match, B, right_blink);
	left_blink.saveObj("C:/avatar/0823_test/left_blink.obj");
	right_blink.saveObj("C:/avatar/0823_test/right_blink.obj");
	left_blink.discard(eyelash_);
	right_blink.discard(eyelash_);
	left_blink.saveObj("C:/avatar/0823_test/left_blink_no_lash.obj");
	right_blink.saveObj("C:/avatar/0823_test/right_blink_no_lash.obj");
	B.saveObj("C:/avatar/0823_test/B.obj");
	B.discard(eyelash_);
	B.saveObj("C:/avatar/0823_test/B_no_lash.obj");
#pragma omp parallel for
	for (int i = 0; i < folder_file.size(); i++)
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
		for (int iter_eyelash = 0; iter_eyelash < res.size()*0.5; iter_eyelash++)
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
		dt_land.setPart(face_part, opt_type, A, A_deform, B);
		dt_land.setPairType(PairType::PAIR_ZERO);
		intVec pair = res;
		cstrVec split_file_name;
		FILEIO::splitString(folder_file[i], split_file_name, '.');
		if (split_file_name[0] == "Ani_eyeBlinkLeft")
		{
			intVec left_eye_match_refine;
			intSet left_eye_part_set(left_eye_part.begin(), left_eye_part.end());
			for (int iter_match = 0; iter_match < left_eye_match.size() / 2; iter_match++)
			{
				int pair_src = left_eye_match[iter_match * 2];
				int pair_dst = left_eye_match[iter_match * 2 + 1];
				float3E pair_dis = A_deform.pos_[pair_src] - A_deform.pos_[pair_dst];
				//if (left_eye_part_set.count(left_eye_match[iter_match * 2]) && left_eye_part_set.count(left_eye_match[iter_match * 2 + 1]))
				if (pair_dis.norm() > dis_thres)
				{
					LOG(INFO) << "part && match conflict: " << left_eye_match[iter_match * 2] << ", " << left_eye_match[iter_match * 2 + 1] << std::endl;
				}
				else
				{
					left_eye_match_refine.push_back(left_eye_match[iter_match * 2]);
					left_eye_match_refine.push_back(left_eye_match[iter_match * 2 + 1]);
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
				if (pair_dis.norm() > dis_thres)
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
	FILEIO::saveJson(output_path + FILEIO::getFileNameWithoutExt(input_obj) + "_delta.json", vertex_value);
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
	intX2Vec face_part =
	{
		left_eye_part_, right_eye_part_,
		mouth_part_, nose_part_,
	};

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
	std::vector<LandGuidedType> opt_type =
	{
		LandGuidedType::XYZ_OPT, LandGuidedType::XYZ_OPT,
		LandGuidedType::XY_OPT_NO_SCALE, LandGuidedType::XY_OPT,
	};
	dt_land.setPart(face_part, opt_type, A, A_deform, B);
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
	//dt_land.Transfer_Weight_Pair = 1e4;
	dt_land.transfer(B.pos_, B_res.pos_);
	//pair.insert(pair.end(), left_eye_match_refine.begin(), left_eye_match_refine.end());
	//ori 1e5 1e3
	//dt_land.Transfer_Weight_Land = 1e1;
	//dt_land.Transfer_Weight_Pair = 1e1;
	A_deform.saveObj("C:/avatar/0823_test/inter.obj");
	A_deform.discard(eyelash_);
	A_deform.saveObj("C:/avatar/0823_test/inter_no_lash.obj");
	A.saveObj("C:/avatar/0823_test/ori.obj");
	A.discard(eyelash_);
	A.saveObj("C:/avatar/0823_test/ori_no_lash.obj");
}

void ExpGen::getCloseShape(MeshCompress& A, MeshCompress& A_deform, intVec& eye_match,
	MeshCompress& B, MeshCompress& B_res)
{
	B_res = B;
	DTGuidedLandmark dt_land;
	intX2Vec face_part =
	{
		left_eye_part_, right_eye_part_,
		mouth_part_, nose_part_,
	};

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
	std::vector<LandGuidedType> opt_type =
	{
		LandGuidedType::XY_ADD_PLANE_ROTATE, LandGuidedType::XY_ADD_PLANE_ROTATE,
		LandGuidedType::XY_OPT_NO_SCALE, LandGuidedType::XY_OPT,
	};
	dt_land.setPart(face_part, opt_type, A, A_deform, B, xy_rotate);
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
	dt_land.Transfer_Weight_Pair = 1e4;
	//dt_land.Transfer_Weight_Land = 1e2;
	dt_land.transfer(B.pos_, B_res.pos_);
	//pair.insert(pair.end(), left_eye_match_refine.begin(), left_eye_match_refine.end());
	//ori 1e5 1e3
	//dt_land.Transfer_Weight_Land = 1e1;
	//dt_land.Transfer_Weight_Pair = 1e1;
	/*
	A_deform.saveObj("C:/avatar/0823_test/inter.obj");
	A_deform.discard(eyelash_);
	A_deform.saveObj("C:/avatar/0823_test/inter_no_lash.obj");
	A.saveObj("C:/avatar/0823_test/ori.obj");
	A.discard(eyelash_);
	A.saveObj("C:/avatar/0823_test/ori_no_lash.obj");
	*/
}

void ExpGen::getTensorCoef(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor, const intVec& up_down_match, const MeshCompress& dst, vecD& coef)
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
				if (start_row + 3 * i + iter_dim == 489)
				{
					LOG(INFO) << "what" << std::endl;
				}
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
	FILEIO::saveEigenDynamic("C:/avatar/0823_test/A.txt", A);
	FILEIO::saveEigenDynamic("C:/avatar/0823_test/B.txt", B.transpose());
	BVLSSolver test(A, B, lower, upper);
	coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;
}

void ExpGen::getTensorCloseCoef(const Tensor& tensor, const intVec& roi, const MeshCompress& src, const MeshCompress& dst, const vecD& weight, vecD& coef)
{
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3;
	double weight_sum_1 = 1e5;
	double weight_pair = 1e-1;
	double sum_extra = 1.02;
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
				B(3 * i + iter_dim) = dst.pos_[vertex_id][iter_dim] - src.pos_[vertex_id][iter_dim];
			}
		}
	}	
	FILEIO::saveEigenDynamic("C:/avatar/0823_test/A.txt", A);
	FILEIO::saveEigenDynamic("C:/avatar/0823_test/B.txt", B.transpose());
	BVLSSolver test(A, B, lower, upper);
	coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;
}


void ExpGen::getTensorCoefDouble(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor_left,
	const Tensor& close_tensor_right, const intVec& up_down_match, const MeshCompress& dst)
{
	vecD shape_coef, exp_coef;
	getTensorCoef(tensor, roi, close_tensor_left, up_down_match, dst, shape_coef);
	MeshCompress shape_A = tensor.template_obj_;
	tensor.interpretIDFloat(shape_A.pos_.data(), shape_coef);
	shape_A.saveObj("C:/avatar/0823_test/shape_A.obj");

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
	getCloseShape(shape_A, shape_A_left, left_eye_match_, B, B_left_blink);
	getCloseShape(shape_A, shape_A_right, right_eye_match_, B, B_right_blink);

	shape_A.discard(eyelash_);
	shape_A.saveObj("C:/avatar/0823_test/shape_A_exp_no_lash.obj");
	shape_A_left.discard(eyelash_);
	shape_A_left.saveObj("C:/avatar/0823_test/shape_A_left_exp_no_lash.obj");
	shape_A_right.discard(eyelash_);
	shape_A_right.saveObj("C:/avatar/0823_test/shape_A_right_exp_no_lash.obj");

	B.saveObj("C:/avatar/0823_test/B.obj");
	B.discard(eyelash_);
	B.saveObj("C:/avatar/0823_test/B_no_lash.obj");

	B_left_blink.saveObj("C:/avatar/0823_test/B_left_blink.obj");
	B_left_blink.discard(eyelash_);
	B_left_blink.saveObj("C:/avatar/0823_test/B_left_blink_no_lash.obj");

	B_right_blink.saveObj("C:/avatar/0823_test/B_right_blink.obj");
	B_right_blink.discard(eyelash_);
	B_right_blink.saveObj("C:/avatar/0823_test/B_right_blink_no_lash.obj");
}
