#include "prepare.h"
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
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ShapeVar.h"
#include "../Config/ResVar.h"
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

using namespace CGP;

void PREPARE::prepareData(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	//prepareTaobaoBS(const_var, res_var);
	//prepareTaobaoBSTensor(const_var, res_var);
	//projectTaobaoToBsCoef(const_var, res_var);
	//prepareTaobao(const_var, res_var);
	prepareMesh(const_var, res_var);
	prepareID(const_var, res_var);
}

void PREPARE::prepareTaobaoBS(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	cstr obj_from = "D:/data/0701_taobaobs/raw/";
	cstr obj_to = "D:/data/0701_taobaobs/pca/";
	SG::needPath(obj_to);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_from, ".obj");

	std::string config_root = "D:/data/star/config/";
	std::string marker = config_root + "fwh_tb.cons";
	intVec src_point = FILEIO::loadIntDynamic("D:/data/0701_taobaobs/select_idx_back.txt");
	src_point.resize(42);
	//discard
	intVec discard = FILEIO::loadIntDynamic(config_root + "skip_2.txt");
	MeshCompress template_obj(config_root + "0_uv.obj");
	MeshCompress with_uv = template_obj;

	cstrVec order_map;
	order_map.push_back("mean_face");
	//auto i cause bugs in != or {} init bugs
	for (cstr i : folder_file)
	{
		cstr file_name = obj_from + i;
		if (!SG::isExist(file_name))
		{
			break;
		}
		MeshCompress src_obj(file_name);
		LOG(INFO) << "i: " << i << std::endl;
		LOG(INFO) << "i != \"mean_face.obj\" " << (i != "mean_face.obj") << std::endl;
		if (i != "mean_face.obj")
		{
			src_obj.flipYZ();
			cstrVec split_file_name;
			FILEIO::splitString(i, split_file_name, '.');
			order_map.push_back(split_file_name[0]);
		}
		else
		{
			//src_obj.rotateX();
		}
		src_obj.discard(discard);
		float3Vec dst_pos, src_pos;
		src_obj.getSlice(src_point, src_pos);
		template_obj.getSlice(src_point, dst_pos);
		float scale = RT::getScale(src_pos, dst_pos);
		RT::scaleInPlace(scale, src_obj.pos_);
		src_obj.getSlice(src_point, src_pos);
		//src_obj.saveObj(cur_root + std::to_string(i) + "_scale.obj");
		LOG(INFO) << "scale: " << scale << std::endl;
		float3E translate;
		RT::getTranslate(src_pos, dst_pos, translate);
		LOG(INFO) << "translate: " << translate.transpose() << std::endl;
		RT::translateInPlace(translate, src_obj.pos_);
		with_uv.replaceVertexBasedData(src_obj);
		with_uv.saveObj(obj_to + "" + i);
	}
	FILEIO::saveFixSize(obj_to + "json_name.txt", order_map, "\n");
}

void PREPARE::prepareBSTensor(const cstr& obj_from, const cstr& obj_to, bool if_mean_zero)
{
	//cstr obj_from = "D:/data/0701_taobaobs/pca/";
	//cstr obj_to = "D:/data/0701_taobaobs/tensor/";
	cstrVec ordered_map;
	FILEIO::loadFixSize(obj_from + "json_name.txt", ordered_map);
	SG::needPath(obj_to);
	int num = ordered_map.size();
	MeshCompress template_mesh(obj_from + ordered_map[0] + ".obj");
	floatVec mean_raw(template_mesh.n_vertex_ * 3, 0);
	if (if_mean_zero)
	{
		SG::safeMemset(mean_raw.data(), sizeof(float)*template_mesh.n_vertex_ * 3);
	}
	else
	{
		SG::safeMemcpy(mean_raw.data(), template_mesh.pos_.data(), sizeof(float)*template_mesh.n_vertex_ * 3);
	}

	floatVec eigen_value(num - 1, 1.0);
	FILEIO::saveToBinary(obj_to + "mean.bin", mean_raw);
	FILEIO::saveToBinary(obj_to + "eigen_value.bin", eigen_value);
	floatVec data_raw(num* template_mesh.n_vertex_ * 3);
	if (if_mean_zero)
	{
		SG::safeMemset(mean_raw.data(), sizeof(float)*template_mesh.n_vertex_ * 3);
		SG::safeMemset(data_raw.data(), sizeof(float)*template_mesh.n_vertex_ * 3);
	}
	else
	{
		for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
		{
			for (int j = 0; j < 3; j++) {
				mean_raw[iter_vertex * 3 + j] = template_mesh.pos_[iter_vertex](j);
				data_raw[iter_vertex * 3 + j] = template_mesh.pos_[iter_vertex](j);
			}
		}
	}

	int shift = template_mesh.n_vertex_ * 3;
	for (int i = 1; i < num; i++)
	{
		MeshCompress iter_input(obj_from + ordered_map[i] + ".obj");
#pragma omp parallel for 
		for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
		{
			for (int j = 0; j < 3; j++) {
				data_raw[i * shift + iter_vertex * 3 + j] = iter_input.pos_[iter_vertex](j)
					- data_raw[iter_vertex * 3 + j];
			}
		}
	}
	template_mesh.saveObj(obj_to + "mean.obj");
	FILEIO::saveToBinary(obj_to + "pca.bin", data_raw);
	json config;
	config["template_"] = "mean.obj";
	config["n_id_"] = ordered_map.size();
	config["n_exp_"] = 1;
	config["n_data_type_"] = 5;
	config["tensor_file_"] = "pca.bin";
	config["eigen_value_"] = "eigen_value.bin";
	FILEIO::saveJson(obj_to + "config.json", config);
}

void PREPARE::projectTaobaoToBsCoef(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	cstr obj_from = "D:/data_20July/0722_coef/";
	cstr obj_to = "D:/data_20July/0722_coef/";
	SG::needPath(obj_to);
	Tensor taobao_tensor;
	JsonHelper::initData("D:/data/0701_taobaobs/tensor/", "config.json", taobao_tensor);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_from, ".obj");
	cstrVec ordered_map;
	FILEIO::loadFixSize("D:/data/0701_taobaobs/pca/json_name.txt", ordered_map);
	for (int i = 0; i < folder_file.size(); i++)
	{
		cstr file_name = obj_from + folder_file[i];
		if (!SG::isExist(file_name))
		{
			break;
		}
		MeshCompress src_obj(file_name);
		//int weight = 0.1;
		//floatVec reg(taobao_tensor.n_id_ - 1, weight);
		//vecF coef;
		//taobao_tensor.fitID(src_obj.pos_, reg, coef);
		//floatVec proj_res_roi = taobao_tensor.interpretID(coef);
		//SG::safeMemcpy(src_obj.pos_.data(), proj_res_roi.data(), proj_res_roi.size() * sizeof(float));
		//src_obj.saveObj(obj_to + folder_file[i]);
		//ceres solver
		ceres::Problem fitting_pca;
		doubleVec scale = {1.0};
		doubleVec translate(3,0), shape_coef(97,0);
		shape_coef[0] = 1.0;
		floatX2Vec pca_refactor(src_obj.n_vertex_, floatVec(taobao_tensor.n_id_ * 3,0));
		int n_vertex = src_obj.n_vertex_;
		for (int iter_vertex = 0; iter_vertex < src_obj.n_vertex_; iter_vertex++)
		{
			for (int iter_id = 0; iter_id < taobao_tensor.n_id_; iter_id++)
			{
				for (int iter_dim = 0; iter_dim < 3; iter_dim++)
				{

					pca_refactor[iter_vertex][iter_id * 3 + iter_dim] = taobao_tensor.data_[iter_id*n_vertex * 3 + iter_vertex*3 + iter_dim];
				}
			}		
		}
#if 0
		for (int iter_vertex = 0; iter_vertex < src_obj.n_vertex_; iter_vertex++)
		{
			const int n_pca = taobao_tensor.n_id_;

			ceres::CostFunction* cost_function =
				new ceres::NumericDiffCostFunction<PCACost,
				ceres::CENTRAL,
				1 /* scale */,
				1,
				3 /* traslate*/,
				97 /*pca value*/>
				(new PCACost(pca_refactor[iter_vertex], src_obj.pos_[iter_vertex], taobao_tensor.n_id_, src_obj.n_vertex_, iter_vertex));
			fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());
	}
#else
			ceres::CostFunction* cost_function =
				new ceres::NumericDiffCostFunction<PCAAreaCost,
				ceres::CENTRAL,
				1779*3 /* scale */,
				1,
				3 /* traslate*/,
				97 /*pca value*/>
				(new PCAAreaCost(taobao_tensor.data_, src_obj.pos_, taobao_tensor.n_id_, src_obj.n_vertex_));
			fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());
#endif

		
		shape_coef[0] = 1;

		fitting_pca.SetParameterUpperBound(&scale[0], 0, 1 + 0.2); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&scale[0], 0, 1 - 0.2); // t_z has to be negative

		fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1+1e-6); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1-1e-6); // t_z has to be negative		
		for (int iter_coef = 1; iter_coef < 97; iter_coef++)
		{
			fitting_pca.SetParameterUpperBound(&shape_coef[0], iter_coef, 1); // t_z has to be negative
			fitting_pca.SetParameterLowerBound(&shape_coef[0], iter_coef, 0); // t_z has to be negative
		}
		ceres::Solver::Options solver_options;
		solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		//solver_options.num_threads = 8;
		solver_options.minimizer_progress_to_stdout = true;
		solver_options.max_num_iterations = 15;
		ceres::Solver::Summary solver_summary;
		Solve(solver_options, &fitting_pca, &solver_summary);
		std::cout << solver_summary.BriefReport() << "\n";
			   
		vecF ceres_coef(96);
		for (int iter_pca = 1; iter_pca < 97; iter_pca++)
		{
			ceres_coef[iter_pca - 1] = shape_coef[iter_pca];
		}

		LOG(INFO) << "ceres pca: " << shape_coef[0] << std::endl << ceres_coef.transpose() << std::endl;
		floatVec proj_res_roi = taobao_tensor.interpretID(ceres_coef);
		SG::safeMemcpy(src_obj.pos_.data(), proj_res_roi.data(), proj_res_roi.size() * sizeof(float));
		src_obj.saveObj(obj_to + folder_file[i]+"_ceres.obj");
		json json_coef;
		json_coef["mean_face"] = 1.0;

		for (int iter_json = 1; iter_json < ordered_map.size(); iter_json++)
		{
			json_coef[ordered_map[iter_json]] = ceres_coef[iter_json - 1];
		}
		std::ofstream out_json(obj_to + folder_file[i] + ".json");
		out_json << json_coef << std::endl;
		out_json.close();
	}



}

void PREPARE::prepareTaobao(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	cstr obj_from = "D:/data/star/uni_raw/";
	cstr obj_to = "D:/data/star/uni/";
	SG::needPath(obj_to);

	std::string config_root = "D:/data/star/config/";
	std::string marker = config_root + "fwh_tb.cons";
	intVec dst_point = FILEIO::loadIntDynamic(config_root + "fwh_landark_sys.txt");
	intVec src_point = FILEIO::loadIntDynamic(config_root + "tb_land_sys.txt");
	dst_point.resize(42);
	src_point.resize(42);
	//discard
	intVec discard = FILEIO::loadIntDynamic(config_root + "skip_2.txt");
	MeshCompress template_obj(config_root + "0_uv.obj");
	MeshCompress with_uv = template_obj;

	for (int i = 0; i < 100; i++)
	{
		cstr file_name = obj_from + "" + std::to_string(i) + ".obj";
		if (!SG::isExist(file_name))
		{
			break;
		}
		MeshCompress src_obj(file_name);
		src_obj.discard(discard);
		float3Vec dst_pos, src_pos;
		src_obj.getSlice(src_point, src_pos);
		template_obj.getSlice(src_point, dst_pos);
		float scale = RT::getScale(src_pos, dst_pos);
		RT::scaleInPlace(scale, src_obj.pos_);
		src_obj.getSlice(src_point, src_pos);
		//src_obj.saveObj(cur_root + std::to_string(i) + "_scale.obj");
		LOG(INFO) << "scale: " << scale << std::endl;
		float3E translate;
		RT::getTranslate(src_pos, dst_pos, translate);
		LOG(INFO) << "translate: " << translate.transpose() << std::endl;
		RT::translateInPlace(translate, src_obj.pos_);
		with_uv.replaceVertexBasedData(src_obj);
		with_uv.saveObj(obj_to + "" + std::to_string(i) + ".obj");
	}

}

void PREPARE::prepareMesh(const std::shared_ptr<ConstVar> ptr_const_var, std::shared_ptr<ResVar> ptr_res_var)
{
	cstr img_root = "D:/data/star/uni_raw/";
	cstr result = "D:/data/star/uni_pp_0806/";
	cstr gender = "male";
	SG::needPath(result);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(img_root, cstrVec{ ".jpg", ".png" });
	for (int i = 0; i < folder_file.size(); i++)
	{
		LOG(INFO) << "starting for image: " << folder_file[i] << std::endl;
		cstrVec file_name_split;
		FILEIO::splitString(folder_file[i], file_name_split, '.');
		json test_config;
		test_config["input_image_"] = (img_root + folder_file[i]);
		test_config["output_dir_"] = (result + "/" + file_name_split[0] + "_");
		test_config["is_debug_"] = true;
		test_config["gender_"] = gender;
		test_config["pp_type_"] = 2;
		ptr_res_var->setInput(test_config);
		std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
		std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
		ptr_rec_mesh->processImage();
	}
}

void PREPARE::prepareID(const std::shared_ptr<ConstVar> ptr_const_var, std::shared_ptr<ResVar> ptr_res_var)
{
	cstr id_root = "D:/data/star/uni_pp/";
	cstr result = "D:/data/star/uni_pp/";
	SG::needPath(result);
	int max_id = 100;	
	int max_pic = 10;
	floatVec ids;
	floatVec all_id;
	for (int i = 0; i < max_id; i++)
	{
		cstr cur_0 = id_root + std::to_string(i) + "_" + std::to_string(0) + "_faceid.txt";
		if (!SG::isExist(cur_0))
		{
			break;
		}
		LOG(INFO) << "starting for image: " << i << std::endl;
		for (int iter_pic = 0; iter_pic < max_pic; iter_pic++)
		{
			cstr cur_image = id_root + std::to_string(i) + "_" + std::to_string(iter_pic) + "_faceid.txt";
			if (SG::isExist(cur_image))
			{
				LOG(INFO) << "get id from: " << cur_image << std::endl;
				floatVec temp_vec = { 1.0f*i, 1.0f*i, 1.0f*iter_pic};
				FILEIO::loadEigenMatToVector(cur_image, temp_vec, true);
				all_id.insert(all_id.end(), temp_vec.begin(), temp_vec.end());
			}
			else
			{
				break;
			}
		}
	}
	FILEIO::saveToBinary(result + "id.bin", all_id);
	json config;
	config["n_id_"] = all_id.size() / 515;
	config["n_dim_"] = 515;
	config["file_"] = "id.bin";
	FILEIO::saveJson(result + "config.json", config);
}

void PREPARE::prepareIDV3(const cstr& id_root, const cstr& result, int max_id)
{
	//cstr id_root = "D:/avatar/guijie_opt3_data/guijie_star/";
	//cstr result = "D:/avatar/guijie_opt3_data/guijie_id/";
	SG::needPath(result);
	//int max_id = 100;
	int max_pic = 10;
	floatVec ids;
	floatVec all_id;
	for (int i = 0; i < max_id; i++)
	{
		bool only_one = false;
		cstr cur_0 = id_root + std::to_string(i) + "_" + std::to_string(0) + "_faceid.txt";
		if (!SG::isExist(cur_0))
		{
			cur_0 = id_root + std::to_string(i) + "_faceid.txt";
			if (!SG::isExist(cur_0))
			{
				break;
			}
			else
			{
				only_one = true;
			}
		}
		LOG(INFO) << "starting for image: " << i << std::endl;

		if (only_one)
		{
			cstr cur_image = id_root + std::to_string(i) + "_faceid.txt";
			if (SG::isExist(cur_image))
			{
				LOG(INFO) << "get id from: " << cur_image << std::endl;
				floatVec temp_vec = { 1.0f*i, 1.0f*i, 1.0f*0 };
				FILEIO::loadEigenMatToVector(cur_image, 512, 1, temp_vec, true);
				all_id.insert(all_id.end(), temp_vec.begin(), temp_vec.end());
			}
			else
			{
				break;
			}
		}
		else
		{
			for (int iter_pic = 0; iter_pic < max_pic; iter_pic++)
			{
				cstr cur_image = id_root + std::to_string(i) + "_" + std::to_string(iter_pic) + "_faceid.txt";
				if (SG::isExist(cur_image))
				{
					LOG(INFO) << "get id from: " << cur_image << std::endl;
					floatVec temp_vec = { 1.0f*i, 1.0f*i, 1.0f*iter_pic };
					FILEIO::loadEigenMatToVector(cur_image, 512, 1, temp_vec, true);
					all_id.insert(all_id.end(), temp_vec.begin(), temp_vec.end());
				}
				else
				{
					break;
				}
			}
		}
	}
	FILEIO::saveToBinary(result + "id.bin", all_id);
	json config;
	config["n_id_"] = all_id.size() / 515;
	config["n_dim_"] = 515;
	config["file_"] = "id.bin";
	FILEIO::saveJson(result + "config.json", config);
}

void PREPARE::projectGuijieToBsCoefOnce()
{
	//TODO change of adding differ position
	cstr obj_from = "D:/avatar/0722_eye_clip/test/";
	cstr obj_to = "D:/avatar/0727_project_00/proj/";
	SG::needPath(obj_to);
	Tensor taobao_tensor;
	JsonHelper::initData("D:/avatar/0727_eye_clip/pca/", "config.json", taobao_tensor);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_from, ".obj");
	cstrVec ordered_map;
	FILEIO::loadFixSize("D:/avatar/0727_eye_clip/raw/json_name.txt", ordered_map);
	intVec whole_eye = FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/left_eye.txt");
	intVec down_eye = FILEIO::loadIntDynamic("D:/avatar/0727_eye_clip/roi/left_down_eye.txt");
	intSet down_eye_set(down_eye.begin(), down_eye.end());
	intVec guijie_left_eye_roi;
	for (int i : whole_eye)
	{
		if (down_eye_set.count(i))
		{
			guijie_left_eye_roi.push_back(i);
		}
	}
	for (int i = 0; i < folder_file.size(); i++)
	{
		cstr file_name = obj_from + folder_file[i];
		if (!SG::isExist(file_name))
		{
			break;
		}
		MeshCompress src_obj(file_name);
		MeshCompress src_obj_back = src_obj;
		ceres::Problem fitting_pca;
		doubleVec scale = { 1.0 };
		doubleVec translate(3, 0), shape_coef(14, 0);
		shape_coef[0] = 1.0;
		
		int n_vertex = src_obj.n_vertex_;
		int num = ordered_map.size();
		const int roi_size_3 = guijie_left_eye_roi.size()*3;
		ceres::CostFunction* cost_function =
			new ceres::NumericDiffCostFunction<PCAAreaCost,
			ceres::CENTRAL,
			(557-262)*3 /* scale */,
			1,
			3 /* traslate*/,
			14 /*pca value*/>
			(new PCAAreaCost(taobao_tensor.data_, src_obj.pos_, taobao_tensor.n_id_, src_obj.n_vertex_, guijie_left_eye_roi));
		fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());

		shape_coef[0] = 1;

		fitting_pca.SetParameterUpperBound(&scale[0], 0, 1 + 1e-6); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&scale[0], 0, 1 - 1e-6); // t_z has to be negative

		fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1 + 1e-6); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1 - 1e-6); // t_z has to be negative		
		for (int iter_coef = 1; iter_coef < 14; iter_coef++)
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

		vecF ceres_coef(num-1);
		for (int iter_pca = 1; iter_pca < num; iter_pca++)
		{
			ceres_coef[iter_pca - 1] = shape_coef[iter_pca];
		}

		LOG(INFO) << "ceres pca: " << shape_coef[0] << std::endl << ceres_coef.transpose() << std::endl;
		floatVec proj_res_roi = taobao_tensor.interpretID(ceres_coef);
		SG::safeMemcpy(src_obj.pos_.data(), proj_res_roi.data(), proj_res_roi.size() * sizeof(float));
		src_obj.saveObj(obj_to + folder_file[i] + "_ceres.obj");
		//diff bs
		MeshCompress face_mean("D:/avatar/0722_eye_clip/raw/mean.obj");
		json json_coef;
		json_coef["mean_face"] = 1.0;
		MeshCompress diff = face_mean;
		double mag_c = 2.0;
#if 1
#pragma omp parallel for
		for (int iter_vertex = 0; iter_vertex < guijie_left_eye_roi.size(); iter_vertex++)
		{
			int idx_vertex = guijie_left_eye_roi[iter_vertex];
			diff.pos_[idx_vertex] += mag_c*(src_obj_back.pos_[idx_vertex] - src_obj.pos_[idx_vertex]);
		}
#else
#pragma omp parallel for
		for (int iter_vertex = 0; iter_vertex < diff.pos_.size(); iter_vertex++)
		{
			diff.pos_[iter_vertex] += mag_c * (src_obj_back.pos_[iter_vertex] - src_obj.pos_[iter_vertex]);
		}
#endif
		diff.saveObj(obj_to + folder_file[i] + "_diff.obj");
		for (int iter_json = 1; iter_json < ordered_map.size(); iter_json++)
		{
			json_coef[ordered_map[iter_json]] = ceres_coef[iter_json - 1];
		}
		std::ofstream out_json(obj_to + folder_file[i] + ".json");
		out_json << json_coef << std::endl;
		out_json.close();
	}



}

void PREPARE::getExpFixAndMovePoints()
{
	cstrVec root = 
	{
		"D:/avatar/exp_server_config/ani_exp_pca/",
		"D:/avatar/exp_server_config/ani_eye_pca/",
		"D:/avatar/exp_server_config/ani_sound_pca/",
		"D:/avatar/exp_server_config/ani_test_pca/",
	};
	double thres = 1e-2;
	for (auto i: root)
	{
		Tensor temp;
		JsonHelper::initData(i, "config.json", temp);
		matI move_idx(temp.n_id_, temp.template_obj_.n_vertex_);
		move_idx.setConstant(0);
		cstrVec name;
		FILEIO::loadFixSize(i + "json_name.txt", name);
//#pragma omp parallel for
		for (int iter_dim = 1; iter_dim  < temp.n_id_; iter_dim ++)
		{
			int count = 0;
			intVec res_vertex;
			for (int iter_vertex = 0; iter_vertex < temp.template_obj_.n_vertex_; iter_vertex++)
			{
				//ÅÐ¶ÏÊÇ·ñÒÆ¶¯
				int idx_m = iter_vertex * 3;
				int idx_start = iter_dim * temp.template_obj_.n_vertex_ * 3 + iter_vertex * 3;
				float3E mean_v = temp.template_obj_.pos_[iter_vertex];
				float3E iter_v = float3E(temp.data_[idx_start], temp.data_[idx_start + 1], temp.data_[idx_start + 2]);
				if ((iter_v- mean_v).norm() > thres)
				{
					move_idx(iter_dim, iter_vertex) = 1;
					res_vertex.push_back(iter_vertex);
					count++;
				}
			}
			LOG(INFO) << "id: " << iter_dim << ", move vertex: " << count << std::endl;
			std::cout << name[iter_dim] << std::endl;
			for (int iter_vertex : res_vertex)
			{
				std::cout << iter_vertex << ",";
			}
			std::cout << std::endl;
		}
		FILEIO::saveEigenDynamic(i + "move_idx.txt", move_idx);
	}
}

void PREPARE::prepareExpGen()
{
	//PREPARE::getExpFixAndMovePoints();
	//BsGenerate get_close_eye;
	//get_close_eye.generateCloseEye();
	//TinyTool::renameFiles();
	//TinyTool::getMatchingFromUnityToMayaGuijie();
	//TESTFUNCTION::testBVLS();
	//TESTFUNCTION::testDeformTranSame();

	//TinyTool::getSysID1to1("D:/data_20July/0717_guijie_sys_tensor/", "config.json", "D:/data/server_pack/guijie_deform_pack/",
	//	"right_eye_match_top3.txt", "left_eye_match_top3.txt");
	//TinyTool::getSysID1to1("D:/data_20July/0717_guijie_sys_tensor/", "config.json", "D:/data/server_pack/guijie_deform_pack/",
	//	"rotate_eye_left.txt", "rotate_eye_right.txt");

	//cstrVec json_vec = {"eye_danfeng.json", "eye_round.json", "eye_taohua.json", "eye_xiachui.json", "eye_xingyan.json"};
	//cstrVec json_vec = { "chunjiaoxia.json", "hou.json", "lunkuochang.json", "tuoyuanchun.json", "zhai.json"};
	//cstrVec json_vec = { "huyang.json", "shizi.json", "suantou.json", "xiaoqiao.json", "xiaoqiao.json", "xila.json", "xuandan.json", "yingzui.json"};
	//cstrVec json_vec = {"fangA.json", "fangB.json", "putong.json", "yuanA.json", "yuanB.json"};

	//TinyTool::getFolderNamesToJson("D:/avatar/exp_server_config/ani_exp/");
	//TinyTool::getFolderNamesToJson("D:/avatar/exp_server_config/ani_eye/");
	//TinyTool::getFolderNamesToJson("D:/avatar/exp_server_config/ani_sound/");
	//TinyTool::getFolderNamesToJson("D:/avatar/exp_server_config/Ani_test/");

	//prepare();
	//TinyTool::getTaobaoLips();
	//TinyTool::getObjToJson("D:/avatar/0805_exp/res_EyeLowerDown/", "eye_delta.json", "D:/avatar/0805_exp/raw/EyeLowerDown.obj");
	//TinyTool::skeletonChange();
	//TinyTool::getSysID1to1("D:/data_20July/0717_guijie_sys_tensor/", "config.json", "D:/data/server_pack/guijie_deform_pack/",	"left_eye.txt", "right_eye.txt");



	//TESTFUNCTION::testLandmarkGuided();
	//TinyTool::skeletonChange();

	//TinyTool::getMeshFileNameToJson("D:/avatar/0716_bs_nl/", "name.json");
	//TinyTool::getSysLandmarkPoint("D:/data/0619_00/star_clean/",
	//	"config.json", "D:/data_20July/0729_taobao_68/", "taobao_68_hand.txt",
	//	"sys_68.txt", "taobao_68_sys.txt");


	//direct using tools
	//TinyTool::getMeshSysInfo("D:/data_20July/0717_guijie/", "mean.obj");
	//TinyTool::getEyelash("D:/avatar/guijie/");
	//TinyTool::getEyelash("D:/avatar/guijie/");
	//TinyTool::resizeTestingImageSize("D:/avatar/testing/");
	//TinyTool::getObjToJson("D:/avatar/0728_test/", "eye_ani.json");
	//system("pause");
	//PREPARE::prepareBSTensor("D:/avatar/0727_eye_clip/raw/", "D:/avatar/0727_eye_clip/pca/");

	//TinyTool::getMirrorID("D:/data_20July/0717_guijie/", "config.json", "D:/avatar/0727_eye_clip/roi/",	"left_eye.txt", "right_eye.txt");
	//TESTFUNCTION::testRatioFace();
	//TESTFUNCTION::testDeformTranSame();
	//TESTFUNCTION::testBatchDeformTranSame();
	//TESTFUNCTION::testProjectionUsingCache();
	//TESTFUNCTION::testSim();
	//TESTFUNCTION::testRatioFace();
	//system("pause");

}

void PREPARE::prepare3dmmAndBsCoefV2()
{
	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();

	cstr img_root = "D:/avatar/guijie_opt2_data/";
	cstr gender = "male";

	cstrVec ordered_map;
	Tensor shape_tensor;
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt2_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt2_data/";
	JsonHelper::initData(img_root + exp_config["shape_bs"].get<cstr>(), "config.json", shape_tensor);
	FILEIO::loadFixSize(img_root + exp_config["shape_bs"].get<cstr>() + "json_name.txt", ordered_map);

	int num = ordered_map.size();
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(img_root, cstrVec{ ".jpg", ".png" }, true);
	for (int i = 0; i < folder_file.size(); i++)
	{
		LOG(INFO) << "starting for image: " << folder_file[i] << std::endl;
		cstrVec file_name_split;
		FILEIO::splitString(folder_file[i], file_name_split, '.');
		json test_config;
		test_config["input_image_"] = folder_file[i];
		test_config["output_dir_"] = file_name_split[0] + "_";
		test_config["is_debug_"] = false;
		test_config["gender_"] = gender;
		test_config["pp_type_"] = 2;
		ptr_res_var->setInput(test_config);
		std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
		std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
		ptr_rec_mesh->processImage(true);
		FILEIO::saveEigenDynamic(file_name_split[0] + "_" + "3dmm.txt", ptr_rec_mesh->coef_3dmm_);
		MeshCompress B;
		ptr_rec_mesh->get3dmmMesh(ptr_rec_mesh->coef_3dmm_, B);
		B.saveObj(file_name_split[0] + "_" + "3dmm.obj");
		//deal with bs
		
		json json_bs = FILEIO::loadJson(file_name_split[0] + ".json");
		vecD random_coef(num - 1);
		random_coef.setConstant(0);
		for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
		{
			std::cout << it.key() << " : " << it.value() << "\n";
		}

		for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
		{
			cstrVec split_txt;
			FILEIO::splitString(it.key(), split_txt, '.');
			auto res = std::find(ordered_map.begin(), ordered_map.end(), split_txt.back());
			if (res == ordered_map.end())
			{
				LOG(WARNING) << "bs value not in list." << std::endl;
				LOG(WARNING) << "bs value: " << it.key() << ", " << it.value() << std::endl;
			}
			else
			{
				int find_pos = res - ordered_map.begin();
				random_coef[find_pos - 1] = it.value();
			}
		}
		FILEIO::saveEigenDynamic(file_name_split[0] + "_" + "bs.txt", random_coef);
	}
}

void PREPARE::prepare3dmmAndBsCoefV3(const cstr& img_root, int n_type)
{	
	cstr server_data_pack_root = "D:/avatar/guijie_opt2_data/";
	cstr gender = "male";
	//get for tensor config
	cstrVec ordered_map;
	Tensor shape_tensor;
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt2_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt2_data/";
	JsonHelper::initData(server_data_pack_root + exp_config["shape_bs"].get<cstr>(), "config.json", shape_tensor);
	FILEIO::loadFixSize(server_data_pack_root + exp_config["shape_bs"].get<cstr>() + "json_name.txt", ordered_map);

	int num = ordered_map.size();	
	cstrVec fix_bs = { "eyebrow_Front", "toothWide", "eyes_PupilBig",  "eyes_PupilSmall"};
	//std::string root = "D:/avatar/guijie_opt3_data/guijie_star/";

	for (int i = 0; i < n_type; i++)
	{
		json json_bs = FILEIO::loadJson(img_root + std::to_string(i) + ".json");
		json json_fix;
		vecD random_coef(num - 1);
		random_coef.setConstant(0);
		for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
		{
			std::cout << it.key() << " : " << it.value() << "\n";
		}

		for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
		{
			cstrVec split_txt;
			FILEIO::splitString(it.key(), split_txt, '.');
			auto res = std::find(ordered_map.begin(), ordered_map.end(), split_txt.back());
			if (res == ordered_map.end())
			{
				LOG(WARNING) << "bs value not in list." << std::endl;
				LOG(WARNING) << "bs value: " << it.key() << ", " << it.value() << std::endl;
				auto res_fix = std::find(fix_bs.begin(), fix_bs.end(), split_txt.back());
				if (res_fix != fix_bs.end())
				{
					json_fix[it.key()] = it.value();
				}
			}
			else
			{
				int find_pos = res - ordered_map.begin();
				random_coef[find_pos - 1] = it.value();
			}
		}
		FILEIO::saveEigenDynamic(img_root + std::to_string(i) + "_" + "bs.txt", random_coef);
		FILEIO::saveJson(img_root + std::to_string(i) + "_" + "fix_bs.json", json_fix);
	}

}

void PREPARE::prepareTest3dmmAndBs(const cstr& img_root)
{
	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();

	cstr gender = "male";

	cstrVec ordered_map;
	Tensor shape_tensor;
	cstr config_root = "D:/avatar/guijie_opt2_data/";
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt2_data/config.json");
	exp_config["root"] = config_root;
	JsonHelper::initData(config_root + exp_config["shape_bs"].get<cstr>(), "config.json", shape_tensor);
	FILEIO::loadFixSize(config_root + exp_config["shape_bs"].get<cstr>() + "json_name.txt", ordered_map);

	int num = ordered_map.size();
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(img_root, cstrVec{ ".jpg", ".png" }, true);
	for (int i = 0; i < folder_file.size(); i++)
	{
		LOG(INFO) << "starting for image: " << folder_file[i] << std::endl;
		cstrVec file_name_split;
		FILEIO::splitString(folder_file[i], file_name_split, '.');
		json test_config;
		test_config["input_image_"] = folder_file[i];
		test_config["output_dir_"] = file_name_split[0] + "_";
		test_config["is_debug_"] = false;
		test_config["gender_"] = gender;
		test_config["pp_type_"] = 2;
		ptr_res_var->setInput(test_config);
		std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
		std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
		ptr_rec_mesh->processImage(true);
		FILEIO::saveEigenDynamic(file_name_split[0] + "_" + "3dmm.txt", ptr_rec_mesh->coef_3dmm_);
		MeshCompress B;
		ptr_rec_mesh->get3dmmMesh(ptr_rec_mesh->coef_3dmm_, B);
		B.saveObj(file_name_split[0] + "_" + "3dmm.obj");	
		FILEIO::saveEigenDynamic(file_name_split[0] + "_" + "faceid.txt", ptr_rec_mesh->faceid_);
	}

}

void PREPARE::prepareEyebrow()
{
	cstr raw_data = "D:/avatar/eyebrow_data/eyebrow/";
	cstr raw_root = "D:/avatar/eyebrow_data/";
#if 1

	CGP::cstrVec raw_file = FILEIO::getFolderFiles(raw_data, ".obj");
	//get mean
	MeshCompress mean_obj(raw_data + raw_file[0]);
	for (int i = 1; i < raw_file.size(); i++)
	{
		MeshCompress iter_i(raw_data + raw_file[i]);
		for (int iter_v = 0; iter_v < iter_i.pos_.size(); iter_v++)
		{
			mean_obj.pos_[iter_v] = mean_obj.pos_[iter_v] + iter_i.pos_[iter_v];
		}
	}

	RT::scaleInPlaceNoShift(1.0/raw_file.size(), mean_obj.pos_);
	mean_obj.saveObj(raw_root + "mean.obj");
#endif

	cstr mean_face = "mean.obj";
	cstr dst_folder = raw_root + "eyebrow_tensor/";

	SG::needPath(dst_folder);
	cstrVec order_map;
	order_map.push_back("mean");
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(raw_data, ".obj");
	boost::filesystem::copy_file(raw_root + mean_face, raw_data + "mean.obj", boost::filesystem::copy_option::overwrite_if_exists);
	for (cstr iter_file : folder_file)
	{
		cstrVec split_file_name;
		FILEIO::splitString(iter_file, split_file_name, '.');
		if (split_file_name[0] != "mean" && split_file_name[0] != "mean_face")
		{
			order_map.push_back(split_file_name[0]);
		}
	}
	FILEIO::saveFixSize(raw_data + "json_name.txt", order_map, "\n");
	FILEIO::saveFixSize(dst_folder + "json_name.txt", order_map, "\n");
	PREPARE::prepareBSTensor(raw_data, dst_folder);

}

void PREPARE::prepareGuijieV3Data(const cstr& img_root, const cstr& id_root, int n_type)
{
	PREPARE::prepare3dmmAndBsCoefV3(img_root, n_type);
	PREPARE::prepareIDV3(img_root, id_root, n_type);
}

void PREPARE::prepare3dmmPartModel()
{
	MeshCompress cto_obj("D:/dota201010/1026_calc/0_3dmm.obj");
	cstr cur_root = "D:/dota201010/1027_bfm_part/";
	SysFinder::findSysBasedOnPosOnly(cto_obj, cur_root, EPSCG3, 5 * 1e-3);
}

void PREPARE::getLandmarkFromJsonRaw()
{
	cstr landmark_in = "D:/dota201104/1105_landmark_json/";
	for (int i = 1; i < 8; i++)
	{
		auto json_in = FILEIO::loadJson(landmark_in + std::to_string(i)+".json");
		//LOG(INFO) << json_in["sd_result"]["items"] << std::endl;
		std::vector<json> landmark_json = json_in["sd_result"]["items"].get<std::vector<json>>();
		//LOG(INFO) << "landmark_json[0]: " << landmark_json[0] << std::endl;
		vecF res_in(68 * 2);
		for (int iter_json = 0; iter_json < landmark_json.size(); iter_json++)
		{
			intVec xy = landmark_json[iter_json]["meta"]["geometry"].get<intVec>();
			int idx = landmark_json[iter_json]["properties"]["point_index"];
			res_in[idx * 2] = xy[0];
			res_in[idx * 2 + 1] = xy[1];
		}
		FILEIO::saveEigenDynamic(landmark_in + std::to_string(i) + "_landmark.txt", res_in);
		cv::Mat img = cv::imread(landmark_in + std::to_string(i) + ".jpg");
		cv::Mat img_canvas = ImageUtils::drawKps(img, res_in);
		cv::imshow("img_canvas", img_canvas);
		cv::waitKey(0);
	}
}

void PREPARE::transferUVGuijieToFwh()
{
	cstr root = "D:/dota201104/1105_guijie_fwh_trans/";
	MeshCompress fwh_from(root + "fit_image.obj");
	MeshCompress guijie_to(root + "move_guijie.obj");
	MeshCompress guijie_deform(root + "deform.obj");
	intVec map_uv_fwh_to_guijie = FILEIO::loadIntDynamic(root + "corre_26.000000.txt");
	guijie_to.material_ = fwh_from.material_;
	intX2Vec map_guijie_to_fwh(guijie_to.n_vertex_);
	for (int i = 0; i < map_uv_fwh_to_guijie.size() / 2; i++)
	{
		int fwh_from = map_uv_fwh_to_guijie[2 * i+1];
		int guijie_to = map_uv_fwh_to_guijie[2 * i ];
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
			float cur_dis = (guijie_deform.pos_[i] - fwh_from.pos_[fwh_idx]).norm();
			min_index = min_dis > cur_dis ? fwh_idx : min_index;
			min_dis = DMIN(min_dis, cur_dis);
		}
		if (min_index < 0)
		{
			min_index = -1;
			min_dis = 1e8;
			for (int iter = 0; iter < fwh_from.pos_.size(); iter++)
			{
				int fwh_idx = iter;
				float cur_dis = (guijie_deform.pos_[i] - fwh_from.pos_[fwh_idx]).norm();
				min_index = min_dis > cur_dis ? fwh_idx : min_index;
				min_dis = DMIN(min_dis, cur_dis);
			}
			//min_index = -1;
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

	guijie_to.saveObj(root + "guijie_transfer_uv.obj");
}

void PREPARE::dragNoseMouth()
{
	intVec area_v0 = FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v0.txt");
	intVec eye_lash_pair = FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_up_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_down_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_up_match.txt", eye_lash_pair);

	intVec eye_idx = FILEIO::loadIntDynamic("D:/dota201201/1201_real/eye_region.txt");
	intVec dst_idx = FILEIO::loadIntDynamic("D:/dota201201/1201_real/mouth_nose.txt");
	MeshCompress from_obj("D:/dota201201/1201_real/tr_06_05.obj");
	MeshCompress to_obj("D:/dota201201/1201_real/isv_pos_ref.obj");
	for (int iter = 0; iter  < 11; iter ++)
	{
		MeshCompress res = from_obj;
		float3Vec dst_pos;
		float shift = 1;
		float y_shift = 0.1*iter;
		for (int i : dst_idx)
		{
			float3E ori_vert = from_obj.pos_[i];
			float3E dst_vert = to_obj.pos_[i];
			float3E res_vert = shift * ori_vert + (1 - shift)*dst_vert;
			res_vert.y() = res_vert.y() + y_shift;
			dst_pos.push_back(res_vert);
		}
		LaplacianDeform to_image;
		intVec fix_pos = res.getReverseSelection(area_v0);
		to_image.init(res, dst_idx, fix_pos, eye_lash_pair);
		to_image.deform(dst_pos, res.pos_);
		res.material_.clear();
		res.saveObj("D:/dota201201/1201_real/tr_06_05/" + std::to_string(y_shift) + ".obj");
	}
}

void PREPARE::dragEyes()
{
	intVec area_v0 = FILEIO::loadIntDynamic("D:/dota201201/1201_real/move_eyes.txt");
	intVec eye_lash_pair = FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_up_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_down_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_up_match.txt", eye_lash_pair);
	intVec area_nose_mouth = FILEIO::loadIntDynamic("D:/dota201201/1201_real/move_eyes.txt");
	LaplacianDeform to_image;
	intVec dst_idx = FILEIO::loadIntDynamic("D:/dota201201/1201_real/eye_region.txt");
	for (int iter = 0; iter < 10; iter++)
	{
		MeshCompress res("D:/dota201201/1201_real/re_05.obj");
		float3Vec dst_pos;
		float shift = 0.1*iter;
		for (int i : dst_idx)
		{
			float3E ori_pos = res.pos_[i];
			if (ori_pos.x() < 0)
			{
				ori_pos.x() += shift;
			}
			else
			{
				ori_pos.x() -= shift;
			}
			dst_pos.push_back(ori_pos);
}
		intVec fix_pos = res.getReverseSelection(area_v0);
		to_image.init(res, dst_idx, fix_pos, eye_lash_pair);
		to_image.deform(dst_pos, res.pos_);
		res.saveObj("D:/dota201201/1201_real/trans/" + std::to_string(shift) + ".obj");
	}
}

void PREPARE::putCTOModelToZero()
{
#if 0
	cstr obj_in = "D:/dota201010/1012_cto/sim.obj";
	MeshCompress cto_obj(obj_in);
	intVec fix_ind = FILEIO::loadIntDynamic("D:/dota201010/1012_cto/cto_mid.txt");
	float3Vec pos_mid;
	MeshTools::getSlice(cto_obj.pos_, fix_ind, pos_mid);
	float3E center_mid;
	MeshTools::getCenter(pos_mid, center_mid);
	RT::translateInPlace(-center_mid, cto_obj.pos_);
	cto_obj.saveObj("D:/dota201010/1012_cto/sim_zero.obj");
	for (auto i : fix_ind)
	{
		LOG(INFO) << "pos: " << cto_obj.pos_[i].transpose() << std::endl;
	}

	MeshCompress cto_obj("D:/dota201010/1012_cto/sim_zero.obj");
	cstr cur_root = "D:/dota201010/1012_cto/";
	SysFinder::findSysBasedOnPosOnly(cto_obj, cur_root, EPSCG3, 5 * 1e-3);

	TinyTool::getSysLandmarkPoint("D:/dota201010/1012_cto/",
		"config.json", "D:/dota201010/1012_cto/", "cto_68.txt",
		"sys_68.txt", "cto_68_sys.txt");

#endif

	//move cto model to neutral scale
	MeshCompress cto_obj("D:/dota201010/1012_cto/sim_zero.obj");
	MeshCompress fwh_obj("D:/dota201010/1012_cto/Neutral.obj");
	intVec cto_68_idx = FILEIO::loadIntDynamic("D:/dota201010/1012_cto/cto_68_sys.txt");
	intVec fwh_68_idx = FILEIO::loadIntDynamic("D:/dota201010/1012_cto/fwh_68_sys.txt");
	float3Vec cto_68, fwh_68;
	MeshTools::getSlice(cto_obj.pos_, cto_68_idx, cto_68);
	MeshTools::getSlice(fwh_obj.pos_, fwh_68_idx, fwh_68);
	double scale = RT::getScale(cto_68, fwh_68);
	RT::scaleInPlace(scale, cto_obj.pos_);
	MeshTools::getSlice(cto_obj.pos_, cto_68_idx, cto_68);
	float3E translate;
	RT::getTranslate(cto_68, fwh_68, translate);
	RT::translateInPlace(translate, cto_obj.pos_);
	cto_obj.saveObj("D:/dota201010/1012_cto/cto_rectify.obj");
}

void PREPARE::getCTOBlendshape()
{
	//test for deformation transfer	
	cstr obj_in = "D:/dota201010/1010_refine/";
	cstr obj_out = "D:/dota201010/1010_res/";
	SG::needPath(obj_out);

	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_in, cstrVec{ ".obj" }, false);

	cstr basic = "D:/dota201010/1010_refine/Neutral.obj";
	cstr q = "D:/dota201010/1012_cto/cto_rectify.obj";
	MeshCompress mesh_q(q);
	mesh_q.material_.clear();

	cstr file_exe = "D:/code/deformation-transfer-win/x64/Release/dtrans_cmd.exe";
	cstr corres = "D:/dota201010/1012_cto/A_B.tricorrs";

	for (auto i : folder_file)
	{		
		if (i == "Neutral.obj")
		{
			boost::filesystem::copy_file(q, obj_out + "Neutral.obj",
				boost::filesystem::copy_option::overwrite_if_exists);
		}
		else
		{
			cstr basic_deform = obj_in + i;
			cstr q_deform = obj_out + i;
			SG::exec(file_exe + " " + basic + " " + q + " " + corres + " " + basic_deform + " " + q_deform);
		}
	}
}

void PREPARE::prepareEyebrowMask()
{
	cstr img_in = "D:/avatar/eyebrow_data/img/";
	cstr img_out = "D:/avatar/eyebrow_data/img_mask/";
	SG::needPath(img_out);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(img_in, cstrVec{ ".png" }, false);
	for (int i = 0; i < 3; i++)
	{
		cv::Mat iter_img = cv::imread(img_in + folder_file[i], cv::IMREAD_UNCHANGED);
		cv::resize(iter_img, iter_img, cv::Size(256, 256));
		cv::Mat binary;
		ImageUtils::removeBackFromAlpha(iter_img, binary, 50);
		cv::imwrite(img_out + std::to_string(i) + ".jpg", binary);
	}

	for (int i = 3; i < folder_file.size(); i++)
	{
		cv::Mat iter_img = cv::imread(img_in + folder_file[i], cv::IMREAD_UNCHANGED);
		cv::resize(iter_img, iter_img, cv::Size(256, 256));
		cv::Mat binary;
		ImageUtils::keepRoiValue(iter_img, binary, 0, 200);
		cv::imwrite(img_out + std::to_string(i) + ".jpg", binary);
	}
}

void PREPARE::moveCTOBlendshapeToZero()
{
	//test for deformation transfer	
	cstr obj_in = "D:/dota201010/1010_res/";
	cstr obj_out = "D:/dota201010/1010_res_fix_zero/";
	SG::needPath(obj_out);
	intVec fix_index = FILEIO::loadIntDynamic("D:/dota201010/1010_ori/fixVertex.txt");
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_in, cstrVec{ ".obj" }, false);
	float3E center_neu = float3E(0, 0, 0);
	float3E mass_center = float3E(0, 0, 0);


	for (auto i : folder_file)
	{
		cstrVec raw_name;
		FILEIO::splitString(i, raw_name, '_');
		LOG(INFO) << "raw_name: " << raw_name.back() << std::endl;
		MeshCompress in_obj(obj_in + i);
		in_obj.material_.clear();
		
		float3Vec part;
		float3E center_iter;
		in_obj.getSlice(fix_index, part);
		RT::getCenter(part, center_iter);
		RT::translateInPlace(center_neu - center_iter - mass_center, in_obj.pos_);
	
		in_obj.saveObj(obj_out + raw_name.back());
	}
}

void PREPARE::preparePolyWinkModel()
{
	//test for deformation transfer	
	cstr obj_in = "D:/dota201010/1010_ori/";
	cstr obj_out = "D:/dota201010/1010_refine/";
	SG::needPath(obj_out);
	intVec keep_index = FILEIO::loadIntDynamic("D:/dota201010/1010_ori/keepVertices.txt");
	intVec fix_index = FILEIO::loadIntDynamic("D:/dota201010/1010_ori/fixVertex.txt");
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(obj_in, cstrVec{ ".obj" }, false);
	float3E center_neu = float3E(0,0,0);
	float3E mass_center = float3E(0, 0, 0);


	for (auto i : folder_file)
	{
		cstrVec raw_name;
		FILEIO::splitString(i, raw_name, '_');
		LOG(INFO) << "raw_name: " << raw_name.back() << std::endl;
		MeshCompress in_obj(obj_in + i);
		in_obj.material_.clear();
		//get neutral pos
		if (in_obj.n_vertex_ > 11510)
		{
			in_obj.keepRoi(keep_index);
		}

		if (raw_name.back() == "Neutral.obj")
		{
			float3Vec part;
			in_obj.getSlice(fix_index, part);
			RT::getCenter(part, center_neu);
			RT::getCenter(in_obj.pos_, mass_center);
		}
	
		{
			float3Vec part;
			float3E center_iter;
			in_obj.getSlice(fix_index, part);
			RT::getCenter(part, center_iter);
			RT::translateInPlace(center_neu - center_iter - mass_center, in_obj.pos_);
		}
		in_obj.saveObj(obj_out + raw_name.back());
		//boost::filesystem::copy_file(obj_in + i, obj_out + raw_name.back(),
		//	boost::filesystem::copy_option::overwrite_if_exists);
	}
}