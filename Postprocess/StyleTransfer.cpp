#include "StyleTransfer.h"
#include "EasyGuijie.h"
#include "../FileIO/FileIO.h"
#include "../MeshDeform/LapDeformWrapper.h"
#include "../MeshDeform/LaplacianDeformation.h"
#include "../Mapping/Mapping.h"

using namespace CGP;

void STYLETRANSFER::generateAvatarWrapper(const std::shared_ptr<ShapeVar> ptr_const_var, std::shared_ptr<CGP::NRResVar> ptr_res_var)
{
	//std::shared_ptr<NRResVar> ptr_res_var = std::make_shared<NRResVar>();
	//json test_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	//ptr_res_var->setInput(test_config);
	//ptr_res_var->setInput(res_json_config);
	std::shared_ptr<RecShapeMesh> ptr_rec_mesh = std::make_shared<RecShapeMesh>(ptr_const_var, ptr_res_var);
	//std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
	ptr_rec_mesh->processImageMidTerm(false);
	return;
}

void STYLETRANSFER::postProcessForGuijieTexBatch(const std::shared_ptr<ExpGen> exp_ptr, const std::shared_ptr<OptV3Gen> optV3_ptr,
	const std::shared_ptr<ShapeVar> ptr_const_var, std::shared_ptr<NRResVar> ptr_res_var, const json& config)
{
	//json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	cstr root = config["nl_root"].get<cstr>();
	cstr output_dir_ = config["output_dir_"].get<cstr>();
	cstrVec input =
	{
		output_dir_,
	};

	//MeshCompress B = data_root + "0127_sl_head/face_interBS_origin_pos.obj";
	MeshCompress B = root + config["B"].get<cstr>();
	MeshCompress eyes = root + config["eyes"].get<cstr>();

	for (auto i : input)
	{
		if (ptr_res_var->model_3dmm_type_ == Type3dmm::MS)
		{
			LOG(INFO) << "using ms version" << std::endl;
			guijieToFWHInstance(i, config);
			transferSimDiff(i, "nricp.obj", B, exp_ptr, ptr_res_var, config);
		}
		else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR)
		{
			MeshCompress template_mesh = ptr_const_var->ptr_data->nr_tensor_.template_obj_;
			ptr_const_var->ptr_data->nr_tensor_.interpretIDFloat(template_mesh.pos_.data(), ptr_res_var->coef_3dmm_);
			template_mesh.saveObj(i + "nricp.obj");
			transferSimDiff(i, "nricp.obj", B, exp_ptr, ptr_res_var, config);
		}
		else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_RAW)
		{
			LOG(ERROR) << "This path should have passed" << std::endl;
		}
		localDeform(i, B, optV3_ptr, config);
	}

	

}

void STYLETRANSFER::transferSimDiff(const cstr& root, const cstr& obj, const MeshCompress& input_B, const std::shared_ptr<ExpGen> exp_ptr,
	std::shared_ptr<NRResVar> res_ptr, const json& config)
{
	//cstr data_root = "D:/avatar/nl_linux/";
	cstr data_root = config["nl_root"].get<cstr>();
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
	else
	{
		LOG(ERROR) << "undefined data" << std::endl;
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

void STYLETRANSFER::localDeform(const cstr& input_root, const MeshCompress& input_B,
	const std::shared_ptr<OptV3Gen> optV3_ptr, const json& config)
{
	//cstr data_root = "D:/avatar/nl_linux/";
	cstr data_root = config["nl_root"].get<cstr>();
	//test for avatar generation v2

	MeshCompress bfm_template = data_root + "1218_3dmm_test/res/3dmm.obj";
	MeshCompress bfm_input = input_root + "3dmm.obj";
	//MeshCompress guijie_template = "D:/dota201201/1222_fwh_guijie/move_guijie_all.obj";
	MeshCompress guijie_template = input_B;
	MeshCompress guijie_input = input_root + "B_deform.obj";
	MeshCompress guijie_output = guijie_input;

	optV3_ptr->optBasedOn3dmm(bfm_input, bfm_template, guijie_input, guijie_template, config, guijie_output);
	guijie_output.saveObj(input_root + "local_deform.obj");
}


void STYLETRANSFER::guijieToFWHInstance(const cstr& obj_root, const json& config)
{
	//cstr data_root = "D:/multiPack/";
	//cstr data_root = "D:/avatar/nl_linux/";
	cstr data_root = config["nl_root"].get<cstr>();
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

void STYLETRANSFER::generateEyesBrow(const json& config)
{
	//json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
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

void STYLETRANSFER::generateTexDst(const json& config)
{
	//cstr root = "D:/avatar/nl_linux/";
	cstr root = config["nl_root"].get<cstr>();
	//json config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
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

	cv::Mat in_image = cv::imread(root_res + "landmark_256_xy.png");

	MeshCompress obj_base_color = obj_base;
	MeshCompress obj_base_pos = obj_base;
	obj_base_color.vertex_color_ = obj_base_color.pos_;
	intSet render_gv2_set(render_gv2_idx.begin(), render_gv2_idx.end());

	//floatVec light_coefs = FILEIO::loadFloatDynamic(root_res + "light_coefs.txt", ',');

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

	fixTexture(root_res, 1.0, config);
}

void STYLETRANSFER::fixTexture(cstr& root_res, double reg_value, const json& config)
{
	//cstr data_root = "D:/multiPack/";
	//cstr data_root = "D:/avatar/nl_linux/";
	cstr data_root = config["nl_root"].get<cstr>();
	Tensor tex_tensor;
	JsonHelper::initData(data_root + "0118_obj_1e4/tensor/", "config.json", tex_tensor);
	MeshCompress dst = root_res + "dst_normal.obj";
	floatVec reg(tex_tensor.n_id_, reg_value);
	vecD coef;
	tex_tensor.fitID(dst.pos_, reg, coef);

	cstr output_dir_ = config["output_dir_"];
	FILEIO::saveEigenDynamic(output_dir_ + "texture_coef.txt", coef);

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
				cv::Vec3d pixel_res = pixel_buff + pixel_diff * coef[i];
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




