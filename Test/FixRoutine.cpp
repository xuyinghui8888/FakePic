#include "FixRoutine.h"
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
#include "../RecMesh/RecShapeMesh.h"
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

using namespace CGP;

void FIXROUTINE::putIsvToDst(const cstr& src_path, const cstr& dst_path)
{
	MeshCompress src = src_path;
	MeshCompress dst = dst_path;
	MeshCompress src_res = src;

	intVec vert_idx(src.n_vertex_, 0);
	iota(vert_idx.begin(), vert_idx.end(), 0);

	double scale;
	float3E scale_center, translate;
	float3E dir = float3E(0, 1, 1);
	MeshTools::putSrcToDst(src, vert_idx, dst, vert_idx, src_res, scale, scale_center, translate);
	//saving test, test pass
	src_res.saveObj(src_path);
}

void FIXROUTINE::putIsvToDst(const cstr& src_path, const cstr& src_eyes_path, const cstr& dst_path)
{
	MeshCompress src = src_path;
	MeshCompress dst = dst_path;
	MeshCompress src_res = src;
	MeshCompress src_eyes = src_eyes_path;

	intVec vert_idx(src.n_vertex_, 0);
	iota(vert_idx.begin(), vert_idx.end(), 0);

	double scale;
	float3E scale_center, translate;
	float3E dir = float3E(0, 1, 1);
	MeshTools::putSrcToDst(src, vert_idx, dst, vert_idx, src_res, scale, scale_center, translate);
	RT::scaleAndTranslateInPlace(scale, scale_center, translate, src_eyes.pos_);
	//saving test, test pass
	src_res.saveObj(src_path);
	src_eyes.saveObj(src_eyes_path);
}

void FIXROUTINE::putIsvToDstWrapper()
{	

	FIXROUTINE::putIsvToDst(
		"D:/dota210317/0330_tenet_isv/head.obj",
		"D:/dota210317/0330_tenet_isv/eyes.obj",
		"D:/dota210305/0312_isv_234/234.png.obj"
	);


	FIXROUTINE::putIsvToDst(
			"D:/multiPack/guijie_version_isv/guijie_v16.obj",
			"D:/multiPack/guijie_version_isv/guijie_v16_eyes.obj",
			"D:/multiPack/guijie_version_ori/guijie_v3.obj"
		);




		FIXROUTINE::putIsvToDst(
			"D:/multiPack/guijie_version_isv/guijie_v17.obj",
			"D:/multiPack/guijie_version_isv/guijie_v17_eyes.obj",
			"D:/multiPack/guijie_version_ori/guijie_v3.obj"
		);	
	
	
	
	
	FIXROUTINE::putIsvToDst(
		"D:/dota210305/0316_close_mouth/head.obj",
		"D:/dota210305/0316_close_mouth/eyes.obj",
		"D:/multiPack/guijie_version/guijie_v4.obj"
	);
	
	
	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v13.obj",
		"D:/multiPack/guijie_version_isv/guijie_v13_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v14.obj",
		"D:/multiPack/guijie_version_isv/guijie_v14_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"

	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v15.obj",
		"D:/multiPack/guijie_version_isv/guijie_v15_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"

	);	
	
	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v10.obj",
		"D:/multiPack/guijie_version_isv/guijie_v10_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v11.obj",
		"D:/multiPack/guijie_version_isv/guijie_v11_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v12.obj",
		"D:/multiPack/guijie_version_isv/guijie_v12_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v9.obj",
		"D:/multiPack/guijie_version_isv/guijie_v9_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);


	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v5.obj",
		"D:/multiPack/guijie_version_isv/guijie_v5_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v6.obj",
		"D:/multiPack/guijie_version_isv/guijie_v6_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v7.obj",
		"D:/multiPack/guijie_version_isv/guijie_v7_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);

	FIXROUTINE::putIsvToDst(
		"D:/multiPack/guijie_version_isv/guijie_v9.obj",
		"D:/multiPack/guijie_version_isv/guijie_v9_eyes.obj",
		"D:/multiPack/guijie_version_ori/guijie_v3.obj"
	);
}

void FIXROUTINE::eyebrowTypeTest()
{
	//PREPARE::prepareEyebrow();
	TESTFUNCTION::eyebrowType();
	//TESTFUNCTION::turnFaceSegIntoToPic();
	TESTFUNCTION::testSegResult();
	TESTFUNCTION::testImageSimilarityLandmark();
	TESTFUNCTION::testSegResult();
	//PREPARE::prepareEyebrowMask();
}

void FIXROUTINE::swapUV()
{
#if 1
	cstr in_file = "D:/multiPack/guijie_version_isv_ori/";
	cstr out_file = "D:/multiPack/guijie_version_isv/";
#else
	cstr in_file = "D:/dota210317/0317_v16/";
	cstr out_file = "D:/dota210317/0317_v16/quad/";
#endif
	cstrVec dst_uv =
	{
		"D:/dotaPTA/0304_fix_uv/head.obj",
		"D:/dotaPTA/0304_fix_uv/eyes.obj",
	};
	cstrVec in_names = FILEIO::getFolderFiles(in_file, ".obj");
	for (auto i : in_names)
	{
		MeshCompress iter_in(in_file + i);
		std::string out_replace = out_file + i;
		MeshCompress obj_template;
		cstr raw_name = FILEIO::getFileNameWithoutExt(i);
		if (iter_in.n_vertex_ == 4230)
		{
			obj_template.loadOri(dst_uv[0]);
			obj_template.pos_ = iter_in.pos_;
			obj_template.setGOption(raw_name);
			obj_template.saveOri(out_replace);
		}
		else if (iter_in.n_vertex_ == 966)
		{
			obj_template.loadOri(dst_uv[1]);
			obj_template.pos_ = iter_in.pos_;
			obj_template.setGOption(raw_name);
			obj_template.saveOri(out_replace);			
		}
		else if (iter_in.n_vertex_ == 154)
		{
			//obj_template.loadOri(quad_mesh_base[2]);
			LOG(ERROR) << "no template match" << std::endl;
		}
		else if (iter_in.n_vertex_ == 237)
		{
			//obj_template.loadOri(quad_mesh_base[3]);
			LOG(ERROR) << "no template match" << std::endl;
		}
		else
		{
			LOG(ERROR) << "no template match" << std::endl;
		}

	}


#if 0   
	//prev test

	EASYGUIJIE::replaceUV("D:/dota210219/0302_isv/rep_uv/guijie_v5.obj",
		"D:/multiPack/0118_template_color/head.obj");

	EASYGUIJIE::replaceUV("D:/dota210219/0302_isv/rep_uv/guijie_v6.obj",
		"D:/multiPack/0118_template_color/head.obj");

	EASYGUIJIE::replaceUV("D:/dota210219/0302_isv/rep_uv/guijie_v7.obj",
		"D:/multiPack/0118_template_color/head.obj");
#endif

}

void FIXROUTINE::swapUVInPlace(const cstr& root)
{
	cstrVec dst_uv =
	{
		"D:/dotaPTA/0304_fix_uv/head.obj",
		"D:/dotaPTA/0304_fix_uv/eyes.obj",
	};
	cstrVec in_names = FILEIO::getFolderFiles(root, ".obj");
	for (auto i : in_names)
	{
		MeshCompress iter_in(root + i);
		std::string out_replace = root + i;
		MeshCompress obj_template;
		cstr raw_name = FILEIO::getFileNameWithoutExt(i);
		if (iter_in.n_vertex_ == 4230)
		{
			obj_template.loadOri(dst_uv[0]);
			obj_template.pos_ = iter_in.pos_;
			obj_template.setGOption(raw_name);
			obj_template.saveOri(out_replace);
		}
		else if (iter_in.n_vertex_ == 966)
		{
			obj_template.loadOri(dst_uv[1]);
			obj_template.pos_ = iter_in.pos_;
			obj_template.setGOption(raw_name);
			obj_template.saveOri(out_replace);
		}
		else if (iter_in.n_vertex_ == 154)
		{
			//obj_template.loadOri(quad_mesh_base[2]);
			LOG(ERROR) << "no template match" << std::endl;
		}
		else if (iter_in.n_vertex_ == 237)
		{
			//obj_template.loadOri(quad_mesh_base[3]);
			LOG(ERROR) << "no template match" << std::endl;
		}
		else
		{
			LOG(ERROR) << "no template match" << std::endl;
		}

	}


#if 0   
	//prev test

	EASYGUIJIE::replaceUV("D:/dota210219/0302_isv/rep_uv/guijie_v5.obj",
		"D:/multiPack/0118_template_color/head.obj");

	EASYGUIJIE::replaceUV("D:/dota210219/0302_isv/rep_uv/guijie_v6.obj",
		"D:/multiPack/0118_template_color/head.obj");

	EASYGUIJIE::replaceUV("D:/dota210219/0302_isv/rep_uv/guijie_v7.obj",
		"D:/multiPack/0118_template_color/head.obj");
#endif

}

void FIXROUTINE::faceGenerationV3()
{
	TESTFUNCTION::serverOptV3FromServer();
}

void FIXROUTINE::testCTO()
{
	// get for CTO blendshape
	//PREPARE::putCTOModelToZero();
	//PREPARE::preparePolyWinkModel();
	//PREPARE::getCTOBlendshape();
	//PREPARE::moveCTOBlendshapeToZero();
}

void FIXROUTINE::faceGenerationV2()
{
	//PREPARE::prepareGuijieV3Data();
	//PREPARE::prepare3dmmAndBsCoefV2();
	//PREPARE::prepareTest3dmmAndBs("D:/avatar/0909_test/");

	//TinyTool::getSysLandmarkPoint("D:/avatar/exp_server_config/guijie_sys_tensor/",
	//	"config.json", "D:/data_20July/0717_guijie_landmark/", "guijie_hand.txt",
	//	"sys_68.txt", "guijie_68_sys.txt");

	//TESTFUNCTION::serverOptV2FromServer();
	//TESTFUNCTION::serverOptV2();
}

void FIXROUTINE::getFaceDataReady()
{
	//这部分不需要生成3dmm+faceid系数，由于本地和服务器处理不一，使用服务器生成的值进行
	//std::string root = "D:/data/server_pack/";
	//JsonData json_data;
	//JsonHelper::initData(root, "config.json", json_data);
	//std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	//std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();
	//TESTFUNCTION::testArcFace(ptr_const_var, ptr_res_var, 
	//	"D:/data/testPic/unit_test/", "5_faceid.png", "5_faceid.txt");
	cstr img_root = "D:/data/testPic/v201118/";
	cstr id_result = "D:/data/testPic/v201118_id/";
	int n_type = 159;
	//PREPARE::prepareTest3dmmAndBs(img_root);
	PREPARE::prepareGuijieV3Data(img_root, id_result, n_type);
}

void FIXROUTINE::fixGuijieV4()
{
	TOPTRANSFER::transferSimDiff(
		"D:/dota210202/0203_v3/mean_face.obj",
		"D:/dota210202/0203_v4/guijie_v4.obj",
		"D:/dota210202/0203_v3/id/",
		"D:/dota210202/0203_v4/id/"
	);

	TOPTRANSFER::transferSimDiff(
		"D:/dota210202/0203_v3/mean_face.obj",
		"D:/dota210202/0203_v4/guijie_v4.obj",
		"D:/dota210202/0203_v3/exp/",
		"D:/dota210202/0203_v4/exp/"
	);

	TOPTRANSFER::transferSimDiff(
		"D:/dota210202/0203_v3/mean_face.obj",
		"D:/dota210202/0203_v4/guijie_v4.obj",
		"D:/dota210202/0203_v3/sound/",
		"D:/dota210202/0203_v4/sound/"
	);
}

void FIXROUTINE::fixFalingwen()
{
	TOPTRANSFER::transferSimDiff(
		"D:/dota210202/0203_v4/head_flw.obj",
		"D:/dota210121/0201_isv_head/mean_face.obj",
		"D:/dota210121/0201_isv_head/id/",
		"D:/dota210202/0203_v3/id/"
	);

	TOPTRANSFER::transferSimDiff(
		"D:/dota210202/0203_v4/head_flw.obj",
		"D:/dota210121/0201_isv_head/mean_face.obj",
		"D:/dota210121/0201_isv_head/exp/",
		"D:/dota210202/0203_v3/exp/"
	);

	TOPTRANSFER::transferSimDiff(
		"D:/dota210202/0203_v4/head_flw.obj",
		"D:/dota210121/0201_isv_head/mean_face.obj",
		"D:/dota210121/0201_isv_head/sound/",
		"D:/dota210202/0203_v3/sound/"
	);
}

void FIXROUTINE::generateAvatarBatch(const std::shared_ptr<ConstVar> ptr_const_var, std::shared_ptr<ResVar> ptr_res_var)
{
	//std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();
	//json test_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	//ptr_res_var->setInput(test_config);
	std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
	std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
	if (ptr_res_var->model_3dmm_type_ == Type3dmm::MS || ptr_res_var->model_3dmm_type_ == Type3dmm::NR)
	{
		LOG(INFO) << "using ms data." << std::endl;
		ptr_rec_mesh->processImage();
	}
	else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_RAW)
	{
		LOG(INFO) << "This pass should skip." << std::endl;
	}
	else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_CPP)
	{
		ptr_rec_mesh->processImage();
		ptr_rec_mesh->processImage(ptr_const_var->ptr_model->deep3d_nr_);
	}
	else if (ptr_res_var->model_3dmm_type_ == Type3dmm::NR_CPP_RAW)
	{
		ptr_rec_mesh->processImage();
		ptr_rec_mesh->processImage(ptr_const_var->ptr_model->deep3d_nr_);
	}
	else
	{
		LOG(ERROR) << "undefined path for generateAvatarBatch" << std::endl;
	}

	return;	
}

void FIXROUTINE::generateAvatar()
{
	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();

   	 	   
	json test_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	ptr_res_var->setInput(test_config);
	std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
	std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
	ptr_rec_mesh->processImage();

	//special for guijieV1
	cv::Mat img_256 = cv::imread(ptr_res_var->output_dir_ + "input_landmark68.png");
	vecF temp;
	MeshCompress guijie_v1("D:/dota201201/1201_real/tr_04.obj");
	MeshCompress guijie_res = guijie_v1;
	vecF landmark_256;
	FILEIO::loadEigenMat(ptr_res_var->output_dir_ + "landmark_xy_256.txt", landmark_256);
	LOG(INFO) << "landmark_256: " << landmark_256.transpose() << std::endl;
	ptr_rec_mesh->processImage();
	return;
	intVec guijie_68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/guijie_68_sys.txt");
	intVec area_v0 = FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v0.txt");
	intVec eye_lash_pair = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", eye_lash_pair);
	ptr_rec_mesh->fitAWithLandmarksToImage(guijie_v1, img_256, landmark_256, guijie_68, area_v0, eye_lash_pair, guijie_res);
	ptr_rec_mesh->processDeform();
	//ptr_rec_mesh->processBasicQRatio();
	ptr_rec_mesh->processPartRatio();
}

void FIXROUTINE::generateTextureBase()
{
	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();
   	 	   
	json test_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	ptr_res_var->setInput(test_config);
	std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
	std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
	ptr_rec_mesh->processImage();

	//special for guijieV1
	cv::Mat img_256 = cv::imread(ptr_res_var->output_dir_ + "input_landmark68.png");
	vecF temp;
	MeshCompress guijie_v1("D:/dota201201/1201_real/tr_04.obj");
	MeshCompress guijie_res = guijie_v1;
	vecF landmark_256;
	FILEIO::loadEigenMat(ptr_res_var->output_dir_ + "landmark_xy_256.txt", landmark_256);
	LOG(INFO) << "landmark_256: " << landmark_256.transpose() << std::endl;
	intVec guijie_68 = FILEIO::loadIntDynamic("C:/code/expgen_aquila/data/guijie_opt3_data/guijie_68_sys.txt");
	intVec area_v0 = FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v0.txt");
	intVec eye_lash_pair = FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_up_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_down_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_up_match.txt", eye_lash_pair);
	//area_v0 movable 
	//ptr_rec_mesh->processImage();
	ptr_rec_mesh->fitAWithLandmarksToImage(guijie_v1, img_256, landmark_256, guijie_68, area_v0, eye_lash_pair, guijie_res);
	ptr_rec_mesh->processDeform();
	//ptr_rec_mesh->processBasicQRatio();
	ptr_rec_mesh->processPartRatio();
}

void FIXROUTINE::generateAvatarUsingBFM()
{
	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();
	   	 	   
	json test_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	ptr_res_var->setInput(test_config);
	std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
	std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
	ptr_rec_mesh->processImage();

	//special for guijieV1
	cv::Mat img_256 = cv::imread(ptr_res_var->output_dir_ + "input_landmark68.png");
	vecF temp;
	MeshCompress guijie_v1("D:/dota201116/1125_get_texture/guijieV1.obj");
	MeshCompress guijie_res = guijie_v1;
	vecF landmark_256;
	FILEIO::loadEigenMat(ptr_res_var->output_dir_ + "landmark_xy_256.txt", landmark_256);
	LOG(INFO) << "landmark_256: " << landmark_256.transpose() << std::endl;
	intVec guijie_68 = FILEIO::loadIntDynamic("C:/code/expgen_aquila/data/guijie_opt3_data/guijie_68_sys.txt");
	intVec area_v0 = FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v0.txt");
	intVec eye_lash_pair = FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_up_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_down_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_up_match.txt", eye_lash_pair);
	//area_v0 movable 
	ptr_rec_mesh->fitAWithLandmarksToImage(guijie_v1, img_256, landmark_256, guijie_68, area_v0, eye_lash_pair, guijie_res);
	ptr_rec_mesh->processDeform();
	//ptr_rec_mesh->processBasicQRatio();
	ptr_rec_mesh->processPartRatio();
}

void FIXROUTINE::prepareImage()
{
	cstr root = "D:/data/server_pack/guijie_refine/";
	auto file_list = FILEIO::getFolderFiles(root, ".jpg");
	for (auto i : file_list)
	{
		cv::Mat img = cv::imread(root + i);
		boost::filesystem::remove(root + i);
		cstr file_name = FILEIO::getFileNameWithoutExt(i);
		cv::imwrite(root + file_name + ".png", img);
	}
}

void FIXROUTINE::prepareTaobaoHardMap()
{
	//TESTFUNCTION::prepareForTaobao35();
	//TESTFUNCTION::getRatioMeshBasedOn3dmm();
	//TESTFUNCTION::subtractMeshDataTaobao();
	//TESTFUNCTION::calcHardRatio("D:/dota201116/1116_taobao/img/", "_0.jpg", 
	//	"D:/dota201116/1116_taobao/subtract/", "D:/dota201116/1116_taobao/res/", 11);
	//PREPARE::dumpGuijieEyelash();
	TESTFUNCTION::movePartVertexForGuijie();
}

void FIXROUTINE::prepareHardMap()
{
#if 0
	PREPARE::prepare3dmmPartModel();
	TinyTool::getSysLandmarkPoint("D:/dota201010/1027_bfm_part/",
		"config.json", "D:/dota201010/1027_bfm_part/", "bfm_hand.txt",
		"sys_68.txt", "bfm_sys.txt");
#endif
	TESTFUNCTION::subtractMeshData();
	//TESTFUNCTION::prepareForGuijie35();
	TESTFUNCTION::calcHardRatio("D:/dota201010/1026_calc/", "", "D:/dota201010/1027_subdata/", 
		"D:/dota201010/1026_landmark/", 23);
	//TESTFUNCTION::testCalculate();
	TESTFUNCTION::changeOfRatio();
	TESTFUNCTION::getDataFromGuijieV3Pack();
}

void FIXROUTINE::cartoonStyleDemo()
{
	//PREPARE::dragEyes();
	//PREPARE::dragNoseMouth();
	//PREPARE::getLandmarkFromJsonRaw();
	//FIXROUTINE::generateTextureBase();
	FIXROUTINE::generateAvatar();
	//FIXROUTINE::getCartoonTexture();
	//TOPTRANSFER::getFWHToGuijieNoLash();
	//PREPARE::transferUVFwhToGuijie();
	//PREPARE::transferUVGuijieToFwh();

}

void FIXROUTINE::getCartoonV2Style(const cstr& input_img, const cstr& output_result)
{
	json init_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	LOG(WARNING) << "direct change json file data " << std::endl;
	init_config["input_image_"] = input_img;
	init_config["output_dir_"] = output_result;
	FILEIO::saveJson("D:/code/cgPlayground/config.json", init_config);
	LOG(WARNING) << "end change json file data " << std::endl;
	FIXROUTINE::generateAvatar();
	TOPTRANSFER::guijieToFWHInstance(output_result);
	TOPTRANSFER::transferSimDiff(output_result, "nricp.obj");
	EASYGUIJIE::transformEyesToMesh(output_result, "local_deform.obj");
	TOPTRANSFER::localDeform(output_result);
	EASYGUIJIE::getEyebrow(output_result);
}

void FIXROUTINE::getNRDemo()
{
	cstr root_res = "D:/dota210202/0203_v5/";
	cstr B = "D:/dota210202/0203_cheek/guijie_v5.obj";
	FIXROUTINE::getCartoonV2Style("D:/data/testPic/cartoon/cartoonPair/01_cartoon_pair.png", root_res + "res_01/", B);
	FIXROUTINE::getCartoonV2Style("D:/data/testPic/cartoon/cartoonPair/05_cartoon_pair.png", root_res + "res_05/", B);
	FIXROUTINE::getCartoonV2Style("D:/data/testPic/cartoon/cartoonPair/08_cartoon_pair.png", root_res + "res_08/", B);
	FIXROUTINE::getCartoonV2Style("D:/data/testPic/cartoon/cartoonPair/11_cartoon_pair.png", root_res + "res_11/", B);
	FIXROUTINE::getCartoonV2Style("D:/data/testPic/cartoon/cartoonPair/26_cartoon_pair.png", root_res + "res_26/", B);
}

void FIXROUTINE::getCartoonV2Style(const cstr& input_img, const cstr& output_result, const cstr& B)
{
	MeshCompress input_B = B;
	json init_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	LOG(WARNING) << "direct change json file data " << std::endl;
	init_config["input_image_"] = input_img;
	init_config["output_dir_"] = output_result;
	FILEIO::saveJson("D:/code/cgPlayground/config.json", init_config);
	LOG(WARNING) << "end change json file data " << std::endl;
	FIXROUTINE::generateAvatar();
	TOPTRANSFER::guijieToFWHInstance(output_result);
	TOPTRANSFER::transferSimDiff(output_result, "nricp.obj", B);
	TOPTRANSFER::localDeform(output_result, B);
	EASYGUIJIE::transformEyesToMesh(output_result, "local_deform.obj");
	EASYGUIJIE::getEyebrow(output_result);
}

void FIXROUTINE::getDesignTop()
{
	TESTFUNCTION::blendTop();
	TESTFUNCTION::calcTopRatio();
}

void FIXROUTINE::guijieExp()
{
	TESTFUNCTION::serverGenExp();	
	//TESTFUNCTION::serverGenExpFromMesh();
}

void FIXROUTINE::getCartoonTexture()
{
	cstr cur_root = "D:/dota201104/1104_cartoon/1_res/";

	Projection test_256(600, 600, 127.5, 127.5);
	floatVec landmark_xy;
	FILEIO::loadFixSizeEigenFormat(cur_root + "landmark_xy_256.txt", landmark_xy);
	MeshCompress fwh_head(cur_root + "bfm_lap_sys.obj");
	intVec landmark_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	int n_num = landmark_xy.size()*0.5;
#if 1
	intSet used_list = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
		39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,54,55,56,57,58,59 };
#else
	intSet used_list = { 36,37,38,39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,54,55,56,57,58,59 };
#endif
	for (int i = 0; i < n_num; i++)
	{
		if (used_list.count(i) && i < 17)
		{
			img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
			float3E pos = fwh_head.pos_[landmark_idx[i]];
			obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
		}
		else
		{
			for (int iter = 0; iter < 3; iter++)
			{
				img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
				float3E pos = fwh_head.pos_[landmark_idx[i]];
				obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
			}
		}
	}
	cvMatD rvec, tvec, r_matrix;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
	cv::solvePnP(obj_points, img_points, test_256.intrisic_,
		dist_coeffs, rvec, tvec);
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = cv::imread(cur_root + "input_landmark68.png");
	float3Vec dst_pos;
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = fwh_head.pos_[landmark_idx[i]];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		LOG(INFO) << "in_camera: " << in_camera << std::endl;
		cvMatD in_image = test_256.intrisic_*in_camera;
		LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		LOG(INFO) << "img_x: " << img_x << std::endl;
		LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);
		//green opt
		//fix z
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - test_256.cx_)*GETD(in_camera_fix, 2, 0) / test_256.fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - test_256.cy_)*GETD(in_camera_fix, 2, 0) / test_256.fy_);
		cvMatD in_image_fix = test_256.intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		if (i < 17)
		{
			dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
		}
		float img_x_fix = GETD(in_image_fix, 0, 0) / GETD(in_image_fix, 2, 0);
		float img_y_fix = GETD(in_image_fix, 1, 0) / GETD(in_image_fix, 2, 0);
		cv::circle(canvas, cv::Point2f(img_x_fix, img_y_fix), 2, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite(cur_root + "2d_3d.png", canvas);
	LaplacianDeform to_image;
	float3Vec dst_select;
	intVec land_select;
	intSet select_idx = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	for (int i : select_idx)
	{
		dst_select.push_back(dst_pos[i]);
		land_select.push_back(landmark_idx[i]);
	}
	intVec move_pos = FILEIO::loadIntDynamic("D:/data/server_pack/taobao_project/move_idx.txt");
	intSet move_set(move_pos.begin(), move_pos.end());
	intVec fix_pos;
	for (size_t i = 0; i < fwh_head.n_vertex_; i++)
	{
		if (!move_set.count(i))
		{
			fix_pos.push_back(i);
		}
	}
	to_image.init(fwh_head, land_select, fix_pos);
	to_image.deform(dst_select, fwh_head.pos_);
	fwh_head.saveObj(cur_root + "fwh_to_image.obj");
}

void FIXROUTINE::getBFMtoNeuRender()
{
	//TESTFUNCTION::getDiscardMappingFromMeshes();
	//注意 deep_3d_base 和 deep_3d_clean在空间中的位置是一致的
	Tensor bfm_all;
	JsonHelper::initData("D:/dota201116/1124_bfm09/deep_3d_base/", "config.json", bfm_all);
	bfm_all.save("D:/dota201116/1124_bfm09/deep_3d_trans/");

	MeshCompress all("D:/dota201116/1124_bfm09/deep_3d_base/pca_-1.obj");
	MeshCompress mean_from_bfm_all_tensor = all;
	MeshCompress part("D:/dota201116/1124_bfm09/deep_3d_clean/pca_-1.obj");
	intVec eye = FILEIO::loadIntDynamic("D:/dota201116/1124_bfm09/eye.txt");
	intVec landmark = FILEIO::loadIntDynamic("D:/dota201116/1124_bfm09/bfm_sys.txt");
	all.discard(eye);
	MeshCompress result_all, result_test;
	double scale;
	float3E translate;
	MeshTools::putSrcToDst(all, landmark, part, landmark, result_all, scale, translate);
	//transform
	int vert_3_size = bfm_all.template_obj_.n_vertex_ * 3;
	floatVec mean_ori(bfm_all.data_.begin(), bfm_all.data_.begin() + vert_3_size);
	SG::safeMemcpy(mean_from_bfm_all_tensor.pos_.data(), mean_ori.data(), vert_3_size * sizeof(float));
	mean_from_bfm_all_tensor.saveObj("D:/dota201116/1124_bfm09/mean_from_bfm_all_tensor.obj");
	result_all.saveObj("D:/dota201116/1124_bfm09/all_trans.obj");
	MeshTools::putSrcWithScaleTranslate(all, scale, translate, result_test);
	result_test.saveObj("D:/dota201116/1124_bfm09/all_trans_test.obj");
}

void FIXROUTINE::selectGuijieVertex()
{
	//v0 - v1 - v2 + 特征点+对称
	MeshSysFinder guijie;
	JsonHelper::initData("D:/avatar/exp_server_config/guijie_sys_tensor/", "config.json", guijie);
	//loading data
	intVec reserve = FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/part_info/left_eye_3.txt");
	FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/part_info/right_eye_3.txt", reserve);
	FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/part_info/mouth.txt", reserve);
	FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/part_info/nose.txt", reserve);

	intSet area_v0 = TDST::vecToSet(FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v0.txt"));
	intSet area_v1 = TDST::vecToSet(FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v1.txt"));
	intSet area_v2 = TDST::vecToSet(FILEIO::loadIntDynamic("D:/dota201116/1125_guijie_select_points/v2.txt"));

	//available area
	intVec area_ava;
	intVec area_mid;
	for (int i : area_v0)
	{
		if (!area_v1.count(i) && !area_v2.count(i) && guijie.template_obj_.pos_[i].x()>0.001)
		{
			area_ava.push_back(i);
		}
		else if (!area_v1.count(i) && !area_v2.count(i) && guijie.template_obj_.pos_[i].x() > -0.001)
		{
			area_mid.push_back(i);
		}
	}

	FILEIO::saveDynamic("D:/dota201116/1125_guijie_select_points/v_ava_left.txt", area_ava, ",");
	int total_num = 300;
	int mid_num = total_num / (1.0*area_mid.size() * 2 + 1.0*area_ava.size() * 2)*area_mid.size() / 2 * 2;
	int part_left = (total_num - mid_num) / 2;
#if 0
	//get one side(left) and random 150
	std::random_shuffle(area_mid.begin(), area_mid.end());
	std::random_shuffle(area_ava.begin(), area_ava.end());	

	intVec left_idx(area_ava.begin(), area_ava.begin() + part_left);
	intVec right_idx;
	for (int i : left_idx)
	{
		right_idx.push_back(guijie.getSysId(i));
	}
	intVec mid_idx(area_mid.begin(), area_mid.begin() + mid_num);
	intVec result = mid_idx;
	result.insert(result.end(), left_idx.begin(), left_idx.end());
	result.insert(result.end(), right_idx.begin(), right_idx.end());
#else
	int mid_skip = area_mid.size() / mid_num;
	intVec result;
	for (int i = 0; i < mid_num; i++)
	{
		result.push_back(area_mid[mid_skip*i]);
	}
	int left_skip = area_ava.size() / part_left;
	for (int i = 0; i < part_left; i++)
	{
		result.push_back(area_ava[left_skip*i]);
		result.push_back(guijie.getSysId(area_ava[left_skip*i]));
	}
#endif
	FILEIO::saveDynamic("D:/dota201116/1125_guijie_select_points/v_300.txt", result, ",");
}

void FIXROUTINE::getTextureBase()
{
	MeshCompress bfm_base("D:/data/server_pack/deep_3d_base/pca_-1.obj");
	floatVec mean_color(bfm_base.n_vertex_ * 3, 0);
	srand((unsigned)time(NULL));
#pragma omp parallel for
	for (int i = 0; i < bfm_base.n_vertex_ * 3; i++)
	{
		mean_color[i] = rand() % 255;
	}
	FILEIO::saveToBinary("D:/dota201201/1216_texturebase/mean_bin.txt", mean_color, bfm_base.n_vertex_ * 3 * sizeof(float));
	FILEIO::saveDynamic("D:/dota201201/1216_texturebase/mean_raw.txt", mean_color, ",");

}
