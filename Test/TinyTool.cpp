#include "TinyTool.h"
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
#include "../RT/Projection.h"
#include "../Mesh/MeshTools.h"
#include "../Skeleton/Skeleton.h"
#include "../Debug/DebugTools.h"

using namespace CGP;

void TinyTool::generatePlane(const cstr& out_file)
{
	int width = 3;
	int height = 3;
	MeshCompress test;
	float length = 1.0f;
	float gap = 1.0f / width;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			test.pos_.push_back(float3E(gap*x, gap*y, 0));
		}
	}
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 1; x < width; x++)
		{
			int tl = y * height + x - 1;
			int tr = y * height + x;
			int bl = (y + 1)*height + x - 1;
			int br = (y + 1)*height + x;
			test.tri_.push_back(tl);
			test.tri_.push_back(br);
			test.tri_.push_back(bl);

			test.tri_.push_back(tl);
			test.tri_.push_back(tr);
			test.tri_.push_back(br);
		}
	}
	test.update();
	test.saveObj(out_file);
}

void TinyTool::resizeTestingImageSize(const cstr& root)
{
	auto res = FILEIO::getFolderFiles(root, FILEIO::FILE_TYPE::IMAGE, true);
	for (auto i : res)
	{
		cv::Mat img = cv::imread(i);
		int height = img.rows;
		int width = img.cols;
		int aim_height = 512;
		int aim_width = width * aim_height * 1.0f / height;
		cv::resize(img, img, cv::Size(aim_width, aim_height));
		cv::imwrite(i, img);
	}
}

void TinyTool::getObjToJson(const cstr& root, const cstr& json_file, const cstr& neutral)
{
	MeshCompress mean(neutral);
	std::vector<float> mean_pos(mean.n_vertex_ * 3);
	SG::safeMemcpy(mean_pos.data(), mean.pos_[0].data(), sizeof(float)*mean.n_vertex_ * 3);

	nlohmann::json vertex_value;
	auto res = FILEIO::getFolderFiles(root, FILEIO::FILE_TYPE::MESH, true);
	
	for (auto i : res)
	{
		MeshCompress in_bs(i);
		std::vector<float> in_bs_temp(in_bs.n_vertex_ * 3);
		SG::safeMemcpy(in_bs_temp.data(), in_bs.pos_[0].data(), sizeof(float)*in_bs.n_vertex_ * 3);

		std::transform(in_bs_temp.begin(), in_bs_temp.end(),
			mean_pos.begin(), in_bs_temp.begin(), std::minus<float>());

		vertex_value["n_vertex"] = in_bs.n_vertex_;	
		cstr item_name = FILEIO::getFileNameWithoutExt(i);
		vertex_value[item_name] = in_bs_temp;
	}
	FILEIO::saveJson(root + json_file, vertex_value);
}

void TinyTool::getMeshFileNameToJson(const cstr& root, const cstr& json_file)
{
	nlohmann::json vertex_value;
	auto res = FILEIO::getFolderFiles(root, FILEIO::FILE_TYPE::MESH, true);
	for (auto i : res)
	{
		cstr item_name = FILEIO::getFileNameWithoutExt(i);
		vertex_value[item_name] = 0.0;
	}
	FILEIO::saveJson(root + json_file, vertex_value);
}

void TinyTool::objSafeCheck(const MeshCompress& src)
{
	LOG(INFO) << SG::checkMesh(src.pos_, src.tri_) << std::endl;
}

void TinyTool::objSafeCheck(const cstr& file_pos)
{
	MeshCompress src(file_pos);
	LOG(INFO) << SG::checkMesh(src.pos_, src.tri_) << std::endl;
}

void TinyTool::discardVertexFolder()
{
	cstr src_path = "D:/data/0606_04/";
	intVec discard = FILEIO::loadIntDynamic(src_path + "skip.txt");
	cstr dst_path = "D:/data/0606_05/";
	for (int i = 0; i < 1; i++)
	{
		MeshCompress src_mesh(src_path + std::to_string(i) + ".obj");
		src_mesh.discard(discard);
		float3E center;
		RT::getCenter(src_mesh.pos_, center);
		RT::translateInPlace(-center, src_mesh.pos_);
		src_mesh.saveObj(dst_path + std::to_string(i) + ".obj");
	}
}

void TinyTool::getMeshSysInfo(const cstr& cur_root, const cstr& mesh)
{
	MeshCompress fwh(cur_root + mesh);
	SysFinder::findSysBasedOnPosOnly(fwh, cur_root);
	//SysFinder::findSysBasedOnPosBiMap(fwh, cur_root);
	//SysFinder::findSysBasedOnUV(fwh, res, root);
	//write config json
	json config;
	config["template_"] = mesh;
	config["mid_"] = "mid.txt";
	config["match_"] = "match.txt";
	FILEIO::saveJson(cur_root + "config.json", config);
}

void TinyTool::getSysLandmarkPoint(const cstr& config_root, const cstr& config_json,
	const cstr& root, const cstr& hand_landmark, 
	const cstr& sys_match, const cstr& result)
{
	MeshSysFinder fwh;
	JsonHelper::initData(config_root, config_json, fwh);
	intVec landmark_by_hand = FILEIO::loadIntDynamic(root + hand_landmark);
	intVec sys_info_68 = FILEIO::loadIntDynamic(root + sys_match);
	intVec landmark_sys = landmark_by_hand;

	for (int i = 0; i < landmark_by_hand.size(); i++)
	{
		int vertex_on_mesh = landmark_by_hand[i];
		int sys_landmark_order = sys_info_68[i];
		if (vertex_on_mesh < 0)
		{
			int vertex_on_mesh_sys = landmark_by_hand[sys_landmark_order];
			if (vertex_on_mesh_sys < 0)
			{
				LOG(ERROR) << "error occur, sys and ori both <0." << std::endl;
			}
			else
			{
				vertex_on_mesh = fwh.getSysId(vertex_on_mesh_sys);
				landmark_sys[i] = vertex_on_mesh;
			}
		}
	}
	FILEIO::saveDynamic(root + result, landmark_sys, ",");
}

void TinyTool::getSysIDReduceSame(const cstr& config_root, const cstr& config_json,
	const cstr& root, const cstr& hand_landmark, const cstr& result)
{
	MeshSysFinder fwh;
	JsonHelper::initData(config_root, config_json, fwh);
	intVec landmark_by_hand = FILEIO::loadIntDynamic(root + hand_landmark);
	intVec landmark_sys = landmark_by_hand;
	fwh.getSysIdsInPlace(landmark_sys);	
	FILEIO::saveDynamic(root + result, landmark_sys, ",");
}

void TinyTool::getSysID1to1(const cstr& config_root, const cstr& config_json,
	const cstr& root, const cstr& hand_landmark, const cstr& result)
{
	MeshSysFinder fwh;
	JsonHelper::initData(config_root, config_json, fwh);
	intVec landmark_by_hand = FILEIO::loadIntDynamic(root + hand_landmark);
	intVec landmark_sys = landmark_by_hand;
	fwh.getMirrorIdsInPlace(landmark_sys);
	FILEIO::saveDynamic(root + result, landmark_sys, ",");
}

void TinyTool::getEyelash(const cstr& cur_root)
{
	intVec face = FILEIO::loadIntDynamic(cur_root + "face.txt");
	intVec lash_left_up = FILEIO::loadIntDynamic(cur_root + "left_up_lash.txt");
	intVec lash_left_down = FILEIO::loadIntDynamic(cur_root + "left_down_lash.txt");
	intVec lash_right_up = FILEIO::loadIntDynamic(cur_root + "right_up_lash.txt");
	intVec lash_right_down = FILEIO::loadIntDynamic(cur_root + "right_down_lash.txt");
	//find match
	MeshCompress template_mesh("D:/avatar/0717bs/mean.obj");
	
	intVec lash_left_up_face = MeshTools::getSrcToDstMatch(template_mesh, lash_left_up, face, 1e-3);
	lash_left_up_face = CalcHelper::keepValueBiggerThan(lash_left_up_face, -0.5);
	FILEIO::saveDynamic(cur_root + "left_up_match.txt", lash_left_up_face, ",");

	intVec lash_left_down_face = MeshTools::getSrcToDstMatch(template_mesh, lash_left_down, face, 1e-3);
	lash_left_down_face = CalcHelper::keepValueBiggerThan(lash_left_down_face, -0.5);
	FILEIO::saveDynamic(cur_root + "left_down_match.txt", lash_left_down_face, ",");

	intVec lash_right_up_face = MeshTools::getSrcToDstMatch(template_mesh, lash_right_up, face, 1e-3);
	lash_right_up_face = CalcHelper::keepValueBiggerThan(lash_right_up_face, -0.5);
	FILEIO::saveDynamic(cur_root + "right_up_match.txt", lash_right_up_face, ",");
	
	intVec lash_right_down_face = MeshTools::getSrcToDstMatch(template_mesh, lash_right_down, face, 1e-3);
	lash_right_down_face = CalcHelper::keepValueBiggerThan(lash_right_down_face, -0.5);
	FILEIO::saveDynamic(cur_root + "right_down_match.txt", lash_right_down_face, ",");
}

void TinyTool::renameFiles()
{
	cstr root = "D:/avatar/0818_guijie_bs/";
	auto files = FILEIO::getFolderFiles(root, { ".obj" }, true);
	for (auto i: files)
	{
		if (FILEIO::tail(i, 9) == "Shape.obj")
		{
			auto new_files = FILEIO::dump(i, 9);
			LOG(INFO) << "new_files: " << new_files << std::endl;
			boost::filesystem::rename(i, new_files + ".obj");
		}
	}
}

void TinyTool::getFileNamesToPCAJson(const cstr& folder)
{
	cstrVec order_map;
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(folder, ".obj");
	for (cstr iter_file : folder_file)
	{
		cstrVec split_file_name;
		FILEIO::splitString(iter_file, split_file_name, '.');
		if (split_file_name[0] != "mean" && split_file_name[0] != "mean_face")
		{
			order_map.push_back(split_file_name[0]);
		}
		else
		{
			order_map.insert(order_map.begin(), split_file_name[0]);
		}
	}
	FILEIO::saveFixSize(folder + "json_name.txt", order_map, "\n");
}

void TinyTool::getTaobaoLips()
{
	const cstr& cur_root = "D:/avatar/0806_taobao_lips/";
	intVec lip_up = FILEIO::loadIntDynamic(cur_root + "up.txt");
	intVec lip_down = FILEIO::loadIntDynamic(cur_root + "down.txt");	
	//find match
	MeshCompress template_mesh(cur_root + "q_match.obj");

	intVec lash_left_up_face = MeshTools::getSrcToDstMatch(template_mesh, lip_up, lip_down, 1e-1);
	lash_left_up_face = CalcHelper::keepValueBiggerThan(lash_left_up_face, -0.5);
	FILEIO::saveDynamic(cur_root + "left_up_match.txt", lash_left_up_face, ",");	
}

void TinyTool::getMatchingFromUnityToMayaCube()
{
	MeshCompress maya("D:/data_20July/0730_fbx_1944/pCube1.obj");
	MeshCompress unity("D:/data_20July/0730_fbx_1944/test_mesh.obj");
	intVec res = MeshTools::getMatchBasedOnUVAndPos(unity, maya, 1e-2, 1e-2);
	json match;
	match["unity_vertex"] = unity.n_vertex_;
	match["maya_vertex"] = maya.n_vertex_;
	match["unity_to_maya"] = res;

	//random move and scale
	RT::scaleInPlace(0.5, maya.pos_);
	RT::translateInPlace(float3E(0.1, 0.1, 0.1), maya.pos_);
	MeshCompress maya_bs("D:/data_20July/0730_fbx_1944/pCube2.obj");
	RT::scaleInPlace(0.5, maya_bs.pos_);
	RT::translateInPlace(float3E(0.1, 0.1, 0.1), maya_bs.pos_);

	std::vector<float> in_bs_temp(maya.n_vertex_ * 3);
	SG::safeMemcpy(in_bs_temp.data(), maya.pos_[0].data(), sizeof(float)*maya.n_vertex_ * 3);
	match["pCube1"] = in_bs_temp;
	SG::safeMemcpy(in_bs_temp.data(), maya_bs.pos_[0].data(), sizeof(float)*maya_bs.n_vertex_ * 3);
	match["pCube2"] = in_bs_temp;
	FILEIO::saveJson("D:/data_20July/0730_fbx_1944/unity_to_maya.json", match);
	maya.saveObj("D:/data_20July/0730_fbx_1944/pcube1_shift.obj");
	maya_bs.saveObj("D:/data_20July/0730_fbx_1944/pcube2_shift.obj");
}

void TinyTool::skeletonChange()
{
	Skeleton taobao_skeleton;
	taobao_skeleton.unitInverse();
}

void TinyTool::getMatchingFromUnityToMayaGuijie()
{
	//cstr root = "D:/data_20July/0731_guijie/";
	cstr root = "D:/dota210604/0604_00/";
	cstr maya_prefix = "maya/";
	cstr unity_prefix = "unity/";
	cstrVec maya_fbx = 
	{
		maya_prefix + "Guijie_head.obj", unity_prefix + "Guijie_head.obj",
		maya_prefix + "Guijie_eyes.obj", unity_prefix + "Guijie_eyes.obj",
		maya_prefix + "Guijie_eyebrow.obj", unity_prefix + "Guijie_eyebrow.obj",
		maya_prefix + "Guijie_tooth.obj", unity_prefix + "Guijie_tooth.obj",
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
		MeshCompress maya(root + maya_fbx[2 * i]);
		MeshCompress unity(root + maya_fbx[2 * i + 1]);
		intVec res = MeshTools::getMatchBasedOnUVAndPos(unity, maya, 5*1e-2, 5 * 1e-2);

		match_iter["unity_vertex"] = unity.n_vertex_;
		match_iter["maya_vertex" ] = maya.n_vertex_;
		match_iter["unity_to_maya"] = res;

		match[signal[i]] = match_iter;
		//RT::translateInPlace(translate, maya.pos_);
		//std::vector<float> in_bs_temp(maya.n_vertex_ * 3);
		//SG::safeMemcpy(in_bs_temp.data(), maya.pos_[0].data(), sizeof(float)*maya.n_vertex_ * 3);
		//match[signal[i]] = in_bs_temp;s
		//maya.saveObj(root + maya_fbx[2 * i] + "_transform.obj");
	}	
	FILEIO::saveJson(root + "unity_to_maya_v1.json", match);
}

void TinyTool::getMatchingFromPartToAll()
{
	cstr root = "D:/dota201201/1214_eyebrow_mapping/";
	cstrVec part_all =
	{
		"eyebrow.obj","head.obj",
		//"eyes_maya.obj", "eyes_unity.obj",
		//"eyebrow_maya.obj", "eyebrow_unity.obj",
		//"tooth_maya.obj", "tooth_unity.obj",
	};

	cstrVec signal =
	{
		"eyebrow",
		//"eyes",
		//"eyebrow",
		//"tooth",
	};
	int n_pair = part_all.size() / 2;
	json match;
	for (int i = 0; i < signal.size(); i++)
	{
		MeshCompress part(root + part_all[2 * i]);
		MeshCompress all(root + part_all[2 * i + 1]);
		intVec part_to_all = MeshTools::getMatchBasedOnPosDstToSrc(all, part, 1e-2);

		match["all_vertex_" + signal[i]] = all.n_vertex_;
		match["part_vertex_" + signal[i]] = part.n_vertex_;
		match["part_to_all_" + signal[i]] = part_to_all;

		//move all to part location
		for (int iter_pos = 0; iter_pos < part_to_all.size(); iter_pos++)
		{
			int idx_part = iter_pos;
			int idx_all = part_to_all[idx_part];
			all.pos_[idx_all] = part.pos_[idx_part];
		}		
		all.saveObj(root + part_all[2 * i] + "_fix.obj");
	}
	FILEIO::saveJson(root + "part_to_all.json", match);
}

void TinyTool::getBSMeshFromDeltaV1File()
{
	cstr source_root = "D:/dota201201/1211_hf_mapping/";
	cstrVec base_mesh =
	{
		"head", "head_maya",
		"eyebrow", "eyebrow_maya",
		"eyes", "eyes_maya",
		"tooth", "tooth_maya",
	};
	json delta = FILEIO::loadJson("D:/dota201201/1215_packexp/1215_hf_delta_test_e2.json");
	//typical type ExpGenGuijie----> Exp: string
	//             ExpGenGuijie----> Exp: json

	if (!delta.count("ExpGenGuijie") || !delta["ExpGenGuijie"].count("Exp"))
	{
		LOG(ERROR) << "missing key" << std::endl;
	}

	cstr exp_value = delta["ExpGenGuijie"]["Exp"];
	json exp_value_json = nlohmann::json::parse(exp_value);		
	DebugTools::printJsonStructure(exp_value_json);
	//没法判断下半程的代码，是有问题的
#if 0
	if (typeid(delta["ExpGenGuijie"]["Exp"]) == typeid(cstr))
	{
		LOG(INFO) << "content ExpGenGuijie Exp: string" << std::endl;
	}
	else if (typeid(delta["ExpGenGuijie"]["Exp"]) == typeid(json))
	{
		DebugTools::printJson(delta["ExpGenGuijie"]);
		LOG(INFO) << "content ExpGenGuijie Exp: json" << std::endl;
		json content = delta["ExpGenGuijie"]["Exp"];
		DebugTools::printJson(content);
	}
#endif


	LOG(INFO) << "typeid: " << (typeid(delta["ExpGenGuijie"]["Exp"]) == typeid(cstr)) << std::endl;
	LOG(INFO) << "typeid: " << (typeid(delta["ExpGenGuijie"]["Exp"]) == typeid(json)) << std::endl;
	
	
	LOG(INFO) << delta.count("Exp") << std::endl;
	LOG(INFO) << delta.count("ExpGenGuijie") << std::endl;
	LOG(INFO) << "typeid: " << (typeid(delta["ExpGenGuijie"]) == typeid(cstr)) << std::endl;
	LOG(INFO) << "typeid: " << (typeid(delta["ExpGenGuijie"]) == typeid(json)) << std::endl;
}

void TinyTool::turnJsonToString()
{
	json template_string = FILEIO::loadJson("D:/dota201224/1228_generate_head/json/template.json");
	json template_no_string = FILEIO::loadJson("D:/dota201224/1228_generate_head/json/template.json");
	json res = FILEIO::loadJson("D:/dota201224/bao_delta(1).json");
	template_string["ExpGenGuijie"]["Exp"] = res.dump();
	FILEIO::saveJson("D:/dota201224/zpack_oss_string.json", template_string);
	template_no_string["ExpGenGuijie"]["Exp"] = res;
	FILEIO::saveJson("D:/dota201224/zpack_oss_no_string.json", template_no_string);
}

