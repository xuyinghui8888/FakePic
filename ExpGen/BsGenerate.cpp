#include "BsGenerate.h"
#include "../Basic/MeshHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../Config/JsonHelper.h"
#ifndef _MINI
#include "../Test/Prepare.h"
#endif

using namespace CGP;
BsGenerate::BsGenerate(const json& config)
{
	if (is_init_ == false)
	{
		init(config);
	}
}

void BsGenerate::init(const json& config)
{
	data_root_ = config["root"].get<cstr>();
	JsonHelper::initData(data_root_ + config["shape_bs"].get<cstr>(), "config.json", shape_tensor_);
	FILEIO::loadFixSize(data_root_ + config["shape_bs"].get<cstr>() + "json_name.txt", ordered_map_);
}

#ifndef _MINI
void BsGenerate::generateEyeTensor()
{

	cstr pca_root = "D:/avatar/0825_close_fix_lash/";
	cstr mean_face = "mean_0804.obj";
	cstrVec src_folder =
	{
		pca_root + "eye/",
		pca_root + "left/",
		pca_root + "right/",
	};

	cstrVec dst_folder =
	{
		pca_root + "eye_pca/",
		pca_root + "left_pca/",
		pca_root + "right_pca/",
	};


	for (int i = 0; i < src_folder.size(); i++)
	{
		SG::needPath(dst_folder[i]);
		cstrVec order_map;
		order_map.push_back("mean_face");
		CGP::cstrVec folder_file = FILEIO::getFolderFiles(src_folder[i], ".obj");
		boost::filesystem::copy_file(pca_root + mean_face, src_folder[i] + "mean_face.obj", boost::filesystem::copy_option::overwrite_if_exists);
		for (cstr iter_file : folder_file)
		{
			cstrVec split_file_name;
			FILEIO::splitString(iter_file, split_file_name, '.');
			if (split_file_name[0] != "mean" && split_file_name[0] != "mean_face")
			{
				order_map.push_back(split_file_name[0]);
			}
		}
		FILEIO::saveFixSize(src_folder[i] + "json_name.txt", order_map, "\n");
		FILEIO::saveFixSize(dst_folder[i] + "json_name.txt", order_map, "\n");
		PREPARE::prepareBSTensor(src_folder[i], dst_folder[i]);
	}
}


void BsGenerate::generateTensor()
{
#if 0
	cstr pca_root = "D:/avatar/0822_guijie_bs/";
	cstr mean_face = "mean_0804.obj";
	cstrVec src_folder =
	{
		"D:/avatar/0822_guijie_bs/bs/",
	};

	cstrVec dst_folder =
	{
		"D:/avatar/0822_guijie_bs/bs_pca/",
	};


	for (int i = 0 ; i < src_folder.size(); i++)
	{
		SG::needPath(dst_folder[i]);
		cstrVec order_map;
		order_map.push_back("mean_face");
		CGP::cstrVec folder_file = FILEIO::getFolderFiles(src_folder[i], ".obj");
		boost::filesystem::copy_file(pca_root + mean_face, src_folder[i] + "mean_face.obj", boost::filesystem::copy_option::overwrite_if_exists);
		for (cstr iter_file : folder_file)
		{
			cstrVec split_file_name;
			FILEIO::splitString(iter_file, split_file_name, '.');
			if (split_file_name[0] != "mean" && split_file_name[0] != "mean_face")
			{
				order_map.push_back(split_file_name[0]);
			}		
		}
		FILEIO::saveFixSize(src_folder[i] + "json_name.txt", order_map, "\n");
		FILEIO::saveFixSize(dst_folder[i] + "json_name.txt", order_map, "\n");
		PREPARE::prepareBSTensor(src_folder[i], dst_folder[i]);
	}
#else
	cstr pca_root = "D:/avatar/0822_guijie_bs/";
	cstr mean_face = "mean_0804.obj";
	cstrVec src_folder =
	{
		"D:/avatar/0824_02/ani_exp/",
		"D:/avatar/0824_02/ani_eye/",
		"D:/avatar/0824_02/ani_sound/",
		"D:/avatar/0824_02/ani_test/",
	};

	cstrVec dst_folder =
	{
		"D:/avatar/0824_02/ani_exp_pca/",
		"D:/avatar/0824_02/ani_eye_pca/",
		"D:/avatar/0824_02/ani_sound_pca/",
		"D:/avatar/0824_02/ani_test_pca/",
	};


	for (int i = 0; i < src_folder.size(); i++)
	{
		SG::needPath(dst_folder[i]);
		cstrVec order_map;
		order_map.push_back("mean_face");
		CGP::cstrVec folder_file = FILEIO::getFolderFiles(src_folder[i], ".obj");
		boost::filesystem::copy_file(pca_root + mean_face, src_folder[i] + "mean_face.obj", boost::filesystem::copy_option::overwrite_if_exists);
		for (cstr iter_file : folder_file)
		{
			cstrVec split_file_name;
			FILEIO::splitString(iter_file, split_file_name, '.');
			if (split_file_name[0] != "mean" && split_file_name[0] != "mean_face")
			{
				order_map.push_back(split_file_name[0]);
			}
		}
		FILEIO::saveFixSize(src_folder[i] + "json_name.txt", order_map, "\n");
		FILEIO::saveFixSize(dst_folder[i] + "json_name.txt", order_map, "\n");
		PREPARE::prepareBSTensor(src_folder[i], dst_folder[i], true);
	}
#endif
}

void BsGenerate::generateFace()
{
	cstr out_folder = "D:/avatar/0805_exp/test_case/";
	cstrVec tensor_folder =
	{
		"D:/avatar/0805_exp/pca/eye_bs/",
		"D:/avatar/0805_exp/pca/face_bs/",
		"D:/avatar/0805_exp/pca/mouth_bs/",
		"D:/avatar/0805_exp/pca/nose_bs/",
		"D:/avatar/0805_exp/pca/all_bs/",
	};

	cstrVec src_folder =
	{
		"D:/avatar/0805_exp/eye_bs/",
		"D:/avatar/0805_exp/face_bs/",
		"D:/avatar/0805_exp/mouth_bs/",
		"D:/avatar/0805_exp/nose_bs/",
		"D:/avatar/0805_exp/all_bs/",
	};

	int part_id = 4;
	Tensor guijie_part;
	JsonHelper::initData(tensor_folder[part_id], "config.json", guijie_part);
	cstrVec ordered_map;
	FILEIO::loadFixSize(src_folder[part_id]+ "json_name.txt", ordered_map);

	int num = ordered_map.size();
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 0.5);
	vecF random_coef(num - 1);
	random_coef.setConstant(0);
	cstrVec set_name = {"EyeUpperDown", "EyeLowerUp"};
	floatVec set_value = {1.0, 1.0};

	for (int i = 0; i < set_name.size(); i++)
	{
		auto res = std::find(ordered_map.begin(), ordered_map.end(), set_name[i]);
		if (res == ordered_map.end())
		{
			LOG(ERROR) << "set value not found." << std::endl;
		}
		int find_pos = res - ordered_map.begin();
		//LOG(INFO) << "find pos: " << set_name[i] << ", " << find_pos << std::endl;
		random_coef[find_pos - 1] = set_value[i];
	}
	MeshCompress face_random = guijie_part.template_obj_;
	floatVec proj_random = guijie_part.interpretID(random_coef);
	SG::safeMemcpy(face_random.pos_.data(), proj_random.data(), proj_random.size() * sizeof(float));
	face_random.saveObj(out_folder + "face_random.obj");
	json out_coef;
	for (int i = 0; i < set_name.size(); i++)
	{
		out_coef[set_name[i]] = set_value[i];
	}
	FILEIO::saveJson(out_folder + "face_random.json", out_coef);
}

void BsGenerate::generateCloseEye()
{
	cstr pca_root = "D:/avatar/0822_close/";
	cstr mean_face = "mean_0804.obj";
	cstrVec src_folder =
	{
		pca_root + "left/",
		pca_root + "right/",
		pca_root + "eye/",
	};

	cstrVec dst_folder =
	{
		pca_root + "left_pca/",
		pca_root + "right_pca/",
		pca_root + "eye_pca/",
	};


	for (int i = 0; i < src_folder.size(); i++)
	{
		SG::needPath(dst_folder[i]);
		cstrVec order_map;
		order_map.push_back("mean_face");
		CGP::cstrVec folder_file = FILEIO::getFolderFiles(src_folder[i], ".obj");
		boost::filesystem::copy_file(pca_root + mean_face, src_folder[i] + "mean_face.obj", boost::filesystem::copy_option::overwrite_if_exists);
		for (cstr iter_file : folder_file)
		{
			cstrVec split_file_name;
			FILEIO::splitString(iter_file, split_file_name, '.');
			if (split_file_name[0] != "mean" && split_file_name[0] != "mean_face")
			{
				order_map.push_back(split_file_name[0]);
			}
		}
		FILEIO::saveFixSize(src_folder[i] + "json_name.txt", order_map, "\n");
		FILEIO::saveFixSize(dst_folder[i] + "json_name.txt", order_map, "\n");
		PREPARE::prepareBSTensor(src_folder[i], dst_folder[i]);
	}
}

#endif

void BsGenerate::generateFace(const cstr& root, const cstr& json_file, int part_id)
{
	cstrVec tensor_folder =
	{
		"D:/avatar/0818_guijie_bs/pca/all_bs/",
	};

	cstrVec src_folder =
	{
		"D:/avatar/0818_guijie_bs/all_bs/",
	};

	Tensor guijie_part;
	JsonHelper::initData(tensor_folder[part_id], "config.json", guijie_part);
	cstrVec ordered_map;
	FILEIO::loadFixSize(src_folder[part_id] + "json_name.txt", ordered_map);

	int num = ordered_map.size();
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 0.5);
	vecF random_coef(num - 1);
	random_coef.setConstant(0);

	json json_bs = FILEIO::loadJson(root + json_file);

	for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it) {
		std::cout << it.key() << " : " << it.value() << "\n";
	}

	for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
	{
		auto res = std::find(ordered_map.begin(), ordered_map.end(), it.key());
		if (res == ordered_map.end())
		{
			LOG(ERROR) << "set value not found." << std::endl;
			LOG(ERROR) << "set key/value: " << it.key() << ", " << it.value() << std::endl;
		}
		int find_pos = res - ordered_map.begin();
		random_coef[find_pos - 1] = it.value();
	}
	MeshCompress face_random = guijie_part.template_obj_;
	floatVec proj_random = guijie_part.interpretID(random_coef);
	SG::safeMemcpy(face_random.pos_.data(), proj_random.data(), proj_random.size() * sizeof(float));
	face_random.saveObj(root + FILEIO::getFileNameWithoutExt(json_file)+".obj");
}

void BsGenerate::generateFace(json& json_bs, MeshCompress& res)
{
	int num = ordered_map_.size();
	vecF random_coef(num - 1);
	random_coef.setConstant(0);
	for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
	{
		std::cout << it.key() << " : " << it.value() << "\n";
	}

	for (json::iterator it = json_bs.begin(); it != json_bs.end(); ++it)
	{
		auto res = std::find(ordered_map_.begin(), ordered_map_.end(), it.key());
		if (res == ordered_map_.end())
		{
			LOG(ERROR) << "set value not found." << std::endl;
			LOG(ERROR) << "set key/value: " << it.key() << ", " << it.value() << std::endl;
		}
		int find_pos = res - ordered_map_.begin();
		random_coef[find_pos - 1] = it.value();
	}
	res = shape_tensor_.template_obj_;
	floatVec proj_random = shape_tensor_.interpretID(random_coef);
	SG::safeMemcpy(res.pos_.data(), proj_random.data(), proj_random.size() * sizeof(float));
}

