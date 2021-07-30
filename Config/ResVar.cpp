#include "ResVar.h"
#include"../CalcFunction/CalcHelper.h"
#include "../Basic/ToDst.h"
using namespace CGP;

ResVar::ResVar()
{
	init();
}

void ResVar::init()
{
	res_mesh.init();
	res_texture.init();
}

void ResVar::setInput(const json& test_config)
{
	if (!SG::isExist(test_config["input_image_"]))
	{
		LOG(ERROR) << test_config["input_image_"] << " file is not exist." << std::endl;
		return;
	}
	input_image_ = cv::imread(test_config["input_image_"]);
	if (input_image_.empty())
	{
		LOG(ERROR) << "image reading error." << std::endl;
		return;
	}
	is_debug_ = test_config["is_debug_"];
	output_dir_ = test_config["output_dir_"];
	if (test_config["gender_"] == "male")
	{
		gender_ = Gender::MALE;
	}
	else
	{
		gender_ = Gender::FEMALE;
	}

	pp_type_ = PostProcessRoutine(test_config["pp_type_"]);	
	SG::needPath(output_dir_);
	model_3dmm_type_ = Type3dmm(test_config["type_3dmm"]);
	debug_data_pack_ = test_config["nl_root"];

	if (model_3dmm_type_ == Type3dmm::NR || model_3dmm_type_ == Type3dmm::NR_RAW)
	{
		if (!test_config.contains("coef_3dmm"))
		{
			LOG(ERROR) << "do not contains for key coef_3dmm" << std::endl;
		}
		else
		{
			floatVec coef_3dmm_vec = test_config["coef_3dmm"].get<floatVec>();
			CalcHelper::vectorToEigen(coef_3dmm_vec, coef_3dmm_);
		}
	}
}

NRResVar::NRResVar()
{
	init();
}

void NRResVar::init()
{

}

void NRResVar::setInput(const json& test_config)
{
	if (!SG::isExist(test_config["input_image_"]))
	{
		LOG(ERROR) << test_config["input_image_"] << " file is not exist." << std::endl;
		return;
	}
	input_image_ = cv::imread(test_config["input_image_"]);
	if (input_image_.empty())
	{
		LOG(ERROR) << "image reading error." << std::endl;
		return;
	}
	load_from_json = test_config["load_from_json"];
	is_debug_ = test_config["is_debug_"];
	output_dir_ = test_config["output_dir_"];
	cstr imgdecode64 = TDST::base64Decode(test_config["landmark_256_xy_img"]);
	//LOG(INFO) << "img decode size:" << imgdecode64.size() << std::endl;
	//cstr imgdecode64 = TDST::base64_decode_(imgBase64);
	std::vector<uchar> img_data(imgdecode64.begin(), imgdecode64.end());
	landmark_256_xy_img_ = cv::imdecode(cv::Mat(img_data), cv::IMREAD_UNCHANGED);
	//cv::imwrite("D:/dota210507/0519_png/base_64_Prog_ut.png", landmark_256_xy_img_);

	floatVec coef_3dmm_vec = test_config["coef_3dmm"].get<floatVec>();
	CalcHelper::vectorToEigen(coef_3dmm_vec, coef_3dmm_);

	intVec landmark_256 = test_config["landmark_256_xy_land"].get<intVec>();
	CalcHelper::vectorToEigen(landmark_256, landmark_256_xy_);
	
	if (test_config["gender_"] == "male")
	{
		gender_ = Gender::MALE;
	}
	else
	{
		gender_ = Gender::FEMALE;
	}

	pp_type_ = PostProcessRoutine(test_config["pp_type_"]);
	SG::needPath(output_dir_);
	model_3dmm_type_ = Type3dmm(test_config["type_3dmm"]);
	debug_data_pack_ = test_config["nl_root"];
}