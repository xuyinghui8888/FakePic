#include "ConstData.h"
#include "../Config/JsonHelper.h"
#include "../FileIO/FileIO.h"
using namespace CGP;
ConstData::ConstData(const JsonData& init_json)
{
	template_obj_.loadObj(init_json.root_ + init_json.template_obj_);
	JsonHelper::initData(init_json.root_ + init_json.fwh_bfm_tensor_, "config.json", fwh_bfm_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.bfm_tensor_, "config.json", bfm_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.bfm_tensor_all_, "config.json", bfm_tensor_all_);
	JsonHelper::initData(init_json.root_ + init_json.nr_tensor_, "config.json", nr_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.fwh_sys_finder_, "config.json", fwh_sys_finder_);
	JsonHelper::initData(init_json.root_ + init_json.fwh_tensor_, "config.json", fwh_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.post_3dmm_, "config.json", fwh_3dmm_);
	JsonHelper::initData(init_json.root_ + init_json.male_id_, "config.json", male_finder_);
	JsonHelper::initData(init_json.root_ + init_json.female_id_, "config.json", female_finder_);
	fwh_68_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.fwh_68_idx_);
	bfw_68_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.bfw_68_idx_);
	bfw_68_all_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.bfw_68_all_idx_);
	taobao_68_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.taobao_68_idx_);
	proj_ = std::make_shared<Projection>(init_json.fxy_, init_json.fxy_, init_json.cxy_, init_json.cxy_);
	fit_idx_contour_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.fit_idx_contour_);
	fit_idx_face_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.fit_idx_face_);
}

ConstData::ConstData(const LinuxJsonData& init_json)
{
	template_obj_.loadObj(init_json.root_ + init_json.template_obj_);
	JsonHelper::initData(init_json.root_ + init_json.fwh_bfm_tensor_, "config.json", fwh_bfm_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.bfm_tensor_, "config.json", bfm_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.bfm_tensor_all_, "config.json", bfm_tensor_all_);
	JsonHelper::initData(init_json.root_ + init_json.nr_tensor_, "config.json", nr_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.fwh_sys_finder_, "config.json", fwh_sys_finder_);
	JsonHelper::initData(init_json.root_ + init_json.fwh_tensor_, "config.json", fwh_tensor_);
	JsonHelper::initData(init_json.root_ + init_json.post_3dmm_, "config.json", fwh_3dmm_);
	JsonHelper::initData(init_json.root_ + init_json.male_id_, "config.json", male_finder_);
	JsonHelper::initData(init_json.root_ + init_json.female_id_, "config.json", female_finder_);
	fwh_68_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.fwh_68_idx_);
	bfw_68_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.bfw_68_idx_);
	bfw_68_all_idx_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.bfw_68_all_idx_);
	//set as default
	taobao_68_idx_ = intVec(68, -1);
	proj_ = std::make_shared<Projection>(init_json.fxy_, init_json.fxy_, init_json.cxy_, init_json.cxy_);
	fit_idx_contour_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.fit_idx_contour_);
	fit_idx_face_ = FILEIO::loadIntDynamic(init_json.root_ + init_json.fit_idx_face_);
}