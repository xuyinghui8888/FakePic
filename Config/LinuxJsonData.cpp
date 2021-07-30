#include "LinuxJsonData.h"
#include "../Basic/SafeGuard.h"
#include "JsonHelper.h"
using namespace CGP;
using namespace rttr;
RTTR_REGISTRATION
{
//registration::class_<JsonData>("CGP::JsonData").constructor<>()
registration::class_<LinuxJsonData>("CGP::LinuxJsonData").constructor<>()
	.property("template_obj_", &CGP::LinuxJsonData::template_obj_)
	.property("landmark68_out_num_", &CGP::LinuxJsonData::landmark68_out_num_)
	.property("bfm_tensor_", &CGP::LinuxJsonData::bfm_tensor_)
	.property("bfm_tensor_all_", &CGP::LinuxJsonData::bfm_tensor_all_)
	.property("nr_tensor_", &CGP::LinuxJsonData::nr_tensor_)
	.property("fwh_sys_finder_", &CGP::LinuxJsonData::fwh_sys_finder_)
	.property("fwh_bfm_tensor_", &CGP::LinuxJsonData::fwh_bfm_tensor_)
	.property("post_3dmm_", &CGP::LinuxJsonData::post_3dmm_)
	.property("male_id_", &CGP::LinuxJsonData::male_id_)
	.property("female_id_", &CGP::LinuxJsonData::female_id_)
	.property("fwh_68_idx_", &CGP::LinuxJsonData::fwh_68_idx_)
	.property("bfw_68_idx_", &CGP::LinuxJsonData::bfw_68_idx_)
	.property("bfw_68_all_idx_", &CGP::LinuxJsonData::bfw_68_all_idx_)
	.property("fxy_", &CGP::LinuxJsonData::fxy_)
	.property("cxy_", &CGP::LinuxJsonData::cxy_)
	.property("fwh_tensor_", &CGP::LinuxJsonData::fwh_tensor_)
	.property("fit_idx_contour_", &CGP::LinuxJsonData::fit_idx_contour_)
	.property("fit_idx_face_", &CGP::LinuxJsonData::fit_idx_face_)
	;
}
