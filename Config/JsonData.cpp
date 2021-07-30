#include "JsonData.h"
#include "../Basic/SafeGuard.h"
#include "JsonHelper.h"
using namespace CGP;
using namespace rttr;
RTTR_REGISTRATION
{
//registration::class_<JsonData>("CGP::JsonData").constructor<>()
registration::class_<JsonData>("CGP::ConstData").constructor<>()
	.property("template_obj_", &CGP::JsonData::template_obj_)
	.property("mtcnn_PNet_", &CGP::JsonData::mtcnn_PNet_)
	.property("mtcnn_RNet_", &CGP::JsonData::mtcnn_RNet_)
	.property("mtcnn_ONet_", &CGP::JsonData::mtcnn_ONet_)
	.property("is_max_face_", &CGP::JsonData::is_max_face_)
	.property("face_min_size_", &CGP::JsonData::face_min_size_)
	.property("facenet_", &CGP::JsonData::facenet_)
	.property("facenet_input_", &CGP::JsonData::facenet_input_)
	.property("facenet_output_", &CGP::JsonData::facenet_output_)
	.property("deep3d_", &CGP::JsonData::deep3d_)
	.property("deep3d_nr_", &CGP::JsonData::deep3d_nr_)
	.property("deep3d_input_", &CGP::JsonData::deep3d_input_)
	.property("deep3d_output_", &CGP::JsonData::deep3d_output_)
	.property("landmark68_", &CGP::JsonData::landmark68_)
	.property("landmark68_input_", &CGP::JsonData::landmark68_input_)
	.property("landmark68_out_num_", &CGP::JsonData::landmark68_out_num_)
	.property("landmark68_out_heatmap_", &CGP::JsonData::landmark68_out_heatmap_)
	.property("bfm_tensor_", &CGP::JsonData::bfm_tensor_)
	.property("bfm_tensor_all_", &CGP::JsonData::bfm_tensor_all_)
	.property("nr_tensor_", &CGP::JsonData::nr_tensor_)
	.property("fwh_sys_finder_", &CGP::JsonData::fwh_sys_finder_)
	.property("fwh_bfm_tensor_", &CGP::JsonData::fwh_bfm_tensor_)
	.property("post_3dmm_", &CGP::JsonData::post_3dmm_)
	.property("male_id_", &CGP::JsonData::male_id_)
	.property("female_id_", &CGP::JsonData::female_id_)
	.property("fwh_68_idx_", &CGP::JsonData::fwh_68_idx_)
	.property("bfw_68_idx_", &CGP::JsonData::bfw_68_idx_)
	.property("bfw_68_all_idx_", &CGP::JsonData::bfw_68_all_idx_)
	.property("fxy_", &CGP::JsonData::fxy_)
	.property("cxy_", &CGP::JsonData::cxy_)
	.property("fwh_tensor_", &CGP::JsonData::fwh_tensor_)
	.property("fit_idx_contour_", &CGP::JsonData::fit_idx_contour_)
	.property("fit_idx_face_", &CGP::JsonData::fit_idx_face_)
	.property("taobao_68_idx_", &CGP::JsonData::taobao_68_idx_)
	;
}
