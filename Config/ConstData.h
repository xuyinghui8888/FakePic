#ifndef CONST_DATA_H
#define CONST_DATA_H
#include <rttr/type>
#include <rttr/registration>
#include <rttr/registration_friend>

#include "FaceID.h"
#include "JsonData.h"
#include "LinuxJsonData.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/Tensor.h"
#include "../Mesh/MeshCompress.h"
#include "../Postprocess/PostProcess.h"
#include "../Sysmetric/Sysmetric.h"
#include "../RT/Projection.h"

namespace CGP
{
	class ConstData
	{
	public:
		ConstData(const JsonData& init_json);
		ConstData(const LinuxJsonData& init_json);
		MeshCompress template_obj_;
		Tensor fwh_bfm_tensor_;
		Tensor bfm_tensor_;
		Tensor bfm_tensor_all_;
		Tensor nr_tensor_;
		Tensor fwh_tensor_;
		MeshSysFinder fwh_sys_finder_;
		MeshPostProcess fwh_3dmm_;
		FaceIDFinder male_finder_;
		FaceIDFinder female_finder_;
		intVec fwh_68_idx_;
		intVec bfw_68_idx_;
		intVec bfw_68_all_idx_;
		intVec taobao_68_idx_;
		//for 3dmm post process movable points in lap
		intVec fit_idx_contour_;
		//for 3dmm post process movable points in lap
		intVec fit_idx_face_;
		std::shared_ptr<Projection> proj_;

		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};	
}

#endif
