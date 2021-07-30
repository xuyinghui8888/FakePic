#ifndef LINUX_JSON_DATA_H
#define LINUX_JSON_DATA_H
#include "../Basic/CGPBaseHeader.h"
#include "JsonBased.h"
#include <rttr/type>
#include <rttr/registration>
#include <rttr/registration_friend>
namespace CGP
{
	//used for not changable programs
	using json = nlohmann::json;
	//jsonData 对应guijie拓扑需要初始化数据类
	class LinuxJsonData: public JsonBaseClass
	{
	public:
		using JsonBaseClass::JsonBaseClass;
		//check data using create data
		cstr template_obj_;
		int landmark68_out_num_;

		cstr bfm_tensor_;
		cstr bfm_tensor_all_;
		cstr nr_tensor_;

		cstr fwh_sys_finder_;
		cstr fwh_bfm_tensor_;
		cstr post_3dmm_;

		cstr male_id_;
		cstr female_id_;

		cstr fwh_68_idx_;
		cstr bfw_68_idx_;
		cstr bfw_68_all_idx_;

		float fxy_;
		float cxy_;

		cstr fwh_tensor_;

		cstr fit_idx_contour_;
		cstr fit_idx_face_;

		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};	
}

#endif
