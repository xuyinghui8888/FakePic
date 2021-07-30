#ifndef __POSTPROCESS_H__
#define __POSTPROCESS_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "../Config/JsonBased.h"
namespace CGP
{
	using json = nlohmann::json;

	class MeshPostProcess : public JsonBaseClass
	{
	private:
		//f_file
		cstr f_bfm_deform_;
		cstr f_bfm_id_trans_;
		cstr f_bfm_mid_;
		cstr f_fwh_deform_;
		cstr f_fwh_id_trans_;
		cstr f_fwh_mid_;
		cstr f_mouth_close_;
		cstr f_fwh_fix_;

	public:
		intVec bfm_deform_;
		intVec bfm_id_trans_;
		intVec bfm_mid_;
		intVec fwh_deform_;
		intVec fwh_id_trans_;
		intVec fwh_mid_;
		intVec mouth_close_;
		intVec fwh_fix_;

		void init();	

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};

}
#endif