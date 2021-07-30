#ifndef __STYLE_TRANSFER_H__
#define __STYLE_TRANSFER_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "../RecMesh/RecShapeMesh.h"
#include "../ExpGen/ExpGen.h"
#include "../OptV2/OptTypeV3.h"

namespace CGP
{
	using json = nlohmann::json;

	namespace STYLETRANSFER
	{
		void generateAvatarWrapper(const std::shared_ptr<ShapeVar> ptr_const_var, std::shared_ptr<CGP::NRResVar> ptr_res_var);
		void postProcessForGuijieTexBatch(const std::shared_ptr<ExpGen> exp_ptr, const std::shared_ptr<OptV3Gen> optV3_ptr,
			const std::shared_ptr<ShapeVar> ptr_const_var, std::shared_ptr<NRResVar> ptr_res_var, const json& config);
		void guijieToFWHInstance(const cstr& root, const json& config);
		void transferSimDiff(const cstr& root, const cstr& obj, const MeshCompress& input_B, const std::shared_ptr<ExpGen> exp_ptr, 
			std::shared_ptr<NRResVar> res_ptr, const json& config);
		void localDeform(const cstr& input_root, const MeshCompress& input_B, const std::shared_ptr<OptV3Gen> optV3_ptr, const json& config);
		void generateEyesBrow(const json& config);
		void generateTexDst(const json& config);
		void fixTexture(cstr& root_res, double reg_value, const json& config);
	}
}
#endif