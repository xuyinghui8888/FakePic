#include "BatchGeneration.h"
#include "../Basic/MeshHeader.h"
#include "../VisHelper/VisHeader.h"
#include "../NRICP/register.h"
#include "../NRICP/demo.h"
#include "../RigidAlign/icp.h"
#include "../Config/Tensor.h"
#include "../Config/TensorHelper.h"
#include "../Config/JsonHelper.h"

#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"

#include "../Sysmetric/Sysmetric.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Metric/Metric.h"
#include "../MNN/MnnModel.h"
#include "../MNN/LandmarkTracking.h"
#include "../MNN/FeatureExtract.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../Solver/BVLSSolver.h"
#include "../ExpGen/ExpGen.h"
#include "../ExpGen/BsGenerate.h"
#include "../OptV2/OptTypeV2.h"
#include "../OptV2/OptTypeV3.h"
#include "../Eyebrow/EyebrowType.h"
#include "../ImageSim/ImageSimilarity.h"
#include "../ImageSim/ImageUtils.h"
#include "../Beauty/Beauty35.h"
#include "../Debug/DebugTools.h"
#include "../Landmark/Landmark68Wrapper.h"
#include "../MeshDeform/ReferenceDeform.h"
#include "../Shell/ShellGenerate.h"
#include "../Bvh/Bvh11.hpp"
#include "../Test/FixRoutine.h"
#include "../Postprocess/StyleTransfer.h"

using namespace CGP;

void BatchGenerate::generateX()
{
	//TOPTRANSFER::generateTextureTensorBatch();
#if 1
	cstr root = "D:/data/server_pack/";
	json json_in = FILEIO::loadJson(root + "config.json");
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);

	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";

	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));

	json shape_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	shape_config["root"] = "D:/avatar/guijie_opt3_data/";

	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(shape_config));

	json res_json_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	std::shared_ptr<CGP::NRResVar> ptr_res_nr = std::make_shared<CGP::NRResVar>();
	ptr_res_nr->setInput(res_json_config);

	std::shared_ptr<ResVar> ptr_res_ms = std::make_shared<ResVar>();
	ptr_res_ms->setInput(res_json_config);

#endif

	if (ptr_res_ms->model_3dmm_type_ == Type3dmm::MS || ptr_res_ms->model_3dmm_type_ == Type3dmm::NR)
	{
		FIXROUTINE::generateAvatarBatch(ptr_const_var, ptr_res_ms);
		TOPTRANSFER::postProcessForGuijieTexBatch(exp_ptr, optV3_ptr, ptr_const_var, ptr_res_ms);
		TOPTRANSFER::generateEyesBrow();
		TOPTRANSFER::generateTexDst();
	}
	else if (ptr_res_ms->model_3dmm_type_ == Type3dmm::NR_CPP)
	{
		//output data
		FIXROUTINE::generateAvatarBatch(ptr_const_var, ptr_res_ms);
		TOPTRANSFER::postProcessForGuijieTexBatch(exp_ptr, optV3_ptr, ptr_const_var, ptr_res_ms);
		TOPTRANSFER::generateEyesBrow();
		TOPTRANSFER::generateTexDst();
	}
	else if (ptr_res_ms->model_3dmm_type_ == Type3dmm::NR_CPP_RAW)
	{
		//output data
		FIXROUTINE::generateAvatarBatch(ptr_const_var, ptr_res_ms);
		TOPTRANSFER::postProcessForGuijieTexBatch(exp_ptr, optV3_ptr, ptr_const_var, ptr_res_ms);
		TOPTRANSFER::generateEyesBrow();
		TOPTRANSFER::generateTexDst();
	}
	else if (ptr_res_ms->model_3dmm_type_ == Type3dmm::NR_RAW)
	{
		MeshCompress template_mesh = ptr_const_var->ptr_data->nr_tensor_.template_obj_;
		ptr_const_var->ptr_data->nr_tensor_.interpretIDFloat(template_mesh.pos_.data(), ptr_res_ms->coef_3dmm_);
		cstr output_dir_ = res_json_config["output_dir_"].get<cstr>();
		template_mesh.saveObj(output_dir_ + "local_deform.obj");
		TOPTRANSFER::generateEyesBrow();
	}
	else
	{
		LOG(ERROR) << "undefined path." << std::endl;
	}
}

void BatchGenerate::generateLinuxTest()
{
	//TOPTRANSFER::generateTextureTensorBatch();
	cstr root = "D:/avatar/nl_linux/";
#if 1
	LinuxJsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ShapeVar> ptr_const_var = std::make_shared<ShapeVar>(json_data);
#endif

	json res_json_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");

	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";

	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));

	json shape_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	shape_config["root"] = "D:/avatar/guijie_opt3_data/";

	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(shape_config));

	std::shared_ptr<CGP::NRResVar> ptr_res_var = std::make_shared<CGP::NRResVar>();
	ptr_res_var->setInput(res_json_config);


	STYLETRANSFER::generateAvatarWrapper(ptr_const_var, ptr_res_var);
	STYLETRANSFER::postProcessForGuijieTexBatch(exp_ptr, optV3_ptr, ptr_const_var, ptr_res_var, res_json_config);
	STYLETRANSFER::generateEyesBrow(res_json_config);
	if (ptr_res_var ->model_3dmm_type_ == Type3dmm::MS || ptr_res_var->model_3dmm_type_ == Type3dmm::NR)
	{
		STYLETRANSFER::generateTexDst(res_json_config);
	}
}
