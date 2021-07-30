#ifndef TOP_TRANSFER_H
#define TOP_TRANSFER_H
#include "../Test/Test.h"
#include "../Test/TinyTool.h"
#include "../ExpGen/ExpGen.h"
#include "../OptV2/OptTypeV3.h"

namespace CGP
{
	//used for testing codes
	namespace TOPTRANSFER
	{
		void correctEyelashPair();
		void fixTexture(double reg_value);
		void fixTexture(cstr& root_res, double reg_value);
		void fixTextureCeres();
		void generateTextureTensor();
		void generateTextureTensorBatch();
		void generateTexDst();
		void transferOnUV();
		void transferSimDiff(const cstr& root, const cstr& obj);
		void transferSimDiff(const cstr& root, const cstr& obj, const MeshCompress& input_B);
		void transferSimDiff(const cstr& root, const cstr& obj, const MeshCompress& input_B, const std::shared_ptr<ExpGen> exp_ptr, std::shared_ptr<ResVar> res_ptr);
		void transferSimDiffTesting(const cstr& root, const cstr& obj, const MeshCompress& input_B, const std::shared_ptr<ExpGen> exp_ptr);
		void transferSimDiff(const cstr& path_A, const cstr& path_B, const cstr& root_A, const cstr& root_res);
		void transferSimDiff(const std::shared_ptr<ExpGen> exp_ptr, const MeshCompress& A, const MeshCompress& A_deform, 
			const MeshCompress& B, MeshCompress& B_deform);
		void putEyelashBack();
		void postProcessForGuijieTex();
		void postProcessForGuijieTexBatch(const std::shared_ptr<ExpGen> exp_ptr, const std::shared_ptr<OptV3Gen> optV3_ptr,
			const std::shared_ptr<ConstVar> ptr_const_var, std::shared_ptr<ResVar> ptr_res_var);
		void generateEyesBrow();	
		void guijieToFWHInstance(const cstr& root);

		void guijieToFWHInstanceTesting(const cstr& root);
		void guijieToFWHInferExample();
		void onlyLandmark();
		void onlyLandmarkDrag();
		void dumpGuijieEyelash();
		void transferUVFwhToGuijie();
		void getMatchFromDF();
		void getMatchFromDFAdv();
		void getMatchFromDFRefine();
		void getMatchFromFile();
		void fromGuijieToFWH();
		void getFWHToGuijieNoLash();
		void getFWHToGuijieV1NoLash();
		void getFWHToGuijieV1NoLashWithRing();
		void cutFWHToHalf(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void moveAndDeformEyelash(const MeshCompress& lash_src, const intVec& eyelash_self_idx, 
			const intVec& match_eye_idx, const intVec& lash_mapping_reverse, MeshCompress& guijie_all);
		void moveAndDeformEyelash(const intVec& lash_idx, const intVec& eyelash_self_idx,
			const intVec& match_eye_idx, const intVec& lash_mapping_reverse, MeshCompress& guijie_all);
		void moveAndDeformEyelash(const intVec& lash_idx, const intVec& match_idx, MeshCompress& guijie_all);
		void copyHeadBS(const json& config);
		void getTemplateBS();
		void serverGenExpFromMesh(const json& config);
		void generateEyebrowBS(const json& config);
		void generateToothBS(const json& config);
		void generateEyeBS(const json& config);
		void getShapeBS(const json& config);
		void getShpaeEyeBS(const json& config);
		void getShpaeEyeBrowBS(const json& config);
		void getToothShapeBS(const json& config);
		void addBS(const json& config);
		void packDeltaForExp(const json& config);
		void getMappingFromMayaToUnity(const json& config);
		void getMDS();
		void localDeform();
		void localDeform(const cstr& input_root);
		void localDeform(const cstr& input_root, const MeshCompress& input_B);
		void localDeform(const cstr& input_root, const MeshCompress& input_B, const std::shared_ptr<OptV3Gen> optV3_ptr);
		void changeBSScale();
		void tenetTest();
		void fromIsvToBase();
		void rawMatching();
	}
}
#endif
