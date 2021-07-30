#ifndef EXP_GEN_H
#define EXP_GEN_H
#include "../Mesh/MeshCompress.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/Tensor.h"
#include "../Sysmetric/Sysmetric.h"
#include "../MeshDeform/DTUtilities.h"

namespace CGP
{	
	class FacePartInfo
	{
	public:
		bool update_ = false;
		int n_obj_;
		cstrVec name_;
		std::vector<MeshCompress> ani_;
		std::vector<SpMat> dt_init_E1_;
		std::vector<vecD> dt_init_E1_B_;
		floatX2Vec omp_float_;
		matI move_idx_;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	class DTSimConfig
	{
	public:
		intX2Vec face_part_;
		std::vector<LandGuidedType> opt_type_;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};


	class ExpGen
	{
	public:
		ExpGen(const json& config);
		void init(const json& config);
		void generatePCA();
		void testExp();
		void setDebug(bool is_debug);
		void setB(MeshCompress& B);
		void getExpOMP(const MeshCompress& B, const cstr& output_path = "");
		void testExpGuided(const cstr& input_obj, const cstr& output_path, int part);
		void getEyeBlink(const MeshCompress& B, FacePartInfo& res, const cstr& output_path = "");
		void fixEyelash(MeshCompress& dst);
		void getTensorCoef(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor, 
			const intVec& up_down_match, const MeshCompress& dst, vecD& coef, const cstr& output_path = "");
		void getTensorCloseCoef(const Tensor& tensor, const intVec& roi, const MeshCompress& src, const MeshCompress& dst, const vecD& weight, vecD& coef);
		void getTensorCoefDouble(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor_left,  const Tensor& close_tensor_right,
			const intVec& up_down_match, const MeshCompress& dst, FacePartInfo& res, const cstr& output_path = "");
		void getEyeTransform(const MeshCompress& left_blink, const MeshCompress& right_blink, floatVec& trans);
		void getCloseShape(const Tensor& tensor, const Tensor& close_tensor, const vecD& coef, 
			intVec& eye_match, MeshCompress& B, MeshCompress& B_res);
		void getCloseShape(MeshCompress& A, MeshCompress& A_deform, intVec& eye_match, MeshCompress& B, MeshCompress& B_res);
		void loadExp(const cstr& root, FacePartInfo& res);
		void getExpGuided(const MeshCompress& B, FacePartInfo& part, const cstr& output_path = "");
		void getExpGuided(const MeshCompress& B, const MeshCompress& A_deform_input, MeshCompress& B_res);
		void getExpGuided(const MeshCompress& B, const MeshCompress& A_input, const MeshCompress& A_deform_input, MeshCompress& B_res);
		void getResultJson(nlohmann::json& res, bool is_compress = true);
		void setJsonValueFromPart(const FacePartInfo& region, nlohmann::json& res, bool is_compress = false);
		void dumpEyelash(MeshCompress& src);
	
	private:

		bool debug_ = false;
		bool is_init_ = false;
		intVec eyelash_;
		intVec left_eye_part_;
		intVec right_eye_part_;
		intVec mouth_part_;
		intVec nose_part_;
		intVec eyelash_eye_pair_;
		intVec left_eye_match_;
		intVec right_eye_match_;
		intVec left_eye_blvs_dis_;
		MeshSysFinder guijie_;
		cstr data_root_;
		double dis_thres_;
		floatVec B_pos_;
		
		Tensor eye_shape_tensor_;
		Tensor left_eye_blink_tensor_;
		Tensor right_eye_blink_tensor_;

		FacePartInfo ani_test_;
		FacePartInfo ani_eye_;
		FacePartInfo ani_exp_;
		FacePartInfo ani_sound_;
		MeshCompress default_A_;

		MeshCompress dt_A_;
		MeshCompress dt_B_;
		MeshCompress dt_B_land_;
		MeshCompress dt_A_deform_;

		floatVec eye_trans_ = { 0,0,0 };
		double fix_thres_;
		intX2Vec xy_rotate_;

		std::vector<vecD> A_part_normal_;
		std::vector<vecD> B_part_normal_;

		DTSimConfig dt_part_config_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}

#endif
