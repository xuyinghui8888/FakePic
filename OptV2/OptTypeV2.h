#ifndef OPT_TYPE_V2
#define OPT_TYPE_V2
#include "../Mesh/MeshCompress.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/Tensor.h"
#include "../Sysmetric/Sysmetric.h"
#include "../MeshDeform/DTUtilities.h"
#include "OptTypeUtils.h"

namespace CGP
{	
	class OptV2Gen
	{

	public:
		OptV2Gen(const json& config);
		void init(const json& config);
		void loadType(const cstr& root, FaceTypeInfo& res);
		void setDebug(bool is_debug);
		void getTensorCoef(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor,
			const intVec& up_down_match, const MeshCompress& dst, vecD& coef, const cstr& output_path = "");
		void dumpEyelash(MeshCompress& src);
		void getBFMMesh(const vecD& coef_3dmm, MeshCompress& res);
		void getGuijieMesh(vecD& coef_bs, MeshCompress& res);
		void getGuijieFromType(json& json_in, MeshCompress& res, json& json_out, MeshCompress& opt_res);
		void getRawBs(const FaceAttPack& in_att, vecD& res);
		void optBsBasedOn3dmm(const FaceAttPack& in_att, const vecD& type_bs, const MeshCompress& type_res,
			vecD& opt_bs, MeshCompress& opt_res);
		void getDriftLandmark(const FaceTypeInfo& part, int type, const intVec& roi, const float3Vec& template_guijie, const float3Vec& att_bfm, 
			float reduce, float3Vec& att_guijie, bool is_shift_center);

		void getDriftLandmarkXY(const FaceTypeInfo& part, int type, const intVec& roi, const float3Vec& template_guijie, const float3Vec& att_bfm,
			float reduce, float3Vec& att_guijie, bool is_shift_center);

		void getDriftLandmarkXYNoScale(const FaceTypeInfo& part, int type, const intVec& roi, const float3Vec& template_guijie, const float3Vec& att_bfm,
			float reduce, float3Vec& att_guijie, bool is_shift_center);

		void getTensorCloseCoef(const Tensor& tensor, const vecD& type_bs, const intVec& fix_tensor, const intVec& roi, const float3Vec& dst_roi, const vecD& weight, vecD& coef);
		
	private:

		bool debug_ = false;
		cstr data_root_;
		MeshCompress default_A_;
		
		intVec landmark_68_bfm_;
		intVec landmark_68_guijie_;

		FaceTypeInfo eye_type_;
		FaceTypeInfo face_type_;
		FaceTypeInfo mouth_type_;
		FaceTypeInfo nose_type_;

		intVec eyelash_;

		cstrVec ordered_map_;
		Tensor shape_tensor_;
		Tensor bfm_tensor_;

		cstr bs_prefix_python_;

		intVec left_eye_region_;
		intVec right_eye_region_;
		intVec nose_region_;
		intVec mouth_region_;
		intVec face_region_;


		bool is_init_ = false;	

	public:
		template<class T>
		void getResultJson(const T& opt_bs, nlohmann::json& res, float cut_thres = 1e-6)
		{
			if (opt_bs.size() != ordered_map_.size() - 1)
			{
				LOG(ERROR) << "opt_bs is wrong." << std::endl;
				return;
			}
			res.clear();
			for (int i = 0; i < opt_bs.size(); i++)
			{
				if (opt_bs[i] > cut_thres)
				{
					res[bs_prefix_python_ + ordered_map_[i + 1]] = opt_bs[i];
				}
			}
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}

#endif
