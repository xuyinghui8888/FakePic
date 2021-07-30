#ifndef MESH_RECONSTRUTION_H
#define MESH_RECONSTRUTION_H
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
namespace CGP
{
	class RecMesh
	{
	private:
		const std::shared_ptr<ConstVar> const_var_;
		std::shared_ptr<ResVar> res_var_;
		cstr result_dir_;
		bool is_debug_;
	public:
		MeshCompress mesh_res_;
		vecF coef_3dmm_;
		vecF faceid_;
		vecF landmark_image_68_xy_;
		vecF landmark_image_5_xy_;
		int match_id_;
		RecMesh(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
			:const_var_(const_var), res_var_(res_var)
		{
			result_dir_ = res_var->output_dir_;
			is_debug_ = res_var->is_debug_;
		}

		void processImage(bool return_before_opt = false);
		void processImage(const std::shared_ptr<NR3D> model_3dmm, bool return_before_opt = false);
		void processDeform();
		void processBasicQRatio();
		void processPartRatio();
		void processFaceDetection(const cv::Mat& input, cv::Rect& res);
		void shiftBBLandmark68(const cv::Mat& img, const cv::Rect& src, int resolution, 
			cv::Mat& dst, cv::Rect& scale_bb);
		void transform(const cv::Point2f& src, const cv::Point2f& center, const double scale,
			const int resolution, cv::Point2f& dst);
		void getLandmarks(const cv::Mat& img, vecF& res);
		void reflectBackToImage(const cv::Rect& landmark_68_crop, int resolution, 
			const vecF& res_256, vecF& res);
		void get5LandFrom68(const vecF& input_68, vecF& output_5);
		void getFaceID(const cv::Mat& img, const vecF& landmark_5_xy, vecF& id);
		void get3dmmCoef(const cv::Mat& img, const vecF& landmark_5_xy, vecF& coef_3dmm);
		void get3dmmCoef(const std::shared_ptr<NR3D> model_3dmm, const cv::Mat& img, const vecF& landmark_5_xy, vecF& coef_3dmm);
		void getPostProcessFor3dmm(const vecF& coef_3dmm, MeshCompress& res);
		void getPostProcessFor3dmmFit68(const vecF& coef_3dmm, MeshCompress& res, 
			const cv::Mat& img_256, const vecF& landmark_256);
		void getPostProcessFor3dmmTexBase(const vecF& coef_3dmm, MeshCompress& res,
			const cv::Mat& img_256, const vecF& landmark_256);
		void fitAWithLandmarksToImage(const MeshCompress& A, const cv::Mat& img_256, 
			const vecF& landmark_256, const intVec& land_68, const intVec& movable, 
			const intVec& pair, MeshCompress& res);
		void getMatchID(const vecF& faceid, int& match_res);
		void getDFResult(const MeshCompress& basic, const int& match_id);
		void getFwhCoefFromProjection(const MeshCompress& bfm, const cv::Mat& img_256,
			const vecF& landmark_xy, vecF& coef_res);
		void getFwhCoefFromCeres(MeshCompress& bfm, const cv::Mat& img_256,
			const vecF& landmark_xy, MeshCompress& fwh, float3E& translate_3dmm_to_fwh);
		void getFwhCoefFromCeres3dmm(MeshCompress& bfm, const cv::Mat& img_256,
			const vecF& landmark_xy, MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx,
			cvMatD& rvec, cvMatD& tvec);
		void adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, 
			MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx);
		void adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, const cvMatD& rvec, const cvMatD& tvec,
			MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx);
		void adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, const cvMatD& rvec, const cvMatD& tvec,
			const intVec& idx, MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx);
		void get3dmmMesh(const vecF& coef_3dmm, MeshCompress& res);
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};	
}

#endif
