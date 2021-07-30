#ifndef MESH_SHAPE_RECONSTRUTION_H
#define MESH_SHAPE_RECONSTRUTION_H
#include "../Config/JsonHelper.h"
#include "../Config/ShapeVar.h"
#include "../Config/ResVar.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
namespace CGP
{
	class RecShapeMesh
	{
	private:
		const std::shared_ptr<ShapeVar> shape_var_;
		std::shared_ptr<NRResVar> nrres_var_;
		cstr result_dir_;
		bool is_debug_;
	public:
		MeshCompress mesh_res_;
		vecF coef_3dmm_;
		vecF faceid_;
		vecF landmark_image_68_xy_;
		vecF landmark_image_5_xy_;
		int match_id_;
		cstr debug_data_pack_;

		RecShapeMesh(const std::shared_ptr<ShapeVar> shape_var, std::shared_ptr<NRResVar> nrres_var)
			:shape_var_(shape_var), nrres_var_(nrres_var)
		{
			result_dir_ = nrres_var_->output_dir_;
			is_debug_ = nrres_var_->is_debug_;
		}

		void processImage(bool return_before_opt = false);
		void processImageMidTerm(bool return_before_opt = false);
		void getPostProcessFor3dmmFit68(const vecF& coef_3dmm, MeshCompress& res,
			const cv::Mat& img_256, const vecF& landmark_256);
		void getPostProcessFor3dmm(const vecF& coef_3dmm, MeshCompress& res);

		void getFwhCoefFromCeres3dmm(MeshCompress& bfm, const cv::Mat& img_256,
			const vecF& landmark_xy, MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx,
			cvMatD& rvec, cvMatD& tvec);

		void adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, const cvMatD& rvec, const cvMatD& tvec,
			MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx);

		struct PCAImageCostRoi
		{
			PCAImageCostRoi(const floatVec& shape_pca, const vecF& dst_xy, const doubleVec& camera, const intVec& vertex_roi, int n_id, int n_vertex)
				: shape_pca_(shape_pca), dst_xy_(dst_xy), camera_(camera), vertex_roi_(vertex_roi), n_id_(n_id), n_vertex_(n_vertex)
			{};

			bool operator()(const double* const scale, const double* const translate, const double* const shape_coeffs, double* residual) const
			{
				double camera_params[10];
				for (int i = 0; i < 10; i++)
				{
					camera_params[i] = camera_[i];
				}
				for (int i = 0; i < vertex_roi_.size(); i++)
				{
					int idx = vertex_roi_[i];
					double xyz[3] = { 0,0,0 };
					for (int iter_pca = 0; iter_pca < n_id_; iter_pca++)
					{
						for (int iter_dim = 0; iter_dim < 3; iter_dim++)
						{
							xyz[iter_dim] += shape_pca_[iter_pca*n_vertex_ * 3 + idx * 3 + iter_dim] * shape_coeffs[iter_pca];
						}
					}
					for (int iter_dim = 0; iter_dim < 3; iter_dim++)
					{
						xyz[iter_dim] = xyz[iter_dim] * scale[0] + translate[iter_dim];
					}
					double p[3] = { 0,0,0 };
					ceres::AngleAxisRotatePoint(camera_params, xyz, p);
					p[0] += camera_[3];
					p[1] += camera_[4];
					p[2] += camera_[5];
					double xp = p[0] / p[2];
					double yp = p[1] / p[2];
					double l1 = camera_[7];
					double l2 = camera_[8];
					double r2 = xp * xp + yp * yp;
					double distortion = 1.0 + r2 * (l1 + l2 * r2);
					// Compute final projected point position.
					double focal = camera_[6];
					double predicted_x = focal * distortion * xp + camera_[9];
					double predicted_y = focal * distortion * yp + camera_[9];
					residual[i * 2 + 0] = predicted_x - dst_xy_[2 * i];
					residual[i * 2 + 1] = predicted_y - dst_xy_[2 * i + 1];
				}
				return true;
			};

		private:
			const floatVec shape_pca_; // pca_data
			const doubleVec camera_;
			const vecF dst_xy_;
			const intVec vertex_roi_;
			const int n_id_;
			const int n_vertex_;
		};


#if 0
		
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

		void adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, 
			MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx);

		void adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, const cvMatD& rvec, const cvMatD& tvec,
			const intVec& idx, MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx);
		void get3dmmMesh(const vecF& coef_3dmm, MeshCompress& res);
#endif
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};	
}

#endif
