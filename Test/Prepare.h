#ifndef PREPARE_DATA_H
#define PREPARE_DATA_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
namespace CGP
{
	//used for testing codes
	namespace PREPARE
	{
		void prepareGuijieV3Data(const cstr& img_root, const cstr& id_root, int n_type);
		void prepareData(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void prepareMesh(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void prepareID(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void prepareIDV3(const cstr& id_root, const cstr& result, int max_id);
		void prepareTaobao(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void prepareTaobaoBS(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void prepareBSTensor(const cstr& obj_from, const cstr& obj_to, bool if_mean_zero = false);
		void projectTaobaoToBsCoef(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void projectGuijieToBsCoefOnce();
		void getExpFixAndMovePoints();
		void prepareExpGen();
		void prepare3dmmAndBsCoefV2();
		void prepare3dmmAndBsCoefV3(const cstr& img_root, int n_type);
		void prepareEyebrow();
		void prepareTest3dmmAndBs(const cstr& img_root);
		void preparePolyWinkModel();
		void prepare3dmmPartModel();
		void moveCTOBlendshapeToZero();
		void prepareEyebrowMask();
		void getCTOBlendshape();
		void putCTOModelToZero();

		void transferUVGuijieToFwh();
		void getLandmarkFromJsonRaw();

		void dragEyes();
		void dragNoseMouth();

		struct PCACost
		{
			PCACost(const floatVec& shape_pca, const float3E& dst_pos, int n_id, int n_vertex, int cur_vertex)
				: shape_pca_(shape_pca), dst_pos_(dst_pos), n_id_(n_id), n_vertex_(n_vertex), cur_vertex_(cur_vertex)
			{};

			bool operator()(const double* const scale, const double* const translate, const double* const shape_coeffs, 
				double* residual) const
			{	
				double x = 0;
				double y = 0;
				double z = 0;
				for (int iter_pca = 0; iter_pca < n_id_; iter_pca++)
				{
					//std::cout << "iter_pca:" << shape_coeffs[0] << std::endl;
					//std::cout << "iter_pca:" << shape_coeffs[1] << std::endl;
					//std::cout << "iter_pca:" << shape_coeffs[2] << std::endl;
					//std::cout << "iter_pca:" << shape_coeffs[3] << std::endl;
					//std::cout << "iter_pca:" << shape_coeffs[iter_pca] << std::endl;
					//std::cout << "x:" << x << std::endl;
					//std::cout << "y:" << x << std::endl;
					//std::cout << "z:" << x << std::endl;
					x += shape_coeffs[iter_pca] * (shape_pca_[iter_pca * 3 + 0]);
					y += shape_coeffs[iter_pca] * (shape_pca_[iter_pca * 3 + 1]);
					z += shape_coeffs[iter_pca] * (shape_pca_[iter_pca * 3 + 2]);
				}
				x = x - dst_pos_.x()*scale[0]+translate[0];
				y = y - dst_pos_.y()*scale[0] + translate[1];
				z = z - dst_pos_.z()*scale[0] + translate[2];
				residual[0] = x * x + y * y + z * z;
				return true;
			};

		private:
			const floatVec shape_pca_; // pca_data
			const float3E dst_pos_;
			const int n_id_;
			const int n_vertex_;
			const int cur_vertex_;
		};	

		struct PCAAreaCost
		{
			PCAAreaCost(const floatVec& shape_pca, const float3Vec& dst_pos, int n_id, int n_vertex, const intVec roi = {})
				: shape_pca_(shape_pca), dst_pos_(dst_pos), n_id_(n_id), n_vertex_(n_vertex), roi_(roi)
			{};

			bool operator()(const double* const scale, const double* const translate, const double* const shape_coeffs, double* residual) const
			{
				intVec index_vec;
				if (roi_.empty())
				{
					index_vec = intVec(n_vertex_, 0);
					std::iota(index_vec.begin(), index_vec.end(), 0);
				}
				else
				{
					index_vec = roi_;
				}
				for (int i = 0; i < index_vec.size(); i++)
				{
					int vertex_id = index_vec[i];
					doubleVec xyz(3, 0);
					for (int iter_pca = 0; iter_pca < n_id_; iter_pca++)
					{
						for (int iter_dim = 0; iter_dim < 3; iter_dim++)
						{
							xyz[iter_dim] += shape_pca_[iter_pca*n_vertex_ * 3+ vertex_id *3+ iter_dim]* shape_coeffs[iter_pca];
						}
					}
					for (int iter_dim = 0; iter_dim < 3; iter_dim++)
					{
						residual[i * 3 + iter_dim] = dst_pos_[vertex_id](iter_dim) - xyz[iter_dim];
					}
				}
				return true;
			};

		private:
			const intVec roi_;//set roi vertex
			const floatVec shape_pca_; // pca_data
			const float3Vec dst_pos_;
			const int n_id_;
			const int n_vertex_;	
		};
		

		struct PCAVertexCostRoi
		{
			PCAVertexCostRoi(const floatVec& shape_pca, const float3Vec& dst_pos, const intVec& vertex_roi, int n_id, int n_vertex)
				: shape_pca_(shape_pca), dst_pos_(dst_pos), vertex_roi_(vertex_roi), n_id_(n_id), n_vertex_(n_vertex)
			{};

			bool operator()(const double* const scale, const double* const translate, const double* const shape_coeffs, double* residual) const
			{
				for (int i = 0; i < vertex_roi_.size(); i++)
				{
					int idx = vertex_roi_[i];
					doubleVec xyz(3, 0);
					for (int iter_pca = 0; iter_pca < n_id_; iter_pca++)
					{
						for (int iter_dim = 0; iter_dim < 3; iter_dim++)
						{
							xyz[iter_dim] += shape_pca_[iter_pca*n_vertex_ * 3 + idx * 3 + iter_dim] * shape_coeffs[iter_pca];
						}
					}
					for (int iter_dim = 0; iter_dim < 3; iter_dim++)
					{
						residual[i * 3 + iter_dim] = (dst_pos_[i](iter_dim)*scale[0] + translate[iter_dim] - xyz[iter_dim]) ;
					}
				}
				return true;
			};

		private:
			const floatVec shape_pca_; // pca_data
			const float3Vec dst_pos_;
			const intVec vertex_roi_;
			const int n_id_;
			const int n_vertex_;
		};

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
					double xyz[3] = {0,0,0};
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

	}
}
#endif
