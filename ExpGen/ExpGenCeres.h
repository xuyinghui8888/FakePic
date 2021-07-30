#ifndef EXP_GEN_H
#define EXP_GEN_H
#include "../Shader/shader.h"
#include "../Mesh/MeshCompress.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/DataStructureHeader.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
namespace CGP
{
	struct PCAVertexCost
	{
		PCAVertexCost(const floatVec& shape_pca, const float3E& dst_pos, int n_id)
			: shape_pca_(shape_pca), dst_pos_(dst_pos), n_id_(n_id)
		{};

		bool operator()(const double* const shape_coeffs, double* residual) const
		{
			doubleVec xyz(3, 0);
			for (int iter_pca = 0; iter_pca < n_id_; iter_pca++)
			{
				for (int iter_dim = 0; iter_dim < 3; iter_dim++)
				{
					xyz[iter_dim] += shape_pca_[iter_pca * 3 + iter_dim] * shape_coeffs[iter_pca];
				}
			}
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				residual[iter_dim] = (dst_pos_(iter_dim) - xyz[iter_dim])*(dst_pos_(iter_dim) - xyz[iter_dim]);
			}
			return true;
		};

	private:
		const floatVec shape_pca_; // pca_data
		const float3E dst_pos_;
		const int n_id_;
	};
	
	class ExpGen
	{
	public:
		ExpGen(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		ExpGen(const json& config);
		void init(const json& config);
		void projectGuijieToBsCoefVertex();
		void generatePCA();
		void testExp();
		void testExpGuided(const cstr& input_obj, const cstr& output_path, int part);
		void getEyeBlink(const cstr& input_obj, const cstr& output_path, int part);
		void getTensorCoef(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor, 
			const intVec& up_down_match, const MeshCompress& dst, vecD& coef);
		void getTensorCloseCoef(const Tensor& tensor, const intVec& roi, const MeshCompress& src, const MeshCompress& dst, const vecD& weight, vecD& coef);
		void getTensorCoefDouble(const Tensor& tensor, const intVec& roi, const Tensor& close_tensor_left,  const Tensor& close_tensor_right,
			const intVec& up_down_match, const MeshCompress& dst);
		void getCloseShape(const Tensor& tensor, const Tensor& close_tensor, const vecD& coef, 
			intVec& eye_match, MeshCompress& B, MeshCompress& B_res);
		void getCloseShape(MeshCompress& A, MeshCompress& A_deform, intVec& eye_match, MeshCompress& B, MeshCompress& B_res);
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
		MeshSysFinder guijie_;
		cstr data_root_;
		double dis_thres_;
	};
}

#endif
