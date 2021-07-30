#ifndef TENSOR_UTILITY_H
#define TENSOR_UTILITY_H
#include "../Basic/CGPBaseHeader.h"
#include "JsonBased.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	//get tensor type
	enum class TensorType
	{
		BFM_DEEP, //set for 3dmm deepFaceReconstruction based model
	};
	class Tensor : public JsonBaseClass
	{
	private:
		int n_data_type_;
		std::string tensor_file_;
		std::string template_;
		std::string eigen_value_;

		int n_vertex_;
	public:
		//register
		int n_id_;
		int n_exp_;
		//eigen value data
		floatVec ev_data_;
		floatVec data_;
		MeshCompress template_obj_;
	public:
		//loading data from file
		void init();
		//saving data to file
		void save(const cstr& root);

		void saveByIdExp(const std::string& res_path, int weight = 0) const;
	
		template <class T1, class T2>
		void fitID(const T1& src, const floatVec& reg, T2& coef)
		{
			if (src.size() % n_vertex_)
			{
				LOG(ERROR) << "sizeof(T1): " << sizeof(T1) << std::endl;
				LOG(ERROR) << "input size: " << src.size() * sizeof(T1) << std::endl;
				LOG(ERROR) << "sizeof(float)*n_vertex_ * 3: " << sizeof(float)*n_vertex_ * 3 << std::endl;
				LOG(ERROR) << "vertex size don't fit." << std::endl;
			}
			//mean set to one
			matF A(n_id_ - 1, n_vertex_ * 3);
			vecF B(n_vertex_ * 3);
			vecF mean(n_vertex_ * 3);
			SG::safeMemcpy(B.data(), src.data(), sizeof(float)*n_vertex_ * 3);
			SG::safeMemcpy(mean.data(), data_.data(), sizeof(float)*n_vertex_ * 3);
			A.setZero();
#pragma omp parallel for
			for (int y = 0; y < n_id_ - 1; y++)
			{
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					A(y, x) = data_[idx];
				}
			}
			matF AAt = A * A.transpose();
#pragma omp parallel for
			for (int y = 0; y < n_id_ - 1; y++)
			{
				AAt(y, y) += reg[y];
			}
			vecF AtB = A * (B - mean);
			vecF res = AAt.ldlt().solve(AtB);
			coef.resize(res.size());
			for (int i = 0; i < coef.size(); i++)
			{
				coef[i] = res[i];
			}
		}

		template <class T1, class T2, class T3>
		void fitID(const T1& src, const floatVec& reg, const T2&ev, T3& coef)
		{
			if (ev.size() < n_id_ - 1)
			{
				LOG(ERROR) << "ev value failed. size is wrong." << std::endl;
				return;
			}

			if (src.size() % n_vertex_)
			{
				LOG(ERROR) << "sizeof(T1): " << sizeof(T1) << std::endl;
				LOG(ERROR) << "input size: " << src.size() * sizeof(T1) << std::endl;
				LOG(ERROR) << "sizeof(float)*n_vertex_ * 3: " << sizeof(float)*n_vertex_ * 3 << std::endl;
				LOG(ERROR) << "vertex size don't fit." << std::endl;
			}
			//mean set to one
			matF A(n_id_ - 1, n_vertex_ * 3 + n_id_ -1);
			vecF B(n_vertex_ * 3 + n_id_ - 1);
			B.setZero();
			vecF mean(n_vertex_ * 3 + n_id_ - 1);
			mean.setZero();
			SG::safeMemcpy(B.data(), src.data(), sizeof(float)*n_vertex_ * 3);
			SG::safeMemcpy(mean.data(), data_.data(), sizeof(float)*n_vertex_ * 3);
			A.setZero();
#pragma omp parallel for
			for (int y = 0; y < n_id_ - 1; y++)
			{
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					A(y, x) = data_[idx];
				}
			}
			for (int y = 0; y < n_id_ -1; y++)
			{
				A(y, n_vertex_ * 3+ y) = 1.0f / ev[y];
			}
			matF AAt = A * A.transpose();
#pragma omp parallel for
			for (int y = 0; y < n_id_ - 1; y++)
			{
				AAt(y, y) += reg[y];
			}
			vecF AtB = A * (B - mean);
			vecF res = AAt.ldlt().solve(AtB);
			coef.resize(res.size());
			for (int i = 0; i < coef.size(); i++)
			{
				coef[i] = res[i];
			}
		}

		template <class T1, class T2, class T3>
		void fitID(const T1& src, const floatVec& reg, const intVec& roi, const T2&ev, T3& coef) const
		{
			if (ev.size() < n_id_ - 1)
			{
				LOG(ERROR) << "ev value failed. size is wrong." << std::endl;
				return;
			}

			if (src.size() % roi.size())
			{
				LOG(ERROR) << "sizeof(T1): " << sizeof(T1) << std::endl;
				LOG(ERROR) << "input size: " << src.size() * sizeof(T1) << std::endl;
				LOG(ERROR) << "sizeof(float)*n_vertex_ * 3: " << sizeof(float)*n_vertex_ * 3 << std::endl;
				LOG(ERROR) << "vertex size don't fit." << std::endl;
			}

			//get mapping
			intVec vec_roi;
			intSet roi_set(roi.begin(), roi.end());
			for (int i = 0; i < n_vertex_; i++)
			{
				if (roi_set.count(i))
				{
					for (int j = 0; j < 3; j++)
					{
						vec_roi.push_back(3*i+j);
					}
				}
			}
			//insert n_id-1
			for (int i = 0; i < n_id_-1; i++)
			{				
				vec_roi.push_back(i+3*n_vertex_);	
			}

			//mean set to one
			matF A(n_id_ - 1, n_vertex_ * 3 + n_id_ - 1);
			A.setZero();
			vecF B(n_vertex_ * 3 + n_id_ - 1);
			B.setZero();
			vecF mean(n_vertex_ * 3 + n_id_ - 1);
			mean.setZero();
			if (src.size() == n_vertex_ || src.size() == n_vertex_ * 3)
			{
				SG::safeMemcpy(B.data(), src.data(), sizeof(float)*n_vertex_ * 3);
			}
			else
			{
				if (src.size() == roi.size())
				{
					for (int i = 0; i < roi.size(); i++)
					{
						int idx = roi[i];
						B[idx * 3 + 0] = src[i].x();
						B[idx * 3 + 1] = src[i].y();
						B[idx * 3 + 2] = src[i].z();
					}
				}
				else
				{
					LOG(ERROR) << "not considered situation occur" << std::endl;
				}
			}
			SG::safeMemcpy(mean.data(), data_.data(), sizeof(float)*n_vertex_ * 3);
#pragma omp parallel for
			for (int y = 0; y < n_id_ - 1; y++)
			{
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					A(y, x) = data_[idx];
				}
			}
			for (int y = 0; y < n_id_ - 1; y++)
			{
				A(y, n_vertex_ * 3 + y) = 1.0f / ev[y];
			}
			//remapping
			int n_remap = vec_roi.size();
			matF A_roi(n_id_ - 1, n_remap  + n_id_ - 1);
			A_roi.setZero();
			vecF B_roi(n_remap + n_id_ - 1);
			B_roi.setZero();
			vecF mean_roi(n_remap + n_id_ - 1);
			mean_roi.setZero();
#pragma omp parallel for
			for (int i = 0; i < vec_roi.size(); i++)
			{
				int all_idx = vec_roi[i];				
				A_roi.col(i) = A.col(all_idx);
				B_roi(i) = B(all_idx);
				mean_roi(i) = mean(all_idx);			
			}

			matF AAt = A_roi * A_roi.transpose();
#pragma omp parallel for
			for (int y = 0; y < n_id_ - 1; y++)
			{
				AAt(y, y) += reg[y];
			}
			vecF AtB = A_roi * (B_roi - mean_roi);
			vecF res = AAt.ldlt().solve(AtB);
			if (isnan(res[0]))
			{
				LOG(INFO) << "went wrong" << std::endl;
			}
			coef.resize(res.size());
			for (int i = 0; i < coef.size(); i++)
			{
				coef[i] = res[i];
			}
		}

		template <class T>
		floatVec interpretID(const T& coef) const
		{
			if (coef.size() != n_id_ - 1)
			{
				LOG(ERROR) << "mean pos is include." << std::endl;
				return {};
			}
			floatVec res(n_vertex_ * 3, 0);
			for (int y = 0; y < n_id_ - 1; y++)
			{
#pragma omp parallel for
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					res[x] += coef[y] * data_[idx];
				}
			}
			floatVec res_add(n_vertex_ * 3);
			std::transform(res.begin(), res.end(),
				data_.begin(), res_add.begin(), std::plus<float>());
			return res_add;
		}

		template <class T, class T_ptr>
		void interpretIDFloat(T_ptr* res_ptr, const T& coef) const
		{
			if (coef.size() < n_id_ - 1)
			{
				LOG(ERROR) << "mean pos is include." << std::endl;
				return;
			}
			floatVec res(n_vertex_ * 3, 0);
			for (int y = 0; y < n_id_ - 1; y++)
			{
#pragma omp parallel for
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					res[x] += coef[y] * data_[idx];
				}
			}
			floatVec res_add(n_vertex_ * 3);
			std::transform(res.begin(), res.end(),
				data_.begin(), res_add.begin(), std::plus<float>());
			SG::safeMemcpy(res_ptr, res_add.data(), n_vertex_ * 3 * sizeof(float));
		}

		template <class T>
		void interpretIDFloatAdd(float* res_ptr, const T& coef) const
		{
			if (coef.size() != n_id_ - 1)
			{
				LOG(ERROR) << "mean pos is include." << std::endl;
				return;
			}
			floatVec res(n_vertex_ * 3, 0);
			for (int y = 0; y < n_id_ - 1; y++)
			{
#pragma omp parallel for
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					res[x] += coef[y] * data_[idx];
				}
			}
			floatVec base(res_ptr, res_ptr + n_vertex_ * 3);
			floatVec res_add(n_vertex_ * 3);
			std::transform(res.begin(), res.end(),
				base.begin(), res_add.begin(), std::plus<float>());
			SG::safeMemcpy(res_ptr, res_add.data(), n_vertex_ * 3 * sizeof(float));
		}

		template <class T>
		floatVec interpretSkipMean(const T& coef)
		{
			if (coef.size() != n_id_ - 1)
			{
				LOG(ERROR) << "size not fit" << std::endl;
				return {};
			}
			floatVec res(n_vertex_ * 3, 0);
			for (int y = 0; y < n_id_ - 1; y++)
			{
#pragma omp parallel for
				for (int x = 0; x < n_vertex_ * 3; x++)
				{
					int shift_size = n_vertex_ * 3;
					int idx = shift_size * n_exp_*(y + 1) + x;
					res[x] += coef[y] * data_[idx];
				}
			}
			return res;
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};



}

#endif
