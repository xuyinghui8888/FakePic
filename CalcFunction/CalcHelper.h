#ifndef CALCULATE_HELPER_H
#define CALCULATE_HELPER_H
#include "../Basic/CGPBaseHeader.h"
#include "../Basic/MeshHeader.h"
namespace CGP
{
	namespace CalcHelper
	{
		//only deal with zero
		void keepRatioStrechX(const floatVec& src, const intVec& roi, int left_idx, int right_idx,
			float scale, floatVec& res);

		double getEigenVectorDis(const MeshCompress& base, intVec& src_idx, const intVec& dst_idx);

		template<class T1, class T2>
		void vectorToEigen(const T1& input, T2& res)
		{
			res.resize(input.size());
			for (int i = 0; i < input.size(); i++)
			{
				res[i] = input[i];
			}
		}

		template<class T>
		T multiMin(const T& src1, const T& src2)
		{
			if (src1.size() != src2.size())
			{
				LOG(ERROR) << "src1 & src2 size not fit." << std::endl;
				return src1.size() > src2.size() ? src1 : src2;
			}

			auto res = src1;
			for (int i = 0; i < src1.size(); i++)
			{
				res[i] = DMIN(src1[i], src2[i]);
			}
			return res;
		}

		template<class T>
		T multiMax(const T& src1, const T& src2)
		{
			if (src1.size() != src2.size())
			{
				LOG(ERROR) << "src1 & src2 size not fit." << std::endl;
				return src1.size() > src2.size() ? src1 : src2;
			}

			auto res = src1;
			for (int i = 0; i < src1.size(); i++)
			{
				res[i] = DMAX(src1[i], src2[i]);
			}
			return res;
		}

		template<class T>
		T multiMinus(const T& src1, const T& src2)
		{
			if (src1.size() != src2.size())
			{
				LOG(ERROR) << "src1 & src2 size not fit." << std::endl;
				return src1.size() > src2.size() ? src1 : src2;
			}
			auto res = src1;
			for (int i = 0; i < src1.size(); i++)
			{
				res[i] = src1[i] - src2[i];
			}
			return res;
		}	

		template<class T>
		double multiMaxDouble(const T& src)
		{
			double max_dis = -INT16_MAX;
			for (int i = 0; i < src.size(); i++)
			{
				max_dis = max_dis > src[i] ? max_dis : src[i];
			}
			return max_dis;
		}

		template<class T>
		double averageValue(const std::vector<T>& src)
		{
			if (src.empty())
			{
				return 0;
			}

			double average = std::accumulate(src.begin(), src.end(), 0.0) / (1.0*src.size());
			return average;
		}
		
		template<class T>
		std::vector<T> removeWithoutHeadTail(const std::vector<T>& src, double ratio)
		{
			if (ratio >= 0.5)
			{
				LOG(WARNING) << "remove ratio is extra large, reset to 0.2 instead." << std::endl;
				ratio = 0.2;
			}
			auto res = src;
			std::sort(res.begin(), res.end());
			int length = res.size();
			int start = int(ratio*length);
			int end = int((1 - ratio)*length);
			end = DMINI(res.size(), end + 1);
			//需要先分配空间再copy
			std::vector<T> part;
			part.resize(end - start);
			std::copy(res.begin() + start , res.begin() + end , part.begin());
			return part;
		}

		template<class T>
		double midPosAverage(const std::vector<T>& src, double ratio)
		{
			auto mid_pos_value = removeWithoutHeadTail(src, ratio);
			return averageValue(mid_pos_value);
		}

		template<class T1, class T2>
		void getMidAverageMulti(const T1& src_eigen_vec, double ratio, T2& res)
		{
			if (src_eigen_vec.empty())
			{
				LOG(WARNING) << "src is empty." << std::endl;
				return;
			}
			else
			{
				int n_dim = src_eigen_vec[0].size();
				doubleX2Vec src_vec(n_dim);
				for (int i = 0; i < src_eigen_vec.size(); i++)
				{
					for (int iter_dim = 0; iter_dim < n_dim; iter_dim++)
					{
						src_vec[iter_dim].push_back(src_eigen_vec[i][iter_dim]);
					}
				}
				res.resize(src_eigen_vec[0].size());
				for (int i = 0; i < n_dim; i++)
				{
					res[i] = midPosAverage(src_vec[i], ratio);
				}
			}
		}

		template<class T1, class T2>
		float getEulerDis(const T1& src, const T2& dst)
		{
			return (src - dst).norm();
		}

		template<class T1, class T2>
		float getMinusCosDis(const T1& src, const T2& dst)
		{
			float length = src.norm()*dst.norm();
			return 1- safeDiv(src.dot(dst), length, INTMAX_MAX);
		}

		template<class T1, class T2>
		int getMinEulerDisByRow(const T1& src, const T2& base)
		{
			if (src.cols() != base.cols())
			{
				LOG(ERROR) << "src cols and base not the same size." << std::endl;
				return -1;
			}
			doubleVec res(base.rows(), INT_MAX);
			for (int i = 0; i < base.rows(); i++)
			{
				res[i] = (src - base.row(i)).norm();
			}
			auto min_max = std::minmax_element(res.begin(), res.end());
			LOG(INFO) << "min_dis: " << *(min_max.first) << ", idx: " << min_max.first - res.begin() << std::endl;
			LOG(INFO) << "max_dis: " << *(min_max.second) << ", idx: " << min_max.second - res.begin() << std::endl;
			return min_max.first - res.begin();
		}
		
		template<class T1, class T2>
		int getMinCosDisByRow(const T1& src, const T2& base)
		{
			if (src.cols()*src.rows() != base.cols())
			{
				LOG(ERROR) << "src cols and base not the same size." << std::endl;
				return -1;
			}
			doubleVec res(base.rows(), INT_MAX);
			for (int i = 0; i < base.rows(); i++)
			{
				float length = src.norm()*base.row(i).norm();
				res[i] = 1 - safeDiv(src.dot(base.row(i)), length, INTMAX_MAX);
			}
			auto min_max = std::minmax_element(res.begin(), res.end());
			LOG(INFO) << "min_dis: " << *(min_max.first) << ", idx: " << min_max.first - res.begin() << std::endl;
			LOG(INFO) << "max_dis: " << *(min_max.second) << ", idx: " << min_max.second - res.begin() << std::endl;
			return min_max.first - res.begin();
		}

		template<class T1, class T2>
		int getMinCosDisByRow(const T1& src, const T2& base, doubleVec& res)
		{
			if (src.cols()*src.rows() != base.cols())
			{
				LOG(ERROR) << "src cols and base not the same size." << std::endl;
				return -1;
			}
			res = doubleVec(base.rows(), INT_MAX);
			for (int i = 0; i < base.rows(); i++)
			{
				float length = src.norm()*base.row(i).norm();
				res[i] = 1 - safeDiv(src.dot(base.row(i)), length, INTMAX_MAX);
			}
			auto min_max = std::minmax_element(res.begin(), res.end());
			LOG(INFO) << "min_dis: " << *(min_max.first) << ", idx: " << min_max.first - res.begin() << std::endl;
			LOG(INFO) << "max_dis: " << *(min_max.second) << ", idx: " << min_max.second - res.begin() << std::endl;
			return min_max.first - res.begin();
		}

		template<class T1, class T2>
		std::vector<T1> keepValueBiggerThan(const std::vector<T1>& src, const T2 thres)
		{
			std::vector <T1> res;
			for (auto i:src)
			{
				if (i > thres)
				{
					res.push_back(i);
				}
			}
			return res;
		}

		template<class T>
		T scaleValue(const T& src, double scale)
		{
			auto dst = src;
			for (int i = 0; i< src.size(); i++)
			{
				dst[i] = scale * src[i];
			}
			return dst;
		}

		template<class T>
		T transformValue(const T& src, double scale, double shift)
		{
			auto dst = src;
			for (int i = 0; i < src.size(); i++)
			{
				dst[i] = scale * src[i] - shift;
			}
			return dst;
		}

		template<class T>
		std::vector<T> mappingToRoi(const std::vector<T>& input, const intVec& roi)
		{
			int n_size = input.size();
			std::vector<T> res;
			for (int i : roi)
			{
				if (i >= n_size)
				{
					LOG(ERROR) << "roi index out of range for input" << std::endl;
				}
				else
				{
					res.push_back(input[i]);
				}
			}
			return res;
		}

		template<class T>
		T expandVector(const T& vec, int n_expansions)
		{
			const int n_vec = vec.size();
			T vec_res = vec;
			vec_res.reserve(n_expansions * n_vec);
			for (int i = 0; i < n_expansions-1; ++i)
			{
				vec_res.insert(vec_res.end(), vec.begin(), vec.begin() + n_vec);
			}
			return vec_res;
		};

		template<class T1, class T2>
		void getEigenAverage(const T1& input, T2& res)
		{
			if (input.empty())
			{
				res.resize(1);
				res.setConstant(0);
				return;
			}
			res = input[0];
			res.setConstant(0);
			for (int i = 0; i < input.size(); i++)
			{
				res = res + input[i];
			}
			res = res * 1.0 / input.size();
		}

		template<class T>
		double getEigenVectorDis(const T& src, const T& dst)
		{
			if (src.empty() || src.size() != dst.size())
			{
				LOG(ERROR) << "size not fit" << std::endl;
				return INT_MAX;
			}
			double error_value = 0;
			for (int i = 0; i < src.size(); i++)
			{
				error_value += (src[i] - dst[i]).norm();
			}
			return error_value;
		}

		template<class T>
		double getEigenVectorDis(const T& src, const T& dst, const intVec& roi)
		{
			if (src.empty() || src.size() != dst.size())
			{
				LOG(ERROR) << "size not fit" << std::endl;
				return INT_MAX;
			}
			double error_value = 0;
			int n_size = dst.size();
			for (int i = 0; i < roi.size(); i++)
			{
				int idx = roi[i];
				error_value += (src[idx] - dst[idx]).norm();
			}
			return error_value;
		}

		template<class T>
		double getEigenVectorDis(const T& src, const intVec& roi, const T& dst)
		{
			if (src.empty() || roi.size() != dst.size())
			{
				LOG(ERROR) << "size not fit" << std::endl;
				return INT_MAX;
			}
			double error_value = 0;
			int n_size = dst.size();
			for (int i = 0; i < roi.size(); i++)
			{
				int idx = roi[i];
				error_value += (src[idx] - dst[i]).norm();
			}
			return error_value;
		}

		template<class T1, class T2>
		void pairSort(const T1& p_input, T2& ind_diff)
		{
			auto p = p_input;
			int length = p.size();
			ind_diff.resize(length);

			std::vector<std::pair<double, int> >value_idx;
			for (int i = 0; i < length; i++)
			{
				// filling the original array
				value_idx.push_back(std::make_pair(p_input[i], i)); // k = value, i = original index
			}
			sort(value_idx.begin(), value_idx.end());

			for (int m = 0; m < length; m++)
			{
				ind_diff[m] = value_idx[m].second;
			}
			//reverse to original order
			auto ind_reorder = ind_diff;
			for (int m = 0; m < length; m++)
			{
				ind_reorder[value_idx[m].second] = m;
			}
			ind_diff = ind_reorder;
		}

		template<class T1, class T2>
		T1 appendVector(const T1& vec, T2& vec_in_stl, typename std::enable_if<std::is_same<T1, T2>::value>::type* = nullptr)
		{
			if (vec_in_stl.empty())
			{
				return vec;
			}

			T1 vec_res = vec;
			vec_res.insert(vec_res.end(), vec_in_stl.begin(), vec_in_stl.end());
			return vec_res;		

		}

		template<class T>
		T appendVector(const T& vec, std::vector<T>& vec_in_stl)
		{
			if (vec_in_stl.empty())
			{
				return vec;
			}
			else if (sameType(vec, vec_in_stl[0]))
			{
				T1 vec_res = vec;

				for (int i = 0; i < vec_in_stl.size(); ++i)
				{
					vec_res.insert(vec_res.end(), vec_in_stl[i].begin(), vec_in_stl[i].end());
				}

				return vec_res;
			}
			else
			{
				LOG(WARNING) << "data format not match. return default instead." << std::endl;
				return vec;
			}
		}

	}
}
#endif
