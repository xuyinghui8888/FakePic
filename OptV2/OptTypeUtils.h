#ifndef OPT_TYPE_UTIL
#define OPT_TYPE_UTIL
#include "../Mesh/MeshCompress.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/Tensor.h"
#include "../Sysmetric/Sysmetric.h"
#include "../MeshDeform/DTUtilities.h"
#include "../Config/FaceID.h"
namespace CGP
{	
	class FaceTypeInfo
	{
	public:
		int n_type_;
		std::vector<vecD> bs_;
		std::vector<vecD> coef_3dmm_;
		std::vector<MeshCompress> mesh_;
		std::vector<float3Vec> landmark_68_;
		void getBs(int input_type, vecD& res) const;
		void getBFM68(int input_type, float3Vec& res) const;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	class FaceStarInfo
	{
	public:
		int n_type_;
		std::vector<vecD> bs_;
		std::vector<vecD> coef_3dmm_;
		std::vector<vecD> faceid_;
		std::vector<MeshCompress> mesh_;
		std::vector<float3Vec> landmark_68_;
		std::vector<json> fix_json_;
		FaceIDFinder tensor_id_;
		void getBs(int input_type, vecD& res) const;
		void getBFM68(int input_type, float3Vec& res) const;
		json getFixJson(int input_type) const;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	class FaceAttPack
	{
	public:
		int eye_type_;
		int face_type_;
		int mouth_type_;
		int nose_type_;
		vecD coef_3dmm_;
		vecD faceid_;
		int match_id_ = -1;
		doubleVec match_dis_;
		double blend_weight_ = 0.8;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};	

	namespace OptTools
	{
		template<class T>
		bool checkAndFixValue(double bound_lower, double bound_upper, T& eigen_vec)
		{
			bool tag = true;
			for (int i = 0; i < eigen_vec.size(); i++)
			{
				if (eigen_vec[i] > bound_upper)
				{
					tag = false;
					LOG(WARNING) << "index " << i << ", out of max " << bound_upper << std::endl;
					eigen_vec[i] = bound_upper;

				}
				else if (eigen_vec[i] < bound_lower)
				{
					tag = false;
					LOG(WARNING) << "index " << i << ", out of min " << bound_lower << std::endl;
					eigen_vec[i] = bound_lower;
				}
			}
			return tag;
		}

		template<class T1, class T2>
		void bubbleSort(const T1& p_input, T2& ind_diff)
		{
			auto p = p_input;
			int length = p.size();
			ind_diff.resize(length);
			for (int m = 0; m < length; m++)
			{
				ind_diff[m] = m;
			}

			for (int i = 0; i < length; i++)
			{
				for (int j = 0; j < length - i - 1; j++)
				{
					if (p[j] > p[j + 1])
					{
						float temp = p[j];
						p[j] = p[j + 1];
						p[j + 1] = temp;

						int ind_temp = ind_diff[j];
						ind_diff[j] = ind_diff[j + 1];
						ind_diff[j + 1] = ind_temp;
					}
				}
			}
		}

		template <typename T>
		std::vector<size_t> sortIndexes(const std::vector<T> &v)
		{

			// initialize original index locations
			std::vector<size_t> idx(v.size());
			iota(idx.begin(), idx.end(), 0);

			// sort indexes based on comparing values in v
			// using std::stable_sort instead of std::sort
			// to avoid unnecessary index re-orderings
			// when v contains elements of equal values 
			std::stable_sort(idx.begin(), idx.end(),
				[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

			return idx;
		}

		template <class T1, class T2>
		void sortIndexesEigen(const T1 &v, T2& idx) {

			doubleVec v_vec(v.size(), 0);
			for (int i = 0; i < v.size(); i++)
			{
				v_vec[i] = v[i];
			}
			auto idx_vec = sortIndexes(v_vec);
			idx.resize(v.size());
			for (int i = 0; i < v.size(); i++)
			{
				idx[i] = idx_vec[i];
			}
		}

	}
}

#endif
