#ifndef EYEBROW_TYPE
#define EYEBROW_TYPE
#include "../Mesh/MeshCompress.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/Tensor.h"
#include "../Sysmetric/Sysmetric.h"
#include "../MeshDeform/DTUtilities.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
namespace CGP
{	
	class EyebrowType
	{
	public:
		EyebrowType(const json& config);
		void init(const json& config);
		void setDebug(bool is_debug);
		void getEyebrowType(json& json_in, json& json_out);
		void getEyebrowTypeOnlyLandmark(json& json_in, json& json_out);
		void setResultDir(const cstr& init);

	private:
		int getTypeFromTensor(const Tensor& tensor, const float3Vec& landmark_pos, vecD& coef);
		int getTypeFromTensor(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef);
		int getTypeFromDist(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef);
		int getTypeFromDistLocalRatio(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef);
		int getTypeFromDistLocalAngle(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef);
		int getTypeFromAngle(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef);
		void getLocalCord(const float3Vec& landmark_pos, const vecF& actual_pos, vecF& local_pos);
		void getLocalCord(const vecF& landmark_pos_vec, const vecF& actual_pos, vecF& local_pos);
		int calculateEyebrowTypeUseLandmark(const floatVec& landmark_106);
		int calculateEyebrowTypeUseLandmark(const floatVec& landmark_106, const intVec& seg);
		floatVec adjust106Landmark(const cvMatU& right_mask, const floatVec& landmark_106);
		void getLocalVec(const float3Vec& landmark_pos, float3Vec& local_vec);
		void getVec(const float3Vec& landmark_pos, float3Vec& local_vec);
		int getTypeFrom(const std::vector<vecD>& dist);
		int getTypeFromOrder(const std::vector<vecD>& dist);
		void initSegmentImage(const intVec& seg);
		void getRightSideOfMask(const cvMatU& seg_2, const cvMatU& seg_3, cvMatU& res);
		void getCombineAndSegRight(const cvMatU& seg_2, const cvMatU& seg_3, cvMatU& res);


	private:
		bool debug_ = false;
		cstr result_dir_ = "";
		cstr data_root_;		
		Tensor eyebrow_tensor_;
		bool is_init_ = false;
		std::vector<cvMatU> canvas_seg_;
		floatVec landmark_106_adjust_ = {};
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}

#endif
