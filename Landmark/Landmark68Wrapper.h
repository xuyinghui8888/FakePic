#ifndef __LANDMARK_68_WRAPPER_H__
#define __LANDMARK_68_WRAPPER_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "../Config/MnnHelper.h"
namespace CGP
{
	//only due with triangle mesh
	class SimLandmarkWrapper
	{
	private:
		std::shared_ptr<MnnHelper> ptr_model = nullptr;		
		bool is_debug_;
	public:
		cstr result_dir_;
		SimLandmarkWrapper(cstr& pack_data_root, cstr result_dir = "", bool is_debug = false);
		void shiftBBLandmark68(const cv::Mat& img, const cv::Rect& src, int resolution, cv::Mat& dst, cv::Rect& scale_bb);
		void getLandmarks(const cv::Mat& img, vecF& res);
		void getLandmark68(const cv::Mat& input, vecF& res);
		void processFaceDetection(const cv::Mat& input, cv::Rect& res);
		void reflectBackToImage(const cv::Rect& landmark_68_crop, int resolution,
			const vecF& res_256, vecF& res);
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}
#endif