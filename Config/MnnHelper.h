#ifndef MNN_HELPER_H
#define MNN_HELPER_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "JsonData.h"
#include "../MNN/mtcnn.h"
#include "../MNN/Facenet.h"
#include "../MNN/DeepRec3d.h"
#include "../MNN/NRRec3d.h"
#include "../MNN/landmark68.h"
namespace CGP
{
	class MnnHelper
	{
	public:
		MnnHelper(const JsonData& init_json);
		std::shared_ptr<MTCNN_SCOPE::MTCNN> mtcnn_ = nullptr;
		std::shared_ptr<FaceNet> facenet_ = nullptr;
		std::shared_ptr<Deep3D> deep3d_ = nullptr;
		std::shared_ptr<NR3D> deep3d_nr_ = nullptr;
		std::shared_ptr<Landmark68> landmark_68_ = nullptr;

		template<class T>
		void getFaceID(const cv::Mat& img, T& res, bool use_mtcnn = true)
		{
			if (use_mtcnn)
			{
				std::vector< MTCNN_SCOPE::FaceInfo> finalBbox;
				mtcnn_->Detect_T(img, finalBbox);
				cv::Mat aligned_res = PREPROCESS_IMAGE::alignToMtcnn(img, finalBbox[0].landmark, false);
				facenet_->inference(aligned_res, 128, 128, res);
			}
			else
			{
				facenet_->inference(img, 128, 128, res);
			}

		}
	};	
}

#endif
