#include "MnnHelper.h"
using namespace CGP;
MnnHelper::MnnHelper(const JsonData& init_json)
{
	//init for mtcnn
	mtcnn_.reset(new MTCNN_SCOPE::MTCNN);
	mtcnn_->init(init_json);
	//init for arcface
	facenet_.reset(new FaceNet);
	facenet_->init(init_json);
	//init for deep3d
	deep3d_.reset(new Deep3D);
	deep3d_->init(init_json);
	//init for nr
	deep3d_nr_.reset(new NR3D);
	deep3d_nr_->init(init_json, init_json.deep3d_nr_);
	//init for landmark68
	landmark_68_.reset(new Landmark68);
	landmark_68_->init(init_json);
}