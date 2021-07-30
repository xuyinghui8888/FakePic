#include "PostProcess.h"
#include "../FileIO/FileIO.h"
#include "../MeshDeform/LaplacianDeformation.h"

using namespace CGP;
using namespace rttr;

RTTR_REGISTRATION
{
registration::class_<MeshPostProcess>("CGP::MeshPostProcess").constructor<>()
	.property("f_bfm_deform_", &CGP::MeshPostProcess::f_bfm_deform_)
	.property("f_bfm_id_trans_", &CGP::MeshPostProcess::f_bfm_id_trans_)
	.property("f_bfm_mid_", &CGP::MeshPostProcess::f_bfm_mid_)
	.property("f_fwh_deform_", &CGP::MeshPostProcess::f_fwh_deform_)
	.property("f_fwh_id_trans_", &CGP::MeshPostProcess::f_fwh_id_trans_)
	.property("f_fwh_mid_", &CGP::MeshPostProcess::f_fwh_mid_)
	.property("f_mouth_close_", &CGP::MeshPostProcess::f_mouth_close_)
	.property("f_fwh_fix_", &CGP::MeshPostProcess::f_fwh_fix_)
	;

}
void MeshPostProcess::init()
{
	bfm_deform_ = FILEIO::loadIntDynamic(root_ + f_bfm_deform_);
	bfm_id_trans_ = FILEIO::loadIntDynamic(root_ + f_bfm_id_trans_);
	bfm_mid_ = FILEIO::loadIntDynamic(root_ + f_bfm_mid_);
	fwh_deform_ = FILEIO::loadIntDynamic(root_ + f_fwh_deform_);
	fwh_id_trans_ = FILEIO::loadIntDynamic(root_ + f_fwh_id_trans_);
	fwh_mid_ = FILEIO::loadIntDynamic(root_ + f_fwh_mid_);
	mouth_close_ = FILEIO::loadIntDynamic(root_ + f_mouth_close_);
	fwh_fix_ = FILEIO::loadIntDynamic(root_ + f_fwh_fix_);

	//safe check
	if (bfm_deform_.size() != fwh_deform_.size())
	{
		//bfm_deform id match fwh_deform id
		LOG(ERROR) << "loading error for bfm && fwh deform pair not match" << std::endl;
	}

	if (fwh_id_trans_.size() != bfm_id_trans_.size())
	{
		//fwh_id_trans_ bfm_id_trans_ match to calculate transform
		LOG(ERROR) << "loading error for fwh_id_trans_ && bfm_id_trans_ deform pair not match" << std::endl;
	}

	if (bfm_mid_.size() != fwh_mid_.size())
	{
		//on mid pos points
		LOG(ERROR) << "loading error for bfm_mid_ && fwh_mid_ deform pair not match" << std::endl;
	}
}



