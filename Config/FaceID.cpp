#include "FaceID.h"
using namespace CGP;
using namespace rttr;
#include "../FileIO/FileIO.h"
#include "../CalcFunction/CalcHelper.h"
RTTR_REGISTRATION
{
registration::class_<FaceIDFinder>("CGP::FaceIDFinder").constructor<>()
	.property("n_dim_", &CGP::FaceIDFinder::n_dim_)
	.property("n_id_", &CGP::FaceIDFinder::n_id_)
	.property("file_", &CGP::FaceIDFinder::file_)
	;

}
void FaceIDFinder::init()
{
	floatVec raw_data;
	FILEIO::loadFixedSizeDataFromBinary(root_ + file_, raw_data);
	if (raw_data.size() != n_dim_ * n_id_)
	{
		LOG(ERROR) << "raw_data size error, size not fit." << std::endl;
		return;
	}
	
	id_info_.resize(n_id_, 3);
	id_data_.resize(n_id_, 512);

#pragma omp parallel for
	for (int i = 0; i < n_id_; i++)
	{
		int shift = i * n_dim_;
		for (int j = 0; j < 3; j++)
		{
			id_info_(i, j) = raw_data[shift + j];
		}
		for (int j = 0; j < n_dim_-3; j++)
		{

			id_data_(i, j) = raw_data[shift + 3 + j];
		}
	}
}

void FaceIDFinder::getMatch(const vecF& input, int& match_id, bool is_debug) const
{
	int match_num = CalcHelper::getMinCosDisByRow(input, id_data_);
	match_id = id_info_(match_num, 1);
	//random match
	//match_id = std::rand() % 11;
	if (is_debug)
	{
		LOG(INFO) << "match info: " << id_info_(match_id, 1) << "_" << id_info_(match_id, 2) << std::endl;
	}	
}

void FaceIDFinder::getMatch(const vecF& input, int& match_id, doubleVec& res, bool is_debug) const
{
	int match_num = CalcHelper::getMinCosDisByRow(input, id_data_, res);
	match_id = id_info_(match_num, 1);
	//random match
	//match_id = std::rand() % 11;
	if (is_debug)
	{
		LOG(INFO) << "match info: " << id_info_(match_id, 1) << "_" << id_info_(match_id, 2) << std::endl;
	}
}



