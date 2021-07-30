#include "OptTypeV2.h"
#include "../Basic/MeshHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../Solver/BVLSSolver.h"
using namespace CGP;

void FaceTypeInfo::getBs(int input_type, vecD& res) const
{
	if (input_type < 0 || input_type >= n_type_)
	{
		LOG(ERROR) << "getBs for type: " << input_type << ", is out of range." << std::endl;
		LOG(WARNING) << "use default 0 instead." << std::endl;
		res = bs_[0];
	}
	else
	{
		res = bs_[input_type];
	}
}

void FaceTypeInfo::getBFM68(int input_type, float3Vec& res) const
{
	if (input_type < 0 || input_type >= n_type_)
	{
		LOG(ERROR) << "getBs for type: " << input_type << ", is out of range." << std::endl;
		LOG(WARNING) << "use default 0 instead." << std::endl;
		res = landmark_68_[0];
	}
	else
	{
		res = landmark_68_[input_type];
	}
}

void FaceStarInfo::getBs(int input_type, vecD& res) const
{
	if (input_type < 0 || input_type >= n_type_)
	{
		LOG(ERROR) << "getBs for type: " << input_type << ", is out of range." << std::endl;
		LOG(WARNING) << "use default 0 instead." << std::endl;
		res = bs_[0];
	}
	else
	{
		res = bs_[input_type];
	}
}

void FaceStarInfo::getBFM68(int input_type, float3Vec& res) const
{
	if (input_type < 0 || input_type >= n_type_)
	{
		LOG(ERROR) << "getBs for type: " << input_type << ", is out of range." << std::endl;
		LOG(WARNING) << "use default 0 instead." << std::endl;
		res = landmark_68_[0];
	}
	else
	{
		res = landmark_68_[input_type];
	}
}

json FaceStarInfo::getFixJson(int input_type) const
{
	if (input_type < 0 || input_type >= n_type_)
	{
		LOG(ERROR) << "getBs for type: " << input_type << ", is out of range." << std::endl;
		LOG(WARNING) << "use default 0 instead." << std::endl;
		return fix_json_[0];
	}
	else
	{
		return fix_json_[input_type];
	}
}




