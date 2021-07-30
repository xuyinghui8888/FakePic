#ifndef METRIC_ERROR_H
#define METRIC_ERROR_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include <rttr/type>
#include <rttr/registration>
#include <rttr/registration_friend>
namespace CGP
{
	namespace Metric
	{
		MeshCompress getError(const MeshCompress& src, const MeshCompress& dst, 
			float max_data, float data_scale = 1.0f);

		MeshCompress getError(const MeshCompress& src, const MeshCompress& dst);
	}
}

#endif
