#ifndef SHAPE_VAR_H
#define SHAPE_VAR_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "ConstData.h"
#include "JsonData.h"
namespace CGP
{
	class ShapeVar
	{
	public:
		ShapeVar(const LinuxJsonData& init_json);
		std::shared_ptr<ConstData> ptr_data = nullptr;
	};
}

#endif
