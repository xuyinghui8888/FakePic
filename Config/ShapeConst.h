#ifndef SHAPE_CONST_H
#define SHAPE_CONST_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "ConstData.h"
#include "MnnHelper.h"
#include "JsonData.h"
namespace CGP
{
	class ShapeConst
	{
	public:	
		ShapeConst(const JsonData& init_json);
		std::shared_ptr<ConstData> ptr_data = nullptr;
	};	
}

#endif
