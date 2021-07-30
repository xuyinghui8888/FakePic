#ifndef CONST_VAR_H
#define CONST_VAR_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "ConstData.h"
#include "MnnHelper.h"
#include "JsonData.h"
namespace CGP
{
	//const是为了原有数据准备的，即data/model混合
	class ConstVar
	{
	public:	
		ConstVar(const JsonData& init_json);
		std::shared_ptr<ConstData> ptr_data = nullptr;
		std::shared_ptr<MnnHelper> ptr_model = nullptr;
	};	
}

#endif
