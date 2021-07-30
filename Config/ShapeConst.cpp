#include "ShapeConst.h"
using namespace CGP;

ShapeConst::ShapeConst(const JsonData& init_json)
{
	ptr_data = std::make_shared<ConstData>(init_json);
}