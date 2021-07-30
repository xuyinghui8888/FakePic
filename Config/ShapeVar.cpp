#include "ShapeVar.h"
using namespace CGP;

ShapeVar::ShapeVar(const LinuxJsonData& init_json)
{
	ptr_data = std::make_shared<ConstData>(init_json);
}