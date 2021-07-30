#include "ConstVar.h"
using namespace CGP;

ConstVar::ConstVar(const JsonData& init_json)
{
	ptr_data = std::make_shared<ConstData>(init_json);
	ptr_model = std::make_shared<MnnHelper>(init_json);
}
