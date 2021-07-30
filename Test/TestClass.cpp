#include "TestClass.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
#include "../Config/ShapeVar.h"

using namespace CGP;
using namespace TESTSTRUCTURE;

void TaobaoSkeleton::print(const cstr& instance)
{
	LOG(INFO) << instance << std::endl;
	LOG(INFO) << "rotate: " << rotate.transpose() << std::endl;
	LOG(INFO) << "trans: " << trans.transpose() << std::endl;
	LOG(INFO) << "scale: " << scale.transpose() << std::endl;
}

double TaobaoSlide::operator() (double lb, double ub)
{
	return DLIP3(value, lb, ub);
}

void TaobaoSlide::print(const cstr& instance)
{
	LOG(INFO) << instance <<": "<< value<< std::endl;
}