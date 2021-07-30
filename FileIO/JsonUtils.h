#ifndef __JSONUTILS_H__
#define __JSONUTILS_H__
#include "../Basic/CGPBaseHeader.h"
namespace JsonTools
{
	using json = nlohmann::json;
	json keepKey(const json& src, const CGP::cstrVec keys);
}
#endif