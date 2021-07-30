#include "JsonUtils.h"
using namespace CGP;
using namespace JsonTools;

json JsonTools::keepKey(const json& src, const CGP::cstrVec keys)
{
	json res;
	for (cstr iter_keys : keys)
	{
		auto pos = src.find(iter_keys);
		if (pos != src.end())
		{
			res[iter_keys] = src[iter_keys];
		}
	}
	return res;
}