#ifndef __GLOGHELPER_H__
#define __GLOGHELPER_H__
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <glog/stl_logging.h>

namespace CGP
{
	void SignalHandle(const char* data, int size);
	class GLogHelper
	{
	public:
		GLogHelper(char* program);
		~GLogHelper();
	};
}
#endif