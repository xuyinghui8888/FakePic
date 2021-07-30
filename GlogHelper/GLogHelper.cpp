#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <boost/filesystem.hpp> 
#include "GLogHelper.h"
#include <gflags/gflags.h>
#include "../Basic/DefinedMacro.h"

namespace CGP
{
	void SignalHandle(const char* data, int size)
	{
		std::string str = std::string(data, size);
		LOG(ERROR) << str;
	}

	GLogHelper::GLogHelper(char* program)
	{
		boost::filesystem::create_directory(LOGDIR);
		google::InitGoogleLogging(program);
		//printing log level > info
		google::SetStderrLogging(google::INFO); 
		FLAGS_colorlogtostderr = true;   
		google::SetLogDestination(google::INFO, LOGDIR"/INFO_");
		google::SetLogDestination(google::WARNING, LOGDIR"/WARNING_");   
		google::SetLogDestination(google::ERROR, LOGDIR"/ERROR_");  
		 //set log buffer time to 5 seconds
		FLAGS_logbufsecs = 0;
		//max log size is 100 mb
		FLAGS_max_log_size = 100;
		FLAGS_stop_logging_if_full_disk = true;
		google::SetLogFilenameExtension("CGP_");
		google::InstallFailureSignalHandler();
		google::InstallFailureWriter(&SignalHandle);
	}

	GLogHelper::~GLogHelper()
	{
		google::ShutdownGoogleLogging();
	}
}
