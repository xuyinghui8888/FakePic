#include "JsonBased.h"
#include "../Basic/SafeGuard.h"
#include "JsonHelper.h"
using namespace CGP;
using namespace rttr;
JsonBaseClass::JsonBaseClass()
{
	root_ = "";
	json_file_ = "";
}

void JsonBaseClass::load(const std::string& root, const std::string& json_name)
{
	root_ = root;
	json_file_ = json_name;
	std::ifstream in_stream(root_ + json_file_);
	if (!in_stream.is_open())
	{
		LOG(ERROR) << "open file failed: " << root_ + json_file_ << std::endl;
	}
	in_stream >> init_;
	in_stream.close();	
	LOG(INFO) << "loading config from: " << root_ + json_file_ << std::endl;
}

void JsonBaseClass::init()
{

}
