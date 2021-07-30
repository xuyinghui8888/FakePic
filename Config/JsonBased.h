#ifndef JSON_BASE_CLASS_H
#define JSON_BASE_CLASS_H
#include "../Basic/CGPBaseHeader.h"
#include <rttr/type>
#include <rttr/registration>
#include <rttr/registration_friend>
namespace CGP
{
	//used for not changable programs
	//用来作为初始化数据组合的基类
	using json = nlohmann::json;
	class JsonBaseClass
	{
	public:
		JsonBaseClass();
		void load(const std::string& root, const std::string& json_name);
		//process init process
		void init();
		std::string json_file_;
		std::string root_;
		json init_;		

		virtual ~JsonBaseClass() {};
		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};	
}

#endif
