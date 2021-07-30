#ifndef AUTO3D_DATA_H
#define AUTO3D_DATA_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonBased.h"
#include <rttr/type>
#include <rttr/registration>
#include <rttr/registration_friend>
namespace AUTO3D
{
	//used for not changable programs
	using json = nlohmann::json;
	class Auto3DData: public CGP::JsonBaseClass
	{
	public:
		using JsonBaseClass::JsonBaseClass;
		//check data using create data
		CGP::cstr mouth_data_;
		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};	
}

#endif
