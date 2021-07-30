#ifndef JSON_TEMPLATE_H
#define JSON_TEMPLATE_H
#include "../Basic/CGPBaseHeader.h"
#include <rttr/type>
#include <rttr/registration>
#include <rttr/registration_friend>
namespace CGP
{
	//used for not changable programs
	using json = nlohmann::json;
	using namespace rttr;
	enum class Gender
	{
		MALE, //0
		FEMALE,//1
	};
	namespace JsonHelper
	{	
		template <class T>
		bool checkData(const T& src, const json& init)
		{
			LOG(INFO) << "loading: " << src.root_ + src.json_file_ << std::endl;
			rttr::type t = rttr::type::get<T>();
			LOG(INFO) << "checking: " <<t.get_name() << std::endl;
			for (auto& prop : t.get_properties())
			{
				std::string iter_name = prop.get_name().to_string();
				if (init.count(iter_name))
				{

				}
				else
				{
					LOG(ERROR) << iter_name << " is not in config.json" << std::endl;
					return false;
				}
			}
			LOG(INFO) << "checkData successed." << std::endl;
			return true;
		}

		template <class T>
		void setData(const json& init, T& src)
		{
			LOG(INFO) << "loading: " << src.root_ + src.json_file_ << std::endl;
			LOG(INFO) << "setting data: " << rttr::type::get<T>().get_name() << std::endl;
			for (auto& prop : rttr::type::get(src).get_properties())
			{
				std::string iter_name = prop.get_name().to_string();
				auto res_iter = init.find(iter_name);
				if (res_iter == init.end())
				{
					LOG(ERROR) << iter_name << " is not in config.json" << std::endl;
					return;
				}
				else
				{
					if (prop.get_type() == type::get<int>())
					{
						int iter_value = *res_iter;
						if (!prop.set_value(src, iter_value))
						{
							LOG(ERROR) << iter_name << " setting value failed." << std::endl;
						}
					}
					else if (prop.get_type() == type::get<float>())
					{
						float iter_value = *res_iter;
						if (!prop.set_value(src, iter_value))
						{
							LOG(ERROR) << iter_name << " setting value failed." << std::endl;
						}
					}
					else if (prop.get_type() == type::get<double>())
					{
						double iter_value = *res_iter;
						if (!prop.set_value(src, iter_value))
						{
							LOG(ERROR) << iter_name << " setting value failed." << std::endl;
						}
					}
					else if (prop.get_type() == type::get<std::string>())
					{
						std::string iter_value = *res_iter;
						if (!prop.set_value(src, iter_value))
						{
							LOG(ERROR) << iter_name << " setting value failed." << std::endl;
						}
					}
					else if (prop.get_type() == type::get<bool>())
					{
						bool iter_value = *res_iter;
						if (!prop.set_value(src, iter_value))
						{
							LOG(ERROR) << iter_name << " setting value failed." << std::endl;
						}
					}
					else
					{
						LOG(ERROR) << "Data type not included." << std::endl;
					}
				}
			}
			LOG(INFO) << "JsonLoader successed for " << rttr::type::get<T>().get_name() << std::endl;
		}
		
		template <class T>
		void initData(const std::string& root, const std::string& file, T& res)
		{
			LOG(INFO) << "loading: " << res.root_ + res.json_file_ << std::endl;
			res.load(root, file);
			if (!checkData(res, res.init_))
			{
				LOG(ERROR) << "loading failed, certain items not found." << std::endl;
				return;
			}
			setData(res.init_, res);
			res.init();
		}
	}
}

#endif
