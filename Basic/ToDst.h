#ifndef __TODST_H__
#define __TODST_H__
#include "CGPBaseHeader.h"
namespace TDST
{
	template<class T>
	std::set<T> vecToSet(const std::vector<T>& input)
	{
		return std::set<T>(input.begin(), input.end());
	}

	template<class T>
	void unpackData(const std::vector<T>& src, T& res)
	{
		for (int i = 0; i < src.size(); i++)
		{
			for (auto iter_data : src[i])
			{
				res.push_back(iter_data);
			}
		}
	}

	bool isBase64(const char c);
	CGP::cstr base64Encode(const char * bytes_to_encode, unsigned int in_len);
	CGP::cstr base64Decode(CGP::cstr const & encoded_string);
	void getBased64FromFiles(const CGP::cstr& file_path, CGP::cstr& res);
}
#endif