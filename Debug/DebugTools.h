#ifndef DEBUG_TOOLS_H
#define DEBUG_TOOLS_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
namespace CGP
{
	//used for testing codes
	namespace DebugTools
	{
		void printJson(const json& init);
		void printJsonStructure(const json& init);
		cv::Mat projectVertexToXY(const std::vector<float3Vec>& src, const std::vector<cv::Scalar>& colors, int image_w = 256, int image_h = 256);
		template<class T>
		void cgPrint(const T& src, const cstr& trash = "", const char& sep = ',')
		{
			std::ostringstream out;
			if (trash != "")
			{
				out << trash << sep;
			}
			for (auto i : src)
			{
				out << i << sep;
			}
			//out << std::endl;
			LOG(INFO) << out.str() << std::endl;
		}
	}
}

#endif
