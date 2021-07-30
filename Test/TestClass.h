#ifndef TEST_CLASS_FUNCTION_H
#define TEST_CLASS_FUNCTION_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
#include "../Config/ShapeVar.h"
namespace CGP
{
	//used for testing codes
	namespace TESTSTRUCTURE
	{
		class TaobaoSkeleton
		{
		public:
			float3E rotate = float3E(0, 0, 0);
			float3E trans = float3E(0, 0, 0);
			float3E scale = float3E(0, 0, 0);
			// print names
			void print(const cstr& instance); 
		};

		class TaobaoSlide
		{
		public:
			double value = 0;
			// https://stackoverflow.com/questions/1936399/c-array-operator-with-multiple-arguments/1936410
			//重载"[]"操作符，返回区间内数据
			double operator() (double lb, double ub);     
			 // print names
			void print(const cstr& instance);
		};
	}
}

#endif
