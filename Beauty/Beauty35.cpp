#include "Beauty35.h"
#include "../FileIO/FileIO.h"
#include "../RT/RT.h"
#include "../Debug/DebugTools.h"
#include "../CalcFunction/CalcHelper.h"
using namespace CGP;
void Beauty35::calculate35BasedOn68(const float3Vec& src, vecF& x_ratio, vecF& y_ratio)
{
	//5ÑÛ
	floatVec dis_x_5_eye;
	//´ÓÍ¼Ïñ×ó±ßµ½Í¼ÏñÓÒ±ß
	intVec eye_idx = {0,36,39,42,45,16};
	intVec mid_idx = { 27,33,8 };
	floatVec order_dist_x;
	for (int i = 1; i < eye_idx.size(); i++)
	{
		order_dist_x.push_back(src[eye_idx[i]].x() - src[eye_idx[i-1]].x());
	}
	float eye_dist = 0.5* order_dist_x[1] + 0.5*order_dist_x[3];
	for (int i = 0; i < order_dist_x.size(); i++)
	{
		order_dist_x[i] = order_dist_x[i] / eye_dist;
	}
	DebugTools::cgPrint(order_dist_x, "x_eye: ");
	CalcHelper::vectorToEigen(order_dist_x, x_ratio);
	floatVec order_dist_y;
	for (int i = 1; i < mid_idx.size(); i++)
	{
		order_dist_y.push_back(src[mid_idx[i]].y() - src[mid_idx[i - 1]].y());
	}
	float nose_dist = order_dist_y[0];
	for (int i = 0; i < order_dist_y.size(); i++)
	{
		order_dist_y[i] = order_dist_y[i] / nose_dist;
	}
	DebugTools::cgPrint(order_dist_y, "y_ting: ");
	CalcHelper::vectorToEigen(order_dist_y, y_ratio);
}

void Beauty35::calculate35BasedOn68(const vecF& src, vecF& x_ratio, vecF& y_ratio)
{
	//xy pattern
	float3Vec temp;
	for (int i = 0; i < src.size()/2; i++)
	{
		temp.push_back(float3E(src[2 * i], src[2 * i + 1], 0));
	}
	calculate35BasedOn68(temp, x_ratio, y_ratio);
}