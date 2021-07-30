#include "DebugTools.h"
#include "../Mesh/MeshTools.h"
#include "../CalcFunction/CalcHelper.h"
using namespace CGP;
cv::Mat DebugTools::projectVertexToXY(const std::vector<float3Vec>& src, const std::vector<cv::Scalar>& colors, int image_w, int image_h)
{
	cv::Mat res(cv::Size(image_w, image_h), CV_8UC3);
	res.setTo(0);
	doubleVec xyz_min, xyz_max;
	MeshTools::getBoundingBox(src, xyz_min, xyz_max);
	//xy transform
	doubleVec xyz_length = CalcHelper::multiMinus(xyz_max, xyz_min);
	double max_dim = CalcHelper::multiMaxDouble(xyz_length);
	double safe_ratio = 1.1f;
	//trans xy to image

	float center_x = xyz_min[0] * 0.5 + xyz_max[0] * 0.5;
	float center_y = xyz_min[1] * 0.5 + xyz_max[1] * 0.5;

	float aim_dim = max_dim * safe_ratio;
	float x_expand_min = center_x - 0.5*aim_dim;
	float y_expand_min = center_y - 0.5*aim_dim;

	double scale = image_w / aim_dim;	
	for (int iter = 0; iter < src.size(); iter++)
	{
		for (int i = 0; i < src[iter].size(); i++)
		{
			float x_3d = src[iter][i].x();
			float y_3d = src[iter][i].y();
			//reverse up-down
			float x_2d = (x_3d - x_expand_min) / (aim_dim)*image_w;
			float y_2d = (1 - (y_3d - y_expand_min) / (aim_dim))*image_w;
			//LOG(INFO) << "x_2d: " << x_2d << std::endl;
			//LOG(INFO) << "y_2d: " << y_2d << std::endl;
			cv::circle(res, cv::Point(x_2d, y_2d), 2, colors[iter], 2, 0);
		}
	}
	return res;
}
void DebugTools::printJson(const json& init)
{
	
	for (auto& el : init.items())
	{
		std::cout << "key: " << el.key() << ", value:" << el.value() << '\n';
	}
}

void DebugTools::printJsonStructure(const json& init)
{
	for (auto& el : init.items())
	{
		std::cout << "key: " << el.key() << ", value size:" << el.value().size() << '\n';
	}
}

