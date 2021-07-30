#include "Projection.h"
#include "../Basic/SafeGuard.h"
using namespace CGP;

Projection::Projection(float fx, float fy, float cx, float cy)
{
	intrisic_ = (cvMatD(3, 3)<< fx, 0, cx,
		0, fy, cy,
		0, 0, 1);
	fx_ = fx;
	fy_ = fy;
	cx_ = cx;
	cy_ = cy;
	//LOG(INFO) << "test for intrinsic_: " << intrisic_ << std::endl;
}

void Projection::getMeshToImageRT(const MeshCompress& src, const intVec& mesh_idx,
	const intVec& used_vec, const vecF& landmark_xy, cvMatD& rvec, cvMatD& tvec)
{
	intSet used_list(used_vec.begin(), used_vec.end());
	int n_num = landmark_xy.size()*0.5;
	float3Vec src_slice;
	src.getSlice(mesh_idx, src_slice);
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	for (int i = 0; i < n_num; i++)
	{
		if (used_list.count(i) && i < 17)
		{
			img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
			float3E pos = src_slice[i];
			obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
		}
		else
		{
			for (int iter = 0; iter < 2; iter++)
			{
				img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
				float3E pos = src_slice[i];
				obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
			}
		}
	}
	cvMatD r_matrix;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
	cv::solvePnP(obj_points, img_points, intrisic_,
		dist_coeffs, rvec, tvec);
	//LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	//LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
}