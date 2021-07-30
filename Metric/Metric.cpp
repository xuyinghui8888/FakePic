#include "Metric.h"
using namespace CGP;
MeshCompress Metric::getError(const MeshCompress& src, const MeshCompress& dst, float max_scale, float data_scale)
{
	if (src.pos_.empty() || dst.pos_.empty())
	{
		LOG(ERROR) << "empty mesh detected." << std::endl;
		return src;
	}
	
	if (src.n_vertex_ != dst.n_vertex_)
	{
		LOG(ERROR) << "src && dst mesh vertex not match" << std::endl;
	}

	MeshCompress delta = src;
	delta.vertex_color_ = delta.pos_;
	float Color[21][3] = { 
	{0,0,0},
	{0,0,0.25},
	{0,0,0.5},
	{0,0,0.75},
	{0,0,1},
	{0,0.25,1},
	{0,0.5,1},
	{0,0.75,1},
	{0,1,1},
	{0,1,0.75},
	{0,1,0.5},//mid
	{0,1,0.25},
	{0,1,0},
	{0.25,1,0},
	{0.5,1,0},
	{0.75,1,0},
	{1,1,0},
	{1,0.5,0},
	{1,0,0},
	{1,0,0.5},
	{1,0,1}
	};
	float sum = 0;
	for (int i = 0; i < src.pos_.size(); i++)
	{
		float dis = ((src.pos_[i] - dst.pos_[i]).norm());
		sum += std::abs(dis)*std::abs(dis);
		dis = DLIP3(dis, -max_scale, max_scale);
		int iter = dis * 10 / max_scale;
		iter += 10;
		for (int j = 0; j < 3; j++)
		{
			delta.vertex_color_[i][j] = Color[iter][j];
		}
	}	
	LOG(INFO) << "mse: " << sum* data_scale*data_scale / (1.0f*src.pos_.size()) << std::endl;
	return delta;
}

MeshCompress Metric::getError(const MeshCompress& src, const MeshCompress& dst)
{
	if (src.pos_.empty() || dst.pos_.empty())
	{
		LOG(ERROR) << "empty mesh detected." << std::endl;
		return src;
	}

	if (src.n_vertex_ != dst.n_vertex_)
	{
		LOG(ERROR) << "src && dst mesh vertex not match" << std::endl;
	}

	MeshCompress delta = src;
	delta.vertex_color_ = delta.pos_;
	float Color[21][3] = {
	{0,0,0},
	{0,0,0.25},
	{0,0,0.5},
	{0,0,0.75},
	{0,0,1},
	{0,0.25,1},
	{0,0.5,1},
	{0,0.75,1},
	{0,1,1},
	{0,1,0.75},
	{0,1,0.5},//mid
	{0,1,0.25},
	{0,1,0},
	{0.25,1,0},
	{0.5,1,0},
	{0.75,1,0},
	{1,1,0},
	{1,0.5,0},
	{1,0,0},
	{1,0,0.5},
	{1,0,1}
	};

	float sum = 0;
	floatVec dis(src.n_vertex_, 0);

	for (int i = 0; i < src.pos_.size(); i++)
	{
		dis[i] = ((src.pos_[i] - dst.pos_[i]).norm());
		sum += std::abs(dis[i])*std::abs(dis[i]);
	}

	auto min_max = std::minmax_element(dis.begin(), dis.end());
	int min_seg_idx = *min_max.first;
	int max_seg_idx = *min_max.second;
	double max_scale = dis[max_seg_idx];
	for (int i = 0; i < src.pos_.size(); i++)
	{
		int iter = dis[i] * 10 / max_scale;
		iter += 10;
		for (int j = 0; j < 3; j++)
		{
			delta.vertex_color_[i][j] = Color[iter][j];
		}
	}
	LOG(INFO) << "mse: " << sum / (1.0f*src.pos_.size()) << std::endl;
	return delta;
}