#include "ObjVis.h"
#include "../RT/RT.h"
using namespace CGP;
void ObjVis::setFixSize(int width, int height, bool debug)
{
	if (width_ < 0 || height_ < 0)
	{
		//not initialized yet
		width_ = width;
		height_ = height;
		mesh_drawer_.init(width, height, debug);
	}
	else if (width_ == width && height_ == height && debug_ == debug)
	{
		LOG(INFO) << "skip for init, since same data is loaded." << std::endl;
	}
	else
	{
		LOG(WARNING) << "mesh_drawer_ is init with different fixed size. reinit for mesh_loader_" << std::endl;
		mesh_drawer_.clear();
		mesh_drawer_.init(width, height, debug);
	}
	width_ = width;
	height_ = height;
	debug_ = debug;
}
cv::Mat ObjVis::drawMesh(const MeshCompress& mesh_data, int width, int height, bool debug)
{
	setFixSize(width, height, debug);
	int n_vertex_buffer = mesh_data.n_tri_ * 3;
	float3Vec vertex_buffer(n_vertex_buffer);
#pragma omp parallel for
	for (int i = 0; i < n_vertex_buffer; i++)
	{
		vertex_buffer[i] = mesh_data.pos_[mesh_data.tri_[i]];
	}
	mesh_drawer_.bindVertices((float*)vertex_buffer.data(), n_vertex_buffer);
	cv::Mat rot = (cv::Mat_<double>(3, 1) << 0, 0, 1);
	cv::Mat trans = (cv::Mat_<double>(3, 1) << 0, 0, 100);
	cv::Mat proj_mat = RT::toPrjMat(rot, trans);
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 600, 0, 0.5*width, 0, 600, 0.5*height, 0, 0, 1);
	LOG(INFO) << "prj_mat" << std::endl << proj_mat << std::endl;
	LOG(INFO) << "camera_matrix" << std::endl << camera_matrix << std::endl;
	LOG(INFO) << "test for width/height" << std::endl << (cv::Mat_<double>(3, 3) << 1.0 / width, 0, 0, 0, 1.0 / height, 0, 0, 0, 1) << std::endl;
	cv::Mat rt = (cv::Mat_<double>(3, 3) << 1.0 / width, 0, 0, 0, 1.0 / height, 0, 0, 0, 1)
		* camera_matrix * proj_mat;
	floatVec mat_buffer(12);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++) 
		{
			mat_buffer[i * 4 + j] = float(rt.at<double>(i, j));
		}
	}
	ucharVec draw_result(width*height * 4);
	mesh_drawer_.drawAndGetContourPixels(n_vertex_buffer, mat_buffer.data(), draw_result.data());
	cv::Mat result(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 255));
#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		SG::safeMemcpy(result.row(height-i-1).data, &draw_result[i*width * 4], width * 4 * sizeof(uchar));
	}
	cv::cvtColor(result, result, cv::COLOR_RGBA2BGRA);
	return result;
}
