#include "Landmark68Wrapper.h"
#include "../FileIO/FileIO.h"
#include "../RT/RT.h"
#include "../Debug/DebugTools.h"
using namespace CGP;
SimLandmarkWrapper::SimLandmarkWrapper(cstr& pack_data_root, cstr result_dir, bool is_debug)
{
	//std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(pack_data_root, "config.json", json_data);
	ptr_model = std::make_shared<MnnHelper>(json_data);
	result_dir_ = result_dir;
	is_debug_ = is_debug;
}

void SimLandmarkWrapper::getLandmark68(const cv::Mat& input, vecF& res)
{
	cv::Rect bb_box;
	processFaceDetection(input, bb_box);
	if (bb_box.area() == 0)
	{
		res.resize(68 * 2);
		res.setConstant(0);
	}
	else
	{
		cv::Mat land68_256;
		cv::Rect scale_bb;
		shiftBBLandmark68(input, bb_box, 256, land68_256, scale_bb);
		vecF landmark_256_xy;
		getLandmarks(land68_256, landmark_256_xy);
		//reflect back to image
		reflectBackToImage(scale_bb, 256, landmark_256_xy, res);
		//get mtcnn format 5 landmarkss
	}

	if (is_debug_)
	{
		cv::Mat img_clone = input.clone();
		for (int i = 0; i < res.size()/2; i++)
		{
			cv::Point p = cv::Point(res[2 * i], res[2 * i + 1]);
			cv::circle(img_clone, p, 1, cv::Scalar(255, 0, 0), 2);
		}
		cv::imwrite(result_dir_ + "whole_landmark68.png", img_clone);
	}
}

void SimLandmarkWrapper::reflectBackToImage(const cv::Rect& landmark_68_crop, int resolution,
	const vecF& res_256, vecF& res)
{
	int n_keypoints = res_256.size() / 2;
	res = res_256;
	if (landmark_68_crop.width != landmark_68_crop.height)
	{
		LOG(ERROR) << "error for landmark 68 problems." << std::endl;
		return;
	}
	float scale = (1.0f*landmark_68_crop.width) / (1.0*resolution);

#pragma omp parallel for
	for (int i = 0; i < n_keypoints; i++)
	{
		float temp_x = res_256[2 * i] * scale + landmark_68_crop.tl().x;
		float temp_y = res_256[2 * i + 1] * scale + landmark_68_crop.tl().y;
		res[2 * i] = temp_x;
		res[2 * i + 1] = temp_y;
	}

}

void SimLandmarkWrapper::shiftBBLandmark68(const cv::Mat& img, const cv::Rect& src, int resolution, cv::Mat& dst, cv::Rect& scale_bb)
{
	float height = src.height*1.0f;
	float width = src.width*1.0f;
	float scale_h = 1.0;
	float reference = 195.0f;
	//cv::Point2f center = cv::Point2f(src.br().x - width * 0.5, src.br().y - height* 1.16 * 0.5 - height*1.16*0.12 );
	cv::Point2f center = cv::Point2f(src.br().x - width * 0.5, src.br().y - height * scale_h * 0.5);
	float scale = 0.5*(scale_h* height + width) / reference;
	scale = scale * 256.0 / 240;

	float scale_tl = MIN(center.x / 128.0, center.y / 128.0);
	float scale_br = MIN((img.cols - center.x) / 128.0, (img.rows - center.y) / 128.0);
	scale = MIN(scale, MIN(scale_tl, scale_br));

	int opt_tl_x = DMAX(center.x - 128.0 * scale, 0);
	int opt_tl_y = DMAX(center.y - 128.0 * scale, 0);
	scale_bb = cv::Rect(opt_tl_x, opt_tl_y, 256 * scale, 256 * scale);
	SG::checkBBox(img, scale_bb);
	scale_bb.width = DMIN(scale_bb.width, scale_bb.height);
	scale_bb.height = scale_bb.width;
	dst = img(scale_bb);
	cv::resize(dst, dst, cv::Size(resolution, resolution));
	if (is_debug_)
	{
		cv::imwrite(result_dir_ + "input_landmark68.png", dst);
	}
}

void SimLandmarkWrapper::getLandmarks(const cv::Mat& img, vecF& res)
{
	ptr_model->landmark_68_->inference(img, 256, 256, res);
	if (is_debug_)
	{
		cv::Mat img_clone = img.clone();
		for (int i = 0; i < 68; i++)
		{
			cv::Point p = cv::Point(res[2 * i], res[2 * i + 1]);
			cv::circle(img_clone, p, 2, cv::Scalar(255, 0, 0), 2);
		}
		cv::imwrite(result_dir_ + "landmark_crop.jpg", img_clone);
		FILEIO::saveEigenDynamic(result_dir_ + "landmark_xy_256.txt", res);
	}
}


void SimLandmarkWrapper::processFaceDetection(const cv::Mat& input, cv::Rect& res)
{
	std::vector< MTCNN_SCOPE::FaceInfo> face_bbox;
	ptr_model->mtcnn_->Detect_T(input, face_bbox);
	if (face_bbox.empty())
	{
		LOG(WARNING) << "No face detected." << std::endl;
		return;
	}
	//get default 0
	res = cv::Rect(face_bbox[0].bbox.xmin, face_bbox[0].bbox.ymin,
		face_bbox[0].bbox.xmax - face_bbox[0].bbox.xmin, face_bbox[0].bbox.ymax - face_bbox[0].bbox.ymin);
	if (is_debug_)
	{
		cv::Mat canvas = input.clone();
		cv::rectangle(canvas, res, cv::Scalar(255, 0, 0), 2);
		cv::imwrite(result_dir_ + "input_bb.png", canvas);
	}
}