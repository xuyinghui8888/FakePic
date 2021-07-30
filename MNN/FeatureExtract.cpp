#ifndef _PREPROCESS_IMAGE_H_
#define _PREPROCESS_IMAGE_H_
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "FaceAlign.h"
#include "../Basic/CGPBaseHeader.h"
namespace PREPROCESS_IMAGE
{
	//Normal face landmark for 112x112,
	static float norm_face[5][2] = {
		{ 30.2946f + 8.0, 51.6963f },
		{ 65.5318f + 8.0, 51.5014f },
		{ 48.0252f + 8.0, 71.7366f },
		{ 33.5493f + 8.0, 92.3655f },
		{ 62.7299f + 8.0, 92.2041f } };

	//Normal face landmark for 112x96,
	static float norm_face_2[5][2] = {
		{ 30.2946f , 51.6963f },
		{ 65.5318f, 51.5014f },
		{ 48.0252f, 71.7366f },
		{ 33.5493f, 92.3655f },
		{ 62.7299f, 92.2041f } };

	//compare input features with labeled features, get classification result with minimal distance and class index
	struct class_info
	{
		double min_distance;
		int index;
	};


	class_info classify(const cv::Mat& img, const  cv::Mat& cmp)
	{
		int rows = cmp.rows;
		cv::Mat broad;
		cv::repeat(img, rows, 1, broad);

		broad = broad - cmp;
		cv::pow(broad, 2, broad);
		cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

		double dis;
		cv::Point point;
		cv::minMaxLoc(broad, &dis, 0, &point, 0);

		return class_info{ dis, point.y };
	}

	void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M)
	{
		float src[10], dst[10];
		memcpy(src, src_5pts, sizeof(float) * 10);
		memcpy(dst, dst_5pts, sizeof(float) * 10);

		float ptmp[2];
		ptmp[0] = ptmp[1] = 0;
		for (int i = 0; i < 5; ++i) {
			ptmp[0] += src[i];
			ptmp[1] += src[5 + i];
		}
		ptmp[0] /= 5;
		ptmp[1] /= 5;
		for (int i = 0; i < 5; ++i) {
			src[i] -= ptmp[0];
			src[5 + i] -= ptmp[1];
			dst[i] -= ptmp[0];
			dst[5 + i] -= ptmp[1];
		}

		float dst_x = (dst[3] + dst[4] - dst[0] - dst[1]) / 2, dst_y = (dst[8] + dst[9] - dst[5] - dst[6]) / 2;
		float src_x = (src[3] + src[4] - src[0] - src[1]) / 2, src_y = (src[8] + src[9] - src[5] - src[6]) / 2;
		float theta = atan2(dst_x, dst_y) - atan2(src_x, src_y);

		float scale = sqrt(pow(dst_x, 2) + pow(dst_y, 2)) / sqrt(pow(src_x, 2) + pow(src_y, 2));
		float pts1[10];
		float pts0[2];
		float _a = sin(theta), _b = cos(theta);
		pts0[0] = pts0[1] = 0;
		for (int i = 0; i < 5; ++i) {
			pts1[i] = scale * (src[i] * _b + src[i + 5] * _a);
			pts1[i + 5] = scale * (-src[i] * _a + src[i + 5] * _b);
			pts0[0] += (dst[i] - pts1[i]);
			pts0[1] += (dst[i + 5] - pts1[i + 5]);
		}
		pts0[0] /= 5;
		pts0[1] /= 5;

		float sqloss = 0;
		for (int i = 0; i < 5; ++i) {
			sqloss += ((pts0[0] + pts1[i] - dst[i])*(pts0[0] + pts1[i] - dst[i])
				+ (pts0[1] + pts1[i + 5] - dst[i + 5])*(pts0[1] + pts1[i + 5] - dst[i + 5]));
		}

		float square_sum = 0;
		for (int i = 0; i < 10; ++i) {
			square_sum += src[i] * src[i];
		}
		for (int t = 0; t < 200; ++t) {
			_a = 0;
			_b = 0;
			for (int i = 0; i < 5; ++i) {
				_a += ((pts0[0] - dst[i])*src[i + 5] - (pts0[1] - dst[i + 5])*src[i]);
				_b += ((pts0[0] - dst[i])*src[i] + (pts0[1] - dst[i + 5])*src[i + 5]);
			}
			if (_b < 0) {
				_b = -_b;
				_a = -_a;
			}
			float _s = sqrt(_a*_a + _b * _b);
			_b /= _s;
			_a /= _s;

			for (int i = 0; i < 5; ++i) {
				pts1[i] = scale * (src[i] * _b + src[i + 5] * _a);
				pts1[i + 5] = scale * (-src[i] * _a + src[i + 5] * _b);
			}

			float _scale = 0;
			for (int i = 0; i < 5; ++i) {
				_scale += ((dst[i] - pts0[0])*pts1[i] + (dst[i + 5] - pts0[1])*pts1[i + 5]);
			}
			_scale /= (square_sum*scale);
			for (int i = 0; i < 10; ++i) {
				pts1[i] *= (_scale / scale);
			}
			scale = _scale;

			pts0[0] = pts0[1] = 0;
			for (int i = 0; i < 5; ++i) {
				pts0[0] += (dst[i] - pts1[i]);
				pts0[1] += (dst[i + 5] - pts1[i + 5]);
			}
			pts0[0] /= 5;
			pts0[1] /= 5;

			float _sqloss = 0;
			for (int i = 0; i < 5; ++i) {
				_sqloss += ((pts0[0] + pts1[i] - dst[i])*(pts0[0] + pts1[i] - dst[i])
					+ (pts0[1] + pts1[i + 5] - dst[i + 5])*(pts0[1] + pts1[i + 5] - dst[i + 5]));
			}
			if (abs(_sqloss - sqloss) < 1e-2) {
				break;
			}
			sqloss = _sqloss;
		}

		for (int i = 0; i < 5; ++i) {
			pts1[i] += (pts0[0] + ptmp[0]);
			pts1[i + 5] += (pts0[1] + ptmp[1]);
		}

		M[0] = _b * scale;
		M[1] = _a * scale;
		M[3] = -_a * scale;
		M[4] = _b * scale;
		M[2] = pts0[0] + ptmp[0] - scale * (ptmp[0] * _b + ptmp[1] * _a);
		M[5] = pts0[1] + ptmp[1] - scale * (-ptmp[0] * _a + ptmp[1] * _b);
	}

	cv::Mat alignToNcnn(cv::Mat& img, float* landmark)
	{

		int image_w = 112; //96 or 112
		int image_h = 112;

		float dst[10] = { 30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
						 51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

		if (image_w == 112)
			for (int i = 0; i < 5; i++)
				dst[i] += 8.0;

		float src[10];
		for (int i = 0; i < 5; i++)
		{
			src[i] = landmark[i];
			src[i + 5] = landmark[i+5];
		}

		float M[6];
		getAffineMatrix(src, dst, M);
		cv::Mat transfer(2, 3, CV_32FC1, M);
		LOG(INFO) << "M:" << transfer << std::endl;
		cv::Mat aligned(112, 112, CV_32FC3);
		cv::Size size(112, 112);
		cv::warpAffine(img, aligned, transfer, size, 1, cv::BORDER_REPLICATE, 0);
		return aligned;
	}

	cv::Mat alignToMtcnn(const cv::Mat& img, float* landmark, bool is_xy)
	{
		int n_keypoints = 5;
		cv::Mat src(n_keypoints, 2, CV_32FC1, norm_face_2);
		float v2[5][2] =
		{ 
			{ landmark[0] , landmark[5]},
			{ landmark[1] , landmark[6]},
			{ landmark[2] , landmark[7]},
			{ landmark[3] , landmark[8]},
			{ landmark[4] , landmark[9]}
		};

		if (is_xy)
		{
			for (int i = 0; i < n_keypoints; i++)
			{
				v2[i][0] = landmark[2 * i];
				v2[i][1] = landmark[2 * i + 1];
			}
		}

		cv::Mat dst(5, 2, CV_32FC1, v2);

		//do similar transformation according normal face
		cv::Mat m = similarTransform(dst, src);

		cv::Mat aligned(112, 112, CV_32FC3);
		cv::Size size(112, 112);

		//get aligned face with transformed matrix and resize to 112*112
		cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
		cv::warpAffine(img, aligned, transfer, size, 1, cv::BORDER_REPLICATE, 0);
		return aligned;
		//cv::imshow("aligned", aligned);
		//cv::waitKey(0);
		//extract feature from aligned face and do classification with labels 
		//cv::Mat output = extract.extractFeature(aligned);

		////draw landmark points
		//for (int j = 0; j < 5; j++)
		//{
		//	cv::Point p(face.landmark.x[j], face.landmark.y[j]);
		//	cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
		//}

		////draw bound ing box
		//cv::Point pt1(face.x0, face.y0);
		//cv::Point pt2(face.x1, face.y1);
		//cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);

		//cv::Point pt3(face.x0, face.y0 - 10);
		//if (result.min_distance < 1.05)
		//{
		//	//show classification name
		//	cv::putText(img, labels[result.index], pt3, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
		//}
	}

	cv::Mat alignToMtcnnExpand(const cv::Mat& img, float* landmark, bool is_xy, int resolution)
	{
		int n_keypoints = 5;
		float scale = resolution * 1.0f / 112.0;
		float norm_face_scaled[5][2] = {
		{ 30.2946f + 8.f, 51.6963f },
		{ 65.5318f + 8.f, 51.5014f },
		{ 48.0252f + 8.f, 71.7366f },
		{ 33.5493f + 8.f, 92.3655f },
		{ 62.7299f + 8.f, 92.2041f } };

		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				norm_face_scaled[i][j] *= scale;
			}
		}


		float y_shift = scale * 112 * 0.12;
		for (int i = 0; i < 5; i++)
		{
			norm_face_scaled[i][1] -= y_shift;
		}

		cv::Mat src(n_keypoints, 2, CV_32FC1, norm_face_scaled);
		float v2[5][2] =
		{
			{ landmark[0] , landmark[5]},
			{ landmark[1] , landmark[6]},
			{ landmark[2] , landmark[7]},
			{ landmark[3] , landmark[8]},
			{ landmark[4] , landmark[9]}
		};

		if (is_xy)
		{
			for (int i = 0; i < n_keypoints; i++)
			{
				v2[i][0] = landmark[2 * i];
				v2[i][1] = landmark[2 * i + 1];
			}
		}

		cv::Mat dst(5, 2, CV_32FC1, v2);

		//do similar transformation according normal face
		cv::Mat m = similarTransform(dst, src);

		cv::Mat aligned(resolution, resolution, CV_32FC3);
		cv::Size size(resolution, resolution);

		//get aligned face with transformed matrix and resize to 112*112
		cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
		cv::warpAffine(img, aligned, transfer, size, 1, cv::BORDER_REPLICATE, 0);
		return aligned;
		//cv::imshow("aligned", aligned);
		//cv::waitKey(0);
		//extract feature from aligned face and do classification with labels 
		//cv::Mat output = extract.extractFeature(aligned);

		////draw landmark points
		//for (int j = 0; j < 5; j++)
		//{
		//	cv::Point p(face.landmark.x[j], face.landmark.y[j]);
		//	cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
		//}

		////draw bound ing box
		//cv::Point pt1(face.x0, face.y0);
		//cv::Point pt2(face.x1, face.y1);
		//cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);

		//cv::Point pt3(face.x0, face.y0 - 10);
		//if (result.min_distance < 1.05)
		//{
		//	//show classification name
		//	cv::putText(img, labels[result.index], pt3, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
		//}
	}
}
#endif
