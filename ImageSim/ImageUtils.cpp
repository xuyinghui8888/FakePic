#include "ImageUtils.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Basic/ToDst.h"
using namespace CGP;

void ImageUtils::toCVMatU(const cv::Mat& src, cv::Mat& dst)
{
	cv::Mat src_gray;
	toGray(src, src_gray);
	src_gray.convertTo(dst, CV_8UC1);
}

void ImageUtils::toGray(const cv::Mat& src, cv::Mat& dst)
{
	if (src.channels() == 4)
	{
		//cv::cvtColor(src, dst, cv::COLOR_RGBA2GRAY);
		std::vector<cv::Mat> channels;
		cv::split(src, channels);
		channels.pop_back();
		cv::merge(channels, dst);
		//cv::imshow("dst", dst);
		//cv::waitKey(0);
		cv::cvtColor(dst, dst, cv::COLOR_RGB2GRAY);
	}
	else if (src.channels() == 3)
	{
		cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
	}
	else if (src.channels() == 1)
	{
		dst = src.clone();
	}
	else
	{
		LOG(ERROR) << "channels not defined." << std::endl;
		dst = src.clone();
	}
}

void ImageUtils::toBinary(const cv::Mat& src, cv::Mat& dst, double thres)
{
	cv::Mat gray;
	toGray(src, gray);
	if (gray.channels() != 1)
	{
		LOG(ERROR) << "gray channels error, not equal to 1." << std::endl;
		dst = src.clone();
	}
	else
	{
		cv::threshold(gray, dst, thres, 255, cv::THRESH_BINARY);
	}
}

void ImageUtils::removeBack(const cv::Mat& src, cv::Mat& dst, double thres)
{
	cv::Mat src_gray;
	toGray(src, src_gray);
	src_gray.convertTo(src_gray, CV_8UC1);
	dst = src_gray.clone();
#pragma omp parallel for
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (GETU(src_gray, y, x) > thres)
			{
				SETU(dst, y, x, 255);
			}
			else
			{
				SETU(dst, y, x, 0);
			}
		}
	}
}

void ImageUtils::keepRoiValue(const cv::Mat& src, cv::Mat& dst, double min_value, double max_value)
{
	cv::Mat src_gray;
	toGray(src, src_gray);
	src_gray.convertTo(src_gray, CV_8UC1);
	dst = src_gray.clone();
#pragma omp parallel for
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (GETU(src_gray, y, x) > min_value && GETU(src_gray, y, x) < max_value)
			{
				SETU(dst, y, x, 255);
			}
			else
			{
				SETU(dst, y, x, 0);
			}
		}
	}
}

void ImageUtils::removeBackFromAlpha(const cv::Mat& src, cv::Mat& dst, double thres)
{
	std::vector<cv::Mat> channels;
	if (src.channels() != 4)
	{
		dst = src.clone();
		LOG(ERROR) << "src is not with 4 channels." << std::endl;
		return;
	}

	cv::split(src, channels);
	cv::cvtColor(src, dst, cv::COLOR_RGBA2GRAY);

#pragma omp parallel for
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (GETU(channels[3], y, x) > thres)
			{
				SETU(dst, y, x, 255);
			}
			else
			{
				SETU(dst, y, x, 0);
			}
		}
	}
}

void ImageUtils::changeRGBKeepAlpha(const cv::Mat& src, cv::Mat& dst, const float3E& dst_color, double thres)
{
	std::vector<cv::Mat> channels;
	if (src.channels() != 4)
	{
		dst = src.clone();
		LOG(ERROR) << "src is not with 4 channels." << std::endl;
		return;
	}

	cv::Vec4b avg_src = cv::mean(src);
	LOG(INFO) << "src_color: " << avg_src << std::endl;
	

	//cv::split(src, channels);
	//cv::cvtColor(src, dst, cv::COLOR_RGBA2GRAY);
	dst = src.clone();
#pragma omp parallel for
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			cv::Vec4b src_value = GETU4(src, y, x);
			for (int c = 0; c < 3; c++)
			{
				float src_value_c = float(src_value[c]) * float(dst_color[0]);
				src_value[c] = safeDiv(src_value_c, float(avg_src[0]), 0);
			}
			SETU4(dst, y, x, src_value);
			//SETU(dst, y, x, 255) = GETU(channels[3], y, x);
			/*
			if (GETU(channels[3], y, x) > thres)
			{
				SETU(dst, y, x, 255);
			}
			else
			{
				SETU(dst, y, x, 0);
			}
			*/
		}
	}
}


void ImageUtils::getMaxContour(const cv::Mat& src, std::vector< std::vector< cv::Point> >& max_contour)
{
	cv::Mat image_binary;
	ImageUtils::toGray(src, image_binary);
	cv::Mat image;
	cv::GaussianBlur(image_binary, image, cv::Size(3, 3), 0);
	cv::Canny(image, image, 100, 220);
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(image, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	//cv::findContours(image, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	image_binary = cv::Scalar::all(0);

	//overall contour
	std::vector<cv::Point> all_points;
	int max_idx = 0;
	int max_num = contours[0].size();
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > max_num)
		{
			max_num = contours[i].size();
			max_idx = i;
		}
	}
	all_points = contours[max_idx];

	//std::vector< cv::Point>  contour;
	//cv::approxPolyDP(cv::Mat(all_points), contour, 3, true);
	//cv::convexHull(cv::Mat(all_points), contour);
	max_contour.clear();
	max_contour.push_back(all_points);
}

cv::Mat ImageUtils::drawContourPolygon(const cv::Mat& src)
{
	cv::Mat image_binary;
	ImageUtils::toGray(src, image_binary);
	cv::Mat image;
	cv::GaussianBlur(image_binary, image, cv::Size(3, 3), 0);
	cv::Canny(image, image, 100, 220);
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(image, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	//cv::findContours(image, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	cv::Mat res = src.clone();
	std::vector< cv::Point>  contour;
	std::vector< cv::Point>  contour_convex;
	//cv::approxPolyDP(cv::Mat(contours), contour, 3, true);
	TDST::unpackData(contours, contour);
	cv::convexHull(cv::Mat(contour), contour_convex);
	cv::polylines(res, contour_convex, true, cv::Scalar::all(255));
	return res;
}

cv::Mat ImageUtils::drawContour(const cv::Mat& src)
{
	cv::Mat image_binary;
	ImageUtils::toGray(src, image_binary);
	cv::Mat image;
	cv::GaussianBlur(image_binary, image, cv::Size(3, 3), 0);
	cv::Canny(image, image, 100, 220);
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(image, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	//cv::findContours(image, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	cv::Mat res = src.clone();
	cv::drawContours(res, contours, -1, cv::Scalar::all(255));
	return res;
}

cv::Rect ImageUtils::maskBox(const cv::Mat& canvas, int thres)
{
	cv::Mat src_binary;
	toBinary(canvas, src_binary);
	std::vector<cv::Point2f> Points;
	cv::findNonZero(src_binary, Points);
	cv::Rect min_Rect = boundingRect(Points);
	return min_Rect;
}

void ImageUtils::cutBasedOnImage(const cv::Mat& src, cv::Mat& dst, int thres)
{
	cv::Rect min_Rect = maskBox(src);
	dst = src(min_Rect);
	//cv::imshow("src", src);
	//cv::imshow("cut", dst);
	//cv::waitKey(0);
}

void ImageUtils::putImageToCenter(const cv::Mat& src, const cv::Size& dst_size, cv::Mat& dst)
{
	dst = src.clone();
	dst = cv::Scalar::all(0);
	cv::resize(dst, dst, dst_size);
	//calculate size
	int src_height = src.rows;
	int src_width = src.cols;

	int aim_height = src_height;
	int aim_width = src_width;

	if (dst_size.height < src_height)
	{
		aim_height = dst_size.height;
		src_width = src_width * (1.0*aim_height / src_height);
	}

	if(dst_size.width<src_width)
	{
		aim_width = dst_size.width;
		aim_height = aim_height * (1.0*aim_width / src_width);
	}

	cv::Mat src_resize;
	cv::resize(src, src_resize, cv::Size(aim_width, aim_height));
	cv::Rect roi = cv::Rect(dst.cols*0.5 - aim_width * 0.5, dst.rows*0.5 - aim_height * 0.5, aim_width, aim_height);
	src_resize.copyTo(dst(roi));

	//cv::imshow("dst", dst);
	//cv::waitKey(0);
}


cv::Mat ImageUtils::drawKps(const cv::Mat &img, const floatVec& kps_xy, double scale, const cv::Scalar& color)
{
	cv::Mat canvas = img.clone();
	if (img.channels() == 1)
	{
		cv::cvtColor(img, canvas, cv::COLOR_GRAY2RGB);
	}
	for (int i = 0; i < kps_xy.size() / 2; i++) {
		const int x = (int)kps_xy[i * 2] * scale;
		const int y = (int)kps_xy[i * 2 + 1] * scale;
		cv::circle(canvas, cv::Point(x, y), 3, color, 2, 0);
	}
	return canvas;
}

cv::Rect ImageUtils::safeExpand(const cv::Mat& canvas, const cv::Rect ori_box, int edge)
{
	int tl_x = ori_box.tl().x;
	int tl_y = ori_box.tl().y;
	int br_x = ori_box.br().x;
	int br_y = ori_box.br().y;
	
	int width = canvas.cols;
	int height = canvas.rows;

	tl_x -= edge;
	tl_y -= edge;
	br_x += edge;
	br_y += edge;

	//in image
	tl_x = DLIP3(tl_x, 0, width - 1);
	tl_y = DLIP3(tl_y, 0, height - 1);
	br_x = DLIP3(br_x, tl_x, width - 1);
	br_y = DLIP3(br_y, tl_y, height - 1);
	return cv::Rect(tl_x, tl_y, br_x - tl_x, br_y - tl_y);
}

floatVec ImageUtils::adjustPosRightBrow(const cv::Mat& mask, const floatVec& ori_106)
{
	//right idx 
	intVec roi = {38,39,40,41,42,68,69,70,71};
	// 38 39 40 41 42
	// 68 69 70 71

	//first get mask around the landmark
	floatVec adjust_106 = ori_106;
	cv::Mat canvas = mask.clone();
	canvas = cv::Scalar::all(0);
	std::vector<cv::Point2f> landmark_pos_point;
	for (int i : roi)
	{
		landmark_pos_point.push_back(cv::Point(ori_106[2 * i], ori_106[2 * i + 1]));
	}
	cv::Rect landmark_box = cv::boundingRect(landmark_pos_point);
	//expand a little
	cv::Rect landmark_box_expand = safeExpand(mask, landmark_box, 10);
	mask(landmark_box_expand).copyTo(canvas(landmark_box_expand));
	//cv::imshow("canvas", canvas);
	//cv::waitKey(0);
	//copy and adjust
	cv::Rect seg_box = maskBox(canvas);
	LOG(INFO) << "seg_box: " << seg_box << std::endl;
	LOG(INFO) << "landmark_box: " << landmark_box << std::endl;
	if (seg_box.area() < 10 || landmark_box.area()<10)
	{
		LOG(WARNING) << "mask box  or landmark_box is empty()" << std::endl;
		return adjust_106;
	}
	//adjust move 38/68
	adjust_106[2 * 38] = seg_box.tl().x;
	adjust_106[2 * 68] = seg_box.tl().x;
	//adjust 42
	adjust_106[2 * 42] = seg_box.br().x;
	float length_after_before = seg_box.width / (1.0*landmark_box.width);
	//adjust x
	intVec upper_idx = { 39 ,40 ,41 };
	intVec lower_idx = { 69, 70, 71 };
	CalcHelper::keepRatioStrechX(ori_106, upper_idx, 38, 42, length_after_before, adjust_106);
	CalcHelper::keepRatioStrechX(ori_106, lower_idx, 68, 42, length_after_before, adjust_106);
	ImageUtils::adjustYUpper(canvas, upper_idx, adjust_106);
	ImageUtils::adjustYLower(canvas, lower_idx, adjust_106);
	ImageUtils::adjustTail(canvas, { 42 }, adjust_106);
	ImageUtils::adjustTop(canvas, 38, 68, 42, adjust_106);
	return adjust_106;
}

void ImageUtils::adjustYUpper(const cv::Mat& mask, const intVec& roi, floatVec& res)
{
	//get min y
	cv::Mat mask_u;
	toCVMatU(mask, mask_u);
	//cv::imshow("mask_u", mask_u);
	//cv::waitKey(0);
	for (int i : roi)
	{
		int x = res[i * 2];
		int min_y = mask_u.rows;
		int max_y = 0;
		for (int y = 0; y < mask_u.rows; y++)
		{
			if (GETU(mask_u, y, x) > 5)
			{
				min_y = min_y > y ? y : min_y;
				max_y = max_y > y ? max_y : y;
			}
		}
		res[2 * i + 1] = min_y;
	}
}

void ImageUtils::adjustYLower(const cv::Mat& mask, const intVec& roi, floatVec& res)
{
	//get min y
	cv::Mat mask_u;
	toCVMatU(mask, mask_u);
	for (int i : roi)
	{
		int x = res[i * 2];
		int min_y = mask_u.rows;
		int max_y = 0;
		for (int y = 0; y < mask_u.rows; y++)
		{
			if (GETU(mask_u, y, x) > 5)
			{
				min_y = min_y > y ? y : min_y;
				max_y = max_y > y ? max_y : y;
			}
		}
		res[2 * i + 1] = max_y;
	}
}

void ImageUtils::adjustTail(const cv::Mat& mask, const intVec& roi, floatVec& res)
{
	//get min y
	cv::Mat mask_u;
	toCVMatU(mask, mask_u);
	for (int i : roi)
	{
		//bounding box Õ‚«–
		int x = res[i * 2]-1;
		int sum = 0;
		int count = 0;
		for (int y = 0; y < mask_u.rows; y++)
		{
			if (GETU(mask_u, y, x) > 5)
			{
				sum += y;
				count++;
			}
		}
		res[2 * i + 1] = sum/(1.0*count);
	}
}

void ImageUtils::adjustTop(const cv::Mat& mask, int up, int low, int tail, floatVec& res)
{
	//get min y
	cv::Mat mask_u;
	toCVMatU(mask, mask_u);
	float dis_ori = res[tail * 2] - res[up * 2];
	//shrink 5%
	float shrink_5 = dis_ori * 0.05 + res[up * 2];

	int x = shrink_5;
	int min_y = mask_u.rows;
	int max_y = 0;
	for (int y = 0; y < mask_u.rows; y++)
	{
		if (GETU(mask_u, y, x) > 5)
		{
			min_y = min_y > y ? y : min_y;
			max_y = max_y > y ? max_y : y;
		}
	}
	res[up * 2 + 1] = min_y;
	res[low * 2 + 1] = max_y;

}





