#ifndef __IMAGE_UTILS_H__
#define __IMAGE_UTILS_H__
#include "../Basic/CGPBaseHeader.h"
namespace CGP
{
	namespace ImageUtils
	{
		void toCVMatU(const cv::Mat& src, cv::Mat& dst);
		void toGray(const cv::Mat& src, cv::Mat& dst);
		void toBinary(const cv::Mat& src, cv::Mat& dst, double thres = 100);
		void removeBack(const cv::Mat& src, cv::Mat& dst, double thres = 100);
		void keepRoiValue(const cv::Mat& src, cv::Mat& dst, double min_value = 100, double max_value = 255);
		void removeBackFromAlpha(const cv::Mat& src, cv::Mat& dst, double thres = 100);
		void changeRGBKeepAlpha(const cv::Mat& src, cv::Mat& dst, const float3E& dst_color, double thres = 100);
		void getMaxContour(const cv::Mat& src, std::vector< std::vector< cv::Point> >& max_contour);
		cv::Mat drawContour(const cv::Mat& src);
		cv::Mat drawContourPolygon(const cv::Mat& src);
		void cutBasedOnImage(const cv::Mat& src, cv::Mat& dst, int thres = 5);
		void putImageToCenter(const cv::Mat& src, const cv::Size& dst_size, cv::Mat& dst);
		cv::Mat drawKps(const cv::Mat& img, const floatVec& kps_xy, double scale, const cv::Scalar& color = cv::Scalar(255, 0, 0));
		floatVec adjustPosRightBrow(const cv::Mat& mask, const floatVec& ori_106);
		cv::Rect safeExpand(const cv::Mat& canvas, const cv::Rect ori_box, int edge);
		cv::Rect maskBox(const cv::Mat& canvas, int thres = 10);
		void adjustYUpper(const cv::Mat& mask, const intVec& roi, floatVec& res);
		void adjustYLower(const cv::Mat& mask, const intVec& roi, floatVec& res);
		void adjustTail(const cv::Mat& mask, const intVec& roi, floatVec& res);
		void adjustTop(const cv::Mat& mask, int up, int low, int tail, floatVec& res);
		
		template<class T>
		cv::Mat drawKps(const cv::Mat& img, const T& kps_xy, const intVec& idx = {})
		{
			cv::Mat canvas = img.clone();
			for (int i = 0; i < kps_xy.size() / 2; i++) {
				const int x = (int)kps_xy[i * 2];
				const int y = (int)kps_xy[i * 2 + 1];
				cv::circle(canvas, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), 2, 0);
				if (!idx.empty() && idx.size() > i)
				{
					cv::putText(canvas, std::to_string(idx[i]), cv::Point(x, y - 3), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);
				}
			}
			return canvas;
		}
	}
}
#endif

