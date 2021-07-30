#include "ImageSimilarity.h"
#include "ImageUtils.h"
using namespace CGP;

float ImageSim::ssimWrapper(const cv::Mat& im1, const cv::Mat& im2, int window, float k1, float k2, float L)
{
	if (im1.size() != im2.size())
	{
		LOG(ERROR) << "img1 and img2 size not fit." << std::endl;
		return 0;
	}

	cv::Mat im1f, im2f;
	im1.convertTo(im1f, CV_32FC1);
	im2.convertTo(im2f, CV_32FC1);
	return ssim(im1f, im2f, window, k1, k2, L);
}

float ImageSim::ssim(const cv::Mat& im1, const cv::Mat& im2, int window, float k1, float k2, float L)
{
	if (im1.size() != im2.size())
	{
		LOG(ERROR) << "img1 and img2 size not fit." << std::endl;
		return 0;
	}

	int ndim = im1.dims;
	float NP = std::pow(window, ndim);
	float cov_norm = NP / (NP - 1);
	float C1 = (k1 * L) * (k1 * L);
	float C2 = (k2 * L) * (k2 * L);

	cv::Mat ux, uy;
	cv::Mat uxx = im1.mul(im1);
	cv::Mat uyy = im2.mul(im2);
	cv::Mat uxy = im1.mul(im2);

	cv::blur(im1, ux, cv::Size(window, window), cv::Point(-1, -1));
	cv::blur(im2, uy, cv::Size(window, window), cv::Point(-1, -1));

	cv::blur(uxx, uxx, cv::Size(window, window), cv::Point(-1, -1));
	cv::blur(uyy, uyy, cv::Size(window, window), cv::Point(-1, -1));
	cv::blur(uxy, uxy, cv::Size(window, window), cv::Point(-1, -1));

	cv::Mat ux_sq = ux.mul(ux);
	cv::Mat uy_sq = uy.mul(uy);
	cv::Mat uxy_m = ux.mul(uy);

	cv::Mat vx = cov_norm * (uxx - ux_sq);
	cv::Mat vy = cov_norm * (uyy - uy_sq);
	cv::Mat vxy = cov_norm * (uxy - uxy_m);

	cv::Mat A1 = 2 * uxy_m;
	cv::Mat A2 = 2 * vxy;
	cv::Mat B1 = ux_sq + uy_sq;
	cv::Mat B2 = vx + vy;

	cv::Mat ssim_map = (A1 + C1).mul(A2 + C2) / (B1 + C1).mul(B2 + C2);

	cv::Scalar mssim = mean(ssim_map);
	ssim_map.convertTo(ssim_map, CV_8UC1, 255, 0);

	//imshow("ssim", ssim_map);
	//cv::waitKey(0);
	return mssim[0];
}

float ImageSim::matchShape(const cv::Mat& im1, const cv::Mat& im2)
{
#ifdef _WIN32
	std::vector< std::vector< cv::Point> > im1_contours, im2_contours;
	ImageUtils::getMaxContour(im1, im1_contours);
	ImageUtils::getMaxContour(im2, im2_contours);
	

	cv::Mat im1_canvas, im2_canvas;
	im1_canvas = im1.clone();
	im2_canvas = im2.clone();

	cv::drawContours(im1_canvas, im1_contours, -1, cv::Scalar::all(255));
	cv::drawContours(im2_canvas, im2_contours, -1, cv::Scalar::all(255));
	//cv::imshow("im1_canvas", im1_canvas);
	//cv::imshow("im2_canvas", im2_canvas);
	//cv::waitKey(0);

	
	return cv::matchShapes(im1_contours[0], im2_contours[0], cv::CONTOURS_MATCH_I2, 0);
#else
	return 0;
#endif
}

float ImageSim::matchShapeImage(const cv::Mat& im1, const cv::Mat& im2)
{
#ifdef _WIN32
	cv::Mat im1_gray, im2_gray;
	ImageUtils::toGray(im1, im1_gray);
	ImageUtils::toGray(im2, im2_gray);

	return cv::matchShapes(im1_gray, im2_gray, cv::CONTOURS_MATCH_I2, 0);
#else
	return 0;
#endif
}
