#ifndef _MOBILE_FACENET_H_
#define _MOBILE_FACENET_H_

#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include "../Config/JsonData.h"

class FaceNet{
  
private:
  int output_dim_ = 0;
  int height_ = 0;
  int width_ = 0;
  std::shared_ptr<MNN::Interpreter> net;
  MNN::Session *session = NULL;
  MNN::CV::ImageProcess::Config config;
  MNN::Tensor* input_tensor = NULL;
  
public:

	void init(const CGP::JsonData& init_json);


	~FaceNet()
	{
		if (session)
		{
		  net->releaseSession(session);
		  session = NULL;
		  net->releaseModel();
		}
	}

  template <class T>	
  int inference(const cv::Mat& img, const int w, const int h, T& res)
  {
	  cvMatU img_gray;
	  if (img.channels() == 4)
	  {
		  cv::Mat temp;
		  cv::cvtColor(img, temp, cv::COLOR_RGBA2GRAY);
		  temp.convertTo(img_gray, CV_8UC1);
	  }
	  else if (img.channels() == 3)
	  {
		  cv::Mat temp;
		  cv::cvtColor(img, temp, cv::COLOR_RGB2GRAY);
		  temp.convertTo(img_gray, CV_8UC1);
	  }
	  else if (img.channels() == 1)
	  {
		  img.convertTo(img_gray, CV_8UC1);
	  }
	  else
	  {
		  LOG(ERROR) << "channels strategy not set." << std::endl;
		  return -1;
	  }
	 
	  if (h != height_ || w != width_)
	  {
		  cv::resize(img_gray, img_gray, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
	  }

	  floatVec input_img(w*h);
#pragma omp parallel for
	  for (int y = 0; y < h; y++)
	  {
		  for (int x = 0; x < w; x++)
		  {
			  input_img[y*width_ + x] = (float(img_gray.at<uchar>(y, x)) - 127.5) / 127.5;
		  }
	  }
	  MNN::CV::ImageProcess* pretreat = MNN::CV::ImageProcess::create(config);
	  std::vector<int> dims{ 1, this->width_, this->height_, 1 };
	  auto tmp = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
	  memcpy(tmp->host<float>(), input_img.data(), tmp->size());
	  input_tensor->copyFromHostTensor(tmp);
	  const int ErrorCode = net->runSession(session);
	  if (ErrorCode)
	  {
		  printf("failed in inference stage! error code: %d\n", ErrorCode);
		  return -1;
	  }

	  // get output
	  const MNN::Tensor* output_tensor = net->getSessionOutput(session, nullptr);
	  const float *ret = output_tensor->host<float>();
	  if (ret == NULL)
	  {
		  return -1;
	  }

	  res.resize(output_dim_);
	  std::memcpy(res.data(), ret, output_dim_ * sizeof(float));
	  return 0;
  }
  
  int get_kps_num() const
  {
    return output_dim_;
  }

  void print_tensor(const MNN::Tensor &mat, const int n);
  int get_max_preds(const std::vector<std::vector<float> >&heatmaps, const int width, 
	  std::vector<std::pair<int, int> > &kps, std::vector<float> &max_vals);


protected:
  int initialize();
};

#endif /* _FACEID_H_ */
