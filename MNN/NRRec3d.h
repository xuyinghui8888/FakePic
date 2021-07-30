#ifndef _NR3D_REC_H_
#define _NR3D_REC_H_

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

class NR3D
{
  
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
	void init(const CGP::JsonData& init_json, const  CGP::cstr model_relative_path);

	~NR3D()
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
	  cvMatF3 img_rgb_float;
	  if (img.channels() == 4)
	  {
		  cv::Mat temp;
		  cv::cvtColor(img, temp, cv::COLOR_RGBA2RGB);
		  temp.convertTo(img_rgb_float, CV_32FC3);
	  }
	  else if (img.channels() == 3)
	  {
		  img.convertTo(img_rgb_float, CV_32FC3);
	  }
	  else if (img.channels() == 1)
	  {
		  cv::Mat temp;
		  cv::cvtColor(img, temp, cv::COLOR_GRAY2RGB);
		  img.convertTo(img_rgb_float, CV_32FC3);
	  }
	  else
	  {
		  LOG(ERROR) << "channels strategy not set." << std::endl;
		  return -1;
	  }
	 
	  if (h != height_ || w != width_)
	  {
		  cv::resize(img, img_rgb_float, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
	  }
	  float scale = 1.0 / 255.0;
	  floatVec input_img(w*h*3);
	  float means[3] = { 0.485f,0.456f,0.406f };
	  float norms[3] = { 1.0/0.229f, 1.0 / 0.224f, 1.0 / 0.225f };
#pragma omp parallel for
	  for (int y = 0; y < h; y++)
	  {
		  for (int x = 0; x < w; x++)
		  {
			  for (int c = 0; c < 3; c++)
			  {
				  float rgb_value = (img_rgb_float.at<cv::Vec3f>(y, x))(c)*scale;
				  //rgb correct
				  float norm_rgb_value = (rgb_value - means[c])*norms[c];
				  //bgr false
				  //float norm_rgb_value = (rgb_value - means[2-c])*norms[2 - c];
				  input_img[3*(y*width_ + x)+c] = norm_rgb_value;
			  }			  
		  }
	  }

#if 0
	  //≤‚ ‘
	  FILEIO::saveDynamic("D:/dota210507/0603_01/img_tensor.txt", input_img, ",");
	  floatVec input_dw = FILEIO::loadFloatDynamic("D:/dota210604/dbg_000020_cur.txt", '\n');
#pragma omp parallel for
	  for (int y = 0; y < h; y++)
	  {
		  for (int x = 0; x < w; x++)
		  {
			  for (int c = 0; c < 3; c++)
			  {
				  input_img[3 * (y*width_ + x) + c] = input_dw[c*h*w + y * width_ + x];
			  }
		  }
	  }

	  //≤‚ ‘
#endif
	  MNN::CV::ImageProcess* pretreat = MNN::CV::ImageProcess::create(config);
	  std::vector<int> dims{ 1, this->width_, this->height_, 3 };
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
	  //id_coeff = coeff[:, : 80] # identity(shape) coeff of dim 80
	  //ex_coeff = coeff[:, 80 : 144] # expression coeff of dim 64
	  //tex_coeff = coeff[:, 144 : 224] # texture(albedo) coeff of dim 80
	  //angles = coeff[:, 224 : 227] # ruler angles(x, y, z) for rotation of dim 3
	  //gamma = coeff[:, 227 : 254] # lighting coeff for 3 channel SH function of dim 27
	  //translation = coeff[:, 254 : ] # translation coeff of dim 3
	  intVec num_out = { 194 };
	  cstrVec name_out = { "552"};
	  res.resize(output_dim_);
	  int count = 0;
	  for (int i = 0; i < num_out.size(); i++)
	  {
		  const MNN::Tensor* output_tensor = net->getSessionOutput(session, name_out[i].c_str());
		  const float *ret = output_tensor->host<float>();
		  if (ret == NULL)
		  {
			  return -1;
		  }
		  std::memcpy(res.data() + count, ret, num_out[i] * sizeof(float));
		  count += num_out[i];
	  }
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
