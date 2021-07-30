#ifndef _LANDMARK_68_H_
#define _LANDMARK_68_H_
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include "../Config/JsonData.h"

class Landmark68
{  
private:
	int output_dim_ = 0;
	int height_ = 0;
	int width_ = 0;
	int n_keypoints = 0;
	int n_heatmap_size = 0;
	std::shared_ptr<MNN::Interpreter> net;
	MNN::Session *session = NULL;
	MNN::CV::ImageProcess::Config config;
	MNN::Tensor* input_tensor = NULL;
  
public:

	void init(const CGP::JsonData& init_json);
	~Landmark68()
	{
		if (session)
		{
		  net->releaseSession(session);
		  session = NULL;
		  net->releaseModel();
		}
	}


	int inference(const cv::Mat& img, const int w, const int h, CGP::vecF& res);
  int get_kps_num() const
  {
    return output_dim_;
  }

  void print_tensor(const MNN::Tensor &mat, const int n);
  int get_max_preds(const std::vector<std::vector<float> >&heatmaps, const int width, 
	  std::vector<std::pair<float, float> > &kps, std::vector<float> &max_vals);


protected:
  int initialize();
};

#endif /* _FACEID_H_ */
