#include "facenet.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace MNN;
using namespace MNN::CV;
using namespace CGP;

void FaceNet::init(const CGP::JsonData& init_json)
{
	//LOG(INFO) << height_ << std::endl;
	height_ = init_json.facenet_input_;
	width_ = init_json.facenet_input_;
	output_dim_ = init_json.facenet_output_;
	cstr model_path = init_json.root_ + init_json.facenet_;
	net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
	initialize();
}

void FaceNet::print_tensor(const Tensor &mat, const int n)
{

  for (int i = 0; i < n; i++){
    cout << mat.host<float>()[i] << ", ";
  }
  cout << endl;
}

int FaceNet::get_max_preds(const vector<vector<float> >&heatmaps, const int width, vector<pair<int, int> > &kps, vector<float> &max_vals){

  const int n = heatmaps.size();
  const int height = heatmaps[0].size()/width;
  for (int i = 0; i < n; i++){
    const vector<float> &ht = heatmaps[i];
    int max_idx = 0;
    for (int k=0; k < ht.size(); k++){
      if (ht[k] > ht[max_idx])
	max_idx = k;
    }
    max_vals.push_back(ht[max_idx]);
    const int x = max_idx%width;
    const int y = min(max_idx/width, height-1);
    kps.push_back(pair<int, int>(x, y));
    // cout<< width<<", " << x << ", " << y << endl;
  }
  return 0;
}

int FaceNet::initialize(){

  MNN::ScheduleConfig session_config;
  session_config.type = MNN_FORWARD_CPU; // @todo gpu?
  session_config.numThread = 4;	// @todo
  session = net->createSession(session_config);
  input_tensor = net->getSessionInput(session, NULL);
  
  config.filterType   = BILINEAR;
  config.sourceFormat = GRAY;
  config.destFormat   = RGB;
  config.wrap         = ZERO;
  
  return 0;
}

