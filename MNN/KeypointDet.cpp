#include "KeypointDet.h"

using namespace std;
using namespace MNN;
using namespace MNN::CV;

void KeypointDetection::print_tensor(const Tensor &mat, const int n)
{
	for (int i = 0; i < n; i++)
    {
		cout << mat.host<float>()[i] << ", ";
	}
	cout << endl;
}

int KeypointDetection::get_max_preds(const vector<vector<float> >&heatmaps, const int width, vector<pair<int, int> > &kps, vector<float> &max_vals)
{
	const int n = heatmaps.size();
	const int height = heatmaps[0].size()/width;
	for (int i = 0; i < n; i++)
	{
		const vector<float> &ht = heatmaps[i];
		int max_idx = 0;
		for (int k=0; k < ht.size(); k++)
		{
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

int KeypointDetection::initialize()
{

	MNN::ScheduleConfig session_config;
	session_config.type = MNN_FORWARD_CPU; // @todo gpu?
	session_config.numThread = 4;	// @todo
	session = net->createSession(session_config);
	input_tensor = net->getSessionInput(session, NULL);
  
	config.filterType   = BILINEAR;
	config.sourceFormat = RGB;
	config.destFormat   = RGB;
	config.wrap         = ZERO;
  
	return 0;
}

int KeypointDetection::inference(const unsigned char *img, const int w, const int h, vector<pair<int, int> >&kps,vector<float>&max_vals){

  // prepare input data
  const float src_ratio = ((float)w)/h;
  const float des_ratio = ((float)this->width)/this->height;
  const float scale= src_ratio>des_ratio ? this->width/float(w):this->height/float(h);
  const int W = w*scale;
  const int H = h*scale;
  const int x0 = (this->width-W)/2;
  const int y0 = (this->height-H)/2;

  cv::Mat src_img;
  src_img.create(h, w, CV_8UC3);
  memcpy(src_img.data, img, w*h*3);  
  cv::resize(src_img, src_img, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

  cv::Mat input_img;
  input_img.create(height, width, CV_32FC3);
  float *data = (float*)input_img.data;
  const float s = 1.0/255.0;
  for (int r = 0; r < height; r++){
    for (int c = 0; c < width; c++){
      const int j = 3*(r*width+c);
      data[j+0] = (0.0-0.485)/0.229;
      data[j+1] = (0.0-0.456)/0.224;
      data[j+2] = (0.0-0.406)/0.225;      
    }
  }
  
  for(int r=0; r < H; r++){
    for (int c=0; c < W; c++){
      const int i = 3*(r*W+c);
      const int j = 3*((r+y0)*width+(c+x0));
      data[j+0] = (src_img.data[i+0]*s-0.485)/0.229;
      data[j+1] = (src_img.data[i+1]*s-0.456)/0.224;
      data[j+2] = (src_img.data[i+2]*s-0.406)/0.225;
    }
  }
  
  // copy to tensor and run
  //std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
  ImageProcess* pretreat=ImageProcess::create(config);
  std::vector<int> dims{1, this->width, this->height, 3};
  //auto tmp = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
  auto tmp = new Tensor(input_tensor, Tensor::TENSORFLOW);
  memcpy(tmp->host<float>(), input_img.data, tmp->size());
  input_tensor->copyFromHostTensor(tmp);

  const int ErrorCode = net->runSession(session);
  if (ErrorCode){
    printf("failed in inference stage! error code: %d\n", ErrorCode);
    return -1;
  }

  // get output
  const Tensor* output_tensor = net->getSessionOutput(session, nullptr);
  const float *ret = output_tensor->host<float>();
  if (ret == NULL){
    return -1;
  }
  const int out_width = output_tensor->width();
  const int out_height = output_tensor->height();
  const int n = out_width*out_height;
  
  vector<vector<float> > heatmaps(num_kps, vector<float>(n));
  for (int s = 0; s<(num_kps-1)/4+1; s++){
    const float *p = &ret[4*n*s];
    const int N = min(4, num_kps-s*4);
    for (int k = 0; k < N; k++){
      for (int i = 0; i < n; i++){
	heatmaps[s*4+k][i] = p[i*4+k];
      }
    }
  }
  get_max_preds(heatmaps, out_width, kps, max_vals);
  for (auto &p: kps){
    p.first = (p.first*4-x0)/scale;
    p.second = (p.second*4-y0)/scale;
  }
  
  delete pretreat;
  return 0;
}
