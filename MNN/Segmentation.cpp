#include <assert.h>
#include <iostream>
#include <vector>
#include <sstream>
#include "Segmentation.h"

using namespace std;
using namespace MNN;
using namespace MNN::CV;

int Segmentation::initialize(){

  MNN::ScheduleConfig session_config;
  session_config.type = MNN_FORWARD_CPU; // @todo gpu?
  session_config.numThread = 4;	// @todo
  session = net->createSession(session_config);
  input_tensor = net->getSessionInput(session, NULL);
  // cout << input_tensor->width() << ", " << input_tensor->height() << ", " << input_tensor->channel() << endl; // 224x224x4
  
  config.filterType   = BILINEAR;
  config.sourceFormat = RGB;
  config.destFormat   = RGB;
  config.wrap         = ZERO;
  
  const float means[3] = {123.68f, 116.78f, 103.94f};
  const float norms[3] = {0.017f, 0.017f, 0.017f};
  memcpy(config.mean, means, sizeof(means));
  memcpy(config.normal, norms, sizeof(norms));
  return 0;
}

const Tensor *Segmentation::inference(const uint8_t *input_img, const int width, const int height, const char *output_name){ 
	// set input
	const int input_w = this->width;
	const int input_h = this->height;
  
	ImageProcess* pretreat=ImageProcess::create(config);
	//std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
	Matrix trans;
	trans.postScale(1.0/input_w, 1.0/input_h);
	trans.postScale(width, height);
	pretreat->setMatrix(trans);

	auto tmp = new Tensor(input_tensor, Tensor::TENSORFLOW);
	pretreat->convert(input_img, width, height, 0, tmp);
	input_tensor->copyFromHostTensor(tmp);
	delete tmp;
  
	// inference
	const int ErrorCode = net->runSession(session);
	if (ErrorCode){
	printf("failed in inference stage! error code: %d\n", ErrorCode);
	return NULL;
	}

	// get output
	const Tensor* output_tensor = net->getSessionOutput(session, output_name);  // 2x128x128
//#ifndef _WIN32
	delete pretreat;
//#endif
	return output_tensor;
}

int Segmentation::inference(const uint8_t *input_img, const int width, const int height, uint8_t *out_mask){
  const Tensor *output_tensor = this->inference(input_img, width, height, "outmask");
  const float *ret = output_tensor->host<float>();
  if (ret == NULL)
    return -1;
  
  const int output_w = this->width;
  const int output_h = this->height;
  for (int i=0; i < output_w*output_h; ++i){
    const float *logits = &(ret[4*i]);
    uint8_t idx = 0;
    idx = logits[1]>logits[idx] ? 1 : idx;
    idx = logits[2]>logits[idx] ? 2 : idx;
    out_mask[i] = (idx==1) ? 127 : ((idx==2) ? 255 : 0);
  }
  return 0;
}
