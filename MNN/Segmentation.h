#ifndef _SEGMENTATION_H_
#define _SEGMENTATION_H_

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

class Segmentation{

private:
  const int height=0;
  const int width=0;
  std::shared_ptr<MNN::Interpreter> net;
  MNN::Session *session = NULL;
  MNN::CV::ImageProcess::Config config;
  MNN::Tensor* input_tensor = NULL;

public:
  Segmentation(const char *model_path, const int img_size): width(img_size), height(img_size){
    net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path));
    initialize();
  }
  ~Segmentation(){
    if (session){
      net->releaseSession(session);
      session = NULL;
      net->releaseModel();
    }
  }
  int inference(const uint8_t *input_img, const int w, const int h, uint8_t *out_mask);
  int get_height()const {
    return height;
  }
  int get_width()const {
    return width;
  } 
  
protected:
  const MNN::Tensor *inference(const uint8_t *input_img, const int width, const int height, const char *output_name=NULL);
  int initialize();
};

#endif /* _SEGMENTATION_H_ */
