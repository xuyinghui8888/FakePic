#ifndef _MNN_MODEL_H_
#define _MNN_MODEL_H_

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "KeypointDet.h"
#include "Segmentation.h"
using namespace std;

class MnnModel{
  
public:
  MnnModel(){}
  int init_seg_model(const char *model_fpath, const int size=224){
    if (seg_model != nullptr)
      delete seg_model;
    seg_model = new Segmentation(model_fpath, size);
    return 0;
  }
  int init_kp_model(const char *model_fpath, const int w=192, const int h=256, const int kp_num=7){
    if (kp_model != nullptr)
      delete kp_model;
    kp_model = new KeypointDetection(model_fpath, w, h, kp_num);
    if (kp_num == 7){
      kps_map = {1,2,3,5,9,10,11};
    }else if(kp_num == 11){
      kps_map = {1,2,3,4,5,6,7,8,9,10,11}; // kps are 1-indexed
    }else{
      // @note kp_num should be 7 or 11
      return 1;
    }
    return 0;
  }
  int inference(const uchar *img, const int w, const int h, uchar *mask,
		vector<float> &kps, vector<float> &kp_vals, vector<int> &idx);
  int inference(const uchar *img, const int w, const int h, uchar *mask);
  int get_kps_num()const{
    return kp_model->get_kps_num();
  }
  static int get_bbox_num(const cv::Mat &mask);
  ~MnnModel(){
#ifndef _WIN32
    if (kp_model){
      delete kp_model;
      kp_model = nullptr;
    }
    if (seg_model){
      delete seg_model;
      seg_model = nullptr;
    }  
#endif  
  }
  
protected:
  bool is_left(const vector<pair<int,int> > &kps)const;
  
private:
  KeypointDetection *kp_model = nullptr;
  Segmentation *seg_model = nullptr;
  vector<int> kps_map;
};

#endif /* _MNN_MODEL_H_ */
