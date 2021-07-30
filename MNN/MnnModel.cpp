#include <queue>
#include <iostream>
#include <assert.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MnnModel.h"
using namespace cv;
using namespace std;

int get_bbox(cv::Mat &img, const int r0, const int c0, vector<int> &bbox){

  int n = 1;
  std::queue<pair<int,int> > pos;
  pos.push(pair<int,int>(r0, c0) );
  img.data[r0*img.cols+c0] = 0;
  bbox = {img.cols-1,img.rows-1,0,0};
  while(pos.size() > 0){
    
    const int r = pos.front().first;
    const int c = pos.front().second;
    bbox[0] = min(bbox[0], c);
    bbox[1] = min(bbox[1], r);
    bbox[2] = max(bbox[2], c);
    bbox[3] = max(bbox[3], r);
    
    pos.pop();
    const pair<int, int> top(max(r-1, 0), c);
    const pair<int, int> bottom(min(r+1, img.rows-1), c);
    const pair<int, int> left(r, max(0, c-1));
    const pair<int, int> right(r, min(img.cols-1, c+1));
    const vector<pair<int,int> > dirs = {top, bottom, left, right};
    for (const auto dir: dirs){
      if (img.data[dir.first*img.cols+dir.second]>0){
	pos.push(dir);
	img.data[dir.first*img.cols+dir.second] = 0;
	n+=1;
      }
    }
  }
  return n;
}

int crop_image(const cv::Mat &img, const cv::Mat &mask, vector<cv::Mat> &imgs, vector<vector<int> > &bboxs){

  // bbox: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
  cv::Mat tmp_mask;
  const int size = 224;
  cv::resize(mask, tmp_mask, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
  for (int r = 0; r < size; r++){
    for (int c = 0; c < size; c++){
      const int i = r*size+c;
      tmp_mask.data[i] = tmp_mask.data[i]<200 && tmp_mask.data[i]>100 ? 1:0;
    }
  }
  // cv::imwrite("./tempt/mask_small.png", tmp_mask*255);

  // @bug
  const float scale_x = float(mask.cols)/size;
  const float scale_y = float(mask.rows)/size;
  const int expend_x = mask.cols*0.05;
  const int expend_y = mask.rows*0.05;
  const int num_tol = (size*size)/100;

  // select only the largest two sub-images
  imgs.resize(2);
  bboxs.resize(2);
  int first_pixel_num = 0;
  int second_pixel_num = 0;
  for (int r = 0; r < size; r++){
    for (int c = 0; c < size; c++){
      vector<int> bbox;
      if (tmp_mask.data[r*size+c]<=0)
	continue;
      const int n = get_bbox(tmp_mask, r, c, bbox);
      int idx = -1;
      if (n <= num_tol){
	continue;
      }
      if (n > first_pixel_num){
	idx = 0;
	second_pixel_num = first_pixel_num;
	bboxs[1] = bboxs[0];
	imgs[1] = imgs[0];
	first_pixel_num = n;
      }else if(n > second_pixel_num){
	idx = 1;
	second_pixel_num = n;
      }
      if (idx < 0){
	continue;
      }
      
      // scale bbox and crop image
      bbox[0] = max(int(bbox[0]*scale_x-expend_x), 0);
      bbox[1] = max(int(bbox[1]*scale_y-expend_y), 0);
      bbox[2] = min(int(bbox[2]*scale_x+expend_x), mask.cols-1);
      bbox[3] = min(int(bbox[3]*scale_y+expend_y), mask.rows-1);
      // cout << bbox[0] << ", " << bbox[1] << ", "  << bbox[2] << ", "  << bbox[3] <<endl;
      // bbox = {17, 22, 17+506, 22+237}; // @todo
      
      bboxs[idx] = bbox;
      imgs[idx] = img.rows>0 ? img(cv::Rect(bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])).clone():img;
    }
  }
  if (first_pixel_num <= 0){
    imgs.clear();
    bboxs.clear();
  }else if (second_pixel_num <= 0){
    imgs.pop_back();
    bboxs.pop_back();
  }
  
  // cv::imwrite("./tempt/crop.png", imgs[0]);
  return imgs.size();
}

int MnnModel::get_bbox_num(const cv::Mat &mask){

  cv::Mat img;
  vector<cv::Mat> imgs;
  vector<vector<int> > bboxs;
  const int n = crop_image(img, mask, imgs, bboxs);
  return n;
}

int MnnModel::inference(const uchar *raw_img, const int w, const int h, uchar *mask,
			vector<float> &kps, vector<float> &kp_vals, vector<int> &idx){
  // prepare inputs
  cv::Mat img;
  img.create(h, w, CV_8UC3);
  memcpy(img.data, raw_img, w*h*3);
  
  // segmentation
  cv::Mat out_mask;
  out_mask.create(seg_model->get_height(), seg_model->get_width(), CV_8U);
  const int r1 = seg_model->inference(img.data, img.cols, img.rows, out_mask.data);
  cv::resize(out_mask, out_mask, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_LINEAR);
  memcpy(mask, out_mask.data, w*h);
    
  // keypoints
  vector<cv::Mat> imgs;
  vector<vector<int> > bboxs;
  //cv::imshow("out_mask", out_mask);
  //cv::waitKey(0);
  const int r2 = crop_image(img, out_mask, imgs, bboxs);
  for (int i = 0; i < imgs.size(); i++){
    vector<std::pair<int,int> > tmp_kps;
    vector<float> tmp_kp_vals;
    kp_model->inference(imgs[i].data, imgs[i].cols, imgs[i].rows, tmp_kps, tmp_kp_vals);

    vector<float> kps_one_foot;
    vector<float> vals_one_foot;
    vector<int> idx_one_foot;
    for(int k = 0; k < tmp_kps.size(); k++){
      if (tmp_kp_vals[k] > 0.001){ // @todo why 0.001?
	kps_one_foot.push_back(tmp_kps[k].first+bboxs[i][0]);
	kps_one_foot.push_back(tmp_kps[k].second+bboxs[i][1]);
	vals_one_foot.push_back(tmp_kp_vals[k]);
	idx_one_foot.push_back(this->kps_map[k]);
      }
    }
    if (kps_one_foot.size() <= 2){
      continue;
    }
    int idx0 = 0;
    if (!is_left(tmp_kps)){
      idx0 = 11;
    }
    for (int k = 0; k < kps_one_foot.size()/2; k++){
      kps.push_back(kps_one_foot[k*2]);
      kps.push_back(kps_one_foot[k*2+1]);
      kp_vals.push_back(vals_one_foot[k]);
      idx.push_back(idx_one_foot[k]+idx0);
    }
  }
  return 0;
}

int MnnModel::inference(const uchar *raw_img, const int w, const int h, uchar *mask){

  // prepare inputs
  cv::Mat img;
  img.create(h, w, CV_8UC3);
  memcpy(img.data, raw_img, w*h*3);
  
  // segmentation
  cv::Mat out_mask;
  out_mask.create(seg_model->get_height(), seg_model->get_width(), CV_8U);
  const int r1 = seg_model->inference(img.data, img.cols, img.rows, out_mask.data);
  cv::resize(out_mask, out_mask, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_LINEAR);
  memcpy(mask, out_mask.data, w*h);  
  return 0;
}

bool MnnModel::is_left(const vector<pair<int,int> > &p)const{
  
  // A = p[0]-p[2]
  // B = p[-1]-p[2]
  // return [AxB].z > 0
  const int n = p.size();
  const float ax = p[0].first-p[2].first;
  const float ay = -(p[0].second-p[2].second);
  const float bx = p[n-2].first-p[2].first;
  const float by = -(p[n-2].second-p[2].second);
  const float cz = ax*by - ay*bx;
  // cout<< "cz = " << cz << endl;
  return cz>0;
}
