#include "Landmark68.h"
#include "../FileIO/FileIO.h"
using namespace std;
using namespace MNN;
using namespace MNN::CV;
using namespace CGP;

void Landmark68::init(const CGP::JsonData& init_json)
{
	//LOG(INFO) << height_ << std::endl;
	height_ = init_json.landmark68_input_;
	width_ = init_json.landmark68_input_;
	output_dim_ = init_json.landmark68_out_heatmap_*2;
	n_keypoints = init_json.landmark68_out_num_;
	n_heatmap_size = init_json.landmark68_out_heatmap_;
	cstr model_path = init_json.root_ + init_json.landmark68_;
	net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
	initialize();
}

void Landmark68::print_tensor(const Tensor &mat, const int n)
{

  for (int i = 0; i < n; i++){
    cout << mat.host<float>()[i] << ", ";
  }
  cout << endl;
}

int Landmark68::get_max_preds(const vector<vector<float> >&heatmaps, const int width, 
	vector<pair<float, float> > &kps, vector<float> &max_vals){

  const int n = heatmaps.size();
  const int height = heatmaps[0].size()/width;
  for (int i = 0; i < n; i++)
  {
    const vector<float> &ht = heatmaps[i];
    int max_idx = 0;
    for (int k=0; k < ht.size(); k++)
	{
		if (ht[k] > ht[max_idx])
		{
			max_idx = k;
		}	  
    }
    max_vals.push_back(ht[max_idx]);
    const int x = max_idx%width;
    const int y = min(max_idx/width, height-1);
	//opt for value
	float x_opt = x, y_opt = y;
	if (CHECKINTRANGE(x, 1, 63) && CHECKINTRANGE(y, 1, 63))
	{
		/*            (x, y-1)
			(x-1, y)   (x, y)   (x+1, y)
				      (x, y+1)
		*/
		intVec x_cor = { x, x, x,x-1, x+1};
		intVec y_cor = { y, y - 1, y + 1,y, y }; 
		floatVec idx;
		for (int iter_corner = 0; iter_corner < 5; iter_corner++)
		{
			idx.push_back(y_cor[iter_corner] * height + x_cor[iter_corner]);
		}
		enum XY
		{
			ORI,
			UP,
			DOWN,
			LEFT,
			RIGHT,
		};
		float x_sgn = SIGN(heatmaps[i][idx[RIGHT]] - heatmaps[i][idx[LEFT]]);
		float y_sgn = SIGN(heatmaps[i][idx[DOWN]] - heatmaps[i][idx[UP]]);
		x_opt += x_sgn * 0.25 ;
		y_opt += y_sgn * 0.25 ;
	}
	x_opt += 0.5;
	y_opt += 0.5;
    kps.push_back(pair<float, float>(x_opt, y_opt));
    // cout<< width<<", " << x << ", " << y << endl;
  }
  return 0;
}

int Landmark68::initialize(){

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

int Landmark68::inference(const cv::Mat& img, const int w, const int h, CGP::vecF& res)
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

	floatVec input_img(w*h * 3);
	float scale = 1.0f / 255.0;
#pragma omp parallel for
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				input_img[3 * (y*width_ + x) + c] = scale * float((img_rgb_float.at<cv::Vec3f>(y, x))(2-c));
			}
		}
	}
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

	int num_out = n_keypoints * n_heatmap_size*n_heatmap_size;
	res.resize(num_out);
	const MNN::Tensor* output_tensor = net->getSessionOutput(session, nullptr);
	const float *ret = output_tensor->host<float>();

	floatX2Vec heatmaps(n_keypoints, vector<float>(n_heatmap_size*n_heatmap_size));
	for (int s = 0; s < (n_keypoints - 1) / 4 + 1; s++) 
	{
		const float *p = &ret[4 * n_heatmap_size*n_heatmap_size *s];
		const int N = min(4, n_keypoints - s * 4);
		for (int k = 0; k < N; k++) {
			for (int i = 0; i < n_heatmap_size*n_heatmap_size; i++) {
				heatmaps[s * 4 + k][i] = p[i * 4 + k];
			}
		}
	}
	std::vector<std::pair<float, float> > kps;
	std::vector<float> max_vals;
	get_max_preds(heatmaps, 64, kps, max_vals);

	res.resize(kps.size() * 2);
	//remap 
	for (int i = 0; i < n_keypoints; i++)
	{
		res[2 * i] = kps[i].first*4;
		res[2 * i+1] = kps[i].second*4;
	}

	return 0;
}

