#ifndef _KEYPOINTDET_H_
#define _KEYPOINTDET_H_
#include "../Basic/CGPBaseHeader.h"
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>

class KeypointDetection
{  

private:

	const int num_kps=0;
	const int height=0;
	const int width=0;
	std::shared_ptr<MNN::Interpreter> net;
	MNN::Session *session = NULL;
	MNN::CV::ImageProcess::Config config;
	MNN::Tensor* input_tensor = NULL;
  
public:

	KeypointDetection(const char *model_path, const int w, const int h, const int num_kps):
    width(w), height(h), num_kps(num_kps)
	{
		net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path));
		initialize();
	}

	~KeypointDetection()
	{
	if (session)
		{
			net->releaseSession(session);
			session = NULL;
			net->releaseModel();
		}
	}

	int inference(const unsigned char *img, const int w, const int h, std::vector<std::pair<int, int> > &kps, std::vector<float> &max_vals);
	
	int get_kps_num()const
	{
		return num_kps;
	}

	void print_tensor(const MNN::Tensor &mat, const int n);

	int get_max_preds(const CGP::floatX2Vec& heatmaps, const int width, std::vector<std::pair<int, int> > &kps, CGP::floatVec &max_vals);

protected:

	int initialize();
};

#endif /* _KEYPOINTDET_H_ */
