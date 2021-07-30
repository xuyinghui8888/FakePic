#include "Eigen/Eigen"
#include <vector>
#include "../Basic/CGPBaseHeader.h"
#ifndef ICP_H
#define ICP_H

#define N_pt 30    // # of points in the datasets
#define N_tests 100    // # of test iterations
#define noise_sigma 0.01    // standard deviation error to be added
#define translation 0.1     // max translation of the test set
#define rotation 0.1        // max rotation (radians) of the test set

namespace ICP
{
	typedef struct {
		Eigen::Matrix4d trans;
		std::vector<float> distances;
		int iter;
	}  ICP_OUT;

	typedef struct {
		std::vector<float> distances;
		std::vector<int> indices;
	} NEIGHBOR;

	Eigen::Matrix4d best_fit_transform(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

	//R*src+T = dst
	void bestFitTransform(const CGP::float3Vec& dst, const CGP::float3Vec& src, CGP::mat4f& T);

	//R*src+T = dst
	void fitTransform(const CGP::float3Vec& dst, const CGP::float3Vec& src, CGP::mat4f& T);

	ICP_OUT icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations = 20, int tolerance = 0.001);

	// throughout method
	NEIGHBOR nearest_neighbot(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst);
	float dist(const Eigen::Vector3d &pta, const Eigen::Vector3d &ptb);

	
}
#endif
