#ifndef REGISTER_H
#define REGISTER_H
#include <Eigen/Eigen>
#include <string>
#include "demo.h"
namespace NRICP
{
	void reg(const std::string& srcName, const std::string& markerName,
		const std::string& dstName, const std::vector<int>& roi,
		const std::string& saveNames, int numIte, 
		vector<demo::Mesh*>& meshes);

	void reg(const std::string& srcName, const std::string& markerName,
		const std::string& dstName, const std::vector<int>& roi, const std::vector<int>& boundaryPts,
		const std::string& saveNames, int numIte,
		vector<demo::Mesh*>& meshes);

	void reg(const std::string& srcName, const std::string& dstName, 
		const std::vector<int>& keyPtIdxSrc, const std::vector<int>& keyPtIdxDst,
		const std::vector<int>& roi,
		const std::string& saveNames, int numIte,
		vector<demo::Mesh*>& meshes);
	void centerize(Eigen::MatrixXd& input);
	void centerize(Eigen::MatrixXd& input, Eigen::VectorXd& meanVec);
}
#endif //DEFS_H