#ifndef PARAMETRIC_COMMON_H
#define PARAMETRIC_COMMON_H

#include "demo.h"
using namespace demo;
using namespace Eigen;

static inline void posToVecx(vector<vec3>& poses, VectorXd& v) {
	auto numPoints = poses.size();
	v.resize(numPoints * 3);
	for (int i = 0; i < numPoints; i++) {
		v(i * 3) = poses[i].x; v(i * 3 + 1) = poses[i].y; v(i * 3 + 2) = poses[i].z;
	}
}
static inline void posToVecx(vector<vec3>& poses, VectorXd& v, vector<bool>& mask) {
	assert(poses.size() == mask.size());
	int count = 0;
	for (int i = 0; i < mask.size(); i++)
		if (mask[i])
			count++;
	auto numPoints = poses.size();
	v.resize(count * 3);
	int index = 0;
	for (int i = 0; i < numPoints; i++) {
		if (mask[i]) {
			v(index * 3) = poses[i].x; v(index * 3 + 1) = poses[i].y; v(index * 3 + 2) = poses[i].z;
			index++;
		}
	}
}
static inline void vecxToPos(VectorXd& v, vector<vec3>& poses) {
	auto numPoints = v.rows() / 3;
	poses.resize(numPoints);
	for (int i = 0; i < numPoints; i++) {
		poses[i] = vec3(v(i * 3), v(i * 3 + 1), v(i * 3 + 2));
	}
}
static inline void updateSrc(Mesh& src, const MatrixXd& uSelected, VectorXd& paras, const MatrixXd& vMor) {
	VectorXd v = uSelected*paras + vMor;
	vecxToPos(v, src.meshPositions);
}
static inline Vector3d transformPt(const double* x, const vec3& srcPt) {
	Matrix4d mat;
	mat << x[0], x[1], x[2], x[3],
		x[4], x[5], x[6], x[7],
		x[8], x[9], x[10], x[11],
		0, 0, 0, 1;
	Affine3d aff(mat);
	Vector3d vPt(srcPt.x, srcPt.y, srcPt.z);
	return aff*vPt;
}
#endif