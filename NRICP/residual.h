#ifndef RESIDUAL_H
#define RESIDUAL_H
#include "ceres/ceres.h"
#include "demo.h"
#include "MyKDtree.h"
#include <set>
#include <Eigen/Dense>
#include "common.h"
#include "dis.h"

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace Eigen;
using namespace demo;

#define NUM_FREEDOM 12//num of freedoms of Affine matrix, i.e.12

struct DataFunctor2 {
public:
	DataFunctor2(const vec3& srcPt, const MyKDtree& dstTree, Mesh& dstMesh, vector<int>& sampleIndices)
		: srcPt(srcPt), dstTree(dstTree), dstMesh(dstMesh), sampleIndices(sampleIndices){}

	bool operator()(const double* const xi, double* r) const {
		Eigen::Vector3d vPt = transformPt(xi, srcPt);
		vec3 transformed(vPt[0], vPt[1], vPt[2]);
		vector<int> k_indices;
		vector<float> k_sqr_dist;
		pcl::PointXYZ p(vPt[0], vPt[1], vPt[2]);
		dstTree.nearestKSearch(vPt, k_indices, k_sqr_dist, 1);
		int triIndex = sampleIndices[k_indices[0]];
		auto& ptIdx1 = dstMesh.meshCells[3*triIndex];
		auto& ptIdx2 = dstMesh.meshCells[3 * triIndex+1];
		auto& ptIdx3 = dstMesh.meshCells[3 * triIndex+2];
		auto nearest = nearestPtInTriangle(transformed, dstMesh.meshPositions[ptIdx1], dstMesh.meshPositions[ptIdx2], dstMesh.meshPositions[ptIdx3]);
		vec3 disp = nearest - transformed;

		r[0] = disp.x;
		r[1] = disp.y;
		r[2] = disp.z;
		return true;
	}
	static ceres::CostFunction* Create(const vec3& srcPt, const MyKDtree& dstTree, Mesh& dstMesh, vector<int>& sampleIndices) {
		return new ceres::NumericDiffCostFunction<DataFunctor2, ceres::CENTRAL, 3, NUM_FREEDOM>(
			new DataFunctor2(srcPt, dstTree, dstMesh, sampleIndices));
	}
private:
	const vec3& srcPt;
	const MyKDtree &dstTree;
	const Mesh& dstMesh;
	const vector<int>& sampleIndices;
};
struct SmoothFunctor2 {
public:
	SmoothFunctor2(const double sqrt_weight, const vec3& p1, const vec3& p2) : sqrt_weight(sqrt_weight), p1(p1), p2(p2) {}

	template <typename T>
	bool operator()(const T* const x1, const T* const x2, T* r) const {
		for (int i = 0; i < NUM_FREEDOM; i++) {
			if(i==0)
				r[i] = T(sqrt_weight)*(T(p1.x)*(x1[i]-T(1)) - T(p2.x)*(x2[i]- T(1)));
			else if(i==5)
				r[i] = T(sqrt_weight)*(T(p1.y)*(x1[i] - T(1)) - T(p2.y)*(x2[i] - T(1)));
			else if (i == 10)
				r[i] = T(sqrt_weight)*(T(p1.z)*(x1[i] - T(1)) - T(p2.z)*(x2[i] - T(1)));
			else if (i % 4 == 0)
				r[i] = T(sqrt_weight)*(T(p1.x)*x1[i] - T(p2.x)*x2[i]);
			else if (i % 4 == 1)
				r[i] = T(sqrt_weight)*(T(p1.y)*x1[i] - T(p2.y)*x2[i]);
			else if (i % 4 == 2)
				r[i] = T(sqrt_weight)*(T(p1.z)*x1[i] - T(p2.z)*x2[i]);
			else
				r[i] = T(sqrt_weight)*(x1[i] - x2[i]);
		}
		return true;
	}
	static ceres::CostFunction* Create(const double sqrt_weight, const vec3& p1, const vec3& p2) {
		return new ceres::AutoDiffCostFunction<SmoothFunctor2, NUM_FREEDOM, NUM_FREEDOM, NUM_FREEDOM>(
			new SmoothFunctor2(sqrt_weight, p1, p2));
	}
private:
	const double sqrt_weight;
	const vec3 &p1, &p2;
};
struct MarkerFunctor {
public:
	MarkerFunctor(const double sqrt_weight, const vec3& srcPt, const vec3& dstPt) : sqrt_weight(sqrt_weight), srcPt(srcPt), dstPt(dstPt) {}

	bool operator()(const double* const xi, double* r) const {
		Eigen::Vector3d vPt = transformPt(xi, srcPt);
		r[0] = sqrt_weight*abs(vPt[0] - dstPt.x);
		r[1] = sqrt_weight*abs(vPt[1] - dstPt.y);
		r[2] = sqrt_weight*abs(vPt[2] - dstPt.z);
		return true;
	}
	static ceres::CostFunction* Create(const double sqrt_weight, const vec3& srcPt, const vec3& dstPt) {
		return new NumericDiffCostFunction<MarkerFunctor, ceres::CENTRAL, 3, NUM_FREEDOM>(
			new MarkerFunctor(sqrt_weight, srcPt, dstPt));
	}
private:
	const double sqrt_weight;
	const vec3& srcPt;
	const vec3& dstPt;
};
struct HeightFunctor {
public:
	HeightFunctor(const double sqrt_weight, const vec3& srcPt) : sqrt_weight(sqrt_weight), srcPt(srcPt) {}

	bool operator()(const double* const xi, double* r) const {
		Eigen::Vector3d vPt = transformPt(xi, srcPt);
		r[0] = sqrt_weight*abs(vPt[2] - srcPt.z);
		return true;
	}
	static ceres::CostFunction* Create(const double sqrt_weight, const vec3& srcPt) {
		return new NumericDiffCostFunction<HeightFunctor, ceres::CENTRAL, 1, NUM_FREEDOM>(
			new HeightFunctor(sqrt_weight, srcPt));
	}
private:
	const double sqrt_weight;
	const vec3& srcPt;
};
#endif