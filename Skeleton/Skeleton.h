#ifndef SKELETON_H
#define SKELETON_H

#include "../Basic/CGPBaseHeader.h"

namespace CGP
{
	class Skeleton
	{
	private:
		doubleX2Vec readDynamic(const cstr& bf);
		doubleX2Vec readDynamicTrans(const cstr& bf);
		doubleVec locToWorld(doubleX2Vec bf, doubleVec world, int joint_idx);
		void locToWorld(const float3Vec& local_pos, const intVec& parent_node, const std::vector<mat3f>& rotate, int joint_idx, float3E& res);
		void locToWorld(const float3Vec& local_pos, const intVec& parent_node, const float3Vec& rotate_xyz, int joint_idx, 
			std::vector<Eigen::Isometry3f>& world_list, std::vector<Eigen::Isometry3f>& bind_list, Eigen::Isometry3f& world_matrix, Eigen::Isometry3f& bind_matrix);
		doubleVec rotationMatrix(float radX, float radY, float radZ);
		void rotationMatrixEigen(float radX, float radY, float radZ, mat3f& rotation_matrix);
		doubleVec mat33Mul(doubleVec mat1, doubleVec mat2);
		doubleVec eulerToWorld(doubleX2Vec bf, doubleX2Vec pos, doubleVec result, int joint_idx);
		double3E eulerToWorld(doubleX2Vec bf, std::vector<mat3f> pos, double3E result, int joint_idx);
		void scaleLocToWorld(doubleVec& scale_div, doubleVec& scale, const intVec& parent_node, int joint_idx, double& res);
	public:
		void unitTest();
		void unitInverse();
	};
}

#endif
