#ifndef MESH_DISTANCE_UTILS_H
#define MESH_DISTANCE_UTILS_H

#include "TriAnyTopology.h"
#include "SurfaceMeshReader.h"
#include "ContactTarget.h"
#include "CNodeBH.h"

namespace SDF
{
	namespace UTILS
	{
		void addFaceCenter(unsigned int iNo0, std::vector<CNodeBH>& aNo, const double fc[3], unsigned int indtri0,
			const unsigned int* paTri, const double* paXYZ);

		void findRangeCubic(int& icnt, double& r0, double& r1, double v0, double v1, double k0, double k1, double k2, double k3);

		void getAppDistInsidePrism(double& dist, double t, const double p[3], const double p0_0[3], const double p0_1[3],
			const double p0_2[3], const double p1_0[3], const double p1_1[3], const double p1_2[3]);

		void getBarnHutAppAreaDistGrad(unsigned int iNo0, const std::vector<CNodeBH>& aNode, const double p[3], const unsigned int* paTri,
			const double* paXYZ, double height, double& tot_area, double& app_dist, double app_grad[3]);

		void getBarnHutAppAreaDistGradIter(unsigned int iNo0, const std::vector<CNodeBH>& aNode, const double p[3], const unsigned int* paTri, 
			const double* paXYZ, double height, double& tot_area, double& app_dist, double app_grad[3]);

		void getDistNormal(double& dist, double n[3], const double p[3], const double p0[3], const double p1[3], const double p2[3]);
		
		double getInterpCoeffInsideProsim(const double p[3], const double p0_0[3], const double p0_1[3], const double p0_2[3],
			const double p1_0[3], const double p1_1[3], const double p1_2[3]);
		
		double getMinVolumePrisms(unsigned int nTri, const unsigned int* paTri, unsigned int nXYZ, const double* paXYZ0,
			const double* paXYZ1);

		void getNewPosGrad(double* pa_pos_new, double* pa_grad_new, double& dh, const double* pa_pos_old, const double* pa_grad_old,
			double height, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo,
			const std::vector<CNodeBH>& aNode);

		void getNormalInsidePrism(double n[3], double t, const double p0_0[3], const double p0_1[3], const double p0_2[3],
			const double p1_0[3], const double p1_1[3], const double p1_2[3]);

		void getPosGrad(double* pa_pos_new, double* pa_grad_new, double& dh, const double* pa_pos_old,
			const double* pa_grad_old,	double height, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, const unsigned int* paTri,
			const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode);

		double getTotalVolumePrisms(unsigned int nTri, const unsigned int* paTri, unsigned int nXYZ, const double* paXYZ0,
			const double* paXYZ1);

		void getValueGradient(const double p[], double grad[], double height, unsigned int nXYZ, const double* paXYZ,
			unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode);
		
		void getValueGradient(const double p[], double grad[], double height, unsigned int nXYZ, const double* paXYZ,
			unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo);

		void GetValueGradientExp(double* pa_pos_new, double* pa_grad_new, double& dh, const double* pa_pos_old, const double* pa_grad_old,
			double height, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo,
			const std::vector<CNodeBH>& aNode);

		void getValueGradientGrowth(const double p[], double grad[], double height, unsigned int nXYZ,
			const double* paXYZ, unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo,
			const std::vector<CNodeBH>& aNode);

		void getValueGradientStopThres(const double p[], double grad[], double height, unsigned int nXYZ, const double* paXYZ,
			unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode);

		void getVolumes(double v[4], const double p[3], const double p0[3], const double p1[3], const double p2[3], const double p3[3]);

		bool isIncludePrism(const double p[3], const double p0_0[3], const double p0_1[3], const double p0_2[3],
			const double p1_0[3], const double p1_1[3], const double p1_2[3], const unsigned int ind[3]);

		bool isInsideTet3D(const double p[3], const double p0[3], const double p1[3], const double p2[3], const double p3[3]);
		
		bool IsInsidePrisms(unsigned int& itri_in, const double p[3], unsigned int nTri, const unsigned int* paTri, unsigned int nXYZ,
			const double* paXYZ0, const double* paXYZ1);

		bool isInsidePrismsHash(unsigned int& itri_in, const double p[3], const CSpatialHashGrid3D& hash, unsigned int nTri,
			const unsigned int* paTri, unsigned int nXYZ, const double* paXYZ0, const double* paXYZ1);

		void makeExtMesh(const double height, std::vector<double*>& aExtXYZ, unsigned int nXYZ, const double* paXYZ, unsigned int nTri,
			const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode, bool is_ext, int layer);

		double minVolumePrism(const double p0_0[3], const double p0_1[3], const double p0_2[3], const double p1_0[3],
			const double p1_1[3], const double p1_2[3], const unsigned int ind[3]);

		void setCentroidWeight(unsigned int iNo0, std::vector<CNodeBH>& aNode, const unsigned int* paTri, const double* paXYZ);

		double volumePrism(const double p0_0[3], const double p0_1[3], const double p0_2[3], const double p1_0[3],
			const double p1_1[3], const double p1_2[3], const unsigned int ind[3]);	

	}
}

#endif