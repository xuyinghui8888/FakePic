#ifndef __DEFORMATIONTRANSFERSAMETOPOLOGY_H__
#define __DEFORMATIONTRANSFERSAMETOPOLOGY_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	class MeshTransfer
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	public:
		MeshTransfer() {}
		~MeshTransfer() {}

		// Initialize the topology and 0th reference mesh A0£¬ pair info from_to from_to
		bool init(int nTriangles, const int3E* pTriangles, int nVertices,
			const float3E* pSrcVertices0, const float3E* pTarVertices0, const intVec& pairInfo = {}, const intVec& fix_points = { 0 });

		//wrapper for data structure
		bool init(const MeshCompress& A, const MeshCompress& A_deform, const intVec& v = {}, const intVec& fix = { 0 });

		intVec initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const intVec& v = {}, double thres = 0.0);

		intVec initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intVec& v = {}, double thres = 0.0);

		// Given B0, Ai, output Bi
		bool transfer(const float3Vec& srcVerts1, float3Vec& tarVerts1);

	private:

		void clear();
		void setupE1A(const float3Vec& tarVerts0);
		void setupE1B(const float3Vec& srcVertsDeformed);
		void setupEaA();
		void setupEaB(const float3Vec& tarVerts0);
		void setupErA();
		void setupErB(const float3Vec& tarVerts0);
		void setupEaA(const intVec& pairInfo, const float3Vec& tarVerts0);
		void vertexVecToPoint(const vecD& x, float3Vec& verts) const;
		void vertexPointToVec(vecD& x, const float3Vec& verts, const int3Vec& faces) const;
	private:

		bool is_init_ = false;
		bool is_check_topology_ = false;

		//face saved by triangle
		int3Vec face_tri_;
		intVec anchor_;
		float3Vec A_vert_pos_;
		float3Vec A_deform_vert_pos_;
		intVec pair_info_;
		//the energy for src-tar triangle correspondences
		SpMat E1_, E1_t_;
		vecD E1_B_;
		//fix energy
		SpMat Ea_, Ea_t_;
		vecD Ea_B_;
		SpMat Ep_, Ep_t_;
		vecD Ep_B_;
		//for isolated-point regularization
		SpMat Er_AtA_;
		vecD Er_AtB_;
		// Ea_t_ * Ea_B_ * Wa + Er_AtB_ * Wr_
		vecD Ea_Er_AtB_;
		// the total energy matrix
		SpMat E_AtA_;
		vecD E_AtB_, res_vertex_pos_;
		SpSolver solver_;
	};

}
#endif