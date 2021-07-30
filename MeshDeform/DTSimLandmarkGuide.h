#ifndef __DEFORMATIONTRANSFERSAMETOPOLOGY_GUIDED_BY_LANDMARK_H__
#define __DEFORMATIONTRANSFERSAMETOPOLOGY_GUIDED_BY_LANDMARK_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "DTUtilities.h"

namespace CGP
{
	class DTGuidedLandmark
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	public:

		double Transfer_Weight_Correspond = 1.0;
		double Transfer_Weight_Anchor = 1e8;
		double Transfer_Weight_Regularization = 1e-8;
		double Transfer_Weight_Pair = 1e3;
		double Transfer_Weight_Land = 1e3;

		SpMat E1_, E1_t_;
		vecD E1_B_;

		DTGuidedLandmark() {}
		~DTGuidedLandmark() {}

		//wrapper for data structure
		void initWeight();

		bool init(const MeshCompress& A, const MeshCompress& A_deform, const intVec& pair_info = {}, const intVec& fix = {0});

		bool init(const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intVec& pair_info = {}, const intVec& fix = { 0 });

		intVec initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const intVec& pair_info = {}, double thres = 0.0);

		intVec initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intVec& pair_info = {}, double thres = 0.0);

		// Given B0, Ai, output Bi
		bool transfer(const float3Vec& srcVerts1, float3Vec& tarVerts1);

		void setPart(const intX2Vec& part_info, const std::vector<LandGuidedType>& type, const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B,
			const intX2Vec& xy_part_info = {}, const std::vector<vecD>& A_normal_fix = {}, const std::vector<vecD>& B_normal_fix = {});

		void getPartNormal(const intX2Vec& part_info, const std::vector<LandGuidedType>& type, const MeshCompress& A, std::vector<vecD>& normal);

		void setPairType(const PairType& init);

		void setDebug(bool init_info);

		void fastSetupE1A(const DTGuidedLandmark& src);
		void fastSetupE1A(const SpMat& E1, const vecD& E1_B);

	private:

		bool setupExExB();
		void clear();
		void setupE1A(const float3Vec& tarVerts0);
		void setupE1B(const float3Vec& srcVertsDeformed);
		void setupEaA();
		void setupEaB(const float3Vec& tarVerts0);
		void setupErA();
		void setupErB(const float3Vec& tarVerts0);
		void setupEaA(const intVec& pairInfo);
		void vertexVecToPoint(const vecD& x, float3Vec& verts) const;
		void vertexPointToVec(vecD& x, const float3Vec& verts, const int3Vec& faces) const;
	

	private:

		bool is_debug_ = false;
		bool is_init_ = false;
		bool is_check_topology_ = false;
		bool is_E1A_init_ = false;
		PairType pair_type_ = PairType::PAIR_DIS;
		//face saved by triangle
		int3Vec face_tri_;		
		intVec anchor_;			
		float3Vec A_vert_pos_;
		float3Vec A_deform_vert_pos_;
		float3Vec B_vert_pos_;
		intVec pair_info_;
		//the energy for src-tar triangle correspondences


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

		//E landmark
		intX2Vec part_info_;
		SpMat Ed_, Ed_t_;
		vecD Ed_B_;
	};

}
#endif