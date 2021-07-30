#ifndef __REFERENCE_DEFORM_H__
#define __REFERENCE_DEFORM_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	
	class Grid 
	{	
	public:
		float3E dim_ = float3E(0, 0, 0);
		float3E center_;
		int3E dense_ = int3E(10, 10, 10);
		int location(int3E pos) const;
		void location(int3E& out, const int pos) const;
		void coord(float3E& out, const int pos) const;
		void getInterpWeight(float3E p, floatVec &w, intVec &pos) const;
		void getLaplaceParameter(float3E& out, int position, intVec &pos, floatVec &w) const;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	class RefDefConfig
	{	
	public:
		int3E density = int3E(20, 20, 20);
		float smooth_weight = 0.3;
		float2E bb_x_expand = float2E(1.2, 1.2);
		float2E bb_y_expand = float2E(1.2, 1.2);
		float2E bb_z_expand = float2E(1.2, 1.2);
	};

	class ReferenceDeform 
	{

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		ReferenceDeform() {}
		~ReferenceDeform() {}

		void init(const Eigen::Vector3f *sourceMesh1, const int mesh1VertSize, const RefDefConfig& config);

		// Scale shows the bounding box scaling to the 8 direction.
		void init(const float3E *sourceMesh1, const int mesh1VertSize,
			const float2E& xScale = float2E(1.0f, 1.0f), 
			const float2E& yScale = float2E(1.0f, 1.0f), 
			const float2E& zScale = float2E(1.0f, 1.0f));

		// dense shows the number of basic cubes in whole gird.
		void setDensity(const int3E& density = int3E(100, 100, 100));

		void setSmoothWeight(float w = 0.3f);

		// 计算meshwraping, 请先输入mesh、网格密度及平滑指数
		void process(Eigen::Vector3f* out, const Eigen::Vector3f* refMesh1, const Eigen::Vector3f* sourceMesh2, 
			const int mesh2VertSize) const;

	private:

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver_;
		Eigen::SparseMatrix<float> sp_;
		float3Vec grid_lap_;
		float3Vec ref_A_;
		int ref_A_n_vertex_;
		//int _m2size;
		Grid grid_;
		float smooth_weight_ = 0.3f;
		void computeBoundingBoxandScale(const float2E& xScale, const float2E& yScale, const float2E& zScale);
	};


}
#endif