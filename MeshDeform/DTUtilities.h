#ifndef __DT_UTILITIES_H__
#define __DT_UTILITIES_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
/*
* Parameters for Deformation Transfer
* Transfer Energy Function:
*	min || Ws * Es + Wi * Ei + Wc * Ec + Wa * Ea + W1 * E1 + Wp * Epb+ Wr * Er ||^2
*		Es: the smoothness term for the target mesh,
*				ONLY for target trangle that cannot find corresponding src traingle.
*		Ei: the identity term for the target mesh
*				ONLY for target trangle that cannot find corresponding src traingle.
*		Ea: the anchor points energy
*				Now the anchor points are automatically selected via the boundaries of the markered components
*		E1: the correspond energy, deformation gradient related
*
*       Ep: the pair energy,
*
*       Er: the regularization energy, 
*
* Note:
*	Each of the weight will be divided by the number of constraints and then applied.
*/

	enum class LandGuidedType
	{
		XYZ_OPT,
		XY_OPT,
		XY_OPT_NO_SCALE,
		XY_NO_ROTATE,
		XY_ADD_PLANE_ROTATE,
	};

	enum class PairType
	{
		PAIR_DIS,
		PAIR_ZERO,
	};

	namespace DTTools
	{
		void fill4VertsOfFace(int id_f, const int3Vec& faces, const float3Vec& verts, 
			int4E& id_v, float3Vec& v);

		void getV(const float3Vec& v, mat3f& res);

		void fillCooSysByMat(tipDVec& cooSys, int row, int nTotalVerts, 
			const int4E& id, const mat34d& T);

		void setPartDelta(const intX2Vec& part_info, const intVec& A_land, const intVec& B_land,
			const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B,
			float3Vec& res, bool is_debug = false, const cstr& result_root = "");

		void setPartOri(const intX2Vec& part_info, const intVec& A_land, const intVec& B_land,
			const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B,
			float3Vec& res, bool is_debug = false, const cstr& result_root = "");

		void fastTransGivenStructure(const SpMat& A, SpMat& At);

		void fastAtAGivenStructure(const SpMat& A, const SpMat& At, SpMat& AtA);

		void getMatrixNamedbyT(const float3Vec& v, mat34d& A);

		template <class T>
		bool hasIllegalData(const T* data, int n)
		{
			for (int i = 0; i < n; i++)
			{
				if (std::isinf(data[i]) || std::isnan(data[i]))
					return true;
			}
			return false;
		}

		template <class T>
		bool hasIllegalTriangle(const T* pTris, int n)
		{
			for (int i = 0; i < n; i++)
			{
				const T& t = pTris[i];
				if (t[0] < 0 || t[1] < 0 || t[2] < 0)
					return true;
				if (t[0] == t[1] || t[0] == t[2] || t[1] == t[2])
					return true;
			}
			return false;
		}

	}	
}
#endif