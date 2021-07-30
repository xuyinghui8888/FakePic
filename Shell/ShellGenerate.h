#ifndef SHELL_GENERATION_H
#define SHELL_GENERATION_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "CNodeBH.h"

namespace SHELLGEN
{
	void makeFurthestMesh(CGP::doubleX2Vec& aExtXYZ, unsigned int nXYZ, const double* paXYZ, unsigned int nTri,
		const unsigned int* paTri,	const SDF::CTriAryTopology& topo, const std::vector<SDF::CNodeBH>& aNode, bool is_ext, double max_height = 10);

	void makeFurthestMeshNormalize(CGP::doubleX2Vec& aExtXYZ, unsigned int nXYZ, const double* paXYZ, unsigned int nTri,
		const unsigned int* paTri, const SDF::CTriAryTopology& topo, const std::vector<SDF::CNodeBH>& aNode, bool is_ext, double max_height = 10);

	void makeFurthestMesh(const CGP::MeshCompress& in, bool is_ext, double max_height, CGP::MeshCompress& res);

	void normalizeInPlace(CGP::doubleVec& src);
}

#endif