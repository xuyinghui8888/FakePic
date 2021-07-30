#ifndef __LAPLACIANDEFORM_H__
#define __LAPLACIANDEFORM_H__
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	enum class LinkType
	{
		KEEP_ZERO,
		KEEP_DIS,
	};

	class LaplacianDeform
	{
	public:
		//TODO need for fix point adjustment for weight
		//secondary wrapper
		void init(CGP::MeshCompress& input, const intVec& handle_idx, const intVec& fix_idx, const intVec& linked_vertex = {});
		void initRoi(CGP::MeshCompress& input, const intVec& handle_idx, const intVec& fix_idx);
		void deform(const float3Vec& handle_pos, float3Vec& res, const LinkType& link_type = LinkType::KEEP_ZERO);

	private:
		void prepareDeform(
			const unsigned int* cells, const int nCells,

			const float* positions, const int nPositions,

			const int* roiIndices, const int nRoi,

			const int unconstrainedBegin,

			bool RSI);

		void doDeform(const float* handlePositions, int nHandlePositions, float* outPositions, const LinkType& link_type = LinkType::KEEP_ZERO);

		void freeDeform();

		void clear();

		bool b_init_ = false;

		intVec handle_idx_;
		intVec fix_idx_;
		intVec link_vertex_;
		float3Vec fix_pos_;
		double fix_weight_ = 1.0;
		double smooth_weight = 10.0;
		double link_weight_ = 1.0;
		intVec idx_to_equation_;
	};
}
#endif