#ifndef FACE_ID_H
#define FACE_ID_H
#include "../Basic/CGPBaseHeader.h"
#include "JsonBased.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	class FaceIDFinder : public JsonBaseClass
	{
	private:
		int n_dim_;
		int n_id_;
		cstr file_;
	public:
		/*
		[0] start order in original order;
		[1] reordered star in female/maile order;
		[2] image for the star;
		[3-514] 512 data
		*/
		matF id_info_;
		//each row is id data
		matF id_data_;

	public:
		void init();
		void getMatch(const vecF& input, int& match_id, bool is_debug = false) const;
		void getMatch(const vecF& input, int& match_id, doubleVec& res, bool is_debug = false) const;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};
}

#endif
