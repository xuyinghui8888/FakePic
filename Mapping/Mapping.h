#ifndef __MAPPING_FUNCTION_H__
#define __MAPPING_FUNCTION_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	namespace MAP
	{
		/*
		mapping defined
		from key---value, if value = -1, matching to empty
		*/
		intVec getReverseMapping(const intVec& src);
		intX2Vec splitMapping(const intVec& src, bool xyzOrder = true, int size = 2);
		intX2Vec passMapping(const intX2Vec& input);
		//index return double map[same_size]
		intVec passIdxThroughMapping(const intVec& idx, const intX2Vec& input_mapping);
		intX2Vec singleToDoubleMap(const intVec& src_to_dst);
		intVec discardRoiKeepOrder(const intVec& src, const intVec& roi);
		intX2Vec discardRoiKeepOrder(const intX2Vec& src, const intVec& reference, const intVec& roi);
		intVec getReverseRoi(const intVec& src, int n_max);
		intVec getInterset(const intX2Vec& src);
		intVec getUnionset(const intX2Vec& src);
		intVec getSubset(const intX2Vec& src);
	}

}
#endif