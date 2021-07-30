#ifndef RIGID_BS_GENERATION_H
#define RIGID_BS_GENERATION_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	using json = nlohmann::json;
	namespace RIGIDBS 
	{
		//using only left or right part
		void getSysDirectMappingFromMesh(const MeshCompress& part_base, const cstr& reference_root,
			const cstr& result_root, const intVec& part_all_map, const json& mesh_mapping, bool is_left,
			const float3E& t_shift);
		void getUsingZeroShift(const MeshCompress& ori, const cstrVec& names, const cstr& result_dir);
		intVec divideIdx(const MeshCompress& part_base, const intVec& roi, bool is_left);
		void getJsonExp(const MeshCompress& part_base, const cstr& result_dir, json& res);
		void getSysDivideAndConquer(const MeshCompress& src, const MeshCompress& ref, 
			const cstr& ref_root, const cstr& res_root, const cstr& temp_root);
		void getSingleBS(const MeshCompress& src, const MeshCompress& ref,
			const cstr& ref_root, const cstr& res_root, const cstr& temp_root);
	}
}

#endif
