#ifndef __SYSMETRIC_FINDER_H__
#define __SYSMETRIC_FINDER_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "../Config/JsonBased.h"
namespace CGP
{
	using json = nlohmann::json;

	namespace SysFinder 
	{
		void findSysBasedOnUV(const MeshCompress& src, const std::string& root);
		//bi-mapping
		void findSysBasedOnPosBiMap(const MeshCompress& src, const std::string& root);
		//force finding pos, not bi-mapping
		void findSysBasedOnPosOnly(const MeshCompress& src, const std::string& root,
		float mid_thres = EPSCG6, float left_right_mapping_thres = EPSCG3);
		void rectifyLandmarkPos(const json& config);
		void getRightToLeft(const json& config);
	}

	class MeshSysFinder : public JsonBaseClass
	{
	private:
		cstr template_;
		cstr mid_;
		cstr match_;
	public:
		intSet mid_ids_;
		intSet left_ids_;
		intSet right_ids_;
		//left to right (in mesh lab)
		intVec match_ids_;

	public:
		void init();
		MeshCompress template_obj_;
		//insert ori and sys in the same vector
		void getSysIdsInPlace(intVec& input);
		void getMirrorIdsInPlace(intVec& input);
		int getSysId(int input);
		void getSysPosLapInPlace(float3Vec& src) const;
		void getSysPosAvgInPlace(float3Vec& src);
		//in guijie sense left right
		void keepOneSideUnchanged(const float3Vec& src, float3Vec& dst, bool is_left);

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		RTTR_ENABLE()
		RTTR_REGISTRATION_FRIEND
	};

}
#endif