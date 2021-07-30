#ifndef TINY_TOOL_H
#define TINY_TOOL_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
namespace CGP
{
	//used for testing codes
	namespace TinyTool
	{
		void generatePlane(const cstr& out_file);
		void objSafeCheck(const MeshCompress& src);
		void objSafeCheck(const cstr& file_pos);
		void discardVertexFolder();
		void getEyelash(const cstr& cur_root);
		void getMeshSysInfo(const cstr& cur_root, const cstr& mesh);
		void getSysLandmarkPoint(const cstr& config_root, const cstr& config_json, 
			const cstr& root, const cstr& hand_landmark, const cstr& sys_match, 
			const cstr& result);

		void getSysIDReduceSame(const cstr& config_root, const cstr& config_json,
			const cstr& root, const cstr& hand_landmark, const cstr& result);
		void getSysID1to1(const cstr& config_root, const cstr& config_json,
			const cstr& root, const cstr& hand_landmark, const cstr& result);
		void resizeTestingImageSize(const cstr& root);
		void getObjToJson(const cstr& root, const cstr& json_file, const cstr& neutral);
		void getMeshFileNameToJson(const cstr& root, const cstr& json_file);
		void getMatchingFromUnityToMayaCube();
		//maya unity mapping
		void getMatchingFromUnityToMayaGuijie();
		void getMatchingFromPartToAll();
		void skeletonChange();
		void getTaobaoLips();
		void renameFiles();
		void getFileNamesToPCAJson(const cstr& folder);
		void getBSMeshFromDeltaV1File();
		void turnJsonToString();
	}
}

#endif
