#ifndef EASY_GUIJIE_H
#define EASY_GUIJIE_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "../Config/JsonBased.h"
namespace CGP
{
	//used for testing codes
	namespace EASYGUIJIE
	{
		void getEyebrow(const cstr& root);
		void getEyebrow(const cstr& root, const cstr& data_dir);
		void getGuijieEyebrow();
		void getGuijieEyebrow(const json& config);
		void transformEyesToMesh(const cstr& root, const cstr& obj);
		void transformEyesToMesh(const cstr& root, const cstr& obj, 
			const cstr& ref_head, const cstr& ref_eyes, float scale, float eyes_shift);
		void replaceUV(const cstr& src, const cstr& dst);
		void generateTexDst(const json& config);
	}
}
#endif
