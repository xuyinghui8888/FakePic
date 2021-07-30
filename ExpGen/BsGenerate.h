#ifndef BSGEN_H
#define BSGEN_H
#include "../Mesh/MeshCompress.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/Tensor.h"
namespace CGP
{
	using json = nlohmann::json;
	class BsGenerate
	{
	public:
		BsGenerate(const json& config);
		void init(const json& config);
		void generateFace(const cstr& root, const cstr& json_file, int part_id);
		void generateFace(json& json_bs, MeshCompress& res);
	public:
#ifndef _MINI
		void generateCloseEye();
		void generateEyeTensor();
		void generateFace();
		void generateTensor();
#endif
	private:
		bool debug_ = false;
		bool is_init_ = false;
		Tensor shape_tensor_;
		cstr data_root_;
		cstrVec ordered_map_;
	};
}

#endif
