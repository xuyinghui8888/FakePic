#ifndef RES_VAR_H
#define RES_VAR_H
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
#include "MeshResult.h"
#include "TextureResult.h"
#include "JsonData.h"
#include "JsonHelper.h"
namespace CGP
{
	enum class PostProcessRoutine
	{
		NORMAL,        //0
		FIT_CONTOUR,   //1 
		FIT_FACE,      //2
		TEX_BASE,      //3  
	};

	enum class Type3dmm
	{
		MS,           //0 using original microsoft 3dmm data
		NR,           //1 using neural rendering data
		NR_RAW,       //2 using neural rendering raw data
		NR_CPP,       //3 using mnn version to add for results
		NR_CPP_RAW,   //4 using mnn version to change results
	};

	class ResVar
	{
	public:
		bool is_debug_;
		cstr output_dir_;
		cv::Mat input_image_;
		cv::Rect bb_box_;
		Gender gender_ = Gender::MALE;
		PostProcessRoutine pp_type_ = PostProcessRoutine::NORMAL;
	public:
		ResVar();
		void init();
		void setInput(const json& test_config);
		MeshResult res_mesh;
		TextureResult res_texture;
		Type3dmm model_3dmm_type_ = Type3dmm::MS;
		vecF coef_3dmm_;
		cstr debug_data_pack_;
	};	

	//neural rendering
	class NRResVar
	{
	//input
	public:
		bool is_debug_;
		bool load_from_json = false;
		cstr output_dir_;
		cv::Mat input_image_;
		cv::Mat landmark_256_xy_img_;
		cv::Rect bb_box_;
		Gender gender_ = Gender::MALE;
		vecF coef_3dmm_;
		vecF landmark_256_xy_;
		PostProcessRoutine pp_type_ = PostProcessRoutine::NORMAL;
		Type3dmm model_3dmm_type_ = Type3dmm::MS;
		cstr debug_data_pack_;

	//result
	public:

	public:
		NRResVar();
		void init();
		void setInput(const json& test_config);
	};
}

#endif
