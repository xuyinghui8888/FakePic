#include "Test.h"
#include "Prepare.h"
#include "../Basic/MeshHeader.h"
#include "../VisHelper/VisHeader.h"
#include "../Config/Tensor.h"
#include "../Config/TensorHelper.h"
#include "../Config/JsonHelper.h"
#include "../Sysmetric/Sysmetric.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Metric/Metric.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../Solver/BVLSSolver.h"
#include "../ExpGen/ExpGen.h"
#include "../ExpGen/BsGenerate.h"
#include "../OptV2/OptTypeV2.h"
#include "../OptV2/OptTypeV3.h"
#include "../Eyebrow/EyebrowType.h"
#include "../ImageSim/ImageSimilarity.h"
#include "../ImageSim/ImageUtils.h"
#include "../Beauty/Beauty35.h"
#include "../Debug/DebugTools.h"
#include "../Landmark/Landmark68Wrapper.h"
#include "../MeshDeform/ReferenceDeform.h"
#include "../Shell/ShellGenerate.h"
#include "../Bvh/Bvh11.hpp"
#include "../Test/TinyTool.h"
#include "../Test/TestClass.h"
#include "../Basic/ToDst.h"

using namespace CGP;

void TESTFUNCTION::ruhuaModel()
{
	MeshCompress face_data = "D:/dota210621/0621_00/res/shape.obj";
	intVec keep_vertex = FILEIO::loadIntDynamic("D:/dota210621/0621_00/res/keep_vertex.txt");
	face_data.keepRoi(keep_vertex);
	face_data.saveObj("D:/dota210621/0621_00/res/face_region.obj");
}

void TESTFUNCTION::normalizeTaobaoMayaRawData()
{
	cstr src_dir = "D:/dota210621/0621_00/";
	cstr dst_dir = "D:/dota210621/0621_00/res/";
	SG::needPath(dst_dir);
	auto file_names = FILEIO::getFolderFiles(src_dir, FILEIO::FILE_TYPE::MESH);
	for (auto i : file_names)
	{
		MeshCompress src;
		src.loadOri(src_dir + i);
		src.flipYZ();
		src.discardMaterial();
		src.saveOri(dst_dir + i);
	}
}

void TESTFUNCTION::getLinearBoundingBoxBrow(float range_m1_0, float range_0_p1, std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res)
{	
	TESTSTRUCTURE::TaobaoSkeleton Brow_top_LT, Brow_mid_LT, Brow_end_LT, Brow_top_RT, Brow_mid_RT, Brow_end_RT;	
	Brow_top_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932581, -1.644810, 0.000000));
	Brow_top_LT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_mid_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932579, -1.644790, 0.000000));
	Brow_mid_LT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_end_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932578, -1.644820, 0.000000));
	Brow_end_LT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_top_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932579, -1.644780, 0.000000));
	Brow_top_RT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_mid_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932578, -1.644780, 0.000000));
	Brow_mid_RT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_end_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932578, -1.644810, 0.000000));
	Brow_end_RT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_top_LT.trans = ((range_0_p1 + range_0_p1) * float3E(0.355213, 0.003036, -1.206520));
	Brow_mid_LT.trans = ((range_0_p1 + range_0_p1) * float3E(0.353503, 0.003021, -1.206520));
	Brow_end_LT.trans = ((range_0_p1 + range_0_p1) * float3E(0.445297, 0.003799, -1.206520));
	Brow_top_RT.trans = ((range_0_p1 + range_0_p1) * float3E(0.355209, 0.003052, 1.206520));
	Brow_mid_RT.trans = ((range_0_p1 + range_0_p1) * float3E(0.353499, 0.003036, 1.206520));
	Brow_end_RT.trans = ((range_0_p1 + range_0_p1) * float3E(0.445293, 0.003815, 1.206520));
	Brow_top_LT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.032788, 1.331450, 0.000000));
	Brow_mid_LT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.032789, 1.331450, 0.000000));
	Brow_end_LT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.032790, 1.331450, 0.000000));
	Brow_top_RT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.032788, 1.331450, 0.000000));
	Brow_mid_RT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.032789, 1.331500, 0.000000));
	Brow_end_RT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.032791, 1.331470, 0.000000));
	Brow_top_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.000000, 0.000000, 0.948776));
	Brow_mid_LT.trans = ((range_m1_0 + range_0_p1) * float3E(-0.452989, -0.003876, 0.948776));
	Brow_end_LT.trans = ((range_m1_0 + range_0_p1) * float3E(-1.544870, -0.013214, 0.948774));
	Brow_top_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.000000, 0.000000, -0.948776));
	Brow_mid_RT.trans = ((range_m1_0 + range_0_p1) * float3E(-0.452985, -0.003876, -0.948778));
	Brow_end_RT.trans = ((range_m1_0 + range_0_p1) * float3E(-1.544860, -0.013214, -0.948780));
	Brow_end_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.946616, -1.636780, 0.000000));
	Brow_end_LT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425935, 0.000000));
	Brow_end_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.946614, -1.636760, 0.000000));
	Brow_end_RT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425935, 0.000000));
	Brow_end_LT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.021393, 2.499910, 0.000000));
	Brow_end_RT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.021393, 2.499910, 0.000000));
	Brow_mid_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.946617, -1.636750, 0.000000));
	Brow_mid_LT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425935, 0.000000));
	Brow_mid_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.946616, -1.636730, 0.000000));
	Brow_mid_RT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425935, 0.000000));
	Brow_mid_LT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.021393, 2.499910, 0.000000));
	Brow_mid_RT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.021393, 2.499910, 0.000000));
	Brow_top_LT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932581, -1.644810, 0.000000));
	Brow_top_LT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_top_RT.trans = ((range_m1_0 + range_0_p1) * float3E(0.932579, -1.644780, 0.000000));
	Brow_top_RT.scale = ((range_m1_0 + range_0_p1) * float3E(0.000000, -0.425934, 0.000000));
	Brow_top_LT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.021392, 2.499910, 0.000000));
	Brow_top_RT.trans = ((range_0_p1 + range_0_p1) * float3E(-0.021392, 2.499910, 0.000000));
	res = { Brow_top_LT, Brow_mid_LT, Brow_end_LT, Brow_top_RT, Brow_mid_RT, Brow_end_RT };
	//return res;
}

void TESTFUNCTION::getLinearBoundingBoxEyeRoi(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res)
{
	TESTSTRUCTURE::TaobaoSkeleton Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT;
	
	TESTSTRUCTURE::TaobaoSlide Dual_EyeBig_Small_, Dual_EyeLowerFront_Outer_, Dual_EyeInUp_Down_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeInWide_Narraw_, Dual_EyeMidUpperUp_Down_, Dual_EyeOutUp_Down_, Dual_EyeOutWide_Narraw_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeAllWide_Narraw_, Dual_EyeForward_Backward_, Dual_EyeUp_Down_;

	double range_0_p1;
	Eye_scale_LT.scale = ((Dual_EyeBig_Small_(-1, 0) + range_0_p1) * float3E(-0.435860, -0.435859, -0.435860));
	Eye_scale_RT.scale = ((Dual_EyeBig_Small_(-1, 0) + range_0_p1) * float3E(-0.435860, -0.435860, -0.435859));
	//EyeFrontLowerDown = ((Dual_EyeLowerFront_Outer_(0, 1) + EyeFrontLowerDown) * 1.000000);
	//EyeFrontUpperUp = ((Dual_EyeUpperFront_Outer_(-1, 0) + EyeFrontUpperUp) * 1.000000);
	EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + range_0_p1) * float3E(-12.437200, 3.597150, 0.256516));
	EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + range_0_p1) * float3E(-12.437200, -3.541290, 0.256638));
	EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(0, 1) + range_0_p1) * float3E(15.736300, 27.218700, -0.713188));
	EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(0, 1) + range_0_p1) * float3E(15.751900, -27.128099, 0.056992));
	EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + range_0_p1) * float3E(-0.001537, -0.007004, 6.259930));
	EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + range_0_p1) * float3E(-0.255201, -0.006882, -6.254790));
	//EyeMiddleLowerDown = ((Dual_EyeMidLowerUp_Down_(-1, 0) + EyeMiddleLowerDown) * 1.000000);
	//EyeMiddleLowerUp = ((Dual_EyeMidLowerUp_Down_(0, 1) + EyeMiddleLowerUp) * 1.000000);
	//EyeMiddleUpperDown = ((Dual_EyeMidUpperUp_Down_(-1, 0) + EyeMiddleUpperDown) * 1.000000);
	//EyeMiddleUpperUp = ((Dual_EyeMidUpperUp_Down_(0, 1) + EyeMiddleUpperUp) * 1.000000);
	//EyeNarrow = ((Dual_EyeWide_Narraw_(-1, 0) + EyeNarrow) * 1.000000);
	EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + range_0_p1) * float3E(0.061870, -21.257299, 1.522710));
	EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + range_0_p1) * float3E(0.000000, 21.256399, -1.524060));
	//EyeOuterLowerDown = ((Dual_EyeLowerFront_Outer_(-1, 0) + EyeOuterLowerDown) * 1.000000);
	//EyeOuterUpperUp = ((Dual_EyeUpperFront_Outer_(0, 1) + EyeOuterUpperUp) * 1.000000);
	EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + range_0_p1) * float3E(-0.641535, -0.090179, -15.995000));
	EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + range_0_p1) * float3E(0.000000, 0.000000, 16.007401));
	EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + range_0_p1) * float3E(0.000000, 20.374800, -0.000114));
	EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + range_0_p1) * float3E(0.000000, -20.374800, 0.000000));
	EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + range_0_p1) * float3E(0.585359, -0.074822, 14.562100));
	EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + range_0_p1) * float3E(0.000000, 0.000000, -14.573700));
	//EyeWide = ((Dual_EyeWide_Narraw_(0, 1) + EyeWide) * 1.000000);
	Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + range_0_p1) * float3E(0.000000, 0.000000, -2.180600));
	Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + range_0_p1) * float3E(0.000000, 0.000000, 2.180600));
	Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(-1, 0) + range_0_p1) * float3E(0.000000, -2.327220, 0.019692));
	Eye_scale_LT.scale = ((Dual_EyeBig_Small_(0, 1) + range_0_p1) * float3E(0.374041, 0.374043, 0.374039));
	Eye_scale_RT.scale = ((Dual_EyeBig_Small_(0, 1) + range_0_p1) * float3E(0.374041, 0.374041, 0.374039));
	EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + range_0_p1) * float3E(0.330012, 0.045853, -15.832900));
	EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + range_0_p1) * float3E(0.330011, -0.045869, 15.832900));
	Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + range_0_p1) * float3E(0.000000, 0.000000, 0.558024));
	Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + range_0_p1) * float3E(0.000000, -0.000214, -0.558090));
	Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(0, 1) + range_0_p1) * float3E(0.000000, -2.327220, 0.019692));
	//EyeSmall = ((Dual_PupilBig_Small_(-1, 0) + EyeSmall) * 1.000000);
	//EyeBig = ((Dual_PupilBig_Small_(0, 1) + EyeBig) * 1.000000);
	Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(0, 1) + range_0_p1) * float3E(-0.659834, 0.005692, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(0, 1) + range_0_p1) * float3E(-0.659830, 0.005676, 0.000000));
	//glass.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.213036, 0.001816, 0.000000));
	Eye_scale_LT.trans = ((Dual_EyeUp_Down_(-1, 0) + range_0_p1) * float3E(0.000000, -1.381740, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeUp_Down_(-1, 0) + range_0_p1) * float3E(0.000000, -1.381760, 0.000000));
	Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(-1, 0) + range_0_p1) * float3E(0.000000, -2.327220, 0.019692));
	//glass.trans = ((Dual_EyeUp_Down_(-1, 0) + range_0_p1) * float3E(-0.010664, -1.246280, 0.000000));
	Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(-1, 0) + range_0_p1) * float3E(0.720415, -0.006134, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(-1, 0) + range_0_p1) *float3E(0.720419, -0.006134, 0.000000));
	//glass.trans = ((Dual_EyeForward_Backward_(-1, 0) + range_0_p1) * float3E(0.284870, -0.002441, 0.000000));
	Eye_scale_LT.trans = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.000000, 1.999390, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.000000, 1.999370, 0.000000));
	Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.000000, -2.327220, 0.019692));
	//glass.trans = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.014067, 1.643800, 0.000000));
}

void TESTFUNCTION::getLinearBoundingBoxEye(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res)
{
	TESTSTRUCTURE::TaobaoSkeleton Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeBig_Small_, Dual_EyeLowerFront_Outer_, Dual_EyeInUp_Down_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeInWide_Narraw_, Dual_EyeMidUpperUp_Down_, Dual_EyeOutUp_Down_, Dual_EyeOutWide_Narraw_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeAllWide_Narraw_, Dual_EyeForward_Backward_, Dual_EyeUp_Down_;

	double eye_small, eyeInDown, eyeInUp, eyeInWide, eyeOutDown, eyeOutNarraw, eyeOutUp, eyeOutWide;
	double eye_narraw, eye_big, eye_wide, eyeInNarraw, eye_backward, eye_down, eye_forward, eye_up;

	Eye_scale_LT.scale = ((Dual_EyeBig_Small_(-1, 0) + eye_small) * float3E(-0.435860, -0.435859, -0.435860));
	Eye_scale_RT.scale = ((Dual_EyeBig_Small_(-1, 0) + eye_small) * float3E(-0.435860, -0.435860, -0.435859));
	//EyeFrontLowerDown = ((Dual_EyeLowerFront_Outer_(0, 1) + EyeFrontLowerDown) * 1.000000);
	//EyeFrontUpperUp = ((Dual_EyeUpperFront_Outer_(-1, 0) + EyeFrontUpperUp) * 1.000000);
	EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + eyeInDown) * float3E(-12.437200, 3.597150, 0.256516));
	EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + eyeInDown) * float3E(-12.437200, -3.541290, 0.256638));
	EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(0, 1) + eyeInUp) * float3E(15.736300, 27.218700, -0.713188));
	EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(0, 1) + eyeInUp) * float3E(15.751900, -27.128099, 0.056992));
	EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + eyeInWide) * float3E(-0.001537, -0.007004, 6.259930));
	EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + eyeInWide) * float3E(-0.255201, -0.006882, -6.254790));
	//EyeMiddleLowerDown = ((Dual_EyeMidLowerUp_Down_(-1, 0) + EyeMiddleLowerDown) * 1.000000);
	//EyeMiddleLowerUp = ((Dual_EyeMidLowerUp_Down_(0, 1) + EyeMiddleLowerUp) * 1.000000);
	//EyeMiddleUpperDown = ((Dual_EyeMidUpperUp_Down_(-1, 0) + EyeMiddleUpperDown) * 1.000000);
	//EyeMiddleUpperUp = ((Dual_EyeMidUpperUp_Down_(0, 1) + EyeMiddleUpperUp) * 1.000000);
	//EyeNarrow = ((Dual_EyeWide_Narraw_(-1, 0) + EyeNarrow) * 1.000000);
	EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + eyeOutDown) * float3E(0.061870, -21.257299, 1.522710));
	EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + eyeOutDown) * float3E(0.000000, 21.256399, -1.524060));
	//EyeOuterLowerDown = ((Dual_EyeLowerFront_Outer_(-1, 0) + EyeOuterLowerDown) * 1.000000);
	//EyeOuterUpperUp = ((Dual_EyeUpperFront_Outer_(0, 1) + EyeOuterUpperUp) * 1.000000);
	EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + eyeOutNarraw) * float3E(-0.641535, -0.090179, -15.995000));
	EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + eyeOutNarraw) * float3E(0.000000, 0.000000, 16.007401));
	EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + eyeOutUp) * float3E(0.000000, 20.374800, -0.000114));
	EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + eyeOutUp) * float3E(0.000000, -20.374800, 0.000000));
	EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + eyeOutWide) * float3E(0.585359, -0.074822, 14.562100));
	EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + eyeOutWide) * float3E(0.000000, 0.000000, -14.573700));
	//EyeWide = ((Dual_EyeWide_Narraw_(0, 1) + EyeWide) * 1.000000);
	Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, 0.000000, -2.180600));
	Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, 0.000000, 2.180600));
	Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, -2.327220, 0.019692));
	Eye_scale_LT.scale = ((Dual_EyeBig_Small_(0, 1) + eye_big) * float3E(0.374041, 0.374043, 0.374039));
	Eye_scale_RT.scale = ((Dual_EyeBig_Small_(0, 1) + eye_big) * float3E(0.374041, 0.374041, 0.374039));
	EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + eyeInNarraw) * float3E(0.330012, 0.045853, -15.832900));
	EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + eyeInNarraw) * float3E(0.330011, -0.045869, 15.832900));
	Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, 0.000000, 0.558024));
	Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, -0.000214, -0.558090));
	Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, -2.327220, 0.019692));
	//EyeSmall = ((Dual_PupilBig_Small_(-1, 0) + EyeSmall) * 1.000000);
	//EyeBig = ((Dual_PupilBig_Small_(0, 1) + EyeBig) * 1.000000);
	Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.659834, 0.005692, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.659830, 0.005676, 0.000000));
	//glass.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.213036, 0.001816, 0.000000));
	Eye_scale_LT.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -1.381740, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -1.381760, 0.000000));
	Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -2.327220, 0.019692));
	//glass.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(-0.010664, -1.246280, 0.000000));
	Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) * float3E(0.720415, -0.006134, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) *float3E(0.720419, -0.006134, 0.000000));
	//glass.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) * float3E(0.284870, -0.002441, 0.000000));
	Eye_scale_LT.trans = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, 1.999390, 0.000000));
	Eye_scale_RT.trans = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, 1.999370, 0.000000));
	Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, -2.327220, 0.019692));
	//glass.trans = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.014067, 1.643800, 0.000000));
}

void TESTFUNCTION::getLinearBoundingBoxEyeLimit(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res)
{
	TESTSTRUCTURE::TaobaoSkeleton Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeBig_Small_, Dual_EyeLowerFront_Outer_, Dual_EyeInUp_Down_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeInWide_Narraw_, Dual_EyeMidUpperUp_Down_, Dual_EyeOutUp_Down_, Dual_EyeOutWide_Narraw_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeAllWide_Narraw_, Dual_EyeForward_Backward_, Dual_EyeUp_Down_;

	double eye_small, eyeInDown, eyeInUp, eyeInWide, eyeOutDown, eyeOutNarraw, eyeOutUp, eyeOutWide;
	double eye_narraw, eye_big, eye_wide, eyeInNarraw, eye_backward, eye_down, eye_forward, eye_up;
	
	//setting
	floatVec dual_limit = { -1, 0, 1};
	floatVec bs_limit = { 0, 1 };

	int count_iter = -1;

	//save single
	cstrVec name_tags_all = { "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" };
	cstrVec name_tags = {};
	getStructureColumnNames({ "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" }, name_tags);
	doubleVec to_csv_data;
	//std::vector<TESTSTRUCTURE::TaobaoSkeleton> res_vec = {};
	//putStructureToVec(res, to_csv_data);
	res = {};

	for (int iter_dual = 0; iter_dual < dual_limit.size(); iter_dual++)
	{
		for (int iter_bs = 0; iter_bs<bs_limit.size();  iter_bs++)
		{
			count_iter++;

			Dual_EyeBig_Small_.value = dual_limit[iter_dual];
			Dual_EyeLowerFront_Outer_.value = dual_limit[iter_dual];
			Dual_EyeInUp_Down_.value = dual_limit[iter_dual];
			Dual_EyeInWide_Narraw_.value = dual_limit[iter_dual];
			Dual_EyeMidUpperUp_Down_.value = dual_limit[iter_dual];
			Dual_EyeOutUp_Down_.value = dual_limit[iter_dual];
			Dual_EyeOutWide_Narraw_.value = dual_limit[iter_dual];
			Dual_EyeAllWide_Narraw_.value = dual_limit[iter_dual];
			Dual_EyeForward_Backward_.value = dual_limit[iter_dual];
			Dual_EyeUp_Down_.value = dual_limit[iter_dual];

			eye_small = bs_limit[iter_bs];
			eyeInDown = bs_limit[iter_bs];
			eyeInUp = bs_limit[iter_bs];
			eyeInWide = bs_limit[iter_bs];
			eyeOutDown = bs_limit[iter_bs];
			eyeOutNarraw = bs_limit[iter_bs];
			eyeOutUp = bs_limit[iter_bs];
			eyeOutWide = bs_limit[iter_bs];

			eye_narraw = bs_limit[iter_bs];
			eye_big = bs_limit[iter_bs];
			eye_wide = bs_limit[iter_bs];
			eyeInNarraw = bs_limit[iter_bs];
			eye_backward = bs_limit[iter_bs];
			eye_down = bs_limit[iter_bs];
			eye_forward = bs_limit[iter_bs];
			eye_up = bs_limit[iter_bs];

			//get value
			Eye_scale_LT.scale = ((Dual_EyeBig_Small_(-1, 0) + eye_small) * float3E(-0.435860, -0.435859, -0.435860));
			Eye_scale_RT.scale = ((Dual_EyeBig_Small_(-1, 0) + eye_small) * float3E(-0.435860, -0.435860, -0.435859));
			//EyeFrontLowerDown = ((Dual_EyeLowerFront_Outer_(0, 1) + EyeFrontLowerDown) * 1.000000);
			//EyeFrontUpperUp = ((Dual_EyeUpperFront_Outer_(-1, 0) + EyeFrontUpperUp) * 1.000000);
			EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + eyeInDown) * float3E(-12.437200, 3.597150, 0.256516));
			EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + eyeInDown) * float3E(-12.437200, -3.541290, 0.256638));
			EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(0, 1) + eyeInUp) * float3E(15.736300, 27.218700, -0.713188));
			EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(0, 1) + eyeInUp) * float3E(15.751900, -27.128099, 0.056992));
			EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + eyeInWide) * float3E(-0.001537, -0.007004, 6.259930));
			EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + eyeInWide) * float3E(-0.255201, -0.006882, -6.254790));
			//EyeMiddleLowerDown = ((Dual_EyeMidLowerUp_Down_(-1, 0) + EyeMiddleLowerDown) * 1.000000);
			//EyeMiddleLowerUp = ((Dual_EyeMidLowerUp_Down_(0, 1) + EyeMiddleLowerUp) * 1.000000);
			//EyeMiddleUpperDown = ((Dual_EyeMidUpperUp_Down_(-1, 0) + EyeMiddleUpperDown) * 1.000000);
			//EyeMiddleUpperUp = ((Dual_EyeMidUpperUp_Down_(0, 1) + EyeMiddleUpperUp) * 1.000000);
			//EyeNarrow = ((Dual_EyeWide_Narraw_(-1, 0) + EyeNarrow) * 1.000000);
			EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + eyeOutDown) * float3E(0.061870, -21.257299, 1.522710));
			EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + eyeOutDown) * float3E(0.000000, 21.256399, -1.524060));
			//EyeOuterLowerDown = ((Dual_EyeLowerFront_Outer_(-1, 0) + EyeOuterLowerDown) * 1.000000);
			//EyeOuterUpperUp = ((Dual_EyeUpperFront_Outer_(0, 1) + EyeOuterUpperUp) * 1.000000);
			EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + eyeOutNarraw) * float3E(-0.641535, -0.090179, -15.995000));
			EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + eyeOutNarraw) * float3E(0.000000, 0.000000, 16.007401));
			EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + eyeOutUp) * float3E(0.000000, 20.374800, -0.000114));
			EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + eyeOutUp) * float3E(0.000000, -20.374800, 0.000000));
			EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + eyeOutWide) * float3E(0.585359, -0.074822, 14.562100));
			EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + eyeOutWide) * float3E(0.000000, 0.000000, -14.573700));
			//EyeWide = ((Dual_EyeWide_Narraw_(0, 1) + EyeWide) * 1.000000);
			Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, 0.000000, -2.180600));
			Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, 0.000000, 2.180600));
			Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, -2.327220, 0.019692));
			Eye_scale_LT.scale = ((Dual_EyeBig_Small_(0, 1) + eye_big) * float3E(0.374041, 0.374043, 0.374039));
			Eye_scale_RT.scale = ((Dual_EyeBig_Small_(0, 1) + eye_big) * float3E(0.374041, 0.374041, 0.374039));
			EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + eyeInNarraw) * float3E(0.330012, 0.045853, -15.832900));
			EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + eyeInNarraw) * float3E(0.330011, -0.045869, 15.832900));
			Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, 0.000000, 0.558024));
			Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, -0.000214, -0.558090));
			Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, -2.327220, 0.019692));
			//EyeSmall = ((Dual_PupilBig_Small_(-1, 0) + EyeSmall) * 1.000000);
			//EyeBig = ((Dual_PupilBig_Small_(0, 1) + EyeBig) * 1.000000);
			Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.659834, 0.005692, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.659830, 0.005676, 0.000000));
			//glass.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.213036, 0.001816, 0.000000));
			Eye_scale_LT.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -1.381740, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -1.381760, 0.000000));
			Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -2.327220, 0.019692));
			//glass.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(-0.010664, -1.246280, 0.000000));
			Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) * float3E(0.720415, -0.006134, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) *float3E(0.720419, -0.006134, 0.000000));
			//glass.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) * float3E(0.284870, -0.002441, 0.000000));
			Eye_scale_LT.trans = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, 1.999390, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, 1.999370, 0.000000));
			Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, -2.327220, 0.019692));
			//glass.trans = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.014067, 1.643800, 0.000000));
			
			//print for sure
			Eye_scale_LT.print("Eye_scale_LT");
			EyeLine_in_LT.print("EyeLine_in_LT");
			EyeLine_out_LT.print("EyeLine_out_LT");
			Eye_scale_RT.print("Eye_scale_RT");
			EyeLine_in_RT.print("EyeLine_in_RT");
			EyeLine_out_RT.print("EyeLine_out_RT");
			res = CalcHelper::appendVector(res, std::vector<TESTSTRUCTURE::TaobaoSkeleton>{ Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT });
#if 0
			//save single
			cstrVec name_tags = { "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" };
			getStructureColumnNames({ "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" }, name_tags);
			doubleVec to_csv_data;
			std::vector<TESTSTRUCTURE::TaobaoSkeleton> res = {Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT};
			putStructureToVec(res, to_csv_data);
			FILEIO::saveVecToCsv("D:/dota210604/0615_00/" +std::to_string(count_iter) +  "_test_vec.csv", to_csv_data, 3 * 3 * 6, name_tags);
#endif
		}
	}
	putStructureToVec(res, to_csv_data);
	FILEIO::saveVecToCsv("D:/dota210604/0617_01/" + std::to_string(count_iter) + "_test_vec.csv", to_csv_data, 3 * 3 * 6, name_tags);
}

void TESTFUNCTION::getLinearBoundingBoxEyeLimitEyeScaleLT(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res)
{
	TESTSTRUCTURE::TaobaoSkeleton Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeBig_Small_, Dual_EyeLowerFront_Outer_, Dual_EyeInUp_Down_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeInWide_Narraw_, Dual_EyeMidUpperUp_Down_, Dual_EyeOutUp_Down_, Dual_EyeOutWide_Narraw_;
	TESTSTRUCTURE::TaobaoSlide Dual_EyeAllWide_Narraw_, Dual_EyeForward_Backward_, Dual_EyeUp_Down_;

	double eye_small, eyeInDown, eyeInUp, eyeInWide, eyeOutDown, eyeOutNarraw, eyeOutUp, eyeOutWide;
	double eye_narraw, eye_big, eye_wide, eyeInNarraw, eye_backward, eye_down, eye_forward, eye_up;
	
	std::map<cstr, TESTSTRUCTURE::TaobaoSlide*> to_slide;
	Dual_EyeBig_Small_.print("Dual_EyeBig_Small_");
	to_slide.insert(std::make_pair("Dual_EyeBig_Small_", &Dual_EyeBig_Small_));
	to_slide["Dual_EyeBig_Small_"]->value = 0.888;
	Dual_EyeBig_Small_.print("Dual_EyeBig_Small_");
	to_slide["Dual_EyeBig_Small_"]->print("Dual_EyeBig_Small_");

	//setting
	floatVec dual_limit = { -1, 0, 1 };
	floatVec bs_limit = { 0, 1 };

	int count_iter = -1;

	//save single
	cstrVec name_tags_all = { "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" };
	cstrVec name_tags = {};
	getStructureColumnNames({ "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" }, name_tags);
	doubleVec to_csv_data;
	//std::vector<TESTSTRUCTURE::TaobaoSkeleton> res_vec = {};
	//putStructureToVec(res, to_csv_data);
	res = {};
	
	floatVec dual_limit_try = { -0.7921, 0.3187, 0.5667 };
	floatVec bs_limit_try = { 0.2333, 0.9233 };


	std::vector<TESTSTRUCTURE::TaobaoSlide> slide_data = {};


	for (int iter_dual = 0; iter_dual < dual_limit.size(); iter_dual++)
	{
		for (int iter_bs = 0; iter_bs < bs_limit.size(); iter_bs++)
		{
			count_iter++;

			Dual_EyeBig_Small_.value = dual_limit[iter_dual];
			Dual_EyeLowerFront_Outer_.value = dual_limit[iter_dual];
			Dual_EyeInUp_Down_.value = dual_limit[iter_dual];
			Dual_EyeInWide_Narraw_.value = dual_limit[iter_dual];
			Dual_EyeMidUpperUp_Down_.value = dual_limit[iter_dual];
			Dual_EyeOutUp_Down_.value = dual_limit[iter_dual];
			Dual_EyeOutWide_Narraw_.value = dual_limit[iter_dual];
			Dual_EyeAllWide_Narraw_.value = dual_limit[iter_dual];
			Dual_EyeForward_Backward_.value = dual_limit[iter_dual];
			Dual_EyeUp_Down_.value = dual_limit[iter_dual];

			eye_small = bs_limit[iter_bs];
			eyeInDown = bs_limit[iter_bs];
			eyeInUp = bs_limit[iter_bs];
			eyeInWide = bs_limit[iter_bs];
			eyeOutDown = bs_limit[iter_bs];
			eyeOutNarraw = bs_limit[iter_bs];
			eyeOutUp = bs_limit[iter_bs];
			eyeOutWide = bs_limit[iter_bs];

			eye_narraw = bs_limit[iter_bs];
			eye_big = bs_limit[iter_bs];
			eye_wide = bs_limit[iter_bs];
			eyeInNarraw = bs_limit[iter_bs];
			eye_backward = bs_limit[iter_bs];
			eye_down = bs_limit[iter_bs];
			eye_forward = bs_limit[iter_bs];
			eye_up = bs_limit[iter_bs];

			//get value
			Eye_scale_LT.scale = ((Dual_EyeBig_Small_(-1, 0) + eye_small) * float3E(-0.435860, -0.435859, -0.435860));
			Eye_scale_RT.scale = ((Dual_EyeBig_Small_(-1, 0) + eye_small) * float3E(-0.435860, -0.435860, -0.435859));
			//EyeFrontLowerDown = ((Dual_EyeLowerFront_Outer_(0, 1) + EyeFrontLowerDown) * 1.000000);
			//EyeFrontUpperUp = ((Dual_EyeUpperFront_Outer_(-1, 0) + EyeFrontUpperUp) * 1.000000);
			EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + eyeInDown) * float3E(-12.437200, 3.597150, 0.256516));
			EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(-1, 0) + eyeInDown) * float3E(-12.437200, -3.541290, 0.256638));
			EyeLine_in_LT.rotate = ((Dual_EyeInUp_Down_(0, 1) + eyeInUp) * float3E(15.736300, 27.218700, -0.713188));
			EyeLine_in_RT.rotate = ((Dual_EyeInUp_Down_(0, 1) + eyeInUp) * float3E(15.751900, -27.128099, 0.056992));
			EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + eyeInWide) * float3E(-0.001537, -0.007004, 6.259930));
			EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(-1, 0) + eyeInWide) * float3E(-0.255201, -0.006882, -6.254790));
			//EyeMiddleLowerDown = ((Dual_EyeMidLowerUp_Down_(-1, 0) + EyeMiddleLowerDown) * 1.000000);
			//EyeMiddleLowerUp = ((Dual_EyeMidLowerUp_Down_(0, 1) + EyeMiddleLowerUp) * 1.000000);
			//EyeMiddleUpperDown = ((Dual_EyeMidUpperUp_Down_(-1, 0) + EyeMiddleUpperDown) * 1.000000);
			//EyeMiddleUpperUp = ((Dual_EyeMidUpperUp_Down_(0, 1) + EyeMiddleUpperUp) * 1.000000);
			//EyeNarrow = ((Dual_EyeWide_Narraw_(-1, 0) + EyeNarrow) * 1.000000);
			EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + eyeOutDown) * float3E(0.061870, -21.257299, 1.522710));
			EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(-1, 0) + eyeOutDown) * float3E(0.000000, 21.256399, -1.524060));
			//EyeOuterLowerDown = ((Dual_EyeLowerFront_Outer_(-1, 0) + EyeOuterLowerDown) * 1.000000);
			//EyeOuterUpperUp = ((Dual_EyeUpperFront_Outer_(0, 1) + EyeOuterUpperUp) * 1.000000);
			EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + eyeOutNarraw) * float3E(-0.641535, -0.090179, -15.995000));
			EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(-1, 0) + eyeOutNarraw) * float3E(0.000000, 0.000000, 16.007401));
			EyeLine_out_RT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + eyeOutUp) * float3E(0.000000, 20.374800, -0.000114));
			EyeLine_out_LT.rotate = ((Dual_EyeOutUp_Down_(0, 1) + eyeOutUp) * float3E(0.000000, -20.374800, 0.000000));
			EyeLine_out_RT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + eyeOutWide) * float3E(0.585359, -0.074822, 14.562100));
			EyeLine_out_LT.rotate = ((Dual_EyeOutWide_Narraw_(0, 1) + eyeOutWide) * float3E(0.000000, 0.000000, -14.573700));
			//EyeWide = ((Dual_EyeWide_Narraw_(0, 1) + EyeWide) * 1.000000);
			Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, 0.000000, -2.180600));
			Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, 0.000000, 2.180600));
			Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(-1, 0) + eye_narraw) * float3E(0.000000, -2.327220, 0.019692));
			Eye_scale_LT.scale = ((Dual_EyeBig_Small_(0, 1) + eye_big) * float3E(0.374041, 0.374043, 0.374039));
			Eye_scale_RT.scale = ((Dual_EyeBig_Small_(0, 1) + eye_big) * float3E(0.374041, 0.374041, 0.374039));
			EyeLine_in_LT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + eyeInNarraw) * float3E(0.330012, 0.045853, -15.832900));
			EyeLine_in_RT.rotate = ((Dual_EyeInWide_Narraw_(0, 1) + eyeInNarraw) * float3E(0.330011, -0.045869, 15.832900));
			Eye_scale_LT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, 0.000000, 0.558024));
			Eye_scale_RT.trans = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, -0.000214, -0.558090));
			Eye_scale_RT.rotate = ((Dual_EyeAllWide_Narraw_(0, 1) + eye_wide) * float3E(0.000000, -2.327220, 0.019692));
			//EyeSmall = ((Dual_PupilBig_Small_(-1, 0) + EyeSmall) * 1.000000);
			//EyeBig = ((Dual_PupilBig_Small_(0, 1) + EyeBig) * 1.000000);
			Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.659834, 0.005692, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.659830, 0.005676, 0.000000));
			//glass.trans = ((Dual_EyeForward_Backward_(0, 1) + eye_backward) * float3E(-0.213036, 0.001816, 0.000000));
			Eye_scale_LT.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -1.381740, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -1.381760, 0.000000));
			Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(0.000000, -2.327220, 0.019692));
			//glass.trans = ((Dual_EyeUp_Down_(-1, 0) + eye_down) * float3E(-0.010664, -1.246280, 0.000000));
			Eye_scale_LT.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) * float3E(0.720415, -0.006134, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) *float3E(0.720419, -0.006134, 0.000000));
			//glass.trans = ((Dual_EyeForward_Backward_(-1, 0) + eye_forward) * float3E(0.284870, -0.002441, 0.000000));
			Eye_scale_LT.trans = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, 1.999390, 0.000000));
			Eye_scale_RT.trans = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, 1.999370, 0.000000));
			Eye_scale_RT.rotate = ((Dual_EyeUp_Down_(0, 1) + eye_up) * float3E(0.000000, -2.327220, 0.019692));
			//glass.trans = ((Dual_EyeUp_Down_(0, 1) + range_0_p1) * float3E(0.014067, 1.643800, 0.000000));

			//print for sure
			Eye_scale_LT.print("Eye_scale_LT");
			EyeLine_in_LT.print("EyeLine_in_LT");
			EyeLine_out_LT.print("EyeLine_out_LT");
			Eye_scale_RT.print("Eye_scale_RT");
			EyeLine_in_RT.print("EyeLine_in_RT");
			EyeLine_out_RT.print("EyeLine_out_RT");
			res = CalcHelper::appendVector(res, std::vector<TESTSTRUCTURE::TaobaoSkeleton>{ Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT });
#if 0
			//save single
			cstrVec name_tags = { "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" };
			getStructureColumnNames({ "Eye_scale_LT", "EyeLine_in_LT", "EyeLine_out_LT", "Eye_scale_RT", "EyeLine_in_RT", "EyeLine_out_RT" }, name_tags);
			doubleVec to_csv_data;
			std::vector<TESTSTRUCTURE::TaobaoSkeleton> res = { Eye_scale_LT, EyeLine_in_LT, EyeLine_out_LT, Eye_scale_RT, EyeLine_in_RT, EyeLine_out_RT };
			putStructureToVec(res, to_csv_data);
			FILEIO::saveVecToCsv("D:/dota210604/0615_00/" + std::to_string(count_iter) + "_test_vec.csv", to_csv_data, 3 * 3 * 6, name_tags);
#endif
		}
	}
	putStructureToVec(res, to_csv_data);
	FILEIO::saveVecToCsv("D:/dota210604/0617_01/" + std::to_string(count_iter) + "_test_vec.csv", to_csv_data, 3 * 3 * 6, name_tags);
}

void TESTFUNCTION::getRandomWalkData(const std::vector<std::vector<TESTSTRUCTURE::TaobaoSkeleton>>& input)
{
	TESTSTRUCTURE::TaobaoSkeleton skeleton_min, skeleton_max;
	skeleton_min.scale = float3E(1.0, 1.0, 1.0);
	skeleton_min.trans = float3E(0, 0, 0);
	skeleton_min.rotate = float3E(0, 0, 0);
	//copy to skeleton_max
	

	//return { skeleton_min, skeleton_max };
}

void TESTFUNCTION::getStructureColumnNames(const cstrVec& input_names, cstrVec& tag)
{
	for (auto i: input_names)
	{
		tag.push_back(i + "_rotate_x");
		tag.push_back(i + "_rotate_y");
		tag.push_back(i + "_rotate_z");
		tag.push_back(i + "_scale_x");
		tag.push_back(i + "_scale_y");
		tag.push_back(i + "_scale_z");
		tag.push_back(i + "_trans_x");
		tag.push_back(i + "_trans_y");
		tag.push_back(i + "_trans_z");
	}
}

void TESTFUNCTION::getTaobaoSkeletonConstrains()
{
	cstr res_root = "D:/dota210507/0518_csv/";
	SG::needPath(res_root);

	TESTSTRUCTURE::TaobaoSkeleton Brow_top_LT, Brow_mid_LT, Brow_end_LT, Brow_top_RT, Brow_mid_RT, Brow_end_RT;
	float range_m1_0, range_0_p1;
	range_m1_0 = 0;
	range_0_p1 = 0;

	doubleVec to_csv_data;
	std::vector<TESTSTRUCTURE::TaobaoSkeleton> res;
	std::vector < TESTSTRUCTURE::TaobaoSkeleton> temp_structure;
	getLinearBoundingBoxBrow(-1, 0, res);
	putStructureToVec(res, to_csv_data);
	
	cstrVec name_tags;
	getStructureColumnNames({ "Brow_top_LT", "Brow_mid_LT", "Brow_end_LT", "Brow_top_RT", "Brow_mid_RT", "Brow_end_R" }, name_tags);
	FILEIO::saveVecToCsv(res_root + "test_vec.csv", to_csv_data, 3 * 3 * 6, name_tags);
}

void TESTFUNCTION::putStructureToVec(const std::vector<TESTSTRUCTURE::TaobaoSkeleton>& src_structure, doubleVec& res)
{
	for (int i = 0; i < src_structure.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			res.push_back(src_structure[i].rotate[j]);
		}

		for (int j = 0; j < 3; j++)
		{
			res.push_back(src_structure[i].scale[j]);
		}

		for (int j = 0; j < 3; j++)
		{
			res.push_back(src_structure[i].trans[j]);
		}
	}
}

void TESTFUNCTION::testBase64()
{
	//https://blog.csdn.net/guo_lei_lamant/article/details/80592120
	std::fstream f;
	f.open("D:/dota210507/0519_png/00_cartoon_pair.png", std::ios::in | std::ios::binary);
	f.seekg(0, std::ios_base::end);
	std::streampos sp = f.tellg();
	int size = sp;
	LOG(INFO) << "size: " << size << std::endl;
	char* buffer = (char*)malloc(sizeof(char)*size);
	f.seekg(0, std::ios_base::beg);//把文件指针移到到文件头位置
	f.read(buffer, size);
	LOG(INFO) << "file size:" << size << std::endl;
	cstr imgBase64 = TDST::base64Encode(buffer, size);

	json codeBase64;
	codeBase64["img"] = imgBase64;
	FILEIO::saveJson("D:/dota210507/0519_png/00_cartoon_pair.json", codeBase64);

	LOG(INFO) << "img base64 encode size:" << imgBase64.size() << std::endl;
	cstr imgdecode64 = TDST::base64Decode(imgBase64);
	LOG(INFO) << "img decode size:" << imgdecode64.size() << std::endl;

	//cstr imgdecode64 = TDST::base64_decode_(imgBase64);
	std::vector<uchar> img_data(imgdecode64.begin(), imgdecode64.end());
	cv::Mat img = cv::imdecode(cv::Mat(img_data), cv::IMREAD_UNCHANGED);
	cv::imwrite("D:/dota210507/0519_png/base_64.png", img);


}

void TESTFUNCTION::generateNRTensor()
{
#if 0
	cstr src_root = "D:/dota210507/bases_id_scaled_ori/";
	cstr res_root = "D:/dota210507/bases_id_scaled/";
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(src_root, ".obj");
	MeshCompress src = "D:/dota210507/bases_id_scaled_ori/000000.obj";
	MeshCompress ref = "D:/dota210507/local_deform(2).obj";
	intVec guijie_neck = FILEIO::loadIntDynamic("D:/dota210507/neck_fix.txt");
	MeshCompress move_res = src;
	double scale;
	float3E ref_to_src;
	MeshTools::putSrcToDst(src, guijie_neck, ref, guijie_neck, move_res, scale, ref_to_src);

	move_res.saveObj("D:/dota210507/trans_0000.obj");
	for (auto i : folder_file)
	{
		MeshCompress iter_mesh_ori(src_root + i);
		MeshCompress iter_mesh_dst = iter_mesh_ori;
		RT::scaleInCenterAndTranslateInPlace(scale, ref_to_src, iter_mesh_dst.pos_);
		iter_mesh_dst.saveObj(res_root + i);
	}
#endif

	TinyTool::getFileNamesToPCAJson("D:/dota210604/0610_dw_base/bases_id_20210610/");
	PREPARE::prepareBSTensor("D:/dota210604/0610_dw_base/bases_id_20210610/", "D:/dota210604/0610_dw_base/nr_resnet/");
}

void TESTFUNCTION::generateTestingJson()
{
	json test_all = FILEIO::loadJson("D:/dota210507/0519_png/template_06.json");
	json test_json = test_all["face_type"];
	cstr img_base64;
	TDST::getBased64FromFiles("D:/dota210507/0519_png/landmark_256_xy.png", img_base64);
	cstr imgdecode64 = TDST::base64Decode(img_base64);
	std::vector<uchar> img_data(imgdecode64.begin(), imgdecode64.end());
	cv::Mat img = cv::imdecode(cv::Mat(img_data), cv::IMREAD_UNCHANGED);
	cv::imwrite("D:/dota210507/0519_png/landmark_256_xy_test.png", img);

	//floatVec coef_3dmm_vec = json_in["coef_3dmm"].get<floatVec>();
	//CalcHelper::vectorToEigen(coef_3dmm_vec, in_att.coef_3dmm_); 
	vecF coef_3dmm_;
	vecI landmark_256_xy_land;
	floatVec coef_3dmm_f;
	intVec landmark_256_xy_land_value;
	FILEIO::loadEigenMat("D:/dota210507/0519_png/landmark_256_xy.txt", landmark_256_xy_land);
	FILEIO::loadEigenMat("D:/dota210507/0519_png/coef_3dmm_.txt", coef_3dmm_);
	CalcHelper::vectorToEigen(coef_3dmm_, coef_3dmm_f);
	CalcHelper::vectorToEigen(landmark_256_xy_land, landmark_256_xy_land_value);
	test_json["landmark_256_xy_img"] = img_base64;
	test_json["landmark_256_xy_land"] = landmark_256_xy_land_value;
	test_json["coef_3dmm"] = coef_3dmm_f;
	test_json["load_from_json"] = true;
	test_all["face_type"] = test_json;
	test_all["type_3dmm"] = 0;
	FILEIO::saveJson("D:/dota210507/0519_png/06.json", test_all);

}

void TESTFUNCTION::testNRTensor()
{
	Tensor nr_tensor;
	JsonHelper::initData("D:/dota210507/0528_dw_base/bases_id_tensor/", "config.json", nr_tensor);
	floatVec res = FILEIO::loadFloatDynamic("D:/dota210507/0601_02/000000_coeff.txt", '\n');
	MeshCompress template_mesh = "D:/dota210507/0528_dw_base/local_deform.obj";
	nr_tensor.interpretIDFloat(template_mesh.pos_.data(), res);
	template_mesh.saveObj("D:/dota210507/0601_02/inter_test.obj");
}

void TESTFUNCTION::testingForPyCDataDiff()
{
	cv::Mat img_raw = cv::imread("D:/dota210507/0603_01/000020_dw.jpg");
	cv::Mat img;
	cv::resize(img_raw, img, cv::Size(224, 224));
	cv::imwrite("D:/dota210507/0603_01/224_base.jpg", img);


	cvMatF3 img_rgb_float;
	if (img.channels() == 4)
	{
		cv::Mat temp;
		cv::cvtColor(img, temp, cv::COLOR_RGBA2RGB);
		temp.convertTo(img_rgb_float, CV_32FC3);
	}
	else if (img.channels() == 3)
	{
		img.convertTo(img_rgb_float, CV_32FC3);
	}
	else if (img.channels() == 1)
	{
		cv::Mat temp;
		cv::cvtColor(img, temp, cv::COLOR_GRAY2RGB);
		img.convertTo(img_rgb_float, CV_32FC3);
	}
	
	floatVec input_dw = FILEIO::loadFloatDynamic("D:/dota210604/dbg_000020_cur.txt", '\n');
	int h = 224;
	int w = 224;
	float means[3] = { 0.485f,0.456f,0.406f };
	float norms[3] = { 0.229f, 0.224f, 0.225f };
	floatVec input_img(w*h * 3);
#pragma omp parallel for
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				(img_rgb_float.at<cv::Vec3f>(y, x))(c) = 255 * (input_dw[c*h*w + y * w + x] * norms[2-c] + means[2 - c]);
				//(img_rgb_float.at<cv::Vec3f>(y, x))(c) = 255*(input_dw[c*h*w + y * w + x]* norms[c] + means[c]);
				//input_img[3 * (y*w + x) + c] = input_dw[c*h*w + y * w + x];
			}
		}
	}

	cv::Mat res;
	img_rgb_float.convertTo(res, CV_8UC3);
	cv::imwrite("D:/dota210507/0603_01/dw_re.jpg", res);

}

