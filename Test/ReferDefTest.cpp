#include "ReferDefTest.h"
#include "../FileIO/FileIO.h"
#include "../Mesh/ObjLoader.h"
#include "../VisHelper/VisHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../MeshDeform/LaplacianDeformation.h"
#include "../NRICP/register.h"
#include "../NRICP/demo.h"
#include "../RT/RT.h"
#include "../RigidAlign/icp.h"
#include "../Config/Tensor.h"
#include "../Config/TensorHelper.h"
#include "../Config/JsonHelper.h"

#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"


#include "../Sysmetric/Sysmetric.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Metric/Metric.h"
#include "../MNN/MnnModel.h"
#include "../MNN/FeatureExtract.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../Mesh/MeshTools.h"
#include "../Test/TinyTool.h"
#include "../ImageSim/ImageUtils.h"

#include "../Test/Prepare.h"
#include "../MNN/KeypointDet.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../FileIO/FileIO.h"
#include "../Test/TinyTool.h"
#include "../Eyebrow/EyebrowType.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Basic/ToDst.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../Debug/DebugTools.h"
#include "../RigidBS/RigidBsGenerate.h"
#include "../ExpGen/ExpGen.h"
#include "../ExpGen/BsGenerate.h"
#include "../Mapping/Mapping.h"
#include "../MeshDeform/LapDeformWrapper.h"
#include "../Postprocess/EasyGuijie.h"
#include "../MeshDeform/ReferenceDeform.h"
#include "../Shell/ShellGenerate.h"

using namespace CGP;

void ReferDefTest::testReferenceDeform()
{
	cstr root = "D:/dota210104/0105_hair_deform_refine/";
	MeshCompress A("D:/dota210104/0105_hair_deform/head_guijie.obj");
	MeshCompress A_deform("D:/dota210104/0111_mofeng/head.obj");
	MeshCompress A_hair("D:/dota210104/0105_hair_deform/Guijie_hair_05.obj");
	MeshCompress A_deform_hair = A_hair;
	ReferenceDeform hair_deformer;
	RefDefConfig config_deformer;
	hair_deformer.init(&A.pos_[0], A.n_vertex_, config_deformer);
	hair_deformer.process(&A_deform_hair.pos_[0], &A_deform.pos_[0], &A_hair.pos_[0], A_deform_hair.n_vertex_);
	A_deform_hair.saveObj("D:/dota210104/0111_mofeng/Guijie_hair_deform_05_10x10x10_refine_x2.obj");
}

void ReferDefTest::changeToSameTopMaleAndFemale()
{
	//cstr root = "D:/dota210104/0114_cloth/";
	cstr root = "D:/dota210317/0317_cloth/";
	MeshCompress male = root + "male.obj";
	MeshCompress female = root + "female.obj";

	intVec res = MeshTools::getMatchAnyUV(male, female, 0.0001);
	MeshCompress female_from_male = male;
	SIMDEFORM::moveHandle(male, female, res, true, female_from_male);
	female_from_male.saveObj(root + "same_female.obj");
}

void ReferDefTest::testClothDeform()
{
	//cstr root = "D:/dota210121/0126_cloth/";
	cstr root = "D:/dota210317/0317_cloth/";

#if 0
	intVec upper_idx = FILEIO::loadIntDynamic(root + "upper.txt");
	MeshCompress male = root + "male.obj";
	MeshCompress female = root + "same_female.obj";

	male.keepRoi(upper_idx);
	female.keepRoi(upper_idx);

	male.saveObj(root + "male_up.obj");
	female.saveObj(root + "female_up.obj");

	male.getBoundingBox();
	MeshCompress male_exp = male;
	male = root + "male_up.obj";
	male.update();
	male_exp.getBoundingBox();
	SHELLGEN::makeFurthestMesh(male, true, 0.25, male_exp);
	male_exp.saveObj(root + "male_dis_0.25.obj");
#else
	
	MeshCompress male = root + "male_dis_0.25.obj";

	MeshCompress female = root + "female_up.obj";

	MeshCompress A_cloth = root + "cloth.obj";
	MeshCompress A_cloth_deform = root + "cloth.obj";
	ReferenceDeform cloth_deformer;
	RefDefConfig config_deformer;
	config_deformer.density = int3E(20, 20, 20);
	config_deformer.smooth_weight = 0.1;
	cloth_deformer.init(&female.pos_[0], female.n_vertex_, config_deformer);
	cloth_deformer.process(&A_cloth_deform.pos_[0], &male.pos_[0], &A_cloth.pos_[0], A_cloth_deform.n_vertex_);
	A_cloth_deform.saveObj(root + "deform_cloth_0.25.obj");
#endif
}



