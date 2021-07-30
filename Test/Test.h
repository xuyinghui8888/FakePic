#ifndef TEST_FUNCTION_H
#define TEST_FUNCTION_H
#include "../Basic/CGPBaseHeader.h"
#include "../Config/JsonHelper.h"
#include "../Config/ConstVar.h"
#include "../Config/ResVar.h"
#include "../Config/ShapeVar.h"
#include "TestClass.h"
namespace CGP
{
	//used for testing codes
	namespace TESTFUNCTION
	{
		void mappingFromOriToDiscard();
		void backProject();
		void testProjectionUsingCache();
		void testProjection();
		void testShader();
		void testDeformTranSame();
		void testBatchDeformTranSame();
		void testBatchDeformTranId();
		void testReg();
		void testRigid();
		void testBaseGen();
		void testRectify();
		void testBatchGen();
		void toMayaNamespace();
		void triToQuadFace(const cstr& in_file, const cstr& out_file);
		void triToQuadInPlace(const cstr& root);
		void testFWH();
		void testExtractLandmarks();
		void testProject();
		void testCorrect26k();
		void testFWHRaw();
		void testPCA();
		void pcaRawToBinary();
		void testSysRectify();
		void testBatchBFMTOFWH();
		void testMoveCSToBasis();
		void testRectifyDeepReconstruction();
		void batchRTGuiJie();
		void transferUV();
		void testSubPCA();
		void testIDMatch();
		void testError();
		void testErrorDT();
		void testMNN();		
		void mesh2tensor();
		void cleanDeep3DPCA();
		void testLandmarkGuided();
		
		void testBVLS();
		void serverGenExp();
		void serverGenExpFromMesh();
		void serverOptV2();
		void serverOptV2FromServer();
		void serverOptV3FromServer();
		void eyebrowType();
		//v1 ssimilarity
		void testImageSimilarity();
		void testImageSimilarityLandmark();

		void testSegResult();
		void turnFaceSegIntoToPic();

		//v1.cpp
		void imageFusion();
		void calcHardRatio(const cstr& img_in, const cstr& img_ext, const cstr& obj_in, 
			const cstr& result_dir, int n_sample);
		void subtractMeshData();
		void subtractMeshDataTaobao();
		void getRatioMeshBasedOn3dmm();
		void prepareForGuijie35();
		void prepareForTaobao35();
		void getDataFromGuijieV3Pack();
		void changeOfRatio();
		void testRatioFace();
		void testCalculate();
		void movePartVertexForGuijie();
		void getDiscardMappingFromMeshes();
		//get new style ratio based on ffhq gan
		void calcTopRatio();
		void blendTop();
		//using ptr version
		void testWithServerPack();
		void testMTCNNVideoStream(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void testMTCNNPic(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void testArcFace(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var,
			const cstr& root, const cstr& img_name, const cstr& com_faceid);
		void test3dmm(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void testLandmark(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var);
		void pcaFromTxt();
		void fixEyelash();
		void testBvh();
		void testShellGen();
		void fitIsvGuijieVersionToFixScale();
		void testWinLinuxColor();
		void normalizeTaobaoMayaRawData();
		void ruhuaModel();
		void getTaobaoSkeletonConstrains();
		void getLinearBoundingBoxBrow(float range_m1_0, float range_0_p1, std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res);
		void getLinearBoundingBoxEyeRoi(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res);
		void getLinearBoundingBoxEye(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res);
		void getLinearBoundingBoxEyeLimit(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res);
		void getLinearBoundingBoxEyeLimitEyeScaleLT(std::vector<TESTSTRUCTURE::TaobaoSkeleton>& res);

		void putStructureToVec(const std::vector<TESTSTRUCTURE::TaobaoSkeleton>& src_structure, doubleVec& res);
		void getStructureColumnNames(const cstrVec& input_names, cstrVec& tag);
		void getRandomWalkData(const std::vector<std::vector<TESTSTRUCTURE::TaobaoSkeleton>>& input);
		
		void testBase64();
		void generateTestingJson();
		void generateNRTensor();
		void testNRTensor();

		void testingForPyCDataDiff();
		void simDumpEyelash();
		void changeEyebrow();

		void testingForTemplate();
	}
}

#endif 
