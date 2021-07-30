#include <vld.h>
#include "../Test/FixRoutine.h"
#include "../Test/BatchGeneration.h"

using namespace CGP;
using namespace AUTO3D;
int main(int argc, char* argv[])
{
	_CrtSetBreakAlloc(205);
	GLogHelper gh(argv[0]);	
	testing::InitGoogleTest(&argc, argv);
	
	//翻转maya模型
	TESTFUNCTION::ruhuaModel();
	TESTFUNCTION::normalizeTaobaoMayaRawData();

	//三角面四边形面转换
	//TESTFUNCTION::triToQuadFace("D:/dota210604/0611_male_isv/tri/", "D:/dota210604/0611_male_isv/quad/");


	//TESTFUNCTION::testingForTemplate();

	std::vector<TESTSTRUCTURE::TaobaoSkeleton> res;
	TESTFUNCTION::getLinearBoundingBoxEyeLimitEyeScaleLT(res);
	TESTFUNCTION::getLinearBoundingBoxEyeLimit(res);

	//TESTFUNCTION::generateNRTensorcvpr2021 human centric video matting challenge();
	//TESTFUNCTION::changeEyebrow();
	//TESTFUNCTION::simDumpEyelash();

	//TinyTool::getMatchingFromUnityToMayaGuijie();
	//TESTFUNCTION::generateNRTensor();
	//TESTFUNCTION::testingForPyCDataDiff();

	BatchGenerate::generateX();
	BatchGenerate::generateLinuxTest();

	TESTFUNCTION::generateTestingJson();
	TESTFUNCTION::testBase64();

	
	//TESTFUNCTION::testWinLinuxColor();
	//test for mapping
	//TOPTRANSFER::fromIsvToBase();
	//TESTFUNCTION::testWinLinuxColor();

	//FIXROUTINE::putIsvToDst("D:/dota210419/0422_aim/fwh.obj", "D:/dota210419/0422_aim/guijie.obj",
	//	"D:/multiPack/0422_test_infer/bfm_lap_sys.obj");

	//TOPTRANSFER::guijieToFWHInstanceTesting("D:/multiPack/0422_test_infer/");
		
	//TOPTRANSFER::rawMatching();
	   	 
	//FIXROUTINE::putIsvToDstWrapper();

	//修正睫毛问题
	//TESTFUNCTION::fixEyelash();

	//ReferDefTest::testClothDeform();

	//FIXROUTINE::swapUVInPlace("D:/dota210305/0312_isv_234/");



	//批量生成部分


	FIXROUTINE::putIsvToDstWrapper();
		
	//FIXROUTINE::swapUV();

	//测试云游戏部分
	//TESTFUNCTION::fitIsvGuijieVersionToFixScale();
	



	//TOPTRANSFER::tenetTest();

	TOPTRANSFER::getTemplateBS();

	//EASYGUIJIE::replaceUV("D:/dota201224/1224_demo/00_cartoon_pair/local_deform.obj", 
	//	"D:/dota210121/0128_bs_flw/change_flw.obj");

	//EASYGUIJIE::transformEyesToMesh("D:/dota201224/1224_demo/05_cartoon_pair/",
	//	"local_deform.obj");	

	
	TESTFUNCTION::testBvh();
	
	//TOPTRANSFER::correctEyelashPair();
	FIXROUTINE::getNRDemo();

	FIXROUTINE::getCartoonV2Style("D:/dota210121/0129_rep_isv/000047.jpg", "D:/dota210121/0129_rep_isv/000047_v7/");

	AUTO3DTEST::testCheekRandom();	
	AUTO3DTEST::testWholeFace();
	
	//AUTO3DTEST::testRightEyes();
	//AUTO3DTEST::testMouth();
	//AUTO3DTEST::generateTensor();
	//TESTFUNCTION::pcaFromTxt();
	//TESTFUNCTION::imageFusion();

	//TOPTRANSFER::getMatchFromDFRefine();
	//ReferDefTest::testClothDeform();
	//TOPTRANSFER::getMDS();

	//TESTFUNCTION::testShellGen();
	//TOPTRANSFER::changeBSScale();
	
	//TESTFUNCTION::triToQuadInPlace("D:/dota210104/0111_generate_head/debug/head_result/");
	//TinyTool::turnJsonToString();
	//TESTFUNCTION::triToQuadFace();
	//EASYGUIJIE::transformEyesToMesh();
	//EASYGUIJIE::getGuijieEyebrow();
	//TOPTRANSFER::transferSimDiff();
	//TOPTRANSFER::putEyelashBack();

	//FIXROUTINE::fixGuijieV4();


	//FIXROUTINE::getCartoonV2Style("D:/dota210121/0129_rep_isv/000009.jpg", "D:/dota210121/0129_rep_isv/000009/");



	{
		//guijieV2 拓扑捏脸
		FIXROUTINE::cartoonStyleDemo();
		TOPTRANSFER::postProcessForGuijieTex();
		MeshCompress q_base = "D:/dota210104/0118_template_color/head.obj";
	
		intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
		FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
		FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
		FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);

		intVec mapping  = q_base.discard(guijie_eyelash);
		FILEIO::saveDynamic("D:/dota210104/0118_template_color/mapping.txt", mapping, ",");
		q_base.saveObj("D:/dota210104/0118_template_color/head_dump_eyelash.obj");


		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/17_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/08_cartoon_pair/");


		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/00_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/01_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/03_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/05_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/07_cartoon_pair/");
	
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/11_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/15_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/26_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/28_cartoon_pair/");
		TOPTRANSFER::localDeform("D:/dota201224/1224_demo/30_cartoon_pair/");

		//TOPTRANSFER::generateTextureTensor();
		TOPTRANSFER::generateTexDst();
	
		//TOPTRANSFER::transferOnUV();
		//TOPTRANSFER::fixTexture(1);
		TOPTRANSFER::fixTexture(1e-1);
		TOPTRANSFER::fixTexture(1e-2);
		TOPTRANSFER::fixTexture(1e-3);
		TOPTRANSFER::fixTexture(1e-4);
		//TOPTRANSFER::fixTextureCeres();
		TOPTRANSFER::fixTexture(1e8);
		TOPTRANSFER::postProcessForGuijieTex();
	}

	//TOPTRANSFER::guijieToFWHInfer();


	//TOPTRANSFER::getFWHToGuijieV1NoLashWithRing();
	//TOPTRANSFER::onlyLandmarkDrag();
	//TOPTRANSFER::onlyLandmark();
	TOPTRANSFER::getMatchFromDFAdv();
	TOPTRANSFER::getMatchFromDF();
	TOPTRANSFER::getMatchFromFile();
	TOPTRANSFER::getFWHToGuijieV1NoLash();
	TOPTRANSFER::fromGuijieToFWH();
	//FIXROUTINE::selectGuijieVertex();
	//FIXROUTINE::getBFMtoNeuRender();
	//FIXROUTINE::getFaceDataReady();

	//FIXROUTINE::getDesignTop();	
	//FIXROUTINE::faceGenerationV3();
	//TinyTool::getMatchingFromPartToAll();
	

	//TESTFUNCTION::toMayaNamespace();
	//TinyTool::getMatchingFromUnityToMayaGuijie();
	//FIXROUTINE::getTextureBase();

	TinyTool::getBSMeshFromDeltaV1File();

	FIXROUTINE::guijieExp();

	//FIXROUTINE::prepareImage();
	//FIXROUTINE::eyebrowTypeTest();
	//FIXROUTINE::prepareHardMap();
	//FIXROUTINE::prepareTaobaoHardMap();
	//FIXROUTINE::faceGenerationV3();

	_CrtDumpMemoryLeaks();
}