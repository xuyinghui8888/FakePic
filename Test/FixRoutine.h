#ifndef FIX_ROUTINE_H
#define FIX_ROUTINE_H

#include "../Test/Test.h"
#include "../Test/TinyTool.h"
#include "../Test/TopTransfer.h"
#include "../Postprocess/EasyGuijie.h"
#include "../Test/Auto3dTest.h"
#include "../Test/ReferDefTest.h"

namespace CGP
{
	//used for testing codes
	namespace FIXROUTINE
	{
		void putIsvToDst(const cstr& src_path, const cstr& dst_path);
		void putIsvToDst(const cstr& src_path, const cstr& src_eyes_path, const cstr& dst_path);
		void putIsvToDstWrapper();
		void swapUV();
		void swapUVInPlace(const cstr& root);
		void fixFalingwen();
		void fixGuijieV4();
		void eyebrowTypeTest();
		void faceGenerationV3();
		void testCTO();
		void faceGenerationV2();
		void prepareHardMap();
		void prepareTaobaoHardMap();
		void guijieExp();
		void prepareImage();
		void generateAvatar();
		void generateAvatarBatch(const std::shared_ptr<ConstVar> ptr_const_var, std::shared_ptr<ResVar> ptr_res_var);

		void generateAvatarUsingBFM();
		void getCartoonTexture();
		void cartoonStyleDemo();
		void getDesignTop();
		void getNRDemo();
		void getFaceDataReady();
		void getBFMtoNeuRender();
		void selectGuijieVertex();
		void generateTextureBase();		
		void getTextureBase();
		void getCartoonV2Style(const cstr& root, const cstr& img);
		void getCartoonV2Style(const cstr& root, const cstr& img, const cstr& B);
	}
}
#endif
