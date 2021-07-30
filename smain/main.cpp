#include "../Test/Test.h"
#include "../Test/Prepare.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Config/DataStructureHeader.h"
#include "../MNN/KeypointDet.h"
#include "../RecMesh/RecMesh.h"
#include "../RecTexture/RecTexture.h"
#include "../FileIO/FileIO.h"
using namespace CGP;

int main(int argc, char* argv[])
{
	_CrtSetBreakAlloc(205);
	GLogHelper gh(argv[0]);	
	testing::InitGoogleTest(&argc, argv);

	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();	

#if 1	   	 	   
	json test_config = FILEIO::loadJson("D:/code/cgPlayground/config.json");
	ptr_res_var->setInput(test_config);
	std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
	std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
	ptr_rec_mesh->processImage();
	ptr_rec_mesh->processDeform();
	//ptr_rec_mesh->processBasicQRatio();
	ptr_rec_mesh->processPartRatio();
#else
	cstr img_root = "D:/data_20July/test_0721/";
	cstr result = "D:/data_20July/0721_taobao/";
	cstr gender = "male";
	bool is_same_folder = true;
	SG::needPath(result);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(img_root, FILEIO::FILE_TYPE::IMAGE);
	for (int i = 0; i < folder_file.size(); i++)
	{
		json test_config;
		test_config["input_image_"] = (img_root + folder_file[i]);
		if (is_same_folder)
		{
			test_config["output_dir_"] = (result + folder_file[i] + "_");
		}
		else
		{
			test_config["output_dir_"] = (result + folder_file[i] + "/");
			SG::needPath(test_config["output_dir_"]);
		}
		test_config["is_debug_"] = true;
		test_config["gender_"] = gender;
		ptr_res_var->setInput(test_config);
		std::shared_ptr<RecMesh> ptr_rec_mesh = std::make_shared<RecMesh>(ptr_const_var, ptr_res_var);
		std::shared_ptr<RecTexture> ptr_rec_texture = std::make_shared<RecTexture>();
		ptr_rec_mesh->processImage();
		ptr_rec_mesh->processDeform();
	}

#endif
	_CrtDumpMemoryLeaks();
}