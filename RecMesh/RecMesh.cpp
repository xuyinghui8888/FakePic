#include "RecMesh.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Config/Tensor.h"
#include "../FileIO/FileIO.h"
#include "../MNN/FeatureExtract.h"
#include "../MeshDeform/LaplacianDeformation.h"
#include "../RT/RT.h"
#include "../Test/Prepare.h"
#include "../Basic/ToDst.h"

using namespace CGP;
void RecMesh::processImage(bool return_before_opt)
{
	//get mtcnn bounding box
	const cv::Mat input = res_var_->input_image_;
	processFaceDetection(input, res_var_->bb_box_);	
	if (res_var_->bb_box_.area() == 0 || res_var_->bb_box_.empty())
	{
		LOG(WARNING) << "no face detected." << std::endl;
	}
	cv::Mat land68_256;
	cv::Rect scale_bb;
	shiftBBLandmark68(input, res_var_->bb_box_, 256, land68_256, scale_bb);
	vecF landmark_256_xy;
	getLandmarks(land68_256, landmark_256_xy);
	//reflect back to image
	reflectBackToImage(scale_bb, 256, landmark_256_xy, landmark_image_68_xy_);
	//get mtcnn format 5 landmarks
	get5LandFrom68(landmark_image_68_xy_, landmark_image_5_xy_);
	//get faceid
	getFaceID(input, landmark_image_5_xy_, faceid_);
	//crop for 3dmm
	get3dmmCoef(input, landmark_image_5_xy_, coef_3dmm_);

	if (return_before_opt)
	{
		return;
	}	

	{
		//for unit test
		FILEIO::saveEigenDynamic(result_dir_ + "coef_3dmm_.txt", coef_3dmm_);
		cv::imwrite(result_dir_ + "landmark_256_xy.png", land68_256);
		FILEIO::saveEigenDynamic(result_dir_ + "landmark_256_xy.txt", landmark_256_xy);
	}

	//post process to drag 3dmm models to fwh models
	if (res_var_->pp_type_ == PostProcessRoutine::FIT_CONTOUR || res_var_->pp_type_ == PostProcessRoutine::FIT_FACE)
	{
		LOG(INFO) << "using PostProcessRoutine::FIT." << std::endl;
		getPostProcessFor3dmmFit68(coef_3dmm_, mesh_res_, land68_256, landmark_256_xy);
	}
	else if (res_var_->pp_type_ == PostProcessRoutine::TEX_BASE)
	{
		LOG(INFO) << "using PostProcessRoutine::TEX_BASE." << std::endl;
		getPostProcessFor3dmmFit68(coef_3dmm_, mesh_res_, land68_256, landmark_256_xy);
		getPostProcessFor3dmm(coef_3dmm_, mesh_res_);
		//getPostProcessFor3dmmTexBase(coef_3dmm_, mesh_res_, land68_256, landmark_256_xy);
		//getPostProcessFor3dmmFit68(coef_3dmm_, mesh_res_, land68_256, landmark_256_xy);
	}
	else
	{
		LOG(INFO) << "using PostProcessRoutine::Normal." << std::endl;
		getPostProcessFor3dmm(coef_3dmm_, mesh_res_);
	}
}

void RecMesh::processImage(const std::shared_ptr<NR3D> model_3dmm, bool return_before_opt)
{
	//get mtcnn bounding box
	const cv::Mat input = res_var_->input_image_;
	processFaceDetection(input, res_var_->bb_box_);
	if (res_var_->bb_box_.area() == 0 || res_var_->bb_box_.empty())
	{
		LOG(WARNING) << "no face detected." << std::endl;
	}
	cv::Mat land68_256;
	cv::Rect scale_bb;
	shiftBBLandmark68(input, res_var_->bb_box_, 256, land68_256, scale_bb);
	vecF landmark_256_xy;
	getLandmarks(land68_256, landmark_256_xy);
	//reflect back to image
	reflectBackToImage(scale_bb, 256, landmark_256_xy, landmark_image_68_xy_);
	//get mtcnn format 5 landmarks
	get5LandFrom68(landmark_image_68_xy_, landmark_image_5_xy_);
	//get faceid
	getFaceID(input, landmark_image_5_xy_, faceid_);
	//crop for 3dmm
	//测试
	get3dmmCoef(model_3dmm, input, landmark_image_5_xy_, coef_3dmm_);
	
	{
		//for unit test
		FILEIO::saveEigenDynamic(result_dir_ + "coef_3dmm_.txt", coef_3dmm_);
		cv::imwrite(result_dir_ + "landmark_256_xy.png", land68_256);
		FILEIO::saveEigenDynamic(result_dir_ + "landmark_256_xy.txt", landmark_256_xy);
		//pass data
		res_var_->coef_3dmm_ = coef_3dmm_;
	}

	
}

void RecMesh::processDeform()
{
	getMatchID(faceid_, match_id_);
	getDFResult(mesh_res_, match_id_);
}

void RecMesh::getMatchID(const vecF& faceid, int& match_res)
{
	const FaceIDFinder& id_finder = res_var_->gender_ == Gender::MALE ?
		const_var_->ptr_data->male_finder_ : const_var_->ptr_data->female_finder_;
	id_finder.getMatch(faceid, match_res, is_debug_);
	if (is_debug_)
	{
		cstr root_basic = res_var_->gender_ == Gender::MALE ?
			"D:/data/testPic/uni_pp_0715/" : "D:/data/star/female_pp/";
		if (SG::isExist(root_basic + std::to_string(match_res) + "_0_3dmm_crop.png"))
		{
			cv::Mat imread = cv::imread(root_basic + std::to_string(match_res) + "_0_3dmm_crop.png");
			cv::imwrite(result_dir_ + "match_pic.jpg", imread);
		}
		else
		{
			LOG(ERROR) << "match pic not found!" << std::endl;
		}

		intVec res = { match_res };
		FILEIO::saveDynamic(result_dir_ + "matchid.txt", res, ",");
	}
}

void RecMesh::getDFResult(const MeshCompress& init_basic, const int& match_id)
{
	cstr file_exe = "D:/code/deformation-transfer-win/x64/Release/dtrans_cmd.exe";
	cstr basic_deform = result_dir_ + "bfm_lap_sys.obj";
	cstr root_basic = res_var_->gender_ == Gender::MALE ?
		"D:/data/testPic/uni_pp_0715/" : "D:/data/testPic/uni_pp_0715/";
	cstr root_q = res_var_->gender_ == Gender::MALE ?
		"D:/data/testPic/uni/" : "D:/data/star/female/";
	cstr basic = root_basic + std::to_string(match_id) + "_0_bfm_lap_sys.obj";
	cstr q = root_q + std::to_string(match_id) + ".obj";
	cstr q_deform = result_dir_ + "q.obj";
	cstr corres = "D:/data/server_pack/taobao_df/A_B.tricorrs";
	if (!SG::isExist(file_exe))
	{
		LOG(ERROR) << file_exe << " not exist." << std::endl;
	}
	if (!SG::isExist(basic))
	{
		LOG(ERROR) << basic << " not exist." << std::endl;
	}
	if (!SG::isExist(q))
	{
		LOG(ERROR) << q << " not exist." << std::endl;
	}
	if (!SG::isExist(corres))
	{
		LOG(ERROR) << corres << " not exist." << std::endl;
	}
	if (!SG::isExist(q))
	{
		LOG(ERROR) << q << " not exist." << std::endl;
	}
	SG::exec(file_exe + " " + basic + " " + q + " " + corres + " " + basic_deform + " " + q_deform);
	MeshCompress q_deform_obj(q_deform);
	//move
	MeshCompress q_basic_obj(q);
	float3E translate;
	RT::getTranslate(q_deform_obj.pos_, q_basic_obj.pos_, translate);
	RT::translateInPlace(translate, q_deform_obj.pos_);
	if (is_debug_)
	{
		q_deform_obj.saveObj(q_deform);
		MeshCompress basic_match(basic);
		basic_match.saveObj(result_dir_ + "basic_match.obj");
		MeshCompress q_match(q);
		q_match.saveObj(result_dir_ + "q_match.obj");
		MeshCompress q_deform(q_deform);
		q_match.pos_ = q_deform.pos_;
		q_match.update();
		//get uv
		q_match.saveObj(result_dir_ + "q.obj");
	}

}

void RecMesh::processBasicQRatio()
{
	MeshCompress basic(result_dir_ + "bfm_lap_sys.obj");
	MeshCompress q(result_dir_ + "q.obj");
	//calculate
	intVec basic_68 = const_var_->ptr_data->fwh_68_idx_;
	intVec q_68 = const_var_->ptr_data->taobao_68_idx_;
	//tranform xy
	float3Vec deform_pos;
	intVec deform_idx;
	intVec sys_info = { 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65,};
	double ratio_up = (q.pos_[q_68[0]].x() - q.pos_[q_68[16]].x());
	double ratio_down = (basic.pos_[basic_68[0]].x() - basic.pos_[basic_68[16]].x());
	double ratio = ratio_up / ratio_down;

	for (int i = 0; i < 8; i++)
	{
		int basic_idx = basic_68[i];
		int basic_sys = basic_68[sys_info[i]];
		int q_idx = q_68[i];
		int q_sys = q_68[sys_info[i]];
		double mid_x = 0.5*(q.pos_[q_idx].x() + q.pos_[q_sys].x());
		double length_x = ratio* (basic.pos_[basic_idx].x() - basic.pos_[basic_sys].x());
		double length_ori = q.pos_[q_idx].x() - q.pos_[q_sys].x();
		deform_idx.push_back(q_idx);
		deform_idx.push_back(q_sys);
		float3E q_deform_pos = q.pos_[q_idx];
		float3E q_deform_sys = q.pos_[q_sys];
		q_deform_pos.x() = mid_x + 0.5*length_x;
		q_deform_sys.x() = mid_x - 0.5*length_x;
		deform_pos.push_back(q_deform_pos);
		deform_pos.push_back(q_deform_sys);
	}
	intVec fix = { 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,473,485,486,487,489,607,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1184,1341,1342,1347,1353,1354,1355,1356,1357,1771,1775, };
	fix.insert(fix.end(), q_68.begin()+17, q_68.end());
	LaplacianDeform q_ratio;
	q_ratio.init(q, deform_idx, fix);
	MeshCompress res = q;
	q_ratio.deform(deform_pos, res.pos_);
	res.saveObj(result_dir_ + "q_deform.obj");
}

void RecMesh::processPartRatio()
{
	MeshCompress basic_ori(result_dir_ + "basic_match.obj");
	MeshCompress basic(result_dir_ + "bfm_lap_sys.obj");
	MeshCompress q(result_dir_ + "q.obj");
	MeshCompress q_ori(result_dir_ + "q_match.obj");
	//calculate
	intVec basic_68 = const_var_->ptr_data->fwh_68_idx_;
	intVec q_68 = const_var_->ptr_data->taobao_68_idx_;

	intVec left_eye = {42,43,44,45,46,47};
	intVec right_eye = { 36,37,38,39,40,41};
	intVec nose = {27,28,29,30,31,32,33,34,35};
	intVec mouth = { 48, 49, 50,51,52,53,54,55,56,57,58,59};
	intVec face = { 0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
	intVec fix = { 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,473,485,486,487,489,607,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1184,1341,1342,1347,1353,1354,1355,1356,1357,1771,1775, };
	intVec mouth_pair = FILEIO::loadIntDynamic("D:/data/server_pack/taobao/left_up_match.txt");
	float3Vec B_ratio_pos;
	intX2Vec opt_part = { left_eye, right_eye, nose, mouth };
	//DTTools::setPartDelta(opt_part, basic_68, q_68, basic_ori, basic, q_ori, B_ratio_pos, is_debug_, result_dir_);
	DTTools::setPartOri(opt_part, basic_68, q_68, basic_ori, basic, q_ori, B_ratio_pos, is_debug_, result_dir_);

	float3Vec deform_pos_wuguan;
	intVec deform_idx_wuguan;
	for (int i = 0; i < opt_part.size(); i++)
	{
		for (int j = 0; j < opt_part[i].size(); j++)
		{
			int idx_in_B = q_68[opt_part[i][j]];
			deform_idx_wuguan.push_back(idx_in_B);
			deform_pos_wuguan.push_back(B_ratio_pos[opt_part[i][j]]);
		}
	}
	LaplacianDeform q_wuguan;
	q_wuguan.init(q, deform_idx_wuguan, fix, mouth_pair);
	MeshCompress res_wuguan = q;
	q_wuguan.deform(deform_pos_wuguan, res_wuguan.pos_, LinkType::KEEP_DIS);
	res_wuguan.saveObj(result_dir_ + "q_wuguan.obj");

	//tranform xy
	float3Vec deform_pos;
	intVec deform_idx;
	intVec sys_info = { 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65, };
	double ratio_up = (q.pos_[q_68[0]].x() - q.pos_[q_68[16]].x());
	double ratio_down = (basic.pos_[basic_68[0]].x() - basic.pos_[basic_68[16]].x());
	double ratio = ratio_up / ratio_down;

	for (int i = 0; i < 8; i++)
	{
		int basic_idx = basic_68[i];
		int basic_sys = basic_68[sys_info[i]];
		int q_idx = q_68[i];
		int q_sys = q_68[sys_info[i]];
		double mid_x = 0.5*(q.pos_[q_idx].x() + q.pos_[q_sys].x());
		double length_x = ratio * (basic.pos_[basic_idx].x() - basic.pos_[basic_sys].x());
		double length_ori = q.pos_[q_idx].x() - q.pos_[q_sys].x();
		deform_idx.push_back(q_idx);
		deform_idx.push_back(q_sys);
		float3E q_deform_pos = q.pos_[q_idx];
		float3E q_deform_sys = q.pos_[q_sys];
		q_deform_pos.x() = mid_x + 0.5*length_x;
		q_deform_sys.x() = mid_x - 0.5*length_x;
		deform_pos.push_back(q_deform_pos);
		deform_pos.push_back(q_deform_sys);
	}
	fix.insert(fix.end(), q_68.begin() + 17, q_68.end());
	LaplacianDeform q_ratio;
	q_ratio.init(q, deform_idx, fix);
	MeshCompress res = q;
	q_ratio.deform(deform_pos, res.pos_);
	res.saveObj(result_dir_ + "q_deform.obj");
}

void RecMesh::getFaceID(const cv::Mat& img, const vecF& landmark_5_xy, vecF& id)
{
	cv::Mat aligned_res = PREPROCESS_IMAGE::alignToMtcnnExpand(img, (float*)(landmark_5_xy.data()), true, 128);
	const_var_->ptr_model->facenet_->inference(aligned_res, 128, 128, id);
	if (is_debug_)
	{
		cv::imwrite(result_dir_ + "arcface_crop.jpg", aligned_res);
		FILEIO::saveEigenDynamic(result_dir_ + "faceid.txt", id);
	}
}

void RecMesh::getLandmarks(const cv::Mat& img, vecF& res)
{
	const_var_->ptr_model->landmark_68_->inference(img, 256, 256, res);
	if (is_debug_)
	{
		cv::Mat img_clone = img.clone();
		for (int i = 0; i < 68; i++)
		{
			cv::Point p = cv::Point(res[2 * i], res[2 * i + 1]);
			cv::circle(img_clone, p, 2, cv::Scalar(255, 0, 0), 2);
		}
		cv::imwrite(result_dir_ + "landmark_crop.jpg", img_clone);
		FILEIO::saveEigenDynamic(result_dir_ + "landmark_xy_256.txt", res);
		if (SG::isExist(result_dir_ + "rep.txt"))
		{
			LOG(WARNING) << "using replaced data." << std::endl;
			intVec rep_256 = FILEIO::loadIntDynamic(result_dir_ + "rep.txt");
			int n_rep = rep_256.size() / 3;
			for (int iter_rep = 0; iter_rep < n_rep; iter_rep++)
			{
				int rep_idx = rep_256[3 * iter_rep];
				int rep_x = rep_256[3 * iter_rep + 1];
				int rep_y = rep_256[3 * iter_rep + 2];
				res[rep_idx * 2 + 0] = rep_x;
				res[rep_idx * 2 + 1] = rep_y;
			}
		}
	}
}

void RecMesh::processFaceDetection(const cv::Mat& input, cv::Rect& res)
{
	std::vector< MTCNN_SCOPE::FaceInfo> face_bbox;
	const_var_->ptr_model->mtcnn_->Detect_T(res_var_->input_image_, face_bbox);
	if (face_bbox.empty())
	{
		LOG(WARNING) << "No face detected." << std::endl;
		return;
	}
	//get default 0
	res_var_->bb_box_ =	cv::Rect (face_bbox[0].bbox.xmin, face_bbox[0].bbox.ymin,
		face_bbox[0].bbox.xmax - face_bbox[0].bbox.xmin, face_bbox[0].bbox.ymax - face_bbox[0].bbox.ymin);
	if (is_debug_)
	{
		cv::Mat canvas = res_var_->input_image_.clone();
		cv::rectangle(canvas, res_var_->bb_box_, cv::Scalar(255, 0, 0), 2);
		cv::imwrite(result_dir_ + "input_bb.png", canvas);
	}
}

void RecMesh::shiftBBLandmark68(const cv::Mat& img, const cv::Rect& src, int resolution, cv::Mat& dst, cv::Rect& scale_bb)
{
	float height = src.height*1.0f;
	float width = src.width*1.0f;
	float scale_h = 1.0;
	float reference = 195.0f;
	//cv::Point2f center = cv::Point2f(src.br().x - width * 0.5, src.br().y - height* 1.16 * 0.5 - height*1.16*0.12 );
	cv::Point2f center = cv::Point2f(src.br().x - width * 0.5, src.br().y - height * scale_h * 0.5);
	float scale = 0.5*(scale_h* height + width) / reference;
	scale = scale * 256.0/ 240 ;
	
	float scale_tl = MIN(center.x / 128.0, center.y / 128.0);
	float scale_br = MIN((img.cols - center.x) / 128.0, (img.rows - center.y) / 128.0);
	scale = MIN(scale, MIN(scale_tl, scale_br));

	int opt_tl_x = DMAX(center.x - 128.0 * scale, 0);
	int opt_tl_y = DMAX(center.y - 128.0 * scale, 0);
	scale_bb = cv::Rect(opt_tl_x, opt_tl_y, 256 * scale, 256 * scale);
	SG::checkBBox(img, scale_bb);
	scale_bb.width = DMIN(scale_bb.width, scale_bb.height);
	scale_bb.height = scale_bb.width;
	dst = img(scale_bb);
	cv::resize(dst, dst, cv::Size(resolution, resolution));
	if (is_debug_)
	{
		cv::imwrite(result_dir_ + "input_landmark68.png", dst);
	}
}

void RecMesh::transform(const cv::Point2f& src, const cv::Point2f& center, const double scale,
	const int resolution, cv::Point2f& dst)
{

}

void RecMesh::reflectBackToImage(const cv::Rect& landmark_68_crop, int resolution, 
	const vecF& res_256, vecF& res)
{
	int n_keypoints = res_256.size() / 2;
	res = res_256;
	if (landmark_68_crop.width != landmark_68_crop.height)
	{
		LOG(ERROR) << "error for landmark 68 problems." << std::endl;
		return;
	}
	float scale = (1.0f*landmark_68_crop.width) / (1.0*resolution);

#pragma omp parallel for
	for (int i = 0; i < n_keypoints ; i++)
	{
		float temp_x = res_256[2 * i] * scale + landmark_68_crop.tl().x;
		float temp_y = res_256[2 * i + 1] * scale + landmark_68_crop.tl().y;
		res[2 * i] = temp_x;
		res[2 * i + 1] = temp_y;
	}

	if (is_debug_)
	{
		cv::Mat img_clone = res_var_->input_image_.clone();
		for (int i = 0; i < n_keypoints; i++)
		{
			cv::Point p = cv::Point(res[2 * i], res[2 * i + 1]);
			cv::circle(img_clone, p, 1, cv::Scalar(255, 0, 0), 2);
		}
		cv::imwrite(result_dir_ + "whole_landmark68.png", img_clone);
	}
}

void RecMesh::get5LandFrom68(const vecF& input_68, vecF& output_5)
{
	enum mtcnn
	{
		LEFT_EYE,
		RIGHT_EYE,
		NOSE,
		LEFT_MOUTH,
		RIGHT_MOUTH,
	};
	int num = 5;
	intX2Vec corr(num, intVec{});
	corr[0] = { 37, 38, 40, 41};
	corr[1] = { 43, 44, 46, 47};
	corr[2] = { 30 };
	corr[3] = { 48 };
	corr[4] = { 54 };
	output_5.resize(num*2);
	for (int i = 0; i < num; i++)
	{
		float x_sum = 0;
		float y_sum = 0;
		for (int iter = 0; iter < corr[i].size(); iter++)
		{
			x_sum += input_68(corr[i][iter] * 2);
			y_sum += input_68(corr[i][iter] * 2+1);
		}
		output_5(2 * i) = x_sum / (1.0f*corr[i].size());
		output_5(2 * i+1) = y_sum / (1.0f*corr[i].size());
	}
	if (is_debug_)
	{
		cv::Mat img_clone = res_var_->input_image_.clone();
		for (int i = 0; i < num; i++)
		{
			cv::Point p = cv::Point(output_5[2 * i], output_5[2 * i + 1]);
			cv::circle(img_clone, p, 2, cv::Scalar(255, 0, 0), 2);
		}
		cv::imwrite(result_dir_ + "whole_landmark5.png", img_clone);
	}
}

void RecMesh::get3dmmCoef(const cv::Mat& img, const vecF& landmark_5_xy, vecF& coef_3dmm)
{
	int height = img.rows;
	int width = img.cols;
	matF point(3, 5);
	vecF B = landmark_5_xy;
	//unit test
	//B << 123.12, 117.58, 176.59, 122.09, 126.99, 144.68, 117.61, 183.43, 163.94, 186.41;
	bool is_flip_y = false;
	if (is_flip_y)
	{
		for (int i = 0; i < 5; i++)
		{
			B(2 * i + 1) = img.rows - 1 - landmark_5_xy(2 * i + 1);
		}
	}
	point << -0.31148657, 0.30979887, 0.0032535, -0.25216928, 0.2484662,
		0.29036078, 0.28972036, -0.04617932, -0.38133916, -0.38128236,
		0.13377953, 0.13179526, 0.55244243, 0.22405732, 0.22235769;
	
	matF A(10, 8);
	A.setConstant(0);
	for (int y = 0; y < 5; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			A(2 * y, x) = point(x, y);
			A(2 * y+ 1, x + 4) = point(x, y);
		}
		A(2 * y, 3) = 1;
		A(2 * y + 1, 7) = 1;
	}

	//matF AtA = A.transpose()*A;
	//vecF AtB = A.transpose()*B;
	//vecF x = AtA.ldlt().solve(AtB);
	vecF x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
	float s = 0.5*(x.head(3).norm() + x.segment(4, 3).norm());
	float t0 = x(3);
	float t1 = x(7);
	float M[6] = {1, 0, 0.5*width - t0, 0, 1, 0.5*height-t1};
	cv::Mat transfer(2, 3, CV_32FC1, M);
	cv::Mat aligned;
	cv::warpAffine(img, aligned, transfer, img.size(), 1, cv::BORDER_REPLICATE, 0);
	int scale_w = 102.0 * width / (1.0f*s);
	int scale_h = 102.0 * height / (1.0f*s);

	if (scale_w < 224 || scale_h < 224)
	{
		float mag_scale = MAX(224.0 / scale_w, 224.0 / scale_h);
		scale_w *= mag_scale;
		scale_h *= mag_scale;
	}

	cv::Mat scale_aligned;
	cv::resize(aligned, scale_aligned, cv::Size(scale_w, scale_h));

	//max scale
	float max_w = scale_aligned.cols / 112.0f * 2;
	float max_h = scale_aligned.rows / 112.0f * 2;

	cv::Rect bbox = cv::Rect(scale_w*0.5 - 112, scale_h*0.5 - 112, 224, 224);
	SG::checkBBox(scale_aligned, bbox);
	if (bbox.width != 224 || bbox.height != 224)
	{
		//TODO
	}
	cv::Mat res = scale_aligned(bbox);
	cv::resize(res, res, cv::Size(224, 224));
	const_var_->ptr_model->deep3d_->inference(res, 224, 224, coef_3dmm);
	if (is_debug_)
	{
		//227 : 254
		floatVec light_coef;
		LOG(INFO) << "lighting: ";
		for (int i = 227; i < 254; i++)
		{
			light_coef.push_back(coef_3dmm[i]);
		}
		FILEIO::saveDynamic(result_dir_ + "light_coefs.txt", light_coef, ",");
		cv::imwrite(result_dir_ + "3dmm_align.png", scale_aligned);
		cv::imwrite(result_dir_ + "3dmm_crop.png", res);
	}
}

void RecMesh::get3dmmCoef(const std::shared_ptr<NR3D> model_3dmm, const cv::Mat& img, const vecF& landmark_5_xy, vecF& coef_3dmm)
{
	int height = img.rows;
	int width = img.cols;
	matF point(3, 5);
	vecF B = landmark_5_xy;
	//unit test
	//B << 123.12, 117.58, 176.59, 122.09, 126.99, 144.68, 117.61, 183.43, 163.94, 186.41;
	bool is_flip_y = false;
	if (is_flip_y)
	{
		for (int i = 0; i < 5; i++)
		{
			B(2 * i + 1) = img.rows - 1 - landmark_5_xy(2 * i + 1);
		}
	}
	point << -0.31148657, 0.30979887, 0.0032535, -0.25216928, 0.2484662,
		0.29036078, 0.28972036, -0.04617932, -0.38133916, -0.38128236,
		0.13377953, 0.13179526, 0.55244243, 0.22405732, 0.22235769;

	matF A(10, 8);
	A.setConstant(0);
	for (int y = 0; y < 5; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			A(2 * y, x) = point(x, y);
			A(2 * y + 1, x + 4) = point(x, y);
		}
		A(2 * y, 3) = 1;
		A(2 * y + 1, 7) = 1;
	}

	//matF AtA = A.transpose()*A;
	//vecF AtB = A.transpose()*B;
	//vecF x = AtA.ldlt().solve(AtB);
	vecF x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
	float s = 0.5*(x.head(3).norm() + x.segment(4, 3).norm());
	float t0 = x(3);
	float t1 = x(7);
	float M[6] = { 1, 0, 0.5*width - t0, 0, 1, 0.5*height - t1 };
	cv::Mat transfer(2, 3, CV_32FC1, M);
	cv::Mat aligned;
	cv::warpAffine(img, aligned, transfer, img.size(), 1, cv::BORDER_REPLICATE, 0);
	int scale_w = 102.0 * width / (1.0f*s);
	int scale_h = 102.0 * height / (1.0f*s);

	if (scale_w < 224 || scale_h < 224)
	{
		float mag_scale = MAX(224.0 / scale_w, 224.0 / scale_h);
		scale_w *= mag_scale;
		scale_h *= mag_scale;
	}

	cv::Mat scale_aligned;
	cv::resize(aligned, scale_aligned, cv::Size(scale_w, scale_h));

	//max scale
	float max_w = scale_aligned.cols / 112.0f * 2;
	float max_h = scale_aligned.rows / 112.0f * 2;

	cv::Rect bbox = cv::Rect(scale_w*0.5 - 112, scale_h*0.5 - 112, 224, 224);
	SG::checkBBox(scale_aligned, bbox);
	if (bbox.width != 224 || bbox.height != 224)
	{
		//TODO
	}
	cv::Mat res = scale_aligned(bbox);

	//测试
	//cv::Mat dw = cv::imread("D:/dota210507/0603_01/000020_dw.jpg");
	//res = dw.clone();
	//测试

	cv::resize(res, res, cv::Size(224, 224));
	model_3dmm->inference(res, 224, 224, coef_3dmm);
	if (is_debug_)
	{
		//227 : 254
		floatVec light_coef;
		LOG(INFO) << "lighting: ";
		for (int i = 227; i < 254; i++)
		{
			light_coef.push_back(coef_3dmm[i]);
		}
		FILEIO::saveDynamic(result_dir_ + "light_coefs.txt", light_coef, ",");
		cv::imwrite(result_dir_ + "3dmm_align.png", scale_aligned);
		cv::imwrite(result_dir_ + "3dmm_crop.png", res);
	}
}

void RecMesh::getPostProcessFor3dmm(const vecF& coef_3dmm, MeshCompress& res)
{
	const Tensor& bfm_tensor = const_var_->ptr_data->bfm_tensor_;
	const Tensor& fwh_bfm_tensor = const_var_->ptr_data->fwh_bfm_tensor_;
	const MeshSysFinder& fwh_sys = const_var_->ptr_data->fwh_sys_finder_;
	
	float mag = 1.0;

	floatVec raw_3dmm = bfm_tensor.interpretID(mag * coef_3dmm.head(80));
	MeshCompress bfm_mesh = bfm_tensor.template_obj_;
	SG::safeMemcpy(bfm_mesh.pos_.data(), raw_3dmm.data(), raw_3dmm.size() * sizeof(float));
	
	floatVec raw_fwh = fwh_bfm_tensor.interpretID(mag * coef_3dmm.head(80));
	MeshCompress fwh_mesh = fwh_bfm_tensor.template_obj_;
	SG::safeMemcpy(fwh_mesh.pos_.data(), raw_fwh.data(), raw_fwh.size() * sizeof(float));

	//needs to fit

	if (is_debug_)
	{
		bfm_mesh.saveObj(result_dir_ + "3dmm.obj");
		fwh_mesh.saveObj(result_dir_ + "3dmm_fwh.obj");
	}

	const MeshPostProcess& pp_data = const_var_->ptr_data->fwh_3dmm_;

	//get translate
	float3Vec fwh_slice, bfm_slice;
	fwh_mesh.getSlice(pp_data.fwh_id_trans_, fwh_slice);
	bfm_mesh.getSlice(pp_data.bfm_id_trans_, bfm_slice);
	float3E translate;
	RT::getTranslate(bfm_slice, fwh_slice, translate);
	translate.x() = 0;
	RT::translateInPlace(translate, bfm_mesh.pos_);
	if (is_debug_)
	{
		bfm_mesh.saveObj(result_dir_ + "3dmm_trans.obj");
	}

	double shift_x = 0;
	for (int i : pp_data.bfm_mid_)
	{
		shift_x += bfm_mesh.pos_[i].x();
	}
	shift_x = shift_x / (1.0*pp_data.bfm_mid_.size());
	LaplacianDeform to_bfm;
	to_bfm.init(fwh_mesh, pp_data.fwh_deform_, pp_data.fwh_fix_, pp_data.mouth_close_);
	float3Vec deform_pos(pp_data.fwh_deform_.size());
	for (int i = 0; i < pp_data.fwh_deform_.size(); i++)
	{
		int idx_fwh = pp_data.fwh_deform_[i];
		int idx_bfm = pp_data.bfm_deform_[i];
		if (idx_bfm > 0)
		{
			deform_pos[i] = bfm_mesh.pos_[idx_bfm];
		}
		else
		{
			float3E right = bfm_mesh.pos_[-idx_bfm];
			float3E left = right;
			left.x() = 2 * shift_x - right.x();
			deform_pos[i] = left;
		}
	}
	res = fwh_mesh;
	to_bfm.deform(deform_pos, res.pos_);
	if (is_debug_)
	{
		res.saveObj(result_dir_ + "bfm_lap.obj");
	}
	fwh_sys.getSysPosLapInPlace(res.pos_);
	if (is_debug_)
	{
		res.saveObj(result_dir_ + "bfm_lap_sys.obj");
	}
}

void RecMesh::getFwhCoefFromCeres(MeshCompress& bfm, const cv::Mat& img_256,
	const vecF& landmark_xy, MeshCompress& fwh_res, float3E& translate_3dmm_to_fwh)
{
	//get 68 landmarks for cropped image, skip for brow && inner mouth
	intSet used_list = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
	39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
		54,55,56,57,58,59 };

	int n_num = landmark_xy.size()*0.5;
	float3Vec bfm_slice_68;
	bfm.getSlice(const_var_->ptr_data->bfw_68_idx_, bfm_slice_68);
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	for (int i = 0; i < n_num; i++)
	{
		if (used_list.count(i) && i < 17)
		{
			img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
			float3E pos = bfm_slice_68[i];
			obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
		}
		else
		{
			for (int iter = 0; iter < 3; iter++)
			{
				img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
				float3E pos = bfm_slice_68[i];
				obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
			}
		}
	}
	cvMatD rvec, tvec, r_matrix;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
	cv::solvePnP(obj_points, img_points, const_var_->ptr_data->proj_->intrisic_,
		dist_coeffs, rvec, tvec);
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = img_256.clone();
	floatVec in_camera_z(n_num, 0);
	float3Vec in_3d(n_num, float3E(0, 0, 0));
	double z_shift = 0;
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = bfm_slice_68[i];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);
#if 1
		cvMatD in_image_from_xy = (cvMatD(3, 1) << landmark_xy[2 * i] * in_camera_z[i], landmark_xy[2 * i + 1] * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		in_3d[i] = pos;

		//LOG(INFO) << "in_camera_3d:" << in_camera_3d << std::endl;
		//LOG(INFO) << "in_world_3d:" << in_world_3d << std::endl;
		//LOG(INFO) << "pos 3d: " << pos << std::endl;
#else
		//used for testing
		cvMatD in_image_from_xy = (cvMatD(3, 1) << img_x * in_camera_z[i], img_y * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		LOG(INFO) << "in_camera_3d:" << in_camera_3d << std::endl;
		LOG(INFO) << "in_world_3d:" << in_world_3d << std::endl;
		LOG(INFO) << "pos 3d: " << pos << std::endl;
#endif
	}
	const Tensor& fwh = const_var_->ptr_data->fwh_tensor_;
#if 1
	ceres::Problem fitting_pca;
	doubleVec scale = { 1.0 };
	doubleVec translate(3, 0), shape_coef(120, 0);
	shape_coef[0] = 1.0;
	floatX2Vec pca_refactor(in_3d.size(), floatVec(fwh.n_id_ * 3, 0));
	int n_vertex = fwh.template_obj_.n_vertex_;
	const doubleVec camera = 
	{ 
		GETD(rvec, 0, 0), GETD(rvec, 1, 0), GETD(rvec, 2, 0),
		GETD(tvec, 0, 0), GETD(tvec, 1, 0), GETD(tvec, 2, 0),
		const_var_->ptr_data->proj_->fx_, 0, 0,
		const_var_->ptr_data->proj_->cx_,
	};
	ceres::CostFunction* cost_function =
		new ceres::NumericDiffCostFunction<PREPARE::PCAImageCostRoi,
		ceres::CENTRAL,
		68 * 2, //vertex*3
		1, /* scale */
		3 /* traslate*/,
		120 /*pca value*/>
		(new PREPARE::PCAImageCostRoi(fwh.data_, landmark_xy, camera, const_var_->ptr_data->fwh_68_idx_, fwh.n_id_, n_vertex));
	fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());

	shape_coef[0] = 1;

	fitting_pca.SetParameterUpperBound(&scale[0], 0, 1 + 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&scale[0], 0, 1 - 1e-6); // t_z has to be negative
	fitting_pca.SetParameterUpperBound(&translate[0], 0, 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&translate[0], 0, -1e-6); // t_z has to be negative
	fitting_pca.SetParameterUpperBound(&translate[0], 1, 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&translate[0], 1, -1e-6); // t_z has to be negative
	fitting_pca.SetParameterUpperBound(&translate[0], 2, 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&translate[0], 2, -1e-6); // t_z has to be negative

	float mag_ev_data = 1.5;
	fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1 + 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1 - 1e-6); // t_z has to be negative		
	for (int iter_coef = 1; iter_coef < 120; iter_coef++)
	{
		fitting_pca.SetParameterUpperBound(&shape_coef[0], iter_coef, mag_ev_data*fwh.ev_data_[iter_coef - 1]); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&shape_coef[0], iter_coef, -mag_ev_data * fwh.ev_data_[iter_coef - 1]); // t_z has to be negative
	}
	ceres::Solver::Options solver_options;
	solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	//solver_options.num_threads = 8;
	solver_options.minimizer_progress_to_stdout = true;
	solver_options.max_num_iterations = 15;
	ceres::Solver::Summary solver_summary;
	Solve(solver_options, &fitting_pca, &solver_summary);
	std::cout << solver_summary.BriefReport() << "\n";

	vecF ceres_coef(119);
	for (int iter_pca = 1; iter_pca < 120; iter_pca++)
	{
		ceres_coef[iter_pca - 1] = shape_coef[iter_pca];
	}

	LOG(INFO) << "ceres pca: " << shape_coef[0] << std::endl << ceres_coef.transpose() << std::endl;
	floatVec raw_fwh = fwh.interpretID(ceres_coef);
	SG::safeMemcpy(fwh_res.pos_.data(), raw_fwh.data(), raw_fwh.size() * sizeof(float));

	/////change uv coordinate
	fwh_res.tex_cor_.resize(fwh_res.n_vertex_);
	for (int i = 0; i < fwh_res.n_vertex_; i++)
	{
		float3E pos = fwh_res.pos_[i];
		//LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		//LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		//pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x() + translate[0], pos.y() + translate[1], pos.z() + translate[2]) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		//in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0)*1.0/255.0;
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green ope
		//cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
		if (img_x > 0 && img_x < 1 && img_y >0 && img_y < 1)
		{
			fwh_res.tex_cor_[i] = float2E(img_x, 1 - img_y);
		}
		else
		{
			fwh_res.tex_cor_[i] = float2E(0,0);
		}
	}
	fwh_res.material_.push_back("img_256.mtl");
	fwh_res.tri_uv_ = fwh_res.tri_;
	fwh_res.update();


#else
	//fit	
	float weight = 0.1;
	floatVec reg(fwh.n_id_ - 1, weight);
	fwh.fitID(in_3d, reg, const_var_->ptr_data->fwh_68_idx_, fwh.ev_data_, coef_res);
	LOG(INFO) << "out coef_res:" << coef_res.transpose() << std::endl;
	floatVec raw_fwh = fwh.interpretID(coef_res);

	ceres::Problem fitting_pca;
	doubleVec scale = { 1.0 };
	doubleVec translate(3, 0), shape_coef(120, 0);
	shape_coef[0] = 1.0;
	floatX2Vec pca_refactor(in_3d.size(), floatVec(fwh.n_id_ * 3, 0));
	int n_vertex = fwh.template_obj_.n_vertex_;
	ceres::CostFunction* cost_function =
		new ceres::NumericDiffCostFunction<PREPARE::PCAVertexCostRoi,
		ceres::CENTRAL,
		68 * 3, //vertex*3
		1, /* scale */
		3 /* traslate*/,
		120 /*pca value*/>
		(new PREPARE::PCAVertexCostRoi(fwh.data_, in_3d, const_var_->ptr_data->fwh_68_idx_, fwh.n_id_, n_vertex));
	fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());

	shape_coef[0] = 1;

	fitting_pca.SetParameterUpperBound(&scale[0], 0, 1 + 0.2); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&scale[0], 0, 1 - 0.2); // t_z has to be negative

	fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1 + 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1 - 1e-6); // t_z has to be negative		
	for (int iter_coef = 1; iter_coef < 120; iter_coef++)
	{
		//fitting_pca.SetParameterUpperBound(&shape_coef[0], iter_coef, 1); // t_z has to be negative
		//fitting_pca.SetParameterLowerBound(&shape_coef[0], iter_coef, 0); // t_z has to be negative
	}
	ceres::Solver::Options solver_options;
	solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	//solver_options.num_threads = 8;
	solver_options.minimizer_progress_to_stdout = true;
	solver_options.max_num_iterations = 15;
	ceres::Solver::Summary solver_summary;
	Solve(solver_options, &fitting_pca, &solver_summary);
	std::cout << solver_summary.BriefReport() << "\n";

	vecF ceres_coef(119);
	for (int iter_pca = 1; iter_pca < 120; iter_pca++)
	{
		ceres_coef[iter_pca - 1] = shape_coef[iter_pca];
	}

	LOG(INFO) << "ceres pca: " << shape_coef[0] << std::endl << ceres_coef.transpose() << std::endl;
	raw_fwh = fwh.interpretID(ceres_coef);
#endif

	for (int i = 0; i < n_num; i++)
	{
		int idx = const_var_->ptr_data->fwh_68_idx_[i];
		float3E pos = float3E(raw_fwh[3 * idx], raw_fwh[3 * idx + 1], raw_fwh[3 * idx + 2]);
		//LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		//LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x()+translate[0], pos.y() + translate[1], pos.z()+translate[2]) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green ope
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
	}
	float3E translate_3dmm_fwh = float3E(0, -translate[1], -translate[2]);
	//RT::translateInPlace(translate_3dmm_fwh, bfm.pos_);
	translate_3dmm_to_fwh = float3E(0, -translate[1], -translate[2]);
	if (is_debug_)
	{
		cv::FileStorage fs(result_dir_ + "test.yml", cv::FileStorage::WRITE);
		fs << "rvec" << rvec;
		fs << "tvec" << tvec;
		cvMatD trans_fwh_bfm = (cvMatD(3, 1) << translate[0], translate[1], translate[2]);
		fs << "trans_fwh_bfm" << trans_fwh_bfm;
		fs << "r_matrix" << r_matrix;
		fs.release();
		cv::imwrite(result_dir_ + "2d_3d.png", canvas);
	}
}

void RecMesh::getFwhCoefFromCeres3dmm(MeshCompress& bfm, const cv::Mat& img_256,
	const vecF& landmark_xy, MeshCompress& fwh_res, float3Vec& deform_pos, intVec& deform_idx,
	cvMatD& rvec, cvMatD& tvec)
{
	intVec used_vec = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
			39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
	54,55,56,57,58,59,61,62,63,65,66,67 };
	int n_num = landmark_xy.size()*0.5;
	float3Vec bfm_slice_68;
	bfm.getSlice(const_var_->ptr_data->bfw_68_idx_, bfm_slice_68);
	cvMatD r_matrix;
	const_var_->ptr_data->proj_->getMeshToImageRT(bfm, const_var_->ptr_data->bfw_68_idx_, used_vec,
		landmark_xy, rvec, tvec);
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = img_256.clone();
	floatVec in_camera_z(n_num, 0);
	float3Vec in_3d(n_num, float3E(0, 0, 0));
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = bfm_slice_68[i];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);
		cvMatD in_image_from_xy = (cvMatD(3, 1) << landmark_xy[2 * i] * in_camera_z[i], landmark_xy[2 * i + 1] * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		in_3d[i] = pos;
	}
	const Tensor& fwh = const_var_->ptr_data->fwh_bfm_tensor_;

	ceres::Problem fitting_pca;
	doubleVec scale = { 1.0 };
	doubleVec translate(3, 0), shape_coef(81, 0);
	shape_coef[0] = 1.0;
	floatX2Vec pca_refactor(in_3d.size(), floatVec(fwh.n_id_ * 3, 0));
	int n_vertex = fwh.template_obj_.n_vertex_;
	const doubleVec camera =
	{
		GETD(rvec, 0, 0), GETD(rvec, 1, 0), GETD(rvec, 2, 0),
		GETD(tvec, 0, 0), GETD(tvec, 1, 0), GETD(tvec, 2, 0),
		const_var_->ptr_data->proj_->fx_, 0, 0,
		const_var_->ptr_data->proj_->cx_,
	};
	ceres::CostFunction* cost_function =
		new ceres::NumericDiffCostFunction<PREPARE::PCAImageCostRoi,
		ceres::CENTRAL,
		68 * 2, //vertex*3
		1, /* scale */
		3 /* traslate*/,
		81 /*pca value*/>
		(new PREPARE::PCAImageCostRoi(fwh.data_, landmark_xy, camera, const_var_->ptr_data->fwh_68_idx_, fwh.n_id_, n_vertex));
	fitting_pca.AddResidualBlock(cost_function, NULL, scale.data(), translate.data(), shape_coef.data());

	shape_coef[0] = 1;

	fitting_pca.SetParameterUpperBound(&scale[0], 0, 1 + 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&scale[0], 0, 1 - 1e-6); // t_z has to be negative
	fitting_pca.SetParameterUpperBound(&translate[0], 0, 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&translate[0], 0, -1e-6); // t_z has to be negative

	float mag_ev_data = 1.5;
	fitting_pca.SetParameterUpperBound(&shape_coef[0], 0, 1 + 1e-6); // t_z has to be negative
	fitting_pca.SetParameterLowerBound(&shape_coef[0], 0, 1 - 1e-6); // t_z has to be negative		
	for (int iter_coef = 1; iter_coef < 81; iter_coef++)
	{
		fitting_pca.SetParameterUpperBound(&shape_coef[0], iter_coef, mag_ev_data * 5); // t_z has to be negative
		fitting_pca.SetParameterLowerBound(&shape_coef[0], iter_coef, -mag_ev_data * 5); // t_z has to be negative
	}
	ceres::Solver::Options solver_options;
	solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	//solver_options.num_threads = 8;
	solver_options.minimizer_progress_to_stdout = true;
	solver_options.max_num_iterations = 15;
	ceres::Solver::Summary solver_summary;
	Solve(solver_options, &fitting_pca, &solver_summary);
	std::cout << solver_summary.BriefReport() << "\n";

	vecF ceres_coef(80);
	for (int iter_pca = 1; iter_pca < 81; iter_pca++)
	{
		ceres_coef[iter_pca - 1] = shape_coef[iter_pca];
	}

	LOG(INFO) << "ceres pca: " << shape_coef[0] << std::endl << ceres_coef.transpose() << std::endl;
	floatVec raw_fwh = fwh.interpretID(ceres_coef);
	SG::safeMemcpy(fwh_res.pos_.data(), raw_fwh.data(), raw_fwh.size() * sizeof(float));
	//translate
	float3E translate_fwh = float3E(0, translate[1], translate[2]);
	RT::translateInPlace(translate_fwh, fwh_res.pos_);
	/////change uv coordinate
	fwh_res.tex_cor_.resize(fwh_res.n_vertex_);
	for (int i = 0; i < fwh_res.n_vertex_; i++)
	{
		float3E pos = fwh_res.pos_[i];
		//LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		//LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		//pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		//in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green ope
		//cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
		if (img_x > 0 && img_x < 1 && img_y >0 && img_y < 1)
		{
			fwh_res.tex_cor_[i] = float2E(img_x, 1 - img_y);
		}
		else
		{
			fwh_res.tex_cor_[i] = float2E(0, 0);
		}
	}
	fwh_res.material_.push_back("img_256.mtl");
	fwh_res.tri_uv_ = fwh_res.tri_;
	fwh_res.update();
	float3Vec dst_pos;
	for (int i = 0; i < n_num; i++)
	{
		int idx = const_var_->ptr_data->fwh_68_idx_[i];
		float3E pos = fwh_res.pos_[idx];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x() , pos.y() , pos.z()) + tvec;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//green opt
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - const_var_->ptr_data->proj_->cx_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - const_var_->ptr_data->proj_->cy_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fy_);
		cvMatD in_image_fix = const_var_->ptr_data->proj_->intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0) , GETD(in_3d_fix, 2, 0)));
	}
	float3E translate_3dmm_fwh = float3E(0, -translate[1], -translate[2]);
	//RT::translateInPlace(translate_3dmm_fwh, bfm.pos_);
	deform_pos = dst_pos;
	deform_idx = const_var_->ptr_data->fwh_68_idx_;

	float3E translate_3dmm_to_fwh = float3E(0, -translate[1], -translate[2]);
	if (is_debug_)
	{
		cv::FileStorage fs(result_dir_ + "test.yml", cv::FileStorage::WRITE);
		fs << "rvec" << rvec;
		fs << "tvec" << tvec;
		cvMatD trans_fwh_bfm = (cvMatD(3, 1) << translate[0], translate[1], translate[2]);
		fs << "trans_fwh_bfm" << trans_fwh_bfm;
		fs << "r_matrix" << r_matrix;
		fs.release();
		cv::imwrite(result_dir_ + "2d_3d.png", canvas);
		//fwh_res.saveObj(result_dir_+ "fit_image.obj");
	}
	//deform
}

void RecMesh::adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, 
	MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx)
{
	//get 68 landmarks for cropped image, skip for brow && inner mouth
	intSet used_list = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
	39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
		54,55,56,57,58,59 };

	int n_num = landmark_xy.size()*0.5;
	float3Vec fwh_slice_68;
	fwh.getSlice(const_var_->ptr_data->fwh_68_idx_, fwh_slice_68);
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	for (int i = 0; i < n_num; i++)
	{
		if (used_list.count(i) && i < 17)
		{
			img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
			float3E pos = fwh_slice_68[i];
			obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
		}
		else
		{
			for (int iter = 0; iter < 2; iter++)
			{
				img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
				float3E pos = fwh_slice_68[i];
				obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
			}
		}
	}
	cvMatD rvec, tvec, r_matrix;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
	cv::solvePnP(obj_points, img_points, const_var_->ptr_data->proj_->intrisic_,
		dist_coeffs, rvec, tvec);
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = img_256.clone();
	cv::Mat canvas_before_opt = img_256.clone();
	floatVec in_camera_z(n_num, 0);
	float3Vec in_3d(n_num, float3E(0, 0, 0));
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = fwh_slice_68[i];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);

		//red 3d
		cv::circle(canvas_before_opt, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas_before_opt, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);


		cvMatD in_image_from_xy = (cvMatD(3, 1) << landmark_xy[2 * i] * in_camera_z[i], landmark_xy[2 * i + 1] * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		in_3d[i] = pos;
	}

	/////change uv coordinate
	fwh.tex_cor_.resize(fwh.n_vertex_);
	for (int i = 0; i < fwh.n_vertex_; i++)
	{
		float3E pos = fwh.pos_[i];
		//LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		//LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		//pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		//in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green ope
		//cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
		if (img_x > 0 && img_x < 1 && img_y >0 && img_y < 1)
		{
			fwh.tex_cor_[i] = float2E(img_x, 1 - img_y);
		}
		else
		{
			fwh.tex_cor_[i] = float2E(0, 0);
		}
	}
	fwh.material_.push_back("img_256.mtl");
	fwh.tri_uv_ = fwh.tri_;
	fwh.update();
	float3Vec dst_pos;
	for (int i = 0; i < n_num; i++)
	{
		int idx = const_var_->ptr_data->fwh_68_idx_[i];
		float3E pos = fwh.pos_[idx];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - const_var_->ptr_data->proj_->cx_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - const_var_->ptr_data->proj_->cy_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fy_);
		cvMatD in_image_fix = const_var_->ptr_data->proj_->intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
		cvMatD in_camera_reproj = r_matrix * in_3d_fix + tvec;
		cvMatD in_image_reproj = const_var_->ptr_data->proj_->intrisic_*in_camera_reproj;
		float img_x = GETD(in_image_reproj, 0, 0) / GETD(in_image_reproj, 2, 0);
		float img_y = GETD(in_image_reproj, 1, 0) / GETD(in_image_reproj, 2, 0);
		//green opt
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
	}
	deform_pos = dst_pos;
	deform_idx = const_var_->ptr_data->fwh_68_idx_;

	if (is_debug_)
	{
		cv::FileStorage fs(result_dir_ + "test_reproject.yml", cv::FileStorage::WRITE);
		fs << "rvec" << rvec;
		fs << "tvec" << tvec;
		fs << "r_matrix" << r_matrix;
		fs.release();
		cv::imwrite(result_dir_ + "re_2d_3d.png", canvas);
		cv::imwrite(result_dir_ + "re_2d_3d_before.png", canvas_before_opt);
		//fwh_res.saveObj(result_dir_+ "fit_image.obj");
	}
	//deform
}

void RecMesh::adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, const cvMatD& rvec, const cvMatD& tvec,
	MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx)
{
	int n_num = landmark_xy.size()*0.5;
	cvMatD r_matrix;
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = img_256.clone();
	cv::Mat canvas_before_opt = img_256.clone();
	floatVec in_camera_z(n_num, 0);
	float3Vec in_3d(n_num, float3E(0, 0, 0));
	float3Vec fwh_slice_68;
	fwh.getSlice(const_var_->ptr_data->fwh_68_idx_, fwh_slice_68);
	double error_dis = 0;
	intSet used_list = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
						39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
						54,55,56,57,58,59 };
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = fwh_slice_68[i];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);

		//red 3d
		cv::circle(canvas_before_opt, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas_before_opt, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);


		cvMatD in_image_from_xy = (cvMatD(3, 1) << landmark_xy[2 * i] * in_camera_z[i], landmark_xy[2 * i + 1] * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		in_3d[i] = pos;
		if (used_list.count(i))
		{
			error_dis += (float2E(img_x, img_y) - float2E(landmark_xy[2 * i], landmark_xy[2 * i + 1])).norm();
		}
	}
	LOG(INFO) << "error: " << 1.0f * error_dis / used_list.size();
	/////change uv coordinate
	fwh.tex_cor_.resize(fwh.n_vertex_);
	for (int i = 0; i < fwh.n_vertex_; i++)
	{
		float3E pos = fwh.pos_[i];
		//LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		//LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		//pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		//in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green ope
		//cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
		if (img_x > 0 && img_x < 1 && img_y >0 && img_y < 1)
		{
			fwh.tex_cor_[i] = float2E(img_x, 1 - img_y);
		}
		else
		{
			fwh.tex_cor_[i] = float2E(0, 0);
		}
	}
	fwh.material_.push_back("img_256.mtl");
	fwh.tri_uv_ = fwh.tri_;
	fwh.update();
	float3Vec dst_pos;
	for (int i = 0; i < n_num; i++)
	{
		int idx = const_var_->ptr_data->fwh_68_idx_[i];
		float3E pos = fwh.pos_[idx];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - const_var_->ptr_data->proj_->cx_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - const_var_->ptr_data->proj_->cy_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fy_);
		cvMatD in_image_fix = const_var_->ptr_data->proj_->intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
		cvMatD in_camera_reproj = r_matrix * in_3d_fix + tvec;
		cvMatD in_image_reproj = const_var_->ptr_data->proj_->intrisic_*in_camera_reproj;
		float img_x = GETD(in_image_reproj, 0, 0) / GETD(in_image_reproj, 2, 0);
		float img_y = GETD(in_image_reproj, 1, 0) / GETD(in_image_reproj, 2, 0);
		//green opt
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
	}
	deform_pos = dst_pos;
	deform_idx = const_var_->ptr_data->fwh_68_idx_;

	if (is_debug_)
	{
		cv::FileStorage fs(result_dir_ + "test_reproject.yml", cv::FileStorage::WRITE);
		fs << "rvec" << rvec;
		fs << "tvec" << tvec;
		fs << "r_matrix" << r_matrix;
		fs.release();
		cv::imwrite(result_dir_ + "re_2d_3d.png", canvas);
		cv::imwrite(result_dir_ + "re_2d_3d_before.png", canvas_before_opt);
		//fwh_res.saveObj(result_dir_+ "fit_image.obj");
	}
	//deform
	if (is_debug_)
	{
		LOG(WARNING) << "data not in pack, this part is in debug part." << std::endl;
		//loading for fwh && guijie map
		//intVec fwh_idx = FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/fwh_move_idx_skip_eyes.txt");
		//intVec gv2_idx = FILEIO::loadIntDynamic("D:/dota201201/1223_test_infer/guijie_move_idx_skip_eyes.txt");
		cstr data_root = "D:/multiPack/";
		intVec fwh_idx = FILEIO::loadIntDynamic(data_root + "1223_test_infer/fwh_move_idx.txt");
		intVec gv2_idx = FILEIO::loadIntDynamic(data_root + "1223_test_infer/guijie_move_idx_all.txt");
		intSet gv2_roi = TDST::vecToSet(FILEIO::loadIntDynamic(data_root + "1223_test_infer/gv2_render_idx.txt"));
		cv::Mat vis_fwh = img_256.clone();
		std::ofstream out_txt(result_dir_ + "image_mapping.txt");
		std::ofstream out_normal(result_dir_ + "fwh_normal_after_RT.txt");
		std::ofstream out_normal_raw(result_dir_ + "fwh_normal_raw.txt");
		int n_count = 0;
		for (int i = 0; i < fwh_idx.size(); i++)
		{
			int idx = fwh_idx[i];
			int idx_guijie = gv2_idx[i];
			//生成gv2_roi的时候需要注释掉后面
			if (gv2_roi.count(idx_guijie))
			{
				float3E pos = fwh.pos_[idx];
				float3E normal = fwh.normal_[idx];
				normal.normalize();
				cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;	
				cvMatD in_camera_normal = r_matrix * (cvMatD(3, 1) << normal.x(), normal.y(), normal.z());
				//LOG(INFO) << "in_camera: " << in_camera_normal << std::endl;
				cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
				cvMatD in_camera_fix = in_camera;
				cvMatD in_image_fix = const_var_->ptr_data->proj_->intrisic_*in_camera_fix;
				cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
				dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
				cvMatD in_camera_reproj = r_matrix * in_3d_fix + tvec;
				cvMatD in_image_reproj = const_var_->ptr_data->proj_->intrisic_*in_camera_reproj;
				float img_x = GETD(in_image_reproj, 0, 0) / GETD(in_image_reproj, 2, 0);
				float img_y = GETD(in_image_reproj, 1, 0) / GETD(in_image_reproj, 2, 0);
				//green opt
				cv::circle(vis_fwh, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
				out_txt << gv2_idx[i] << "," << int(img_x) << "," << int(img_y) << ",";
				out_normal << GETD(in_camera_normal, 0, 0) << "," << GETD(in_camera_normal, 1, 0) << "," << GETD(in_camera_normal, 2, 0) << ",";
				out_normal_raw << normal.x() << "," << normal.y() << "," << normal.z() << ",";
				n_count++;
			}
		}
		LOG(INFO) << "n_count selected: " << n_count << std::endl;
		out_txt.close();
		out_normal.close();
		out_normal_raw.close();
		cv::imwrite(result_dir_ + "vis_fwh_move.png", vis_fwh);
	}
}

void RecMesh::adjustFace2d3d(const cv::Mat& img_256, const vecF& landmark_xy, const cvMatD& rvec, const cvMatD& tvec,
	const intVec& idx_68, MeshCompress& fwh, float3Vec& deform_pos, intVec& deform_idx)
{
	int n_num = landmark_xy.size()*0.5;
	cvMatD r_matrix;
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = img_256.clone();
	cv::Mat canvas_before_opt = img_256.clone();
	floatVec in_camera_z(n_num, 0);
	float3Vec in_3d(n_num, float3E(0, 0, 0));
	float3Vec fwh_slice_68;
	fwh.getSlice(idx_68, fwh_slice_68);
	double error_dis = 0;
	intSet used_list = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
						39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
						54,55,56,57,58,59 };
	intVec add_list = {17,18,19,20,21,22,23,24,25,26};
	used_list.insert(add_list.begin(), add_list.end());
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = fwh_slice_68[i];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);

		//red 3d
		cv::circle(canvas_before_opt, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas_before_opt, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);


		cvMatD in_image_from_xy = (cvMatD(3, 1) << landmark_xy[2 * i] * in_camera_z[i], landmark_xy[2 * i + 1] * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		in_3d[i] = pos;
		if (used_list.count(i))
		{
			error_dis += (float2E(img_x, img_y) - float2E(landmark_xy[2 * i], landmark_xy[2 * i + 1])).norm();
		}
	}
	LOG(INFO) << "error: " << 1.0f * error_dis / used_list.size();
	/////change uv coordinate
	fwh.tex_cor_.resize(fwh.n_vertex_);
	for (int i = 0; i < fwh.n_vertex_; i++)
	{
		float3E pos = fwh.pos_[i];
		//LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		//LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		//pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		//in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0)*1.0 / 255.0;
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green ope
		//cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
		if (img_x > 0 && img_x < 1 && img_y >0 && img_y < 1)
		{
			fwh.tex_cor_[i] = float2E(img_x, 1 - img_y);
		}
		else
		{
			fwh.tex_cor_[i] = float2E(0, 0);
		}
	}
	fwh.material_.push_back("img_256.mtl");
	fwh.tri_uv_ = fwh.tri_;
	fwh.update();
	float3Vec dst_pos;
	for (int i = 0; i < n_num; i++)
	{
		int idx = idx_68[i];
		float3E pos = fwh.pos_[idx];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - const_var_->ptr_data->proj_->cx_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - const_var_->ptr_data->proj_->cy_)*GETD(in_camera_fix, 2, 0) / const_var_->ptr_data->proj_->fy_);
		cvMatD in_image_fix = const_var_->ptr_data->proj_->intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
		cvMatD in_camera_reproj = r_matrix * in_3d_fix + tvec;
		cvMatD in_image_reproj = const_var_->ptr_data->proj_->intrisic_*in_camera_reproj;
		float img_x = GETD(in_image_reproj, 0, 0) / GETD(in_image_reproj, 2, 0);
		float img_y = GETD(in_image_reproj, 1, 0) / GETD(in_image_reproj, 2, 0);
		//green opt
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);
	}
	deform_pos = dst_pos;
	deform_idx = idx_68;

	if (is_debug_)
	{
		fwh.saveMtl(result_dir_ + "img_256.mtl", "input_landmark68");
		cv::FileStorage fs(result_dir_ + "test_reproject.yml", cv::FileStorage::WRITE);
		fs << "rvec" << rvec;
		fs << "tvec" << tvec;
		fs << "r_matrix" << r_matrix;
		fs.release();
		cv::imwrite(result_dir_ + "re_2d_3d.png", canvas);
		cv::imwrite(result_dir_ + "re_2d_3d_before.png", canvas_before_opt);
		//fwh_res.saveObj(result_dir_+ "fit_image.obj");
	}
	//deform
}

void RecMesh::getFwhCoefFromProjection(const MeshCompress& bfm, const cv::Mat& img_256, 
	const vecF& landmark_xy, vecF& coef_res)
{
	//get 68 landmarks for cropped image, skip for brow && inner mouth
	intSet used_list = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
	39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
		54,55,56,57,58,59 };

	int n_num = landmark_xy.size()*0.5;
	coef_res.resize(80);
	float3Vec bfm_slice_68;
	bfm.getSlice(const_var_->ptr_data->bfw_68_idx_, bfm_slice_68);
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	for (int i = 0; i < n_num; i++)
	{
		if (used_list.count(i) && i < 17)
		{
			img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
			float3E pos = bfm_slice_68[i];
			obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
		}
		else
		{
			for (int iter = 0; iter < 2; iter++)
			{
				img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
				float3E pos = bfm_slice_68[i];
				obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
			}
		}
	}
	cvMatD rvec, tvec, r_matrix;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
	cv::solvePnP(obj_points, img_points, const_var_->ptr_data->proj_->intrisic_,
		dist_coeffs, rvec, tvec);
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = img_256.clone();
	floatVec in_camera_z(n_num, 0);
	float3Vec in_3d(n_num, float3E(0, 0, 0));
	double z_shift = 0.0864024460;
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = bfm_slice_68[i];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);
#if 1
		cvMatD in_image_from_xy = (cvMatD(3, 1) << landmark_xy[2 * i] * in_camera_z[i], landmark_xy[2 * i + 1] * in_camera_z[i], 1* in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		in_3d[i] = float3E(GETD(in_world_3d, 0, 0), GETD(in_world_3d, 1, 0), GETD(in_world_3d, 2, 0) + z_shift);
		//LOG(INFO) << "in_camera_3d:" << in_camera_3d << std::endl;
		//LOG(INFO) << "in_world_3d:" << in_world_3d << std::endl;
		//LOG(INFO) << "pos 3d: " << pos << std::endl;
#else
		//used for testing
		cvMatD in_image_from_xy = (cvMatD(3, 1) << img_x * in_camera_z[i], img_y * in_camera_z[i], 1 * in_camera_z[i]);
		cvMatD in_camera_3d = const_var_->ptr_data->proj_->intrisic_.inv()*in_image_from_xy;
		cvMatD in_world_3d = r_matrix.t()*(in_camera_3d - tvec);
		LOG(INFO) << "in_camera_3d:" << in_camera_3d << std::endl;
		LOG(INFO) << "in_world_3d:" << in_world_3d << std::endl;
		LOG(INFO) << "pos 3d: " << pos << std::endl;
#endif
	}

#if 0
	//fit
	const Tensor& fwh_bfm_tensor = const_var_->ptr_data->fwh_bfm_tensor_;
	float weight = 0.0;
	floatVec reg(fwh_bfm_tensor.n_id_ - 1, weight);
	floatVec ev(fwh_bfm_tensor.n_id_ - 1, 10.0);
	for (int i = 0; i < ev.size(); i++)
	{
		//ev[i] = 10.0 / (i+1);
	}

	intVec temp_16 = const_var_->ptr_data->fwh_68_idx_;
	temp_16.resize(17);
	in_3d.resize(17);
	fwh_bfm_tensor.fitID(in_3d, reg, temp_16, ev, coef_res);
	LOG(INFO) << "out coef_res:" << coef_res.transpose() << std::endl;
	floatVec raw_fwh = fwh_bfm_tensor.interpretID(coef_res);
#else
	//fit
	const Tensor& fwh = const_var_->ptr_data->fwh_tensor_;
	float weight = 0.1;
	floatVec reg(fwh.n_id_ - 1, weight);
	fwh.fitID(in_3d, reg, const_var_->ptr_data->fwh_68_idx_, fwh.ev_data_, coef_res);
	LOG(INFO) << "out coef_res:" << coef_res.transpose() << std::endl;
	floatVec raw_fwh = fwh.interpretID(coef_res);
#endif

	for (int i = 0; i < n_num; i++)
	{
		int idx = const_var_->ptr_data->fwh_68_idx_[i];
		float3E pos = float3E(raw_fwh[3 * idx], raw_fwh[3 * idx + 1], raw_fwh[3 * idx + 2]);
		LOG(INFO) << "pos after opt: " << pos.transpose() << std::endl;
		LOG(INFO) << "pos before opt: " << in_3d[i].transpose() << std::endl;
		pos.z() = pos.z() - z_shift;
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		//LOG(INFO) << "in_camera: " << in_camera << std::endl;
		in_camera_z[i] = GETD(in_camera, 2, 0);
		cvMatD in_image = const_var_->ptr_data->proj_->intrisic_*in_camera;
		//LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		//LOG(INFO) << "img_x: " << img_x << std::endl;
		//LOG(INFO) << "img_y: " << img_y << std::endl;
		//green opt
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 255, 0), 2);		
	}

	cv::imwrite(result_dir_ + "2d_3d.png", canvas);
}
void RecMesh::get3dmmMesh(const vecF& coef_3dmm, MeshCompress& res)
{
	const Tensor& bfm_tensor = const_var_->ptr_data->bfm_tensor_;
	floatVec raw_3dmm = bfm_tensor.interpretID(coef_3dmm.head(80));
	res = bfm_tensor.template_obj_;
	SG::safeMemcpy(res.pos_.data(), raw_3dmm.data(), raw_3dmm.size() * sizeof(float));
}

void RecMesh::getPostProcessFor3dmmFit68(const vecF& coef_3dmm, MeshCompress& res, 
	const cv::Mat& img_256, const vecF& landmark_256)
{
	const Tensor& bfm_tensor = const_var_->ptr_data->bfm_tensor_;
	const Tensor& fwh_bfm_tensor = const_var_->ptr_data->fwh_bfm_tensor_;
	const MeshSysFinder& fwh_sys = const_var_->ptr_data->fwh_sys_finder_;

	float mag = 1.0;

	floatVec raw_3dmm = bfm_tensor.interpretID(mag * coef_3dmm.head(80));
	MeshCompress bfm_mesh = bfm_tensor.template_obj_;
	SG::safeMemcpy(bfm_mesh.pos_.data(), raw_3dmm.data(), raw_3dmm.size() * sizeof(float));
	if (is_debug_)
	{
		bfm_mesh.saveObj(result_dir_ + "3dmm.obj");
	}

	MeshCompress fwh_mesh = fwh_bfm_tensor.template_obj_;
	
	float3Vec dst_pos_full;
	intVec dst_idx_full;
	cvMatD rvec, tvec;
	getFwhCoefFromCeres3dmm(bfm_mesh, img_256, landmark_256, fwh_mesh, dst_pos_full, dst_idx_full, rvec, tvec);
	const MeshPostProcess& pp_data = const_var_->ptr_data->fwh_3dmm_;
	//need transform 3dmm to fwh
	//get translate
	float3Vec fwh_slice, bfm_slice;
	fwh_mesh.getSlice(pp_data.fwh_id_trans_, fwh_slice);
	bfm_mesh.getSlice(pp_data.bfm_id_trans_, bfm_slice);
	float3E translate;
	RT::getTranslate(bfm_slice, fwh_slice, translate);
	translate.x() = 0;
	RT::translateInPlace(translate, bfm_mesh.pos_);
	if (is_debug_)
	{
		bfm_mesh.saveObj(result_dir_ + "3dmm_trans.obj");
	}

	if (is_debug_)
	{
		fwh_mesh.saveObj(result_dir_ + "3dmm_fwh.obj");
	}

	double shift_x = 0;
	for (int i : pp_data.bfm_mid_)
	{
		shift_x += bfm_mesh.pos_[i].x();
	}
	shift_x = shift_x / (1.0*pp_data.bfm_mid_.size());
	LaplacianDeform to_bfm;
	to_bfm.init(fwh_mesh, pp_data.fwh_deform_, pp_data.fwh_fix_, pp_data.mouth_close_);
	float3Vec deform_pos(pp_data.fwh_deform_.size());
	for (int i = 0; i < pp_data.fwh_deform_.size(); i++)
	{
		int idx_fwh = pp_data.fwh_deform_[i];
		int idx_bfm = pp_data.bfm_deform_[i];
		if (idx_bfm > 0)
		{
			deform_pos[i] = bfm_mesh.pos_[idx_bfm];
		}
		else
		{
			float3E right = bfm_mesh.pos_[-idx_bfm];
			float3E left = right;
			left.x() = 2 * shift_x - right.x();
			deform_pos[i] = left;
		}
	}
	res = fwh_mesh;
	to_bfm.deform(deform_pos, res.pos_);
	if (is_debug_)
	{
		res.saveObj(result_dir_ + "bfm_lap.obj");
	}

	fwh_sys.getSysPosLapInPlace(res.pos_);
	if (is_debug_)
	{
		res.saveObj(result_dir_ + "bfm_lap_sys_pre.obj");
	}
	
	intVec used_vec = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
			39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
	54,55,56,57,58,59,61,62,63,65,66,67 };
	const_var_->ptr_data->proj_->getMeshToImageRT(res, const_var_->ptr_data->fwh_68_idx_, used_vec,
		landmark_256, rvec, tvec);
	adjustFace2d3d(img_256, landmark_256, rvec, tvec, res, dst_pos_full, dst_idx_full);

	if (is_debug_)
	{
		res.saveObj(result_dir_ + "bfm_lap_sys_adjust_uv.obj");
	}	
	//currently not useful for now. effect for landmark not ready
	LaplacianDeform to_image;
	intVec move_pos;
	intSet select_idx;
	if (res_var_->pp_type_ == PostProcessRoutine::FIT_CONTOUR)
	{
		move_pos = const_var_->ptr_data->fit_idx_contour_;
		select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15 };
	}
	else if (res_var_->pp_type_ == PostProcessRoutine::FIT_FACE)
	{
		move_pos = const_var_->ptr_data->fit_idx_face_;
		select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15,27,28,29,30,31,32,33,34,35,
						36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
						57,58,59};
	}
	else if (res_var_->pp_type_ == PostProcessRoutine::TEX_BASE)
	{
		move_pos = const_var_->ptr_data->fit_idx_face_;
		select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15,27,28,29,30,31,32,33,34,35,
						36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
						57,58,59 };
	}
	else
	{
		LOG(ERROR) << "pp_type_ not defined." << std::endl;
	}
	intVec fix_pos = res.getReverseSelection(move_pos);
	float3Vec dst_pos;
	intVec dst_idx;
	for (int i : select_idx)
	{
		dst_pos.push_back(dst_pos_full[i]);
		dst_idx.push_back(dst_idx_full[i]);
	}
	for (int i = 0; i < dst_idx.size(); i++)
	{
		dst_pos[i].z() = res.pos_[dst_idx[i]].z();
	}
	
	to_image.init(res, dst_idx, fix_pos, pp_data.mouth_close_);
	to_image.deform(dst_pos, res.pos_);
	adjustFace2d3d(img_256, landmark_256, rvec, tvec, res, dst_pos_full, dst_idx_full);
	if (is_debug_)
	{
		res.saveObj(result_dir_ + "fit_image.obj");
	}

	fwh_sys.getSysPosLapInPlace(res.pos_);
	if (is_debug_)
	{
		//LOG(WARNING) << "should use bfm_lap_sys_fit.obj, but add fit instead." << std::endl;
		res.saveObj(result_dir_ + "bfm_lap_sys.obj");
	}
 }

 void RecMesh::getPostProcessFor3dmmTexBase(const vecF& coef_3dmm, MeshCompress& res,
	 const cv::Mat& img_256, const vecF& landmark_256)
 {
	 const Tensor& bfm_tensor = const_var_->ptr_data->bfm_tensor_all_;
	 const Tensor& fwh_bfm_tensor = const_var_->ptr_data->fwh_bfm_tensor_;
	 const MeshSysFinder& fwh_sys = const_var_->ptr_data->fwh_sys_finder_;

	 float mag = 1.0;

	 floatVec raw_3dmm = bfm_tensor.interpretID(mag * coef_3dmm.head(80));
	 MeshCompress bfm_mesh = bfm_tensor.template_obj_;
	 SG::safeMemcpy(bfm_mesh.pos_.data(), raw_3dmm.data(), raw_3dmm.size() * sizeof(float));
	 if (is_debug_)
	 {
		 bfm_mesh.saveObj(result_dir_ + "3dmm.obj");
	 }

	 float3Vec dst_pos_full, dst_pos;
	 intVec dst_idx_full;
	 cvMatD rvec, tvec;
	 intVec used_landmark = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
		 39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
		54,55,56,57,58,59,61,62,63,65,66,67 };
	 const_var_->ptr_data->proj_->getMeshToImageRT(bfm_mesh, const_var_->ptr_data->bfw_68_all_idx_, used_landmark,
		 landmark_256, rvec, tvec);
	
	 if (is_debug_)
	 {
		 cv::FileStorage fs(result_dir_ + "test.yml", cv::FileStorage::WRITE);
		 fs << "rvec" << rvec;
		 fs << "tvec" << tvec;
		 fs.release();
		 //cv::imwrite(result_dir_ + "2d_3d.png", canvas);
		 //fwh_res.saveObj(result_dir_+ "fit_image.obj");
	 }
	 
	 res = bfm_mesh;
	 adjustFace2d3d(img_256, landmark_256, rvec, tvec, const_var_->ptr_data->bfw_68_all_idx_, 
		 bfm_mesh, dst_pos_full, dst_idx_full);
	 
	 intVec used_vec = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
		 39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
 54,55,56,57,58,59,61,62,63,65,66,67 };
	 intVec dst_idx;
	 for (int i : used_vec)
	 {
		 dst_pos.push_back(dst_pos_full[i]);
		 dst_idx.push_back(dst_idx_full[i]);
	 }

	 LaplacianDeform to_bfm;
	 to_bfm.init(res, dst_idx, {0});
	 
	 LaplacianDeform to_image;
	 to_image.init(res, dst_idx, { 0 });
	 to_image.deform(dst_pos, res.pos_);
	 adjustFace2d3d(img_256, landmark_256, rvec, tvec, const_var_->ptr_data->bfw_68_all_idx_,
		 res, dst_pos_full, dst_idx_full);
	 if (is_debug_)
	 {
		 res.saveObj(result_dir_ + "fit_image.obj");
	 }
 }

 void RecMesh::fitAWithLandmarksToImage(const MeshCompress& A, const cv::Mat& img_256,
	 const vecF& landmark_256, const intVec& land_68, const intVec& movable, 
	 const intVec& pair, MeshCompress& res)
 {
	 MeshSysFinder guijie_sys;
	 JsonHelper::initData("D:/code/expgen_aquila/data/exp_server_config/guijie_sys_tensor/", "config.json", guijie_sys);	 
	 float3Vec dst_pos_full;
	 intVec dst_idx_full;
	 res = A;
	 cvMatD rvec, tvec;
	 //intVec used_vec = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38, 39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
	 //54,55,56,57,58,59,61,62,63,65,66,67 };
	 intVec used_vec = { 36,37,38, 39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,
	 54,55,56,57,58,59,61,62,63,65,66,67 };
	 intVec add_list = { 17,18,19,20,21,22,23,24,25,26 };
	 used_vec.insert(used_vec.end(), add_list.begin(), add_list.end());
	 const_var_->ptr_data->proj_->getMeshToImageRT(res, land_68, used_vec, landmark_256, rvec, tvec);
	 adjustFace2d3d(img_256, landmark_256, rvec, tvec, land_68, res, dst_pos_full, dst_idx_full);

	 if (is_debug_)
	 {
		 res.saveObj(result_dir_ + "bfm_lap_sys_adjust_uv.obj");
	 }
	 //currently not useful for now. effect for landmark not ready
	 LaplacianDeform to_image;
	 intVec move_pos;
	 intSet select_idx;
	 if (res_var_->pp_type_ == PostProcessRoutine::FIT_CONTOUR)
	 {
		 move_pos = movable;
		 select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15 };
	 }
	 else if (res_var_->pp_type_ == PostProcessRoutine::FIT_FACE)
	 {
		 move_pos = movable;
		 select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15,27,28,29,30,31,32,33,34,35,
						 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
						 57,58,59 };
	 }
	 else if (res_var_->pp_type_ == PostProcessRoutine::TEX_BASE)
	 {
		 move_pos = movable;
		 select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15,27,28,29,30,31,32,33,34,35,
						 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
						 57,58,59 };
	 }
	 else
	 {
		 LOG(ERROR) << "pp_type_ not defined." << std::endl;
	 }
	 intVec fix_pos = res.getReverseSelection(move_pos);
	 float3Vec dst_pos;
	 intVec dst_idx;
	 for (int i : select_idx)
	 {
		 dst_pos.push_back(dst_pos_full[i]);
		 dst_idx.push_back(dst_idx_full[i]);
	 }
	 for (int i = 0; i < dst_idx.size(); i++)
	 {
		 dst_pos[i].z() = res.pos_[dst_idx[i]].z();
	 }

	 to_image.init(res, dst_idx, fix_pos, pair);
	 to_image.deform(dst_pos, res.pos_);
	 adjustFace2d3d(img_256, landmark_256, rvec, tvec, land_68, res, dst_pos_full, dst_idx_full);
	 if (is_debug_)
	 {
		 res.saveObj(result_dir_ + "fit_image.obj");
	 }

	 guijie_sys.getSysPosLapInPlace(res.pos_);
	 auto ori_uv = A;
	 ori_uv.pos_ = res.pos_;
	 if (is_debug_)
	 {
		 res.saveObj(result_dir_ + "bfm_lap_sys.obj");
		 ori_uv.saveObj(result_dir_ + "isv_pos_ref.obj");
	 }


 }
