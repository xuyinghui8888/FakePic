#include "Test.h"
#include "../Basic/MeshHeader.h"
#include "../VisHelper/VisHeader.h"
#include "../NRICP/register.h"
#include "../NRICP/demo.h"
#include "../RigidAlign/icp.h"
#include "../Config/Tensor.h"
#include "../Config/TensorHelper.h"
#include "../Config/JsonHelper.h"
#include "../Sysmetric/Sysmetric.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Metric/Metric.h"
#include "../MNN/MnnModel.h"
#include "../MNN/LandmarkTracking.h"
#include "../MNN/FeatureExtract.h"
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

using namespace CGP;

void TESTFUNCTION::getDataFromGuijieV3Pack()
{
	//test
	cstr root = "D:/data/server_pack/guijie_star/";
	cstr root_out = "D:/dota201010/1020_ratio/";
	//test for avatar generation v3	
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt3_data/";
	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(exp_config));
	SG::needPath(root_out);
	//while (true)
	{
		for (int i = 0; i < 23; i++)
		{
			cstr file_id = std::to_string(i);
			MeshCompress opt_type;
			optV3_ptr->getGuijieMesh(optV3_ptr->getStarInfo().bs_[i], opt_type);

			//SG::needPath(root_out + file_id[i] + "_res");

			opt_type.saveObj(root_out + file_id + "_opt.obj");
			optV3_ptr->dumpEyelash(opt_type);
			opt_type.saveObj(root_out + file_id + "_opt_no_lash.obj");

			MeshCompress match_att_bfm;
			vecD match_3dmm;
			FILEIO::loadEigenMat("D:/avatar/guijie_opt3_data/guijie_star/" + std::to_string(i) + "_3dmm.txt", 80, 1, match_3dmm);
			optV3_ptr->getBFMMesh(match_3dmm, match_att_bfm);
			match_att_bfm.saveObj(root_out + file_id + "_3dmm.obj");
		}
	}
}

void TESTFUNCTION::prepareForGuijie35()
{
	cstr obj_in = "D:/dota201010/1020_ratio/";
	cstr obj_out = "D:/dota201010/1026_calc/";
	SG::needPath(obj_out);
	intVec guijie_68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/guijie_68_sys.txt");
	intVec bfm_68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/part_68_refine.txt");
	floatVec scale_vec(23, 0);
#pragma omp parallel for
	for (int i = 0; i < 23; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress bfm(obj_in + fileid + "_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + "_opt.obj");
		MeshCompress guijie_move = guijie;
		double scale;
		float3E translate;
		MeshTools::putSrcToDst(guijie, guijie_68, bfm, bfm_68, guijie_move, scale, translate);
		scale_vec[i] = scale;
		guijie_move.saveObj(obj_out + fileid + "_opt.obj");
		bfm.saveObj(obj_out + fileid + "_3dmm.obj");
	}
	DebugTools::cgPrint(scale_vec);

	double scale_avg = CalcHelper::averageValue(scale_vec);
	double mid_avg = CalcHelper::midPosAverage(scale_vec, 0.2);


	LOG(INFO) << "avg: " << scale_avg << std::endl;
	LOG(INFO)<<"avg: "<< mid_avg <<std::endl;
	//use fixed size

#pragma omp parallel for
	for (int i = 0; i < 23; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress bfm(obj_in + fileid + "_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + "_opt.obj");
		MeshCompress guijie_move = guijie;
		MeshTools::putSrcToDstFixScale(guijie, guijie_68, bfm, bfm_68, guijie_move, mid_avg);
		guijie_move.saveObj(obj_out + fileid + "_opt.obj");
		bfm.saveObj(obj_out + fileid + "_3dmm.obj");
	}	
}

void TESTFUNCTION::prepareForTaobao35()
{
	cstr obj_in = "D:/dota201116/1116_taobao/taobao/";
	cstr obj_out = "D:/dota201116/1116_taobao/rec/";
	SG::needPath(obj_out);
	intVec taobao_68 = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/taobao_68_sys.txt");
	intVec bfm_68 = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/bfm_sys_adjust.txt");
	int n_sample = 11;
	floatVec scale_vec(n_sample, 0);
#pragma omp parallel for
	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress bfm(obj_in + fileid + "_0_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + ".obj");
		MeshCompress guijie_move = guijie;
		double scale;
		float3E translate;
		MeshTools::putSrcToDst(guijie, taobao_68, bfm, bfm_68, guijie_move, scale, translate);
		scale_vec[i] = scale;
		guijie_move.saveObj(obj_out + fileid + ".obj");
		bfm.saveObj(obj_out + fileid + "_3dmm.obj");
	}
	DebugTools::cgPrint(scale_vec);

	double scale_avg = CalcHelper::averageValue(scale_vec);
	double mid_avg = CalcHelper::midPosAverage(scale_vec, 0.2);

	LOG(INFO) << "avg: " << scale_avg << std::endl;
	LOG(INFO) << "avg: " << mid_avg << std::endl;
	//use fixed size

#pragma omp parallel for
	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress bfm(obj_in + fileid + "_0_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + ".obj");
		MeshCompress guijie_move = guijie;
		MeshTools::putSrcToDstFixScale(guijie, taobao_68, bfm, bfm_68, guijie_move, mid_avg);
		guijie_move.saveObj(obj_out + fileid + "_opt.obj");
		bfm.saveObj(obj_out + fileid + "_3dmm.obj");
	}
}

void TESTFUNCTION::subtractMeshDataTaobao()
{
	cstr obj_in = "D:/dota201116/1116_taobao/rec/";
	cstr obj_out = "D:/dota201116/1116_taobao/subtract/";
	SG::needPath(obj_out);
	int n_sample = 11;
	intVec taobao_68 = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/taobao_68_sys.txt");
	intVec bfm_68 = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/bfm_sys_adjust.txt");

	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress taobao(obj_in + fileid + "_3dmm.obj");
		MeshCompress q(obj_in + fileid + "_opt.obj");
		float3Vec bfm_order, guijie_order;
		taobao.getSlice(bfm_68, bfm_order);
		q.getSlice(taobao_68, guijie_order);
		for (auto iter_land : taobao_68)
		{
			LOG(INFO) << iter_land << ", " << q.pos_[iter_land].transpose() << std::endl;
		}

		taobao.keepRoi(bfm_68);
		q.keepRoi(taobao_68);
		taobao.pos_ = bfm_order;
		q.pos_ = guijie_order;
		taobao.saveObj(obj_out + fileid + "_3dmm.obj");
		q.saveObj(obj_out + fileid + "_opt.obj");
		cv::Mat vis = DebugTools::projectVertexToXY(
			{ taobao.pos_, q.pos_ },
			{ cv::Scalar(255.0,0.0,0.0), cv::Scalar(0.0,255.0,0.0) },
			512, 512
		);
		cv::imwrite(obj_out + fileid + ".png", vis);
	}
}

void TESTFUNCTION::getRatioMeshBasedOn3dmm()
{
	cstr obj_in = "D:/dota201116/1116_taobao/rec/";
	cstr obj_out = "D:/dota201116/1116_taobao/stretch/";
	SG::needPath(obj_out);
	int n_sample = 11;
	intVec taobao_68 = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/taobao_68_sys.txt");
	intVec bfm_68 = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/bfm_sys_adjust.txt");

	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress taobao(obj_in + fileid + "_3dmm.obj");
		MeshCompress q(obj_in + fileid + "_opt.obj");
		float3Vec bfm_order, guijie_order;
		taobao.getSlice(bfm_68, bfm_order);
		q.getSlice(taobao_68, guijie_order);
		for (auto iter_land : taobao_68)
		{
			LOG(INFO) << iter_land << ", " << q.pos_[iter_land].transpose() << std::endl;
		}
		for (int j = 0; j < 11; j++)
		{
			double scale = 0.1*j;
			float3Vec inter_bfm_order = bfm_order;
			for (int k = 0; k < inter_bfm_order.size(); k++)
			{
				inter_bfm_order[k] = (1 - scale)*bfm_order[k] + scale * guijie_order[k];
			}
			//get for scale dist
	
			LaplacianDeform q_ratio;
			q_ratio.init(q, taobao_68, { 0 });
			MeshCompress res = q;
			MeshCompress q_res = q;
			q_ratio.deform(inter_bfm_order, q_res.pos_);
			q_res.saveObj(obj_out + fileid + "_" + std::to_string(scale) + "_deform.obj");
			MeshCompress q_move = q_res;
			double q_scale = 0;
			float3E translate;
			MeshTools::putSrcToDst(q_res, taobao_68, taobao, bfm_68, q_move, q_scale, translate);
			q_move.saveObj(obj_out + fileid + "_" + std::to_string(scale) + "_move.obj");
		}
	}
}

void TESTFUNCTION::subtractMeshData()
{
	cstr obj_in = "D:/dota201010/1026_calc/";
	cstr obj_out = "D:/dota201010/1027_subdata/";
	SG::needPath(obj_out);
	int n_sample = 23;
	intVec guijie_68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/guijie_68_sys.txt");
	//intVec bfm_68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/part_68_refine.txt");
	intVec bfm_68 = FILEIO::loadIntDynamic("D:/dota201010/1027_bfm_part/bfm_sys_adjust.txt");

	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress bfm(obj_in + fileid + "_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + "_opt.obj");
		float3Vec bfm_order, guijie_order;
		bfm.getSlice(bfm_68, bfm_order);
		guijie.getSlice(guijie_68, guijie_order);
		for (auto iter_land : guijie_68)
		{
			LOG(INFO) << iter_land << ", " << guijie.pos_[iter_land].transpose() << std::endl;
		}

		bfm.keepRoi(bfm_68);
		guijie.keepRoi(guijie_68);
		bfm.pos_ = bfm_order;
		guijie.pos_ = guijie_order;
		bfm.saveObj(obj_out + fileid + "_3dmm.obj");
		guijie.saveObj(obj_out + fileid + "_opt.obj");
		cv::Mat vis = DebugTools::projectVertexToXY(
			{ bfm.pos_, guijie.pos_ },
			{ cv::Scalar(255.0,0.0,0.0), cv::Scalar(0.0,255.0,0.0)},
			512, 512
			);
		cv::imwrite(obj_out + fileid + ".png", vis);
	}
}

void TESTFUNCTION::calcHardRatio(const cstr& img_in, const cstr& img_ext, const cstr& obj_in,
	const cstr& result_dir, int n_sample)
{
	//cstr img_in = "D:/dota201010/1026_calc/";
	//cstr obj_in = "D:/dota201010/1027_subdata/";
	cstr data_root = "D:/data/server_pack/";
	//cstr result_dir = "D:/dota201010/1026_landmark/";
	SG::needPath(result_dir);
	SimLandmarkWrapper mnn_tool(data_root, result_dir, true);

	//mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59,60};
	//1019version
	//mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59 };
	intVec left_eye_region = { 42,43,44,45,46,47 };
	intVec right_eye_region = { 36, 37,38,39,40,41 };
	intVec nose_region = { 27, 28,29,30,31,32,33,34,35 };
	intVec mouth_region = { 48,49,50,51,52,53,54,55,56,57,58,59, 61,62,63,65,66,67 };
	intVec face_region = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };	

	float3Vec v_left_eye_scale(n_sample), v_right_eye_scale(n_sample), v_nose_scale(n_sample), v_mouth_scale(n_sample), v_face_scale(n_sample);
	floatXVec v_bfm_x_ting(n_sample), v_bfm_y_ting(n_sample), v_guijie_x_ting(n_sample), v_guijie_y_ting(n_sample);
//mnn 不支持并行
//#pragma omp parallel for
	for (int i = 0; i < n_sample; i++)
	{		
		cstr fileid = std::to_string(i);
		cv::Mat image = cv::imread(img_in + fileid + img_ext);
		vecF land68;
		mnn_tool.result_dir_ = result_dir + std::to_string(i) + "_";
		mnn_tool.getLandmark68(image, land68);
		MeshCompress bfm(obj_in + fileid + "_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + "_opt.obj");
		float3Vec bfm_68_pos = bfm.pos_;
		float3Vec guijie_68_pos = guijie.pos_;
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, left_eye_region, v_left_eye_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, right_eye_region, v_right_eye_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, nose_region, v_nose_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, mouth_region, v_mouth_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, face_region, v_face_scale[i]);
		Beauty35::calculate35BasedOn68(bfm_68_pos, v_bfm_x_ting[i], v_bfm_y_ting[i]);
		Beauty35::calculate35BasedOn68(guijie_68_pos, v_guijie_x_ting[i], v_guijie_y_ting[i]);
	}
	float3E left_eye_scale, right_eye_scale, nose_scale, mouth_scale, face_scale;
	CalcHelper::getMidAverageMulti(v_left_eye_scale, 0.2, left_eye_scale);
	CalcHelper::getMidAverageMulti(v_right_eye_scale, 0.2, right_eye_scale);
	CalcHelper::getMidAverageMulti(v_nose_scale, 0.2, nose_scale);
	CalcHelper::getMidAverageMulti(v_mouth_scale, 0.2, mouth_scale);
	CalcHelper::getMidAverageMulti(v_face_scale, 0.2, face_scale);
	vecF avg_bfm_x_ting, avg_bfm_y_ting, avg_guijie_x_ting, avg_guijie_y_ting;
	CalcHelper::getEigenAverage(v_bfm_x_ting, avg_bfm_x_ting);
	CalcHelper::getEigenAverage(v_bfm_y_ting, avg_bfm_y_ting);
	CalcHelper::getEigenAverage(v_guijie_x_ting, avg_guijie_x_ting);
	CalcHelper::getEigenAverage(v_guijie_y_ting, avg_guijie_y_ting);
	LOG(INFO) << "left_eye_scale: " << left_eye_scale.transpose() << std::endl;
	LOG(INFO) << "right_eye_scale: " << right_eye_scale.transpose() << std::endl;
	LOG(INFO) << "nose_scale: " << nose_scale.transpose() << std::endl;
	LOG(INFO) << "mouth_scale: " << mouth_scale.transpose() << std::endl;
	LOG(INFO) << "face_scale: " << face_scale.transpose() << std::endl;

	LOG(INFO) << "avg_bfm_x_ting: " << avg_bfm_x_ting.transpose() << std::endl;
	LOG(INFO) << "avg_bfm_y_ting: " << avg_bfm_y_ting.transpose() << std::endl;
	LOG(INFO) << "avg_guijie_x_ting: " << avg_guijie_x_ting.transpose() << std::endl;
	LOG(INFO) << "avg_guijie_y_ting: " << avg_guijie_y_ting.transpose() << std::endl;

}

void TESTFUNCTION::calcTopRatio()
{
	cstr img_in = "D:/dota201010/1026_calc/";
	cstr obj_in = "D:/dota201010/1027_subdata/";
	cstr data_root = "D:/data/server_pack/";
	cstr result_dir = "D:/dota201010/1026_landmark/";
	SG::needPath(result_dir);
	SimLandmarkWrapper mnn_tool(data_root, result_dir, true);

	//mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59,60};
	//1019version
	//mouth_region_ = { 48,49,50,51,52,53,54,55,56,57,58,59 };
	intVec left_eye_region = { 42,43,44,45,46,47 };
	intVec right_eye_region = { 36, 37,38,39,40,41 };
	intVec nose_region = { 27, 28,29,30,31,32,33,34,35 };
	intVec mouth_region = { 48,49,50,51,52,53,54,55,56,57,58,59, 61,62,63,65,66,67 };
	intVec face_region = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

	int n_sample = 23;
	float3Vec v_left_eye_scale(n_sample), v_right_eye_scale(n_sample), v_nose_scale(n_sample), v_mouth_scale(n_sample), v_face_scale(n_sample);
	//mnn 不支持并行
	//#pragma omp parallel for
	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		cv::Mat image = cv::imread(img_in + fileid + ".png");
		vecF land68;
		mnn_tool.result_dir_ = result_dir + std::to_string(i) + "_";
		mnn_tool.getLandmark68(image, land68);
		MeshCompress bfm(obj_in + fileid + "_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + "_opt.obj");
		float3Vec bfm_68_pos = bfm.pos_;
		float3Vec guijie_68_pos = guijie.pos_;
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, left_eye_region, v_left_eye_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, right_eye_region, v_right_eye_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, nose_region, v_nose_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, mouth_region, v_mouth_scale[i]);
		MeshTools::getScale3Dim(guijie_68_pos, bfm_68_pos, face_region, v_face_scale[i]);
		vecF x_bfm_68, y_bfm_68, x_land68, y_land68;
		Beauty35::calculate35BasedOn68(bfm_68_pos, x_bfm_68, y_bfm_68);
		Beauty35::calculate35BasedOn68(land68, x_land68, y_land68);
	}
	float3E left_eye_scale, right_eye_scale, nose_scale, mouth_scale, face_scale;
	CalcHelper::getMidAverageMulti(v_left_eye_scale, 0.2, left_eye_scale);
	CalcHelper::getMidAverageMulti(v_right_eye_scale, 0.2, right_eye_scale);
	CalcHelper::getMidAverageMulti(v_nose_scale, 0.2, nose_scale);
	CalcHelper::getMidAverageMulti(v_mouth_scale, 0.2, mouth_scale);
	CalcHelper::getMidAverageMulti(v_face_scale, 0.2, face_scale);

}

void TESTFUNCTION::blendTop()
{
	cstr obj_in = "D:/dota201116/1130_retop/";
	cstr result_dir = "D:/dota201116/1130_retop/res/";
	SG::needPath(result_dir);
	MeshCompress obj_0 = obj_in + "mean_0820.obj";
	MeshCompress obj_1 = obj_in + "guijie_v1.obj";
	int n_num = 10;
	float add = 1.0 / n_num;
	for (int i = 0; i < n_num+1; i++)
	{
		float weight = add * i;
		MeshCompress res = obj_0;
		for (int iter_vertex = 0; iter_vertex < res.n_vertex_; iter_vertex++)
		{
			res.pos_[iter_vertex] = weight * obj_0.pos_[iter_vertex] + (1 - weight)*obj_1.pos_[iter_vertex];
		}
		res.saveObj(result_dir + std::to_string(weight) + ".obj");
	}

}

void TESTFUNCTION::testCalculate()
{
	//intVec test(10, 0);
	//std::iota(test.begin(), test.end(), 0);
	//intVec dst = CalcHelper::removeWithoutHeadTail(test, 0.2);
	//DebugTools::cgPrint(dst);
	//LOG(INFO) << "average: " << CalcHelper::averageValue(dst) << std::endl;
	cstr test_data = "D:/dota201010/1026_summary/test_data.txt";
	matF raw_scale;
	FILEIO::loadEigenMat(test_data, 23, 15, raw_scale);
	for (int x = 0; x < 15; x++)
	{
		floatVec col_vec(raw_scale.col(x).data(), raw_scale.col(x).data() + 23);
		//LOG(INFO) << "average: " << CalcHelper::midPosAverage(col_vec, 0) << std::endl;
		std::cout<<CalcHelper::midPosAverage(col_vec, 0.2) << ",";
	}
}

void TESTFUNCTION::movePartVertexForGuijie()
{
	cstr ref_part = "D:/dota201116/1116_taobao/df/deform_0.700000.obj";
	cstr src_obj = "D:/dota201116/1116_taobao/guijie/0_opt.obj";
	intVec mapping = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/guijie/mapping.txt");
	MeshCompress guijie_all(src_obj);
	MeshCompress guijie_ref_part(ref_part);
	MeshCompress guijie_ref_all = guijie_all;
	//mapping 
	intVec mapping_move_idx = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/guijie/move_idx.txt");
	intVec mapping_move_idx_s = FILEIO::loadIntDynamic("D:/dota201116/1116_taobao/guijie/move_idx_s.txt");
	intSet mapping_move_set(mapping_move_idx.begin(), mapping_move_idx.end());
	intSet mapping_move_set_s(mapping_move_idx_s.begin(), mapping_move_idx_s.end());

	intVec rt_idx_all, rt_idx_part;

	for (int i = 0; i < mapping.size(); i++)
	{
		int all_idx = i;
		int part_idx = mapping[all_idx];
		if (part_idx >= 0)
		{
			guijie_ref_all.pos_[all_idx] = guijie_ref_part.pos_[part_idx];
		}

		if (part_idx >= 0 && mapping_move_set.count(part_idx) && !mapping_move_set_s.count(part_idx))
		{
			rt_idx_all.push_back(all_idx);
			rt_idx_part.push_back(part_idx);
		}
	}

	//calculate 
	MeshCompress guijie_ref_part_scale = guijie_ref_part;
	MeshTools::putSrcToDst(guijie_ref_part, rt_idx_part, guijie_all, rt_idx_all, guijie_ref_part_scale);
	guijie_ref_part_scale.saveObj("D:/dota201116/1116_taobao/guijie/guijie_from_dst_scale.obj");

	guijie_ref_all.saveObj("D:/dota201116/1116_taobao/guijie/guijie_from_dst.obj");

	intVec fix_idx;
	intVec move_idx;
	float3Vec move_pos;
	for (int i = 0; i < mapping.size(); i++)
	{
		int all_idx = i;
		int part_idx = mapping[all_idx];
		if (part_idx >= 0 && !mapping_move_set.count(part_idx))
		{
			fix_idx.push_back(all_idx);
		}

		if (part_idx >= 0 && mapping_move_set.count(part_idx))
		{
			move_idx.push_back(all_idx);
			move_pos.push_back(guijie_ref_part_scale.pos_[part_idx]);
		}
	}	

	//move part model
	MeshCompress guijie_lap = guijie_all;
	LaplacianDeform q_ratio;
	q_ratio.init(guijie_lap, move_idx, fix_idx);	
	q_ratio.deform(move_pos, guijie_lap.pos_);
	guijie_lap.saveObj("D:/dota201116/1116_taobao/guijie/lap_stretch.obj");	
}

void TESTFUNCTION::changeOfRatio()
{
	cstr obj_in = "D:/dota201010/1027_subdata/";
	cstr obj_out = "D:/dota201010/1029_transform/";
	SG::needPath(obj_out);
	intVec left_eye_region = { 42,43,44,45,46,47 };
	intVec right_eye_region = { 36, 37,38,39,40,41 };
	intVec nose_region = { 27, 28,29,30,31,32,33,34,35 };
	intVec mouth_region = { 48,49,50,51,52,53,54,55,56,57,58,59, 61,62,63,65,66,67 };
	intVec face_region = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
	int n_sample = 23;
	float3E left_eye_scale = float3E(0.514537, 0.285134, 0.57191);
	float3E right_eye_scale = float3E(0.504435, 0.290754, 0.62284);
	float3E nose_scale = float3E(0.92548, 1.06958, 1.21885);
	float3E mouth_scale = float3E(0.972104, 1.23458, 1.52146);
	float3E face_scale = float3E(0.741417,	1.08814,	1.13339	);

	for (int i = 0; i < n_sample; i++)
	{
		cstr fileid = std::to_string(i);
		MeshCompress bfm(obj_in + fileid + "_3dmm.obj");
		MeshCompress guijie(obj_in + fileid + "_opt.obj");
		cv::Mat vis = DebugTools::projectVertexToXY(
			{ bfm.pos_, guijie.pos_ },
			{ cv::Scalar(255.0,0.0,0.0), cv::Scalar(0.0,255.0,0.0) },
			512, 512);
		cv::imwrite(obj_out + fileid + ".png", vis);
		//scale 
		RT::scaleInPlace(left_eye_scale, left_eye_region, guijie.pos_);
		RT::scaleInPlace(right_eye_scale, right_eye_region, guijie.pos_);
		RT::scaleInPlace(nose_scale, nose_region, guijie.pos_);
		RT::scaleInPlace(mouth_scale, mouth_region, guijie.pos_);
		RT::scaleInPlace(face_scale, face_region, guijie.pos_);
		
		//move eye pos
		float3E right_eye_center = guijie.pos_[0] * 0.5 + guijie.pos_[27] * 0.5;
		float3E left_eye_center = guijie.pos_[16] * 0.5 + guijie.pos_[27] * 0.5;
		RT::translateRoiCenterInPlace(right_eye_center, right_eye_region, guijie.pos_);
		RT::translateRoiCenterInPlace(left_eye_center, left_eye_region, guijie.pos_);
		cv::Mat vis_transform = DebugTools::projectVertexToXY(
			{ bfm.pos_, guijie.pos_ },
			{ cv::Scalar(255.0,0.0,0.0), cv::Scalar(0.0,255.0,0.0) },
			512, 512);
		cv::imwrite(obj_out + fileid + "_trans.png", vis_transform);
		
		//calculate diff
		//int n_point = guijie.pos_.size();
		int n_point = 68;
		cstrVec tag(n_point*3*2, "test");
		
		for (int i = 0; i < n_point*3; i++)
		{
			tag[i] = "bfm_" + std::to_string(i);
		}

		for (int i = 0; i < n_point * 3; i++)
		{
			tag[i+ n_point * 3] = "diff_" + std::to_string(i);
		}

		floatVec delta(n_point*2*3);
		//delta前面保存bfm_pos, 后面存储delta值
		int count = 0;
		for (int iter_pos = 0; iter_pos < n_point; iter_pos++)
		{
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				delta[iter_pos * 3 + iter_dim] = bfm.pos_[iter_pos][iter_dim];
			}
		}

		int shift = n_point * 3;
		for (int iter_pos = 0; iter_pos < n_point; iter_pos++)
		{
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				delta[iter_pos * 3 + iter_dim + shift] = guijie.pos_[iter_pos][iter_dim]- bfm.pos_[iter_pos][iter_dim];
			}
		}
		//暂时不需要进行放缩，3dmm出来的位置在-1/1之间
		//auto delta_norm = CalcHelper::transformValue(delta, 2.0 / 511.0, -1);
		auto expand_delta = CalcHelper::expandVector(delta, 1000);
		FILEIO::saveVecToCsv(obj_out + "test_vec.csv", expand_delta, n_point*3*2, tag);
	}
}

void TESTFUNCTION::getDiscardMappingFromMeshes()
{
	MeshCompress whole("D:/data/server_pack/deep_3d_bfm_09/bfm_mean.obj");
	intVec eye_roi = FILEIO::loadIntDynamic("D:/data/server_pack/deep_3d_bfm_09/eye.txt");
	intVec mapping = whole.discard(eye_roi);
	FILEIO::saveDynamic("D:/data/server_pack/deep_3d_bfm_09/mapping.txt", mapping, ",");
	whole.saveObj("D:/data/server_pack/deep_3d_bfm_09/bfm_part.obj");
}

void TESTFUNCTION::imageFusion()
{
	cstr root = "D:/dota210104/0118_1e4/";
	cstr root_txt = "D:/dota210104/0118_coef/coeff_1e4/coeff/";
	cstr root_res = "D:/dota210104/0118_1e4_res/";
	SG::needPath(root_res);
	cstrVec image_files = FILEIO::getFolderFiles(root, { ".png" }, false);
	cstrVec weight_files = FILEIO::getFolderFiles(root_txt, { ".txt" }, false);

	for (int iter_weight = 0; iter_weight < weight_files.size(); iter_weight++)
	{

		floatVec weight = FILEIO::loadFloatDynamic(root_txt + weight_files[iter_weight], '\n');
		cstr raw_txt = FILEIO::getFileNameWithoutExt(weight_files[iter_weight]);
		weight.resize(16);
		//doubleVec weight = { -0.03781,0.14911,0.28226,-0.15111,-0.15111,-0.09900,-0.69211,-2.75234,0.50472,1.97301,0.52097,-0.22397,-0.04207,-0.45951,0.18434,0.52097, };
		//doubleVec weight = { 0.030529166,0.202374514,0.342170857,0.005063353,0.005063353,-0.061717811,-0.555405244,-2.460024655,0.245517995,1.840056926,0.437659658,-0.299362801,-0.13072907,-0.515482426,0.098842047,0.437659658, };
		//doubleVec weight = { -0.063298554,0.091281561,0.113896299,-0.368105508,-0.368105508,-0.198107939,-0.643460304,-2.515917718,0.968833715,1.540563405,0.458845235,-0.053197751,0.076056132,-0.258839428,0.248420127,0.458845235, };
		cvMatD3 res(512, 512);
		res.setTo(cv::Vec3d(0, 0, 0));
		cv::Mat image_mean = cv::imread("D:/dota210104/0117_pca_mean/mean.png");
		cv::resize(image_mean, image_mean, cv::Size(512, 512));
		cv::Mat res_int8;
#pragma omp parallel for
		for (int y = 0; y < 512; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < 512; x++)
			{
				cv::Vec3d pixel_buff;
				pixel_buff[0] = float(GETU3(image_mean, y, x)(0));
				pixel_buff[1] = float(GETU3(image_mean, y, x)(1));
				pixel_buff[2] = float(GETU3(image_mean, y, x)(2));
				SETD3(res, y, x, pixel_buff);
			}
		}


		//res.convertTo(res_int8, CV_8UC3);
		//cv::cvtColor(res, res_int8, CV_8UC3);
		//cv::imshow("res_int8", res_int8);
		//cv::waitKey(0);

		for (int i = 0; i < weight.size(); i++)
		{
			cv::Mat in_image = cv::imread(root + image_files[i]);
			cv::resize(in_image, in_image, cv::Size(512, 512));
#pragma omp parallel for
			for (int y = 0; y < 512; y++)
			{
#pragma omp parallel for
				for (int x = 0; x < 512; x++)
				{
					cv::Vec3d pixel_buff = GETD3(res, y, x);
					cv::Vec3d pixel_diff;
					pixel_diff[0] = float(GETU3(in_image, y, x)(0)) - float(GETU3(image_mean, y, x)(0));
					pixel_diff[1] = float(GETU3(in_image, y, x)(1)) - float(GETU3(image_mean, y, x)(1));
					pixel_diff[2] = float(GETU3(in_image, y, x)(2)) - float(GETU3(image_mean, y, x)(2));

					SETD3(res, y, x, pixel_buff + pixel_diff * weight[i]);
				}
			}
			//res.convertTo(res_int8, CV_8UC3);
			//cv::cvtColor(res, res_int8, CV_8UC3);
			//cv::imshow("res_int8_iter", res_int8);
			//cv::imwrite(root_res + std::to_string(i) + ".png", res_int8);
			//cv::waitKey(0);
		}


		res.convertTo(res_int8, CV_8UC3);
		cv::resize(res_int8, res_int8, cv::Size(2048, 2048));
		cv::imwrite(root_res + raw_txt + "_fusion.png", res_int8);
		//cv::cvtColor(res, res_int8, CV_8UC3);

		//cv::imshow("res_int8", res_int8);
		//cv::waitKey(0); 
	}
}

void TESTFUNCTION::pcaFromTxt()
{

	floatVec data = FILEIO::loadFloatDynamic("D:/dota210104/0117_pca_mean/tex_mean.txt", '\n');
	//cv::Mat img_base = cv::imread("D:/dota210104/0117_pca/pca_mean.png");
	cvMatD3 img_base(2048, 2048);
#pragma omp parallel for
	for (int y = 0; y < 2048; y++)
	{
#pragma omp parallel for
		for (int x = 0; x < 2048; x++)
		{
			int ind = (y * 2048 + x) * 3;
			SETD3(img_base, y, x, cv::Vec3d(data[ind], data[ind + 1], data[ind + 2]));
		}
	}

	cv::Mat res_int8;
	img_base.convertTo(res_int8, CV_8UC3);
	cv::imwrite("D:/dota210104/0117_pca_mean/mean.png", res_int8);

	cstrVec txt_files = FILEIO::getFolderFiles("D:/dota210104/0117_pca_base/", ".txt");
	for (int i = 0; i < txt_files.size(); i++)
	{
		floatVec data_pca = FILEIO::loadFloatDynamic("D:/dota210104/0117_pca_base/" + txt_files[i], '\n');
		cvMatD3 img_pca_1e2 = img_base.clone();
		cvMatD3 img_pca_1e3 = img_base.clone();
		cvMatD3 img_pca_1e4 = img_base.clone();
#pragma omp parallel for
		for (int y = 0; y < 2048; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < 2048; x++)
			{
				int ind = (y * 2048 + x) * 3;
				cv::Vec3d value = GETD3(img_pca_1e2, y, x);
				cv::Vec3d value_pca = cv::Vec3d(data_pca[ind], data_pca[ind + 1], data_pca[ind + 2]);
				SETD3(img_pca_1e2, y, x, value + 100.0 * (value_pca - value));
				SETD3(img_pca_1e3, y, x, value + 1000.0 * (value_pca - value));
				SETD3(img_pca_1e4, y, x, value + 10000.0 * (value_pca - value));
			}
		}
		cv::Mat res_pca;
		img_pca_1e2.convertTo(res_pca, CV_8UC3);
		cstr raw_name = FILEIO::getFileNameWithoutExt(txt_files[i]);
		cv::imwrite("D:/dota210104/0117_pca_base/" + raw_name +"_1e2.png", res_pca);

		img_pca_1e3.convertTo(res_pca, CV_8UC3);
		cv::imwrite("D:/dota210104/0117_pca_base/" + raw_name + "_1e3.png", res_pca);

		img_pca_1e4.convertTo(res_pca, CV_8UC3);
		cv::imwrite("D:/dota210104/0117_pca_base/" + raw_name + "_1e4.png", res_pca);
	}
	
}

void TESTFUNCTION::fixEyelash()
{
	cstrVec in_files = 
	{
		"D:/multiPack/guijie_version_isv_ori_fix_eyelash/guijie_v17.obj",
	};

	cstrVec out_files =
	{
		"D:/multiPack/guijie_version_isv_ori_fix_eyelash/guijie_v17.obj",
	};

	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";
	std::shared_ptr<ExpGen> exp_ptr;
	exp_ptr.reset(new ExpGen(exp_config));

	for (int i = 0; i < in_files.size(); i++)
	{
		MeshCompress base = in_files[i];
		exp_ptr->fixEyelash(base);
		base.saveObj(out_files[i]);
	}
}

void TESTFUNCTION::testShellGen()
{
	MeshCompress cube = "D:/dota210121/0125_cube/cube_shape.obj";
	cube.getBoundingBox();
	MeshCompress cube_res = cube;
	SHELLGEN::makeFurthestMesh(cube, true, 1, cube_res);
	cube_res.saveObj("D:/dota210121/0125_cube/cube_shape_1_norm.obj");

	MeshCompress male_body = "D:/dota210121/0126_cloth/male_up.obj";
	male_body.getBoundingBox();
	for (int i = 0; i < 10; i++)
	{
		MeshCompress male_body_res = male_body;
		double dis = 0.005 + 0.005*i;
		SHELLGEN::makeFurthestMesh(male_body, true, dis, male_body_res);
		male_body_res.saveObj("D:/dota210121/0126_cloth/male_dis_" + std::to_string(dis) + ".obj");
	}
}

void TESTFUNCTION::testBvh()
{
	/*	
	LOG(INFO) << "working" << std::endl;
	cstr from_bvh = "D:/code/deep-motion-editing/retargeting/datasets/train_v0222/eric/VTmale01_Anim_RSP_Paper.bvh";
	cstr to_bvh = "D:/code/deep-motion-editing/retargeting/datasets/train_v0222/eric/VTmale01_Anim_RSP_Paper_ut.bvh";
	bvh11::BvhObject bvh_blender(from_bvh, 1.0, true);
	bvh_blender.WriteGuijieFile(to_bvh);
	*/

	cstrVec src_folder =
	{
		"D:/dota210219/0223_data_back/eric/",
		"D:/dota210219/0223_data_back/guijie/",
	};

	cstrVec dst_folder =
	{
		"D:/dota210219/0223_fix_bvh/eric/",
		"D:/dota210219/0223_fix_bvh/guijie/",
	};

	int n_size = src_folder.size();

	for (size_t i = 0; i < n_size; i++)
	{
		SG::needPath(dst_folder[i]);
		cstrVec res = FILEIO::getFolderFiles(src_folder[i], {".bvh"}, false);
		for (int iter_file = 0; iter_file < res.size(); iter_file++)
		{
			bvh11::BvhObject bvh_blender(src_folder[i] + res[iter_file], 1.0, true);
			bvh_blender.WriteGuijieFile(dst_folder[i] + res[iter_file]);
		}
	}		
}

void TESTFUNCTION::testWinLinuxColor()
{
	cstr from_dir = "D:/avatar/nl_linux/cb_2101/";
	cstr to_dir = "D:/avatar/nl_linux/cb_2101_png/";
	SG::needPath(to_dir);
	cstrVec image_files = FILEIO::getFolderFiles(from_dir, { ".png" }, false);

	for (auto i: image_files)
	{
		cv::Mat img_win = cv::imread(from_dir + i, cv::IMREAD_UNCHANGED);
		cv::imwrite(to_dir + i, img_win);
	}

	cv::Mat img_linux = cv::imread("C:/code/expgen_aquila/result/test/dst_fusion_1.000000_normal.png", cv::IMREAD_UNCHANGED);
	cv::Mat img_win = cv::imread("D:/dota210507/0507_tt/dst_fusion_1.000000_normal.png", cv::IMREAD_UNCHANGED);
	std::cout << GETU4(img_linux, 0, 0) << std::endl;
	std::cout << GETU4(img_win, 0, 0) << std::endl;
}




void TESTFUNCTION::fitIsvGuijieVersionToFixScale()
{
#if 0
	//获取landmark信息
	cstr result_root = "D:/dota210219/0226_isv/";
	//TinyTool::getMeshSysInfo("D:/dota210219/0226_isv/", "clean.obj");

	TinyTool::getSysLandmarkPoint("D:/dota210219/0226_isv/",
		"config.json", "D:/dota210219/0226_isv/land68/", "guijie_hand.txt",
		"sys_68.txt", "guijie_68_sys.txt");
#endif

#if 0
	// put yyx_isv to dst pos	
	cstr root_res = "D:/dota210219/0303_yyx_isv/";
	SG::needPath(root_res);

	MeshCompress yyx = "D:/dota210219/0226_yyx_isv/move_dst_pro.obj";
	MeshCompress isv = "D:/dota210219/0303_isv/head.obj";

	//subtract data sl: structure line
	intVec guijie68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/guijie_68_sys.txt");
	intVec yyx68 = FILEIO::loadIntDynamic("D:/dota210219/0226_isv/land68/guijie_68_sys.txt");

	//fitting is not ready
	MeshCompress move_yyx68;
	MeshTools::putSrcToDst(yyx, yyx68, isv, guijie68, move_yyx68);
	   
	move_yyx68.saveObj(root_res + "move_yyx.obj");
#endif
	cstr root_res = "D:/dota210219/0303_yyx_isv/";
	SG::needPath(root_res);
	MeshCompress guijie_v4 = "D:/multiPack/guijie_version/guijie_v4.obj";
	MeshCompress isv = "D:/dota210219/0303_isv/head.obj";
	//subtract data sl: structure line
	intVec guijie68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/guijie_68_sys.txt");
	intVec yyx68 = FILEIO::loadIntDynamic("D:/dota210219/0226_isv/land68/guijie_68_sys.txt");
	//fitting is not ready
	MeshCompress move_yyx68;
	MeshTools::putSrcToDst(isv, yyx68, guijie_v4, guijie68, move_yyx68, float3E(0,1,1));

	move_yyx68.saveObj(root_res + "guijie_v8.obj");


	//put eyes
	MeshCompress temp_res = isv;
	MeshCompress eyes = "D:/dota210219/0303_isv/eye.obj";
	double scale;
	float3E scale_center, translate;
	MeshTools::putSrcToDst(isv, guijie68, move_yyx68, guijie68, temp_res, scale, scale_center, translate);
	MeshCompress scale_eyes = eyes;
	RT::scaleAndTranslateInPlace(scale, scale_center, translate, eyes.pos_);
	eyes.saveObj(root_res + "guijie_v8_eyes.obj");
}

void TESTFUNCTION::simDumpEyelash()
{
	cstr root = "D:/dota210604/0610_dw_tensor_test/";
	intVec eyelash = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/left_up_lash.txt", eyelash);
	FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_down_lash.txt", eyelash);
	FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_up_lash.txt", eyelash);
	
	//MeshCompress face = "D:/dota210604/0608_18/local_deform.obj";
	MeshCompress face = root + "local_deform.obj";
	MeshCompress eyelash_mesh = face;
	eyelash_mesh.keepRoi(eyelash);

	face.discard(eyelash);
	face.saveObj(root + "render_face.obj");
	eyelash_mesh.saveObj(root + "render_eyelash.obj");

}

void TESTFUNCTION::changeEyebrow()
{
	cv::Mat src = cv::imread("D:/dota210604/0609_18_render/00.png", -1);
	cv::Mat dst = src.clone();
	float3E dst_color = float3E(20, 20, 20);
	ImageUtils::changeRGBKeepAlpha(src, dst, dst_color);
	cv::imwrite("D:/dota210604/0609_18_render/00_opt_dark_20.png", dst);
}


class A {};

enum E : int {};

template <class T>
T f(T i)
{
	static_assert(std::is_integral<T>::value, "Integral required.");
	return i;
}


template<typename T, typename U>
typename std::enable_if<std::is_same<T, U>::value, void>::type func(T& t, U& u) {
	std::cout << "same\n";
}
template<typename T, typename U>
typename std::enable_if<!std::is_same<T, U>::value, void>::type func(T& t, U& u) {
	std::cout << "different\n";
}

void TESTFUNCTION::testingForTemplate()
{
	// https://en.cppreference.com/w/cpp/types/is_integral
	// https://zhuanlan.zhihu.com/p/21314708
	// https://stackoverflow.com/questions/11861610/decltype-comparison
	std::cout << std::boolalpha;
	std::cout << std::is_integral<A>::value << '\n';
	std::cout << std::is_integral_v<E> << '\n';
	std::cout << std::is_integral_v<float> << '\n';
	std::cout << std::is_integral_v<int> << '\n';
	std::cout << std::is_integral_v<const int> << '\n';
	std::cout << std::is_integral_v<bool> << '\n';
	std::cout << f(123) << '\n';
	int a, b;
	float c;
	func(a, b);
	func(a, c);
}