#include "Auto3dTest.h"
#include "../Basic/CGPBaseHeader.h"
#include "../Basic/MeshHeader.h"
#include "../Auto3d/Auto3dLoader.h"
#include "../Config/JsonHelper.h" 
#include "../Auto3d/Auto3DData.h"
#include "../Debug/DebugTools.h"
#include "../ImageSim/ImageUtils.h"
#include "../Solver/BVLSSolver.h"
#include "../MeshDeform/LapDeformWrapper.h"
#include "../CalcFunction/CalcHelper.h"
#include "../Test/TinyTool.h"
#include "../Test/Prepare.h"
#include "../Mapping/Mapping.h"

using namespace AUTO3D;
using namespace CGP;

void AUTO3DTEST::testMouth()
{
	cstr root = "D:/code/auto3dData/";
	//Auto3DData test;
	//JsonHelper::initData(root, "config.json", test);

	//subtract data sl: structure line
	intVec mouth_sl = FILEIO::loadIntDynamic(root + "guijie/mouth_structure_line.txt");
	cstrVec exp_files = FILEIO::getFolderFiles(root + "guijie/ani_exp/", { ".obj" }, true);
	cstrVec shape_files = FILEIO::getFolderFiles(root + "guijie/head_shapebs/", { ".obj" }, true);

	cstr exp_files_output = root + "guijie/ani_exp_img/";
	cstr shape_files_output = root + "guijie/shape_img/";
	SG::needPath(exp_files_output);
	SG::needPath(shape_files_output);
	
#if 0
	//get for data
	for (int i = 0; i < exp_files.size(); i++)
	{
		MeshCompress obj(exp_files[i]);
		float3Vec res;
		MeshTools::getSlice(obj.pos_, mouth_sl, res);
		doubleVec xyz_min, xyz_max;
		MeshTools::getBoundingBox(res, xyz_min, xyz_max);
		cv::Mat img_res = DebugTools::projectVertexToXY({ res }, { cv::Scalar(0, 255, 0) });
		cstr raw_name = FILEIO::getFileNameWithoutExt(exp_files[i]);
		cv::imwrite(exp_files_output + raw_name + ".png", img_res);
		cv::Mat img_contour_res = ImageUtils::drawContourPolygon(img_res);
		cv::imwrite(exp_files_output + raw_name + "_contour.png", img_contour_res);
	}

	for (int i = 0; i < shape_files.size(); i++)
	{
		MeshCompress obj(shape_files[i]);
		float3Vec res;
		MeshTools::getSlice(obj.pos_, mouth_sl, res);
		doubleVec xyz_min, xyz_max;
		MeshTools::getBoundingBox(res, xyz_min, xyz_max);
		cv::Mat img_res = DebugTools::projectVertexToXY({ res }, { cv::Scalar(0, 255, 0) });
		cstr raw_name = FILEIO::getFileNameWithoutExt(shape_files[i]);
		cv::imwrite(shape_files_output + raw_name + ".png", img_res);
		cv::Mat img_contour_res = ImageUtils::drawContourPolygon(img_res);
		cv::imwrite(shape_files_output + raw_name + "_contour.png", img_contour_res);
	}
#endif
	
	float3Vec dst_roi;
	MeshCompress obj(root + "guijie/ani_exp/" + "Ani_mouthLowerDownLeft.obj");
	MeshTools::getSlice(obj.pos_, mouth_sl, dst_roi);


	//LOG(INFO) << "test.mouth_: " << test.mouth_data_ << std::endl;
	//eyelash pairs
#if 0
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);
#else
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
#endif
	Tensor shape_tensor;
	JsonHelper::initData("D:/code/auto3dData/guijie/mouth_tensor/", "config.json", shape_tensor);

	intVec roi = mouth_sl;
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3;
	int n_var = shape_tensor.n_id_ - 1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(-2);
	upper.setConstant(2);
	vecD weight = vecD::Ones(n_constrain);
	intVec fix_tensor = { 24 };
	for (int i : fix_tensor)
	{
		lower[i] = -1e-5;
		upper[i] = 1e-5;
	}

	LOG(INFO) << upper.transpose() << std::endl;
	int shift = shape_tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < roi.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = roi[i];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = shape_tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim] * weight(i);
				B(3 * i + iter_dim) = (dst_roi[i][iter_dim] - shape_tensor.data_[vertex_id * 3 + iter_dim])*weight(i);
			}
		}
	}

	if (true)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	vecD coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;

	MeshCompress res_obj = shape_tensor.template_obj_;

	floatVec raw_bs = shape_tensor.interpretID(coef);
	SG::safeMemcpy(res_obj.pos_.data(), raw_bs.data(), raw_bs.size() * sizeof(float));
	res_obj.saveObj("D:/dota210104/0113_auto3d/interBS.obj");

	//close mouth
	MeshCompress mean = "D:/dota210104/0113_auto3d/mean_0820.obj";
	MeshCompress dst_res = res_obj;

	
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(res_obj.pos_, mouth_sl, dst_roi) << std::endl;
	intVec mouth_close = FILEIO::loadIntDynamic(root + "guijie/mouth_close.txt");
	SIMDEFORM::moveHandle(res_obj, mouth_close, mean, mouth_close, mouth_sl, guijie_eyelash, dst_res);
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(dst_res.pos_, mouth_sl, dst_roi) << std::endl;
	dst_res.saveObj("D:/dota210104/0113_auto3d/dst_res.obj");
}

void AUTO3DTEST::testWholeFace()
{
	cstr root = "D:/code/auto3dData/";
	cstr root_res = "D:/dota210202/0203_head_round/";
	SG::needPath(root_res);
	
	Tensor shape_tensor;
	JsonHelper::initData("D:/dota210202/0203_v4/tensor/", "config.json", shape_tensor);

	//subtract data sl: structure line
	intVec face_sl = FILEIO::loadIntDynamic("D:/dota210121/0127_sl_head/select_move.txt");
	MeshCompress src = shape_tensor.template_obj_;
	MeshCompress dst_raw = root_res + "guijie_v6.obj";
	MeshCompress dst = dst_raw;
	float3Vec dst_roi;
	intVec guijie68 = FILEIO::loadIntDynamic("D:/avatar/guijie_opt3_data/guijie_68_sys.txt");
	MeshTools::putSrcToDst(dst_raw, guijie68, src, guijie68, dst);
	dst.saveObj(root_res + "move_dst.obj");
	MeshTools::getSlice(dst.pos_, face_sl, dst_roi);
#if 0
	//eyelash only
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);
#else
	//eyelash pair
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
#endif

	intVec roi = face_sl;
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3 + 1;
	int n_var = shape_tensor.n_id_ - 1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1);
	vecD weight = vecD::Ones(n_constrain);
	intVec fix_tensor = {  };
	for (int i : fix_tensor)
	{
		lower[i] = -1e-5;
		upper[i] = 1e-5;
	}

	LOG(INFO) << upper.transpose() << std::endl;
	int shift = shape_tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < roi.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = roi[i];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = shape_tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim] * weight(i);
				B(3 * i + iter_dim) = (dst_roi[i][iter_dim] - shape_tensor.data_[vertex_id * 3 + iter_dim])*weight(i);
			}
		}
	}

	float weight_sum_1 = 1e7;
	double sum_extra = 4.00;
	//last
	for (int j = 0; j < n_var; j++)
	{
		A(n_constrain - 1, j) = weight_sum_1 * 1.0;
		B(n_constrain - 1) = weight_sum_1 * sum_extra;
	}

	if (true)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	vecD coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;

	MeshCompress res_obj = shape_tensor.template_obj_;

	floatVec raw_bs = shape_tensor.interpretID(coef);
	SG::safeMemcpy(res_obj.pos_.data(), raw_bs.data(), raw_bs.size() * sizeof(float));
	res_obj.saveObj(root_res + "face_interBS.obj");


	//lap
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	SIMDEFORM::moveHandle(res_obj, guijie68, dst, guijie68, loading_fix_part_expand, guijie_eyelash, res_obj);
	res_obj.saveObj(root_res + "face_interBS_lap.obj");

	//move back
	intVec small_head_fix = FILEIO::loadIntDynamic("D:/dota210202/0203_head/small_head_fix.txt");
	intVec small_head_fix_inv_trans = FILEIO::loadIntDynamic("D:/dota210202/0203_head/small_head_fix_reverse_trans.txt");
	intVec eye_keep = FILEIO::loadIntDynamic("D:/dota210202/0203_head/eye_region.txt");
	small_head_fix = MAP::getUnionset({ small_head_fix, eye_keep });
	intVec move_idx = MAP::getSubset({ small_head_fix_inv_trans , eye_keep });
	SIMDEFORM::moveHandle(res_obj, move_idx, dst, move_idx, small_head_fix, guijie_eyelash, res_obj);
	res_obj.saveObj(root_res + "face_interBS_lap_move_all.obj");



#if 0
	intVec move_guijie_all = FILEIO::loadIntDynamic("D:/dota210121/0127_sl_head/select_move.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");

	SIMDEFORM::moveHandle(res_obj, move_guijie_all, dst_move, move_guijie_all, loading_fix_part_expand, guijie_eyelash, res_obj);

	res_obj.saveObj(root_res + "lap_interBS.obj");


	//close mouth
	MeshCompress mean = "D:/dota210104/0113_auto3d/mean_0820.obj";
	MeshCompress dst_res = res_obj;

	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(res_obj.pos_, right_eye_sl, dst_roi) << std::endl;
	SIMDEFORM::moveHandle(res_obj, right_eye_sl, dst_isv_res, right_eye_sl, {}, guijie_eyelash, dst_res);
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(dst_res.pos_, right_eye_sl, dst_roi) << std::endl;
	dst_res.saveObj(root_res + "fit_res.obj");

	MeshTools::putSrcToDst(dst_res, right_eye_sl, dst_isv, right_eye_sl, dst_res);
	dst_res.saveObj(root_res + "move_back_fit_res.obj");

	intVec right_eye_sl_inner = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_sl_inner.txt");
	intVec right_eye_sl_outer = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_sl_outer.txt");

	//isv制作
	MeshCompress dst_isv_adjust = dst_isv;
	SIMDEFORM::moveHandle(dst_res, right_eye_sl_inner, dst_isv, right_eye_sl_inner, {}, guijie_eyelash, dst_res);
	SIMDEFORM::moveHandle(dst_res, right_eye_sl_outer, dst_isv, right_eye_sl_outer, {}, guijie_eyelash, dst_res);
	SIMDEFORM::moveHandle(dst_res, right_eye_sl, dst_isv, right_eye_sl, {}, guijie_eyelash, dst_res);

	dst_res.saveObj(root_res + "move_back_rep_data.obj");

	MeshSysFinder guijie_sys;
	JsonHelper::initData("D:/avatar/exp_server_config/guijie_sys_tensor/", "config.json", guijie_sys);
	intVec right_eye_rep_region = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_rep_region.txt");
	intVec left_eye_rep_region = right_eye_rep_region;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region);
	intVec right_eye_rep_region_exp = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_rep_region_expand.txt");
	intVec right_eye_rep_region_exp_fix = MAP::getReverseRoi(right_eye_rep_region_exp, dst_res.n_vertex_);
	intVec left_eye_rep_region_exp_fix = right_eye_rep_region_exp_fix;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region_exp_fix);

	intVec face_vertex = FILEIO::loadIntDynamic("D:/dota210121/0126_guijie_resource/maya/head_vertex.txt");
	intVec face_fix = MAP::getInterset({ left_eye_rep_region_exp_fix , right_eye_rep_region_exp_fix, face_vertex });

	//expand for eye
	//get sys
	for (int i = 0; i < left_eye_rep_region.size(); i++)
	{
		dst_res.pos_[left_eye_rep_region[i]] = dst_res.pos_[right_eye_rep_region[i]];
		dst_res.pos_[left_eye_rep_region[i]].x() = 2 * 0 - dst_res.pos_[right_eye_rep_region[i]].x();
	}

	MeshCompress isv_eyebrow = dst_isv;

	dst_res.saveObj(root_res + "dst_res_sys.obj");
	intVec eye_rep_region = MAP::getUnionset({ left_eye_rep_region, right_eye_rep_region });
	SIMDEFORM::moveHandle(dst_isv, eye_rep_region, dst_res, eye_rep_region, face_fix, guijie_eyelash, dst_isv);

	dst_isv.saveObj(root_res + "dir_rep_data.obj");


	//eyelash only
	intVec guijie_eyelash_region = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash_region);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash_region);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash_region);

	SIMDEFORM::replaceHandle(dst_isv, guijie_eyelash_region, isv_eyebrow, guijie_eyelash_region, dst_isv);

	dst_isv.saveObj(root_res + "dir_rep_data_keep_lash.obj");



#if 0
	intVec left_eye_rep_region = right_eye_rep_region;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region);
	double mid_shift = 0;
	for (int i : guijie_sys.mid_ids_)
	{
		mid_shift += dst_isv.pos_[i].x();
	}
	double mid_shift_value = mid_shift / (1.0* guijie_sys.mid_ids_.size());
	LOG(INFO) << "mid: " << mid_shift_value << std::endl;

	for (int i = 0; i < left_eye_rep_region.size(); i++)
	{
		dst_isv.pos_[left_eye_rep_region[i]] = dst_isv.pos_[right_eye_rep_region[i]];
		dst_isv.pos_[left_eye_rep_region[i]].x() = 2 * mid_shift_value - dst_isv.pos_[right_eye_rep_region[i]].x();
	}
	dst_isv.saveObj(root_res + "dir_rep_data_sys.obj");
#endif
#endif
}

void AUTO3DTEST::testCheek()
{
	cstr root = "D:/code/auto3dData/";
	cstr root_res = "D:/dota210202/0203_cheek/";
	SG::needPath(root_res);

	//subtract data sl: structure line
	intVec cheek_sl = FILEIO::loadIntDynamic(root + "guijie/cheek_sl.txt");
	MeshCompress src = "D:/dota210121/0127_sl_head/head.obj";
	MeshCompress dst = "D:/dota210121/0127_sl_head/head_deform.obj";
	MeshCompress dst_move = dst;
	MeshTools::putSrcToDst(dst, cheek_sl, src, cheek_sl, dst_move);
	dst_move.saveObj("D:/dota210121/0127_sl_head/head_deform_move.obj");
	float3Vec dst_roi;
	MeshTools::getSlice(dst_move.pos_, cheek_sl, dst_roi);
#if 0
	//eyelash only
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);
#else
	//eyelash pair
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
#endif

	Tensor shape_tensor;
	JsonHelper::initData("D:/dota210121/0126_guijie_resource/debug/head_tensor/", "config.json", shape_tensor);

	intVec roi = cheek_sl;
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3 + 1;
	int n_var = shape_tensor.n_id_ - 1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1);
	vecD weight = vecD::Ones(n_constrain);
	intVec fix_tensor = {  };
	for (int i : fix_tensor)
	{
		lower[i] = -1e-5;
		upper[i] = 1e-5;
	}

	LOG(INFO) << upper.transpose() << std::endl;
	int shift = shape_tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < roi.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = roi[i];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = shape_tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim] * weight(i);
				B(3 * i + iter_dim) = (dst_roi[i][iter_dim] - shape_tensor.data_[vertex_id * 3 + iter_dim])*weight(i);
			}
		}
}

	float weight_sum_1 = 1e7;
	double sum_extra = 2.00;
	//last
	for (int j = 0; j < n_var; j++)
	{
		A(n_constrain - 1, j) = weight_sum_1 * 1.0;
		B(n_constrain - 1) = weight_sum_1 * sum_extra;
	}

	if (true)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	vecD coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;

	MeshCompress res_obj = shape_tensor.template_obj_;

	floatVec raw_bs = shape_tensor.interpretID(coef);
	SG::safeMemcpy(res_obj.pos_.data(), raw_bs.data(), raw_bs.size() * sizeof(float));
	res_obj.saveObj(root_res + "interBS.obj");

	//movemove

	intVec move_guijie_all = FILEIO::loadIntDynamic("D:/dota210121/0127_sl_head/select_move.txt");
	intVec loading_fix_part_expand = FILEIO::loadIntDynamic("D:/dota201201/1220_fwh_guijie/fix_guijie_expand.txt");
	
	SIMDEFORM::moveHandle(res_obj, move_guijie_all, dst_move, move_guijie_all, loading_fix_part_expand, guijie_eyelash, res_obj);

	res_obj.saveObj(root_res + "lap_interBS.obj");


#if 0
	//close mouth
	MeshCompress mean = "D:/dota210104/0113_auto3d/mean_0820.obj";
	MeshCompress dst_res = res_obj;

	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(res_obj.pos_, right_eye_sl, dst_roi) << std::endl;
	SIMDEFORM::moveHandle(res_obj, right_eye_sl, dst_isv_res, right_eye_sl, {}, guijie_eyelash, dst_res);
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(dst_res.pos_, right_eye_sl, dst_roi) << std::endl;
	dst_res.saveObj(root_res + "fit_res.obj");

	MeshTools::putSrcToDst(dst_res, right_eye_sl, dst_isv, right_eye_sl, dst_res);
	dst_res.saveObj(root_res + "move_back_fit_res.obj");

	intVec right_eye_sl_inner = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_sl_inner.txt");
	intVec right_eye_sl_outer = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_sl_outer.txt");

	//isv制作
	MeshCompress dst_isv_adjust = dst_isv;
	SIMDEFORM::moveHandle(dst_res, right_eye_sl_inner, dst_isv, right_eye_sl_inner, {}, guijie_eyelash, dst_res);
	SIMDEFORM::moveHandle(dst_res, right_eye_sl_outer, dst_isv, right_eye_sl_outer, {}, guijie_eyelash, dst_res);
	SIMDEFORM::moveHandle(dst_res, right_eye_sl, dst_isv, right_eye_sl, {}, guijie_eyelash, dst_res);

	dst_res.saveObj(root_res + "move_back_rep_data.obj");

	MeshSysFinder guijie_sys;
	JsonHelper::initData("D:/avatar/exp_server_config/guijie_sys_tensor/", "config.json", guijie_sys);
	intVec right_eye_rep_region = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_rep_region.txt");
	intVec left_eye_rep_region = right_eye_rep_region;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region);
	intVec right_eye_rep_region_exp = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_rep_region_expand.txt");
	intVec right_eye_rep_region_exp_fix = MAP::getReverseRoi(right_eye_rep_region_exp, dst_res.n_vertex_);
	intVec left_eye_rep_region_exp_fix = right_eye_rep_region_exp_fix;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region_exp_fix);

	intVec face_vertex = FILEIO::loadIntDynamic("D:/dota210121/0126_guijie_resource/maya/head_vertex.txt");
	intVec face_fix = MAP::getInterset({ left_eye_rep_region_exp_fix , right_eye_rep_region_exp_fix, face_vertex });

	//expand for eye
	//get sys
	for (int i = 0; i < left_eye_rep_region.size(); i++)
	{
		dst_res.pos_[left_eye_rep_region[i]] = dst_res.pos_[right_eye_rep_region[i]];
		dst_res.pos_[left_eye_rep_region[i]].x() = 2 * 0 - dst_res.pos_[right_eye_rep_region[i]].x();
	}

	MeshCompress isv_eyebrow = dst_isv;

	dst_res.saveObj(root_res + "dst_res_sys.obj");
	intVec eye_rep_region = MAP::getUnionset({ left_eye_rep_region, right_eye_rep_region });
	SIMDEFORM::moveHandle(dst_isv, eye_rep_region, dst_res, eye_rep_region, face_fix, guijie_eyelash, dst_isv);

	dst_isv.saveObj(root_res + "dir_rep_data.obj");


	//eyelash only
	intVec guijie_eyelash_region = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash_region);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash_region);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash_region);

	SIMDEFORM::replaceHandle(dst_isv, guijie_eyelash_region, isv_eyebrow, guijie_eyelash_region, dst_isv);

	dst_isv.saveObj(root_res + "dir_rep_data_keep_lash.obj");



#if 0
	intVec left_eye_rep_region = right_eye_rep_region;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region);
	double mid_shift = 0;
	for (int i : guijie_sys.mid_ids_)
	{
		mid_shift += dst_isv.pos_[i].x();
	}
	double mid_shift_value = mid_shift / (1.0* guijie_sys.mid_ids_.size());
	LOG(INFO) << "mid: " << mid_shift_value << std::endl;

	for (int i = 0; i < left_eye_rep_region.size(); i++)
	{
		dst_isv.pos_[left_eye_rep_region[i]] = dst_isv.pos_[right_eye_rep_region[i]];
		dst_isv.pos_[left_eye_rep_region[i]].x() = 2 * mid_shift_value - dst_isv.pos_[right_eye_rep_region[i]].x();
	}
	dst_isv.saveObj(root_res + "dir_rep_data_sys.obj");
#endif
#endif
}

void AUTO3DTEST::testCheekRandom()
{
	cstr root = "D:/code/auto3dData/";
	cstr root_res = "D:/dota210202/0203_head_round/";
	SG::needPath(root_res);

	//subtract data sl: structure line
	intVec cheek_sl = FILEIO::loadIntDynamic(root + "guijie/cheek_sl.txt");
	MeshCompress src = root_res + "face_interBS_c1.obj";
	
	//eyelash pair
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
	
	intVec expand_cheek = FILEIO::loadIntDynamic("D:/dota210202/0203_cheek/expand_cheek.txt");

	for (int iter_mag = 0; iter_mag < 10; iter_mag++)
	{
		MeshCompress guijie_dst = src;
		guijie_dst.generateNormal();
		for (int i = 0; i < guijie_dst.n_vertex_; i++)
		{
			float3E iter_norm = guijie_dst.normal_[i].normalized();
			iter_norm.x() = iter_norm.x() / (1.5+iter_norm.y());
			iter_norm.y() = 0;
			iter_norm.z() = 0;
			guijie_dst.pos_[i] = guijie_dst.pos_[i] + 0.001* iter_mag * iter_norm;
		}

		intVec mouth_sl = FILEIO::loadIntDynamic(root + "guijie/mouth_structure_line.txt");
		//move back
		intVec small_head_fix = FILEIO::loadIntDynamic("D:/dota210202/0203_head/small_head_fix.txt");
		intVec eye_keep = FILEIO::loadIntDynamic("D:/dota210202/0203_head/eye_region.txt");
		small_head_fix = MAP::getUnionset({ mouth_sl, small_head_fix });
		SIMDEFORM::moveHandle(src, expand_cheek, guijie_dst, expand_cheek, small_head_fix, guijie_eyelash, src);
		src.saveObj(root_res + "expand_cheek_ss_" + std::to_string(iter_mag)+".obj");
	}

}

void AUTO3DTEST::testRightEyes()
{
	cstr root = "D:/code/auto3dData/";
	cstr root_res = "D:/dota210121/0127_eyeclose/";
	SG::needPath(root_res);
	//subtract data sl: structure line
	intVec right_eye_sl = FILEIO::loadIntDynamic(root + "guijie/right_eye_sl.txt");	
	MeshCompress src = "D:/code/auto3dData/guijie/mean_0820.obj";
	MeshCompress dst_isv = "D:/dota210121/0126_guijie_resource/maya/head_smooth.obj";
	MeshCompress dst_isv_res = dst_isv;
	MeshTools::putSrcToDst(dst_isv, right_eye_sl, src, right_eye_sl, dst_isv_res);
	
	src.saveObj(root_res + "src.obj");
	dst_isv.saveObj(root_res + "dst_isv.obj");
	dst_isv_res.saveObj(root_res + "dst_move.obj");

	float3Vec dst_roi;
	MeshTools::getSlice(dst_isv_res.pos_, right_eye_sl, dst_roi);

#if 0
	//eyelash only
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash);
#else
	//eyelash pair
	intVec guijie_eyelash = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/left_down_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_up_match.txt", guijie_eyelash);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/pair_info/right_down_match.txt", guijie_eyelash);
#endif

	Tensor shape_tensor;
	JsonHelper::initData("D:/code/auto3dData/guijie/face_shape_ani_pca/", "config.json", shape_tensor);

	intVec roi = right_eye_sl;
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = roi.size() * 3 + 1;
	int n_var = shape_tensor.n_id_ - 1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1);
	vecD weight = vecD::Ones(n_constrain);
	intVec fix_tensor = {  };
	for (int i : fix_tensor)
	{
		lower[i] = -1e-5;
		upper[i] = 1e-5;
	}

	LOG(INFO) << upper.transpose() << std::endl;
	int shift = shape_tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < roi.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = roi[i];
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = shape_tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim] * weight(i);
				B(3 * i + iter_dim) = (dst_roi[i][iter_dim] - shape_tensor.data_[vertex_id * 3 + iter_dim])*weight(i);
			}
		}
	}

	float weight_sum_1 = 1e7;
	double sum_extra = 1.00;
	//last
	for (int j = 0; j < n_var; j++)
	{
		A(n_constrain - 1, j) = weight_sum_1 * 1.0;
		B(n_constrain - 1) = weight_sum_1 * sum_extra;
	}

	if (true)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	vecD coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;

	MeshCompress res_obj = shape_tensor.template_obj_;

	floatVec raw_bs = shape_tensor.interpretID(coef);
	SG::safeMemcpy(res_obj.pos_.data(), raw_bs.data(), raw_bs.size() * sizeof(float));
	res_obj.saveObj(root_res + "interBS.obj");

#if 1
	//close mouth
	MeshCompress mean = "D:/dota210104/0113_auto3d/mean_0820.obj";
	MeshCompress dst_res = res_obj;

	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(res_obj.pos_, right_eye_sl, dst_roi) << std::endl;
	SIMDEFORM::moveHandle(res_obj, right_eye_sl, dst_isv_res, right_eye_sl, {}, guijie_eyelash, dst_res);
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(dst_res.pos_, right_eye_sl, dst_roi) << std::endl;
	dst_res.saveObj(root_res + "fit_res.obj");

	MeshTools::putSrcToDst(dst_res, right_eye_sl, dst_isv, right_eye_sl, dst_res);
	dst_res.saveObj(root_res + "move_back_fit_res.obj");

	intVec right_eye_sl_inner = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_sl_inner.txt");
	intVec right_eye_sl_outer = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_sl_outer.txt");

	//isv制作
	MeshCompress dst_isv_adjust = dst_isv;
	SIMDEFORM::moveHandle(dst_res, right_eye_sl_inner, dst_isv, right_eye_sl_inner, {}, guijie_eyelash, dst_res);
	SIMDEFORM::moveHandle(dst_res, right_eye_sl_outer, dst_isv, right_eye_sl_outer, {}, guijie_eyelash, dst_res);
	SIMDEFORM::moveHandle(dst_res, right_eye_sl, dst_isv, right_eye_sl, {}, guijie_eyelash, dst_res);

	dst_res.saveObj(root_res + "move_back_rep_data.obj");

	MeshSysFinder guijie_sys;
	JsonHelper::initData("D:/avatar/exp_server_config/guijie_sys_tensor/", "config.json", guijie_sys);
	intVec right_eye_rep_region = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_rep_region.txt");
	intVec left_eye_rep_region = right_eye_rep_region;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region);
	intVec right_eye_rep_region_exp = FILEIO::loadIntDynamic("D:/code/auto3dData/guijie/right_eye_rep_region_expand.txt");
	intVec right_eye_rep_region_exp_fix = MAP::getReverseRoi(right_eye_rep_region_exp, dst_res.n_vertex_);
	intVec left_eye_rep_region_exp_fix = right_eye_rep_region_exp_fix;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region_exp_fix);

	intVec face_vertex = FILEIO::loadIntDynamic("D:/dota210121/0126_guijie_resource/maya/head_vertex.txt");
	intVec face_fix = MAP::getInterset({ left_eye_rep_region_exp_fix , right_eye_rep_region_exp_fix, face_vertex });
	
	//expand for eye
	//get sys
	for (int i = 0; i < left_eye_rep_region.size(); i++)
	{
		dst_res.pos_[left_eye_rep_region[i]] = dst_res.pos_[right_eye_rep_region[i]];
		dst_res.pos_[left_eye_rep_region[i]].x() = 2 * 0 - dst_res.pos_[right_eye_rep_region[i]].x();
	}

	MeshCompress isv_eyebrow = dst_isv;

	dst_res.saveObj(root_res + "dst_res_sys.obj");
	intVec eye_rep_region = MAP::getUnionset({ left_eye_rep_region, right_eye_rep_region });
	SIMDEFORM::moveHandle(dst_isv, eye_rep_region, dst_res, eye_rep_region, face_fix, guijie_eyelash, dst_isv);

	dst_isv.saveObj(root_res + "dir_rep_data.obj");


	//eyelash only
	intVec guijie_eyelash_region = FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_down_lash.txt");
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/left_up_lash.txt", guijie_eyelash_region);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_down_lash.txt", guijie_eyelash_region);
	FILEIO::loadIntDynamic("D:/avatar/exp_server_config/eye_lash/right_up_lash.txt", guijie_eyelash_region);

	SIMDEFORM::replaceHandle(dst_isv, guijie_eyelash_region, isv_eyebrow, guijie_eyelash_region, dst_isv);

	dst_isv.saveObj(root_res + "dir_rep_data_keep_lash.obj");



#if 0
	intVec left_eye_rep_region = right_eye_rep_region;
	guijie_sys.getMirrorIdsInPlace(left_eye_rep_region);
	double mid_shift = 0;
	for (int i : guijie_sys.mid_ids_)
	{
		mid_shift += dst_isv.pos_[i].x();
	}
	double mid_shift_value = mid_shift / (1.0* guijie_sys.mid_ids_.size());
	LOG(INFO) << "mid: " << mid_shift_value << std::endl;

	for (int i = 0; i < left_eye_rep_region.size(); i++)
	{
		dst_isv.pos_[left_eye_rep_region[i]] = dst_isv.pos_[right_eye_rep_region[i]];
		dst_isv.pos_[left_eye_rep_region[i]].x() = 2 * mid_shift_value - dst_isv.pos_[right_eye_rep_region[i]].x();
	}
	dst_isv.saveObj(root_res + "dir_rep_data_sys.obj");
#endif
#endif
}

void AUTO3DTEST::testCloseMouth()
{
	//mouth cloth
	cstr root = "D:/code/auto3dData/";
	intVec mouth_close = FILEIO::loadIntDynamic(root + "guijie/mouth_close.txt");
	intVec mouth_sl = FILEIO::loadIntDynamic(root + "guijie/mouth_structure_line.txt");
	MeshCompress mean = "D:/dota210104/0113_auto3d/mean_0820.obj";
	MeshCompress dst = "D:/dota210104/0113_auto3d/Ani_mouthLowerDownLeft.obj";
	
	LOG(INFO) << "init value: " << CalcHelper::getEigenVectorDis(mean.pos_, dst.pos_, mouth_sl) << std::endl;

	MeshCompress dst_res, dst_res_x2;
	intVec eye_lash_pair = FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_down_match.txt");
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/left_up_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_down_match.txt", eye_lash_pair);
	FILEIO::loadIntDynamic("D:/code/expgen_aquila/data/exp_server_config/pair_info/right_up_match.txt", eye_lash_pair);
	SIMDEFORM::moveHandle(dst, mouth_close, mean, mouth_close, mouth_sl, eye_lash_pair, dst_res);
	dst_res.saveObj("D:/dota210104/0113_auto3d/ani_res.obj");
	LOG(INFO) << "first pass value: " << CalcHelper::getEigenVectorDis(dst_res.pos_, dst.pos_, mouth_sl) << std::endl;
	dst_res_x2 = dst_res;
	SIMDEFORM::moveHandle(dst_res, mouth_sl, dst_res, mouth_sl, mouth_close, eye_lash_pair, dst_res_x2);
	dst_res_x2.saveObj("D:/dota210104/0113_auto3d/ani_res_x2.obj");

	float3Vec res;
	MeshTools::getSlice(dst_res_x2.pos_, mouth_sl, res);
	doubleVec xyz_min, xyz_max;
	MeshTools::getBoundingBox(res, xyz_min, xyz_max);
	cv::Mat img_res = DebugTools::projectVertexToXY({ res }, { cv::Scalar(0, 255, 0) });
	LOG(INFO) << "second pass value: " << CalcHelper::getEigenVectorDis(dst_res_x2.pos_, dst.pos_, mouth_sl) << std::endl;
	cv::imwrite("D:/dota210104/0113_auto3d/ani_res_x2.png", img_res);
}

void AUTO3DTEST::selectMouthObj()
{
	cstr root = "D:/code/auto3dData/";	

	intVec mouth_sl = FILEIO::loadIntDynamic(root + "guijie/mouth_structure_line.txt");
	cstrVec exp_files = FILEIO::getFolderFiles(root + "guijie/ani_exp/", { ".obj" }, true);
	cstrVec shape_files = FILEIO::getFolderFiles(root + "guijie/head_shapebs/", { ".obj" }, true);

	MeshCompress mean = "D:/dota210104/0113_auto3d/mean_0820.obj";
	mean.discardMaterial();

	cstr dst_root = "D:/code/auto3dData/guijie/mouth_raw_data/";
	cstr dst_ub_root = "D:/code/auto3dData/guijie/mouth_upper_bound/";
	cstr dst_lb_root = "D:/code/auto3dData/guijie/mouth_lower_bound/";
	SG::needPath(dst_root);
	SG::needPath(dst_ub_root);
	SG::needPath(dst_lb_root);
	
	//FILEIO::copyFile("D:/dota210104/0113_auto3d/mean_0820.obj", dst_root + "mean.obj");
	mean.saveObj(dst_root + "mean.obj");

	double value_lb = 1;
	double value_ub = 70;
	for (int i = 0; i < exp_files.size(); i++)
	{
		MeshCompress obj(exp_files[i]);
		cstr raw_name = FILEIO::getFileNameWithoutExt(exp_files[i]);
		double error_value = CalcHelper::getEigenVectorDis(mean.pos_, obj.pos_, mouth_sl);
		LOG(INFO)<<raw_name<<": "<< error_value << std::endl;
		if (error_value > value_lb && error_value<value_ub)
		{
			LOG(INFO) << "copy " << raw_name << std::endl;
			//FILEIO::copyFile(exp_files[i], dst_root + raw_name + ".obj");
			obj.discardMaterial();
			obj.saveObj(dst_root + raw_name + ".obj");
		}
		else if (error_value > value_ub)
		{
			LOG(INFO) << "exceed: " << raw_name << std::endl;
			LOG(INFO) << "value: " << error_value << std::endl;
			obj.discardMaterial();
			obj.saveObj(dst_ub_root + raw_name + ".obj");
		}
	}

	for (int i = 0; i < shape_files.size(); i++)
	{
		MeshCompress obj(shape_files[i]);
		cstr raw_name = FILEIO::getFileNameWithoutExt(shape_files[i]);
		double error_value = CalcHelper::getEigenVectorDis(mean.pos_, obj.pos_, mouth_sl);
		LOG(INFO) << raw_name << ": " << error_value << std::endl;
		if (error_value > value_lb && error_value < value_ub)
		{
			LOG(INFO) << "copy " << raw_name << std::endl;
			//FILEIO::copyFile(shape_files[i], dst_root + raw_name + ".obj");
			obj.discardMaterial();
			obj.saveObj(dst_root + raw_name + ".obj");
		}
		else if (error_value > value_ub)
		{
			LOG(INFO) << "exceed: " << raw_name << std::endl;
			LOG(INFO) << "value: " << error_value << std::endl;
			obj.discardMaterial();
			obj.saveObj(dst_ub_root + raw_name + ".obj");
		}
	}
}

void AUTO3DTEST::generateTensor()
{
#if 0
	TinyTool::getFileNamesToPCAJson("D:/code/auto3dData/guijie/mouth_select_data/");
	PREPARE::prepareBSTensor("D:/code/auto3dData/guijie/mouth_select_data/", 
		"D:/code/auto3dData/guijie/mouth_tensor/");
#else
	TinyTool::getFileNamesToPCAJson("D:/dota210202/0203_v4/pca/");
	PREPARE::prepareBSTensor("D:/dota210202/0203_v4/pca/",
		"D:/dota210202/0203_v4/tensor/");
#endif
}
