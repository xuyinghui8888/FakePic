#include "Test.h"
#include "../Basic/MeshHeader.h"
#include "../VisHelper/VisHeader.h"
#include "../NRICP/register.h"
#include "../NRICP/demo.h"
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
#include "../FileIO/JsonUtils.h"

using namespace CGP;

void TESTFUNCTION::testBVLS()
{
	//testing data from https://people.sc.fsu.edu/~jburkardt/f77_src/bvls/bvls_prb_output.txt
	{
		matD A(2, 2);
		vecD B(2), lower(2), upper(2);
		A << 0.829509, 0.415307, 0.561695, 0.661187E-01;
		//A = A.transpose();
		B << 0.218418, 0.956318;
		lower << 1, 3;
		upper << 2, 4;
		BVLSSolver test(A, B, lower, upper);
		test.solve();
		LOG(INFO) << "if converge: " << test.converged() << std::endl;
		LOG(INFO) << "getSolution: " << test.getSolution().transpose() << std::endl;
	}

	{
		matD A(2, 4);
		vecD B(2), lower(4), upper(4);
		A << 0.829509, 0.415307, 0.257578, 0.438290E-01, 0.561695, 0.661187E-01, 0.109957, 0.633966;
		B << 0.218418, 0.956318;
		lower << 0,0,0,0;
		upper << 10,10,10,10;
		BVLSSolver test(A, B, lower, upper);
		test.solve();
		LOG(INFO) << "if converge: " << test.converged() << std::endl;
		LOG(INFO) << "getSolution: " << test.getSolution().transpose() << std::endl;
	}
}

void TESTFUNCTION::testShader()
{
	int draw_size = 200;
	//test for drawing
	ObjVis mesh_drawer;
	MeshCompress mesh_data;
	mesh_data.loadObj("Data/bumps.obj");
	mesh_drawer.setFixSize(draw_size, draw_size, true);
	cv::Mat mesh_screen = mesh_drawer.drawMesh(mesh_data, draw_size, draw_size, true);
	LOG(INFO) << "mesh_screen depth: " << mesh_screen.depth() << std::endl;
	LOG(INFO) << "mesh_screen size: " << mesh_screen.size << std::endl;
	cv::imshow("show for mesh_screen: ", mesh_screen);
	cv::waitKey(0);
}

void TESTFUNCTION::testDeformTranSame()
{
	MeshCompress template_render("D:/data/0715_project_test_gdg_test_1717/bfm_lap_sys.obj");
	float3E translate = float3E(0.000000000, 150.939240, -2.90438318);
	float guijie_scale = 37.24;
	std::string cur_root = "D:/avatar/0717bs/";
	std::string result_root = "D:/avatar/0722_bs_keep_updown_static_fix/";
	SG::needPath(result_root);
	//get A A' B B'
	MeshCompress mean_face, mean_face_dst, delta_src, delta_dst;
	mean_face.loadObj(cur_root + "mean.obj");
	mean_face_dst.loadObj(cur_root + "Ani_eyeBlinkRight.obj");
	delta_src.loadObj(cur_root + "Ani_eyeSquintLeft.obj");

	//for rendering
	RT::translateInPlace(-translate, mean_face.pos_);
	RT::translateInPlace(-translate, mean_face_dst.pos_);
	RT::translateInPlace(-translate, delta_src.pos_);

	float render_scale = template_render.scale_ / mean_face.scale_;
	RT::scaleInPlaceNoShift(render_scale, mean_face.pos_);
	RT::scaleInPlaceNoShift(render_scale, mean_face_dst.pos_);
	RT::scaleInPlaceNoShift(render_scale, delta_src.pos_);
	mean_face.saveObj(result_root + "mean_face_ori.obj");
	mean_face_dst.saveObj(result_root + "mean_face_dst_ori.obj");
	delta_src.saveObj(result_root + "delta_src_ori.obj");

	intVec res = FILEIO::loadIntDynamic("D:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_down_match.txt", res);
	//FILEIO::loadIntDynamic("D:/avatar/guijie/right_eye_close.txt", res);
	intVec fix = FILEIO::loadIntDynamic("D:/avatar/guijie/fix_head_sys.txt");

	intVec right_eye_round = FILEIO::loadIntDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_round.txt");
	intSet right_skip = { 3161,3162,3163,3164,3166,3167,3169,3170,3171,3172,3173,3174,3175,3176,3179,3180,3182,3183,3515,3516,3517,3518,3519,3520,3521,3522,3523,3524,3525,3526,3527,3528,3529,3530,3531,3532,3533,3534,3535,3536,3537,3538,3539,3540,3541,3542,3543,3544,3545,3546,3547,3548,3549,3550,3551,3552,3553,3554,3555,3556,3557,3558,3559,3560,3561,3562,3563,3564,3565,3566,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576,3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3621,3622,3628,3629,3635,3636,3659,3660,3661,3677,3678,3687,3688,3694,3695, };
	intSet right_down = { 2101,2102,2103,2104,2263,2264,2265,2266,2307,2308,2309,2310,2311,2312,2313,2314,2317,2318,2319,2320,2321,2322,3001,3002,3003,3004,3005,3006,3007,3171,3172,3173,3174,3175,3176,3177,3178,3179,3180,3181,3186,3194,3195,3196,3523,3528,3531,3533,3539,3623,3624,3625,3626,3627,3628,3636,3695,3743,3744,3752,3772,3773,3774,3778,3779,3780,3781,3782,3785,3786,3787,3788,3790,3791,3792,3793,3794,3795,3796,3797,3798,3799,3800,3802,3803,3804,3805,3806,3807,3808,3809,3810,3812,3813,3814,3816,3817, };
	//find pair
	float3Vec dir;
	intVec up, down;
	for (int i = 0; i < right_eye_round.size(); i++)
	{
		int idx = right_eye_round[i];
		if (!right_skip.count(idx))
		{
			if (!right_down.count(idx))
			{
				up.push_back(idx);
			}
			else
			{
				down.push_back(idx);
			}
		}		
	}

	//intVec match = MeshTools::getSrcToDstMatchKeepSign(mean_face_dst, up, down, 0.05, true);
	intVec match = MeshTools::getSrcToDstMatchKeepSignTopN(mean_face_dst, up, 5, down, 0.05, true);
	match = CalcHelper::keepValueBiggerThan(match, -0.5);

	FILEIO::saveDynamic("D:/data/server_pack/guijie_deform_pack/right_eye_match_top3.txt", match, ",");

	res.insert(res.end(), match.begin(), match.end());

	delta_dst = delta_src;
	MeshTransfer trans_exp;
	trans_exp.init(mean_face, mean_face_dst, res, fix);
	//trans_exp.initDynamicFix(mean_face, mean_face_dst, res, 0.001);
	trans_exp.transfer(delta_src.pos_, delta_dst.pos_);
	delta_dst.saveObj(result_root + "delta_dst_ori.obj");

	RT::scaleInPlaceNoShift(1.0 / render_scale, delta_dst.pos_);
	//float3E translate_diff;
	//RT::getTranslate(delta_dst.pos_, fix, delta_src.pos_, fix, translate_diff);

	RT::translateInPlace(translate, delta_dst.pos_);
	delta_dst.saveObj(result_root + "delta_dst.obj");
}

void TESTFUNCTION::testBatchDeformTranSame()
{
	MeshCompress template_render("D:/data/0715_project_test_gdg_test_1717/bfm_lap_sys.obj");
	float3E translate = float3E(0.000000000, 150.939240, -2.90438318);
	float guijie_scale = 37.24;
	std::string cur_root = "D:/avatar/0717bs/";
	std::string result_root = "D:/avatar/0722_batch_dynamic_fix/";
	SG::needPath(result_root);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(cur_root, FILEIO::FILE_TYPE::MESH);
	
	
	//get A A' B B'
	MeshCompress mean_face, mean_face_dst, delta_src, delta_dst;
	mean_face.loadObj(cur_root + "mean.obj");
	delta_src.loadObj(cur_root + "Ani_eyeSquintLeft.obj");
	RT::translateInPlace(-translate, mean_face.pos_);
	RT::translateInPlace(-translate, delta_src.pos_);
	float render_scale = template_render.scale_ / mean_face.scale_;
	RT::scaleInPlaceNoShift(render_scale, mean_face.pos_);
	RT::scaleInPlaceNoShift(render_scale, delta_src.pos_);

	//trans_exp.init(mean_face, mean_face_dst, res, fix);
	intVec res = FILEIO::loadIntDynamic("D:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_down_match.txt", res);
	//FILEIO::loadIntDynamic("D:/avatar/guijie/right_eye_close.txt", res);
	intVec fix = FILEIO::loadIntDynamic("D:/avatar/guijie/fix_head_sys.txt");

	for (auto i : folder_file)
	{
		MeshTransfer trans_exp;
		mean_face_dst.loadObj(cur_root + i);
		RT::translateInPlace(-translate, mean_face_dst.pos_);
		RT::scaleInPlaceNoShift(render_scale, mean_face_dst.pos_);
		intVec fix = trans_exp.initDynamicFix(mean_face, mean_face_dst, delta_src, res, 0.001);
		//trans_exp.init(mean_face, mean_face_dst, res, fix);
		//for rendering
		delta_dst = delta_src;
		trans_exp.transfer(delta_src.pos_, delta_dst.pos_);
//#pragma omp parallel for
//		for (int j = 0; j < fix.size(); j++)
//		{
//			int fix_idx = fix[j];
//			if((delta_src.pos_[fix_idx] - mean_face.pos_[fix_idx]).norm()>1e-8)
//			{
//				delta_dst.pos_[fix_idx] = delta_src.pos_[fix_idx];
//			}
//		}
		delta_dst.saveObj(result_root + i);		
	}	
}

void TESTFUNCTION::testBatchDeformTranId()
{
	MeshCompress template_render("D:/data/0715_project_test_gdg_test_1717/bfm_lap_sys.obj");
	float3E translate = float3E(0.000000000, 150.939240, -2.90438318);
	float guijie_scale = 37.24;
	std::string cur_root = "D:/avatar/0717bs/";
	std::string result_root = "D:/avatar/0722_batch_dynamic_fix_tid/";
	SG::needPath(result_root);
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(cur_root, FILEIO::FILE_TYPE::MESH);


	//get A A' B B'
	MeshCompress mean_face, mean_face_dst, delta_src, delta_dst;
	mean_face.loadObj(cur_root + "mean.obj");
	delta_src.loadObj(cur_root + "Ani_eyeSquintLeft.obj");
	RT::translateInPlace(-translate, mean_face.pos_);
	RT::translateInPlace(-translate, delta_src.pos_);
	float render_scale = template_render.scale_ / mean_face.scale_;
	RT::scaleInPlaceNoShift(render_scale, mean_face.pos_);
	RT::scaleInPlaceNoShift(render_scale, delta_src.pos_);

	//trans_exp.init(mean_face, mean_face_dst, res, fix);
	intVec res = FILEIO::loadIntDynamic("D:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_down_match.txt", res);
	//FILEIO::loadIntDynamic("D:/avatar/guijie/right_eye_close.txt", res);
	intVec fix = FILEIO::loadIntDynamic("D:/avatar/guijie/fix_head_sys.txt");
	MeshTransfer trans_exp;
	trans_exp.initDynamicFix(mean_face, delta_src, res, 0.001);
	for (auto i : folder_file)	{
		
		mean_face_dst.loadObj(cur_root + i);
		RT::translateInPlace(-translate, mean_face_dst.pos_);
		RT::scaleInPlaceNoShift(render_scale, mean_face_dst.pos_);

		//trans_exp.init(mean_face, mean_face_dst, res, fix);
		//for rendering
		delta_dst = delta_src;
		trans_exp.transfer(mean_face_dst.pos_, delta_dst.pos_);
		delta_dst.saveObj(result_root + i);
	}
}

void TESTFUNCTION::testReg()
{
	std::string cur_root = "D:/data/0521_04/";
	std::string whole_mesh = cur_root + "all_step0_lap.obj";
	std::string part_mesh = cur_root + "part_0_st.obj";
	std::string marker = cur_root + "whole_part.cons";
	std::string save_name = cur_root + "nricp_test_00.obj";

	intVec all_move = FILEIO::loadIntDynamic(cur_root + "round_6.txt");
	intVec eye_left = FILEIO::loadIntDynamic(cur_root + "skip_left.txt");
	intVec eye_right = FILEIO::loadIntDynamic(cur_root + "skip_right.txt");
	intVec eye_left_bound = FILEIO::loadIntDynamic(cur_root + "eye_left.txt");
	intVec eye_right_bound = FILEIO::loadIntDynamic(cur_root + "eye_right.txt");

	intVec all_select;
	for (int i : all_move)
	{
		if (std::find(eye_left.begin(), eye_left.end(), i) == eye_left.end() && std::find(eye_right.begin(), eye_right.end(), i) == eye_right.end())
		{
			all_select.push_back(i);
		}
		else
		{
		
		}
	}
	all_select.insert(all_select.end(), eye_left_bound.begin(), eye_left_bound.end());
	all_select.insert(all_select.end(), eye_right_bound.begin(), eye_right_bound.end());

	std::vector<demo::Mesh*> res;
	NRICP::reg(whole_mesh, marker, part_mesh, all_move, save_name, 10, res);
}

void TESTFUNCTION::testRigid()
{
	MeshCompress src("D:/data/0520_3dmm/part_0.obj");
	MeshCompress dst("D:/data/0520_3dmm/whole_0.obj");
	std::string marker = "D:/data/0518_3dmm/whole_part.cons";
	intVec src_point, dst_point;
	int num = FILEIO::getPair(marker, dst_point, src_point);
	float3Vec dst_pos, src_pos;
	src.getSlice(src_point, src_pos);
	dst.getSlice(dst_point, dst_pos);
	float scale = RT::getScale(src_pos, dst_pos);
	RT::scaleInPlace(scale, src.pos_);
	float3E src_center;
	RT::getCenter(src.pos_, src_center);
	LOG(INFO) << "Print center: " << src_center << std::endl;
	src.saveObj("D:/data/0520_3dmm/part_0_s.obj");
	//get slice for scaled pos
	src.getSlice(src_point, src_pos);
	dst.getSlice(dst_point, dst_pos);
	float3E translate;
	RT::getTranslate(src_pos, dst_pos, translate);
	LOG(INFO) << "translate: " << translate << std::endl;
	//set x to zero
	translate[0] = 0;
	RT::translateInPlace(translate, src.pos_);
	src.saveObj("D:/data/0520_3dmm/part_0_st.obj");
	LOG(INFO) << "fix transform by base: " << std::endl << "scale: " << scale << std::endl << "translate :" << translate << std::endl;
	return;
	mat4f trans;
	//ICP::bestFitTransform(src_pos, src_pos, trans);
	ICP::fitTransform(src_pos, src_pos, trans);
	LOG(INFO) << "trans: "<<std::endl << trans << std::endl;
	RT::transformInPlace(trans, src.pos_);
	src.saveObj("D:/data/0518_3dmm/tb01_close_revise_rts.obj");
}

void TESTFUNCTION::testBaseGen()
{
	std::string cur_root = "D:/data/0521_00/";
	MeshCompress part(cur_root + "part_0_st.obj");
	MeshCompress all(cur_root + "whole_0.obj");
	std::string marker = cur_root + "whole_part.cons";
	intVec all_move = FILEIO::loadIntDynamic(cur_root + "round_3.txt");
	intVec all_point, part_point;
	intVec all_fix;
	for (int i = 0; i < all.n_vertex_; i++)
	{
		if (std::find(all_move.begin(), all_move.end(), i) == all_move.end())
		{
			all_fix.push_back(i);
		}
	}
	int num = FILEIO::getPair(marker, all_point, part_point);
	//lap deform
	LaplacianDeform nicp;	
	MeshCompress all_result = all;
	nicp.init(all_result, all_point, all_fix);
	float3Vec deform_pos;
	for (size_t i = 0; i < all_point.size(); i++)
	{
		//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
		float3E temp = part.pos_[part_point[i]];
		deform_pos.push_back(temp);
	}
	nicp.deform(deform_pos, all_result.pos_);
	all_result.saveObj(cur_root + "all_step0_lap.obj");
	
}

void TESTFUNCTION::testRectify()
{
	std::string cur_root = "D:/data/0521_07/";
	MeshCompress all_ori(cur_root + "whole_0.obj");
	MeshCompress all_icp(cur_root + "nricp_test_00.obj3.obj");
	
	intVec roi;
	
	intVec round_0 = FILEIO::loadIntDynamic(cur_root + "round_0.txt");
	intVec round_6 = FILEIO::loadIntDynamic(cur_root + "round_6.txt");
	
	intVec eye_left = FILEIO::loadIntDynamic(cur_root + "skip_left.txt");
	intVec eye_right = FILEIO::loadIntDynamic(cur_root + "skip_right.txt");
	intVec eye_left_bound = FILEIO::loadIntDynamic(cur_root + "eye_left.txt");
	intVec eye_right_bound = FILEIO::loadIntDynamic(cur_root + "eye_right.txt");


	intVec all_point;

	for (int i = 0; i < all_ori.n_vertex_; i++)
	{
		if (std::find(round_0.begin(), round_0.end(), i) == round_0.end())
		{
			//not in round 0 handle points
			all_point.push_back(i);
		}
		else
		{
			if (std::find(round_6.begin(), round_6.end(), i) != round_6.end())
			{
				if (std::find(eye_left.begin(), eye_left.end(), i) == eye_left.end() && std::find(eye_right.begin(), eye_right.end(), i) == eye_right.end())
				{
					all_point.push_back(i);
				}
				else
				{

				}
			}
		}
	}
	all_point.insert(all_point.end(), eye_left_bound.begin(), eye_left_bound.end());
	all_point.insert(all_point.end(), eye_right_bound.begin(), eye_right_bound.end());
	//lap deform
	LaplacianDeform nicp;
	MeshCompress all_result = all_ori;
	nicp.init(all_result, all_point, {});
	float3Vec deform_pos;
	for (size_t i = 0; i < all_point.size(); i++)
	{
		//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
		float3E temp = all_icp.pos_[all_point[i]];
		deform_pos.push_back(temp);
	}
	nicp.deform(deform_pos, all_result.pos_);
	all_result.saveObj(cur_root + "all_step1_lap.obj");
}

void TESTFUNCTION::cleanDeep3DPCA()
{

	MeshCompress mean_fwh("D:/data/0603_00/fwh_pca/basis_0.obj");
	MeshCompress src_bfm("D:/data/0603_01/cs_rt/cs_0.obj");
	intVec fwh_id = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_id.txt");
	intVec bfw_id = FILEIO::loadIntDynamic("D:/data/0623_01/bfm_id.txt");
	float3Vec bfm_pos, fwh_pos;
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	float3E translate;
	RT::getTranslate(bfm_pos, fwh_pos, translate);
	//fix transpose for all bfm meshed
	MeshCompress deep_3d_0("D:/code/Deep3DFaceReconstruction-pytorch/output_pca/pca_0.obj");
	MeshCompress basis_0("D:/data/0623_01/basis_0.obj");
	float3Vec deep_3d_0_pos, basis_0_pos;
	deep_3d_0.getSlice(bfw_id, deep_3d_0_pos);
	basis_0.getSlice(bfw_id, basis_0_pos);
	//scale from deep_3d to basis
	float scale = RT::getScale(deep_3d_0_pos, basis_0_pos);
	RT::scaleInPlace(scale, deep_3d_0.pos_);
	deep_3d_0.getSlice(bfw_id, deep_3d_0_pos);
	float3E translate_deep;
	RT::getTranslate(deep_3d_0_pos, basis_0_pos, translate_deep);
	RT::translateInPlace(translate_deep + translate, deep_3d_0.pos_);
	deep_3d_0.saveObj("D:/data/0623_01/pca_0_rt.obj");

	cstr deep_3d_folder = "D:/code/Deep3DFaceReconstruction-pytorch/output_pca/";
	cstr deep_3d_rt_folder = "D:/data/0623_01/deep3d_rt_clean/";
	SG::needPath(deep_3d_rt_folder);
	MeshCompress deep_3d_0_anchor("D:/code/Deep3DFaceReconstruction-pytorch/output_pca/pca_0.obj");

	intVec basis_left_eye = FILEIO::loadIntDynamic("D:/data/0518_3dmm/left_eye.txt");
	intVec basis_right_eye = FILEIO::loadIntDynamic("D:/data/0518_3dmm/right_eye.txt");
	intVec both_eye = basis_left_eye;
	both_eye.insert(both_eye.end(), basis_right_eye.begin(), basis_right_eye.end());


	for (int i = -1; i < 80; i++)
	{
		std::string deep_3d_file = deep_3d_folder + "pca_" + std::to_string(i) + ".obj";		
		MeshCompress deep_3d_iter(deep_3d_file);
#pragma omp parallel for
		for (int iter_ver = 0; iter_ver < deep_3d_iter.n_vertex_; iter_ver++)
		{
			deep_3d_iter.pos_[iter_ver] = (deep_3d_iter.pos_[iter_ver] - deep_3d_0_anchor.pos_[iter_ver])*scale + deep_3d_0.pos_[iter_ver];
		}
		deep_3d_iter.discard(both_eye);
		deep_3d_iter.saveObj(deep_3d_rt_folder+ "pca_" + std::to_string(i) + ".obj");
	}


#if 0

	cstr cur_root = "";
	intVec basis_left_eye = FILEIO::loadIntDynamic("D:/data/0518_3dmm/left_eye.txt");
	intVec basis_right_eye = FILEIO::loadIntDynamic("D:/data/0518_3dmm/right_eye.txt");
	intVec both_eye = basis_left_eye;
	both_eye.insert(both_eye.end(), basis_right_eye.begin(), basis_right_eye.end());
	basis_mesh.discard(both_eye);
	basis_mesh.saveObj(cur_root + "/clean/part_" + std::to_string(i) + ".obj");
	

#endif


}

void TESTFUNCTION::mappingFromOriToDiscard()
{
	intVec bfw_discard = FILEIO::loadIntDynamic("D:/data/0713_point/bfw_discard.txt");
	intVec all_68 = FILEIO::loadIntDynamic("D:/data/0713_point/68_whole.txt");
	intVec part_68 = all_68;
	MeshCompress all_mesh("D:/data/0713_mapping/basis_0.obj");
	MeshCompress part_mesh("D:/data/0713_mapping/part_0.obj");
	intVec map_all_part = all_mesh.getDiscardMap(bfw_discard);
	int n_vertex = all_mesh.n_vertex_;
	int n_part = part_mesh.n_vertex_;
	int n_landmark = all_68.size();
	intVec part_visited(n_part, 0);
	for (size_t i = 0; i < n_landmark; i++)
	{
		int all_id = all_68[i];
		int part_id = map_all_part[all_id];
		if (part_id < 0)
		{
			//matching pos is not valid
			floatVec match_dis(n_part, INT_MAX);
#pragma omp parallel for
			for (int j = 0; j < n_part; j++)
			{
				if (part_visited[j])
				{

				}
				else
				{
					match_dis[j] = (part_mesh.pos_[j] - all_mesh.pos_[all_id]).norm();
				}
			}
			auto res = std::minmax_element(match_dis.begin(), match_dis.end());
			int match_id = res.first - match_dis.begin();
			part_68[i] = match_id;
			part_visited[match_id] = 1;
		}
		else
		{
			part_68[i] = part_id;
			part_visited[part_id] = 1;
		}

	}
	part_68[39] = 6036;
	part_68[42] = 9528;
	FILEIO::saveDynamic("D:/data/0713_mapping/part_68_refine.txt", part_68, ",");
}

void TESTFUNCTION::backProject()
{
	MeshCompress all_mesh("D:/data/0713_mapping/basis_0.obj");
	intVec discard_id = FILEIO::loadIntDynamic("D:/data/0713_mapping/discard.txt");
	all_mesh.discard(discard_id);
	all_mesh.saveObj("D:/data/0713_mapping/basis_0_dis.obj");
	MeshCompress part_mesh("D:/data/0713_mapping/part_0.obj");
	int num_all = all_mesh.n_vertex_;
	int num_part = part_mesh.n_vertex_;
	intVec all_vertex(num_all, 0);
	intVec part_vertex(num_part, 0);
	intVec all_vertex_part(num_all, -1);

	for (int i = 0; i < num_part; i++)
	{
		if (part_vertex[i])
		{
			//already visited
		}
		else
		{
			floatVec match_dis(num_all, INT_MAX);
#pragma omp parallel for
			for (int j = 0; j < num_all; j++)
			{
				if (all_vertex[j])
				{

				}
				else
				{
					match_dis[j] = (part_mesh.pos_[i] - all_mesh.pos_[j]).norm();					
				}
			}
			auto res = std::minmax_element(match_dis.begin(), match_dis.end());
			part_vertex[i] = 1;
			int match_id = res.first - match_dis.begin();
			all_vertex[match_id] = 1;			
			all_vertex_part[match_id] = i;
		}
	}
	intVec discard_vertex;
	for (int i = 0; i < num_all; i++)
	{
		if (all_vertex_part[i] < 0)
		{
			discard_vertex.push_back(i);
		}
	}
	FILEIO::saveDynamic("D:/data/0713_mapping/discard.txt", discard_vertex, ",");
}

void TESTFUNCTION::testBatchGen()
{
	std::string cur_root = "D:/data/0602_02/";
	int num = 80;
	for (int i = 0; i < num; i++)
	{
		std::string part_file = cur_root + "/basis_faces/basis_" + std::to_string(i) + ".obj";
		std::string whole_file = cur_root + "out_resolve_dang.obj";
		std::string marker = cur_root+ "fwh_part.cons";
		MeshCompress basis_mesh(part_file);
		intVec basis_left_eye = FILEIO::loadIntDynamic("D:/data/0518_3dmm/left_eye.txt");
		intVec basis_right_eye = FILEIO::loadIntDynamic("D:/data/0518_3dmm/right_eye.txt");
		intVec both_eye = basis_left_eye;
		both_eye.insert(both_eye.end(), basis_right_eye.begin(), basis_right_eye.end());
		basis_mesh.discard(both_eye);
		basis_mesh.saveObj(cur_root + "/clean/part_"+std::to_string(i)+".obj");

#if 0
		MeshCompress part_mesh(cur_root + "/clean/part_" + std::to_string(i) + ".obj");
		MeshCompress dst("D:/data/0520_3dmm/whole_0.obj");
		intVec src_point, dst_point;
		int num = FILEIO::getPair(marker, dst_point, src_point);
		float3Vec dst_pos, src_pos;
		part_mesh.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float scale = RT::getScale(src_pos, dst_pos);
		RT::scaleInPlace(scale, part_mesh.pos_);
		float3E src_center;
		RT::getCenter(part_mesh.pos_, src_center);
		//get slice for scaled pos
		part_mesh.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float3E translate;
		RT::getTranslate(src_pos, dst_pos, translate);
		//set x to zero
		translate[0] = 0;
		RT::translateInPlace(translate, part_mesh.pos_);
		part_mesh.saveObj(cur_root+"result/part_"+std::to_string(i)+"_st.obj");
#else
		//pos reverse
		MeshCompress part_mesh(cur_root + "/clean/part_" + std::to_string(i) + ".obj");
		MeshCompress dst(whole_file);
		intVec src_point, dst_point;
		int num = FILEIO::getPair(marker, dst_point, src_point);
		float3Vec dst_pos, src_pos;
		part_mesh.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float scale = RT::getScale(dst_pos, src_pos);
		RT::scaleInPlace(scale, dst.pos_);
		float3E src_center;
		RT::getCenter(part_mesh.pos_, src_center);
		//get slice for scaled pos
		part_mesh.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float3E translate;
		RT::getTranslate(dst_pos, src_pos, translate);
		//set x to zero
		translate[0] = 0;
		RT::translateInPlace(translate, dst.pos_);
		part_mesh.saveObj(cur_root + "result/part_" + std::to_string(i) + "_st.obj");
		dst.saveObj(cur_root + "result/whole_" + std::to_string(i) + "_st.obj");
#endif
		MeshCompress part(cur_root + "result/part_" + std::to_string(i) + "_st.obj");
		MeshCompress all(cur_root + "result/whole_" + std::to_string(i) + "_st.obj");

		intVec all_move = FILEIO::loadIntDynamic(cur_root + "round_fwh_0.txt");
		intVec all_point, part_point;
		intVec all_fix;
		for (int i = 0; i < all.n_vertex_; i++)
		{
			if (std::find(all_move.begin(), all_move.end(), i) == all_move.end())
			{
				all_fix.push_back(i);
			}
		}
		int num_pt = FILEIO::getPair(marker, all_point, part_point);
		//lap deform
		LaplacianDeform nicp;
		MeshCompress all_result = all;
		nicp.init(all_result, all_point, all_fix);
		float3Vec deform_pos;
		for (size_t i = 0; i < all_point.size(); i++)
		{
			//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
			float3E temp = part.pos_[part_point[i]];
			deform_pos.push_back(temp);
		}
		nicp.deform(deform_pos, all_result.pos_);
		all_result.saveObj(cur_root + "result/all_step"+std::to_string(i)+"_lap.obj");


		std::string whole_mesh = cur_root + "result/all_step" + std::to_string(i) + "_lap.obj";
		std::string part_mesh_file = cur_root + "/result/part_" + std::to_string(i) + "_st.obj";
		std::string save_name = cur_root + "result/nicp_" + std::to_string(i) + "_nicp.obj";

		intVec all_move_lap = FILEIO::loadIntDynamic(cur_root + "round_fwh_0.txt");
		intVec eye_left = FILEIO::loadIntDynamic(cur_root + "eye_left.txt");
		intVec eye_right = FILEIO::loadIntDynamic(cur_root + "eye_right.txt");
		intVec left_ear = FILEIO::loadIntDynamic(cur_root + "left_ear.txt");
		intVec right_ear = FILEIO::loadIntDynamic(cur_root + "right_ear.txt");
		left_ear = {};
		right_ear = {};
		intVec fwh_fix = FILEIO::loadIntDynamic(cur_root + "fwh_fix.txt");
		intVec all_select;
		for (int i : all_move_lap)
		{
			if (std::find(eye_left.begin(), eye_left.end(), i) == eye_left.end() && std::find(eye_right.begin(), eye_right.end(), i) == eye_right.end())
			{
				all_select.push_back(i);
			}
			else
			{

			}
		}
		//all_select.insert(all_select.end(), eye_left_bound.begin(), eye_left_bound.end());
		//all_select.insert(all_select.end(), eye_right_bound.begin(), eye_right_bound.end());

		std::vector<demo::Mesh*> res;
		std::vector<int> mesh_boundary = FILEIO::loadIntDynamic(cur_root + "boundary.txt");
		if (!boost::filesystem::exists(cur_root + "result/nicp_" + std::to_string(i) + "_nicp.obj3.obj"))
		{
			NRICP::reg(whole_mesh, marker, part_mesh_file, all_move_lap, mesh_boundary, save_name, 10, res);
			//NRICP::reg(whole_mesh, marker, part_mesh_file, all_move_lap, save_name, 10, res);
			//NRICP::reg(whole_mesh, marker, part_mesh_file, all_move_lap, save_name, 10, res);
		}

		MeshCompress all_ori(cur_root + "result/whole_" + std::to_string(i) + "_st.obj");
		MeshCompress all_icp(cur_root + "result/nicp_" + std::to_string(i) + "_nicp.obj3.obj");

		intVec roi;

		intVec round_0 = FILEIO::loadIntDynamic(cur_root + "round_fwh_2.txt");
		intVec round_6 = {};

		all_point.clear();

		for (int i : round_0)
		{
			if (std::find(eye_left.begin(), eye_left.end(), i) == eye_left.end()
				&& std::find(eye_right.begin(), eye_right.end(), i) == eye_right.end())
			{
				//not in round 0 handle points
				all_point.push_back(i);
			}
			
		}
		//all_point.insert(all_point.end(), eye_left_bound.begin(), eye_left_bound.end());
		//all_point.insert(all_point.end(), eye_right_bound.begin(), eye_right_bound.end());
		//all_point = round_0;
		//lap deform
		LaplacianDeform recify;
		all_result = all_ori;
		intVec rec_bound = fwh_fix;
		rec_bound.insert(rec_bound.end(), right_ear.begin(), right_ear.end());
		recify.init(all_result, all_point, rec_bound);
		deform_pos.clear();
		for (size_t i = 0; i < all_point.size(); i++)
		{
			//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
			float3E temp = all_icp.pos_[all_point[i]];
			deform_pos.push_back(temp);
		}
		recify.deform(deform_pos, all_result.pos_);
		all_result.saveObj(cur_root + "result/rectify_" + std::to_string(i) + ".obj");

	}
}

void TESTFUNCTION::triToQuadFace(const cstr& in_file, const cstr& out_file)
{
	//std::string in_file = "D:/dota201224/1228_generate_head/debug/toothbs_result/";
	//std::string out_file = "D:/dota201224/1228_generate_head/toFbx/tooth_ani/";
	SG::needPath(out_file);
	cstrVec quad_mesh_base =
	{
		"D:/multiPack/1217_hf_quad/base/head.obj",//4230
		"D:/multiPack/1217_hf_quad/base/eyes.obj",//966
		"D:/multiPack/1217_hf_quad/base/eyebrow.obj",//154
		"D:/multiPack/1217_hf_quad/base/tooth.obj",//237
	};

	cstrVec in_names = FILEIO::getFolderFiles(in_file, ".obj");
	for (auto i: in_names)
	{
		MeshCompress iter_in(in_file + i);
		std::string out_replace = out_file +i;
		MeshCompress obj_template;
		cstr raw_name = FILEIO::getFileNameWithoutExt(i);
		if (iter_in.n_vertex_ == 4230)
		{
			obj_template.loadOri(quad_mesh_base[0]);
		}
		else if (iter_in.n_vertex_ == 966)
		{
			obj_template.loadOri(quad_mesh_base[1]);
		}
		else if (iter_in.n_vertex_ == 154)
		{
			obj_template.loadOri(quad_mesh_base[2]);
		}
		else if (iter_in.n_vertex_ == 237)
		{
			obj_template.loadOri(quad_mesh_base[3]);
		}
		else
		{
			LOG(ERROR) << "no template match" << std::endl;
		}

		obj_template.pos_ = iter_in.pos_;
		obj_template.setGOption(raw_name);
		obj_template.saveOri(out_replace);
	}
}

void TESTFUNCTION::triToQuadInPlace(const cstr& root)
{
	std::string in_file = root;
	cstrVec quad_mesh_base =
	{
		"D:/dota201201/1217_hf_quad/base/head.obj",//4230
		"D:/dota201201/1217_hf_quad/base/eyes.obj",//966
		"D:/dota201201/1217_hf_quad/base/eyebrow.obj",//154
		"D:/dota201201/1217_hf_quad/base/tooth.obj",//237
	};

	cstrVec in_names = FILEIO::getFolderFiles(in_file, { ".obj" }, true);
	for (auto i : in_names)
	{
		MeshCompress iter_in(i);
		std::string out_replace = i;
		MeshCompress obj_template;
		cstr raw_name = FILEIO::getFileNameWithoutExt(i);
		if (iter_in.n_vertex_ == 4230)
		{
			obj_template.loadOri(quad_mesh_base[0]);
		}
		else if (iter_in.n_vertex_ == 966)
		{
			obj_template.loadOri(quad_mesh_base[1]);
		}
		else if (iter_in.n_vertex_ == 154)
		{
			obj_template.loadOri(quad_mesh_base[2]);
		}
		else if (iter_in.n_vertex_ == 237)
		{
			obj_template.loadOri(quad_mesh_base[3]);
		}
		else
		{
			LOG(ERROR) << "no template match" << std::endl;
		}

		obj_template.pos_ = iter_in.pos_;
		obj_template.setGOption(raw_name);
		obj_template.saveOri(out_replace);
	}
}

void TESTFUNCTION::toMayaNamespace()
{
	std::string in_file = "D:/dota201201/1217_hf_nielian/nielian_ori/";
	std::string out_file = "D:/dota201201/1217_hf_nielian/nielian_maya/";
	SG::needPath(out_file);
	cstrVec in_names = FILEIO::getFolderFiles(in_file, ".obj");
	for (auto i : in_names)
	{
		MeshCompress iter_in(in_file + i);
		cstr raw_name = FILEIO::getFileNameWithoutExt(i);
		iter_in.setGOption(raw_name);
		std::string out_replace = out_file + i;		
		iter_in.saveObj(out_replace);
	}
}

void TESTFUNCTION::testFWH()
{
	Tensor src_tensor;
	JsonHelper::initData("D:/data/server_pack/fwh/", "config.json", src_tensor);
	src_tensor.saveByIdExp("D:/data/0528_03/", 2);


	json config;
	std::ifstream fin("D:/data/0528_01/config.json");
	fin >> config;
	fin.close();
	std::string cur_root = config["root"];
	std::string src_mesh_prefix = config["src_mesh"];
	std::string dst_mesh_prefix = config["dst_mesh"];
	std::string src_point_idx = config["src_point"];
	std::string dst_point_idx = config["dst_point"];
	std::string result_root = config["result"];
	intVec src_point, dst_point;
	src_point = FILEIO::loadIntDynamic(cur_root + src_point_idx);
	dst_point = FILEIO::loadIntDynamic(cur_root + dst_point_idx);
	result_root = cur_root + result_root;
	std::ofstream out_con(result_root + "fwh_28k.cons");
	out_con << src_point.size() << std::endl;
	for (int i = 0; i < src_point.size(); i++)
	{
		out_con << src_point[i] << "," << dst_point[i] << std::endl;
	}
	std::ofstream out_con_reverse(result_root + "26k_fwh.cons");
	out_con_reverse << src_point.size() << std::endl;
	for (int i = 0; i < src_point.size(); i++)
	{
		out_con_reverse << dst_point[i] << ","<< src_point[i] << std::endl;
	}
	out_con.close();
	int num = config["num"];
	for (int i = 0; i < num; i++)
	{
		//move dst to src
		MeshCompress src(cur_root + src_mesh_prefix + std::to_string(i) + ".obj");
		MeshCompress dst(cur_root + dst_mesh_prefix + std::to_string(i) + ".obj");

		int num = src_point.size();
		float3Vec dst_pos, src_pos;
		src.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float scale = RT::getScale(dst_pos, src_pos);
		RT::scaleInPlace(scale, dst.pos_);
		float3E src_center;
		RT::getCenter(src.pos_, src_center);
		//get slice for scaled pos
		src.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float3E translate;
		RT::getTranslate(dst_pos, src_pos, translate);
		//set x to zero
		translate[0] = 0;
		RT::translateInPlace(translate, dst.pos_);
		src.saveObj(result_root + src_mesh_prefix + std::to_string(i)+ " _st.obj");
		dst.saveObj(result_root + dst_mesh_prefix + std::to_string(i) + "_st.obj");

		LOG(INFO) << "finish transform keep src still, move dst." << std::endl;

		//lap deform
		LaplacianDeform nicp;
		nicp.init(src, src_point, {});
		float3Vec deform_pos;
		for (size_t i = 0; i < src_point.size(); i++)
		{
			//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
			float3E temp = dst.pos_[dst_point[i]];
			deform_pos.push_back(temp);
		}
		nicp.deform(deform_pos, src.pos_);
		src.saveObj(result_root + src_mesh_prefix + std::to_string(i) + "_lap.obj");
		LOG(INFO) << "finish deform src to dst." << std::endl;

		intVec move_region;
		for (int i = 0; i < src.n_vertex_; i++)
		{
			move_region.push_back(i);
		}

		std::vector<demo::Mesh*> res;
#if 0
		NRICP::reg(result_root + src_mesh_prefix + std::to_string(i) + "_lap.obj", 
			result_root + dst_mesh_prefix + std::to_string(i) + "_st.obj",
			src_point, dst_point, move_region, result_root + std::to_string(i) + "_nicp.obj", 10, res);
#endif
		//NRICP::reg(whole_mesh, marker, part_mesh_file, all_move_lap, save_name, 10, res);
		LOG(INFO) << "end of nicp process." << std::endl;

		//reproject
		MeshCompress raw_src(result_root + std::to_string(i) + "_nicp.obj3.obj");
		for (int weight = 0; weight < 5; weight++)
		{
			floatVec reg(src_tensor.n_id_ - 1, weight);
			floatVec coef_res;
			src_tensor.fitID(raw_src.pos_, reg, coef_res);
			floatVec proj_res = src_tensor.interpretID(coef_res);
			SG::safeMemcpy(src.pos_.data(), proj_res.data(), proj_res.size() * sizeof(float));
			src.saveObj(result_root + src_mesh_prefix + std::to_string(i) + "_proj_"+ std::to_string(weight)+"_.obj");
			LOG(INFO) << "end of project process." << std::endl;
		}

#if 0

		MeshCompress all_ori(cur_root + "result/whole_" + std::to_string(i) + "_st.obj");
		MeshCompress all_icp(cur_root + "result/nicp_" + std::to_string(i) + "_nicp.obj3.obj");

		intVec roi;

		intVec round_0 = FILEIO::readIntDynamic(cur_root + "round_0.txt");
		intVec round_6 = FILEIO::readIntDynamic(cur_root + "round_6.txt");

		all_point.clear();

		for (int i = 0; i < all_ori.n_vertex_; i++)
		{
			if (std::find(round_0.begin(), round_0.end(), i) == round_0.end())
			{
				//not in round 0 handle points
				all_point.push_back(i);
			}
			else
			{
				if (std::find(round_6.begin(), round_6.end(), i) != round_6.end())
				{
					if (std::find(eye_left.begin(), eye_left.end(), i) == eye_left.end() && std::find(eye_right.begin(), eye_right.end(), i) == eye_right.end())
					{
						all_point.push_back(i);
					}
					else
					{

					}
				}
			}
		}
		all_point.insert(all_point.end(), eye_left_bound.begin(), eye_left_bound.end());
		all_point.insert(all_point.end(), eye_right_bound.begin(), eye_right_bound.end());
		//lap deform
		LaplacianDeform recify;
		all_result = all_ori;
		recify.init(all_result, all_point, {});
		deform_pos.clear();
		for (size_t i = 0; i < all_point.size(); i++)
		{
			//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
			float3E temp = all_icp.pos_[all_point[i]];
			deform_pos.push_back(temp);
		}
		recify.deform(deform_pos, all_result.pos_);
		all_result.saveObj(cur_root + "result/rectify_" + std::to_string(i) + ".obj");
#endif
	}
}

void TESTFUNCTION::testExtractLandmarks()
{
	json fwh;
	fwh["src_top"] = "D:/data/0528_00/id_0_exp_0.obj";
	fwh["src_land_id"] = "D:/data/0528_00/fwh_land.txt";
	fwh["dst_top"] = "D:/data/0528_00/id_0_exp_0.obj";
	fwh["dst_sys"] = "D:/data/0528_00/match.txt";
	fwh["dst_mid"] = "D:/data/0528_00/mid.txt";
	fwh["dst_land_id"] = "D:/data/0528_00/fwh_id.txt";
	fwh["landmark_mid"] = "D:/data/0528_00/landmark_mid.txt";
	fwh["landmark_match"] = "D:/data/0528_00/landmark_match.txt";
	SysFinder::rectifyLandmarkPos(fwh);


	json file_path;
	file_path["src_top"] = "D:/data/0521_08/whole_0.obj";
	file_path["src_land_id"] = "D:/data/0521_08/whole_land.txt";
	file_path["dst_top"] = "D:/data/0527_02/26k.obj";
	file_path["dst_sys"] = "D:/data/0527_02/match.txt";
	file_path["dst_mid"] = "D:/data/0527_02/mid.txt";
	file_path["dst_land_id"] = "D:/data/0527_02/26k_id.txt";
	file_path["landmark_mid"] = "D:/data/0527_02/landmark_mid.txt";
	file_path["landmark_match"] = "D:/data/0527_02/landmark_match.txt";
	SysFinder::rectifyLandmarkPos(file_path);
}

void TESTFUNCTION::testCorrect26k()
{
	MeshCompress src("D:/data/0529_04/26k_refine.obj");
	MeshCompress dst("D:/data/0529_04/out_resolve.obj");
	intVec need_fix = FILEIO::loadIntDynamic("D:/data/0529_04/wrong_region_pro.txt");
	intVec left_ear = FILEIO::loadIntDynamic("D:/data/0529_04/left_ear.txt");
	intVec right_ear = FILEIO::loadIntDynamic("D:/data/0529_04/right_ear.txt");
	intVec mouth_fix = FILEIO::loadIntDynamic("D:/data/0529_04/fix_region.txt");
	//get
	LaplacianDeform nicp;
	intVec handle_points;
	intSet fix_set(need_fix.begin(), need_fix.end());
	intSet left_set(left_ear.begin(), left_ear.end());
	intSet right_set(right_ear.begin(), right_ear.end());
	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (!fix_set.count(i) && !left_set.count(i) && !right_set.count(i))
		{
			handle_points.push_back(i);
		}
	}
	nicp.init(src, handle_points, mouth_fix);
	float3Vec deform_pos;
	for (size_t i = 0; i < handle_points.size(); i++)
	{
		//float3E temp = Loader.mesh_data_.pos_[handle[i]] + float3E(0, 0.1, 0);
		float3E temp = dst.pos_[handle_points[i]];
		deform_pos.push_back(temp);
	}
	nicp.deform(deform_pos, src.pos_);
	src.saveObj("D:/data/0529_04/out_resolve_fix_ear.obj");
	//second phase
	LaplacianDeform second_close_lip;
	intVec up = { 1733,1734,1738,1739,1740,1901,5167,5168,5171,5172,5328,9690,9697,9700,9706,10004,16474,16481,16482,16489,16782, };
	intVec down = { 2033,2168,2169,2174,2185,5453,5571,5574,5583,10536,10542,10565,10568,17290,17297,17316,17319, };
	float mid_up_z = 0;
	float mid_down_z = 0;
	for (auto i: up)
	{
		mid_up_z += src.pos_[i].z();
		handle_points.push_back(i);
	}
	mid_up_z = mid_up_z / up.size();

	for (auto i : down)
	{
		mid_down_z += src.pos_[i].z();
		handle_points.push_back(i);
	}

	mid_down_z = mid_down_z / down.size();
	float mid_z = mid_up_z * 0.5 + mid_down_z * 0.5;
	for (auto i : up)
	{
		float3E temp = src.pos_[i];
		temp.z() = mid_z;
		deform_pos.push_back(temp);
	}
	for (auto i : down)
	{
		float3E temp = src.pos_[i];
		temp.z() = mid_z;
		deform_pos.push_back(temp);
	}

	second_close_lip.init(src, handle_points, {});
	second_close_lip.deform(deform_pos, src.pos_);
	src.saveObj("D:/data/0529_03/out_resolve_fix_second.obj");


	//get mid

}

void TESTFUNCTION::testBatchBFMTOFWH()
{
	MeshSysFinder fwh;
	JsonHelper::initData("D:/data/0528_00/fwh_sys/", "config.json", fwh);	
	cstr result = "D:/data/0603_01/bfm_pos/";
	std::string cur_root = "D:/data/server_pack/bfm_fwh/";
	Tensor src_tensor;
	JsonHelper::initData(cur_root, "config.json", src_tensor);
	MeshCompress mean_fwh("D:/data/0603_00/fwh_pca/basis_0.obj");
	MeshCompress src_bfm("D:/data/0603_01/cs_rt/cs_0.obj");
	fwh.getSysPosLapInPlace(src_bfm.pos_);
	src_bfm.saveObj(result + "cs_0_sys.obj");
	intVec fwh_id = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_id.txt");
	float3Vec bfm_pos, fwh_pos;
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	//float scale = RT::getScale(bfm_pos, fwh_pos);
	//RT::scaleInPlace(scale, src_bfm.pos_);
	//get slice for scaled pos
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	float3E translate;
	RT::getTranslate(bfm_pos, fwh_pos, translate);
	//set x to zero
	LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, src_bfm.pos_);
	src_bfm.saveObj(result + "cs_0_trans.obj");
	int weight = 0;
	floatVec reg(src_tensor.n_id_ - 1, weight);
	floatVec coef_roi;
	intVec all_roi = FILEIO::loadIntDynamic("D:/data/0603_00/round_fwh_2.txt");
	intVec second_drag = FILEIO::loadIntDynamic("D:/data/0603_01/fwh_drag_s.txt");
	fwh.getSysIdsInPlace(second_drag);
	intVec fwh_fix = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_fix_neck_s_ear.txt");
	fwh.getSysIdsInPlace(fwh_fix);
	//intVec certain_set = { 64,66,67 };
	for (int i = 53; i < 80; i++)
	//for(int i: certain_set)
	{
		src_bfm.loadObj("D:/data/0603_01/cs_rt/cs_"+std::to_string(i)+ ".obj");
		MeshCompress res = src_bfm;
		RT::translateInPlace(translate, src_bfm.pos_);
		src_bfm.saveObj(result + "cs_" + std::to_string(i) + ".obj");
		src_tensor.fitID(src_bfm.pos_, reg, all_roi, src_tensor.ev_data_, coef_roi);
		if (isnan(coef_roi[0]))
		{
			LOG(INFO) << "went wrong" << std::endl;
		}
		std::cout << "fit with roi." << std::endl;
		std::for_each(coef_roi.begin(), coef_roi.end(), [](auto c) {std::cout << c << ","; });
		cout << std::endl;
		floatVec proj_res_roi = src_tensor.interpretID(coef_roi);
		SG::safeMemcpy(res.pos_.data(), proj_res_roi.data(), proj_res_roi.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(i) + "_roi.obj");
		LaplacianDeform to_bfm;
		float3Vec deform_pos;
		to_bfm.init(res, second_drag, fwh_fix);
		for (int iter_handle : second_drag)
		{
			deform_pos.push_back(src_bfm.pos_[iter_handle]);
		}
		to_bfm.deform(deform_pos, res.pos_);
		res.saveObj(result + "proj_" + std::to_string(i) + "_lap.obj");
		fwh.getSysPosLapInPlace(res.pos_);
		res.saveObj(result + "proj_" + std::to_string(i) + "_sys.obj");
		RT::translateInPlace(-translate, res.pos_);
		res.saveObj(result + "proj_" + std::to_string(i) + "_bfm.obj");
		LOG(INFO) << "end of project process." << std::endl;
	}
}

void TESTFUNCTION::testProject()
{
	MeshSysFinder fwh;
	JsonHelper::initData("D:/data/0528_00/fwh_sys/", "config.json", fwh);
	intVec test_sys_finder = { 0,1427,9 };
	fwh.getSysIdsInPlace(test_sys_finder);
	cstr result = "D:/data/0603_00/bfm_3nd/";
	std::string cur_root = "D:/data/server_pack/bfm_fwh/";
	Tensor src_tensor;
	JsonHelper::initData(cur_root, "config.json", src_tensor);
	//FILEIO::saveToBinary(cur_root + "id_40.bin", src_tensor.data_, sizeof(float)*120*src_tensor.template_obj_.n_vertex_);
	//FILEIO::saveToBinary(cur_root + "id_all.bin", src_tensor.data_);
	//src_tensor.saveByIdExp("D:/data/0603_00/tensor_test/");
	MeshCompress src_bfm("D:/data/0603_00/cs/cs_0.obj");
	MeshCompress mean_fwh("D:/data/0603_00/fwh_pca/basis_0.obj");
	intVec fwh_id = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_id.txt");

	float3Vec bfm_pos, fwh_pos;
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	//float scale = RT::getScale(bfm_pos, fwh_pos);
	//RT::scaleInPlace(scale, src_bfm.pos_);
	//get slice for scaled pos
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	float3E translate;
	RT::getTranslate(bfm_pos, fwh_pos, translate);
	//set x to zero
	translate[0] = 0;
	RT::translateInPlace(translate, src_bfm.pos_);
	src_bfm.saveObj(result + "cs_0_trans.obj");
	intVec all_roi = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_drag.txt");
	intVec second_drag = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_drag.txt");
	fwh.getSysIdsInPlace(second_drag);
	intVec fwh_fix = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_fix_neck_s.txt");
	fwh.getSysIdsInPlace(fwh_fix);
	//intVec fwh_fix = {};
	//src_bfm.loadObj(result + "cs_a_0.obj");
	MeshCompress res = src_bfm;
	int weight = 0;
	for (int iter_rep = 0; iter_rep < 5; iter_rep++)
	{
		floatVec reg(src_tensor.n_id_ - 1, weight);
		floatVec coef_res, coef_res_var, coef_roi;
		src_tensor.fitID(src_bfm.pos_, reg, coef_res);
		src_tensor.fitID(src_bfm.pos_, reg, src_tensor.ev_data_, coef_res_var);
		src_tensor.fitID(src_bfm.pos_, reg, all_roi, src_tensor.ev_data_, coef_roi);
		floatVec proj_res = src_tensor.interpretID(coef_res);
		std::cout << "fit with no var." << std::endl;
		std::for_each(coef_res.begin(), coef_res.end(), [](auto c) {std::cout << c << ","; });
		cout << std::endl;
		std::cout << "fit with var." << std::endl;
		std::for_each(coef_res_var.begin(), coef_res_var.end(), [](auto c) {std::cout << c << ","; });
		std::cout << endl;
		std::cout << "fit with roi." << std::endl;
		std::for_each(coef_roi.begin(), coef_roi.end(), [](auto c) {std::cout << c << ","; });
		cout << std::endl;
		floatVec proj_res_var = src_tensor.interpretID(coef_res_var);
		floatVec proj_res_roi = src_tensor.interpretID(coef_roi);
		SG::safeMemcpy(res.pos_.data(), proj_res.data(), proj_res.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_.obj");
		SG::safeMemcpy(res.pos_.data(), proj_res_var.data(), proj_res.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_var_.obj");
		SG::safeMemcpy(res.pos_.data(), proj_res_roi.data(), proj_res.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_roi_.obj");
		LaplacianDeform to_bfm;
		float3Vec deform_pos;
		to_bfm.init(res, second_drag, fwh_fix);
		for (int iter_handle:second_drag)
		{
			deform_pos.push_back(src_bfm.pos_[iter_handle]);
		}
		to_bfm.deform(deform_pos, res.pos_);
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_lap.obj");
		fwh.getSysPosLapInPlace(res.pos_);
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_sys.obj");
		src_bfm = res;
		LOG(INFO) << "end of project process." << std::endl;
	}
}

void TESTFUNCTION::testPCA()
{
	int fwh_dim = 149;
	//int bfm_dim = 76;
	int bfm_dim = 0;
	cstr input_fwh = "D:/data/0601_00/fwh_use/";
	cstr prefix_fwh = "Tester_";
	cstr input_bfm = "D:/data/0603_00/bfm_use/";
	cstr prefix_bfm = "proj_";
	cstr pca = "D:/data/0611_00/pca/";
	MeshCompress template_mesh("D:/data/0601_00/fwh/id_0_exp_0.obj");
	matD data_raw(fwh_dim+bfm_dim, template_mesh.n_vertex_ * 3);
	for (int i = 0; i < fwh_dim; i++)
	{
		MeshCompress iter_input(input_fwh + prefix_fwh + std::to_string(i) + "_center.obj");
//#pragma omp parallel for 
		for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
		{
			for (int j = 0; j < 3; j++) {
				data_raw(i, iter_vertex * 3 + j) = iter_input.pos_[iter_vertex](j);
			}
		}
	}

	for (int i = 0; i < bfm_dim; i++)
	{
		MeshCompress iter_input(input_bfm + prefix_bfm + std::to_string(i) + "_sys.obj");
		//#pragma omp parallel for 
		for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
		{
			for (int j = 0; j < 3; j++) {
				data_raw(i + fwh_dim, iter_vertex * 3 + j) = iter_input.pos_[iter_vertex](j);
			}
		}
	}

	vecD meanFace;
	NRICP::centerize(data_raw, meanFace);
	Eigen::BDCSVD <Eigen::MatrixXd> svd(data_raw, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto& eigVectors = svd.matrixV().leftCols(fwh_dim + bfm_dim -1);
	LOG(INFO) << "eigVectors rows,cols:" << eigVectors.rows() << " " << eigVectors.cols() << endl;
	std::ofstream fout(pca + "eigs.txt");
	fout << eigVectors.rows() << " " << eigVectors.cols() << endl;
	fout << svd.singularValues().transpose().leftCols(fwh_dim + bfm_dim -1) / sqrt(fwh_dim + bfm_dim - 1) << endl;
	fout << eigVectors << endl;
	fout.flush();
	fout.close();
	std::ofstream fout_mean(pca + "mean.txt");
	//matF meanFaceFloat() ;
	//SG::safeMemcpy(template_mesh.pos_.data(), meanFaceFloat.data(), template_mesh.n_vertex_ * 3 * sizeof(float));
	//template_mesh.saveObj(pca + prefix + std::to_string(1) + "_mean.obj");
	fout_mean << meanFace << std::endl;
	fout_mean.close();
	//matD pca_value = data_raw
}

void TESTFUNCTION::pcaRawToBinary()
{
	json config;
	config["pca_raw"] = "D:/data/0611_00/pca/";
	config["binary_res"] = "D:/data/0611_00/pca_binary/";
	config["pca_vis"] = "D:/data/0611_00/fwh_pca_vis/";
	config["template"] = "D:/data/0603_00/pca/template.obj";
	config["b_vis"] = true;
	tensorHelper::rawToBinary(config);
}

void TESTFUNCTION::testSysRectify()
{
	MeshSysFinder fwh;
	JsonHelper::initData("D:/data/0528_00/fwh_sys/", "config.json", fwh);
	MeshCompress non_sys("D:/data/0601_00/fwh_raw/Tester_1.obj");
	MeshCompress avg_sys = non_sys;
	fwh.getSysPosAvgInPlace(avg_sys.pos_);
	fwh.getSysPosLapInPlace(non_sys.pos_);
	non_sys.saveObj("D:/data/0601_00/fwh_sys_test/Tester_0_center_r.obj");
	avg_sys.saveObj("D:/data/0601_00/fwh_sys_test/Tester_0_r.obj");
}

void TESTFUNCTION::testFWHRaw()
{
	cstr input = "D:/data/0601_00/fwh_raw/";
	cstr output = "D:/data/0601_00/fwh_sys_test/";
	cstr pca = "D:/data/0601_00/fwh_pca/";
	cstr prefix = "Tester_";
	intVec left_right_sys = FILEIO::loadIntDynamic("D:/data/0528_00/match.txt");
	intVec mid = FILEIO::loadIntDynamic("D:/data/0528_00/mid.txt");
	MeshCompress template_mesh(input + prefix + std::to_string(1) + ".obj");
	matD data_raw(150, template_mesh.n_vertex_ * 3);
	MeshSysFinder fwh;
	JsonHelper::initData("D:/data/0528_00/fwh_sys/", "config.json", fwh);
	for (int i = 0; i < 150; i++)
	{
		MeshCompress iter_input(input + prefix + std::to_string(i+1) + ".obj");
		MeshCompress sys_result = iter_input;
		fwh.getSysPosLapInPlace(sys_result.pos_);
		sys_result.saveObj(output + prefix + std::to_string(i) + "_lap.obj");
		fwh.getSysPosAvgInPlace(iter_input.pos_);
		iter_input.saveObj(output + prefix + std::to_string(i) + ".obj");
		//move mid line
		float3E center;
		RT::getCenter(sys_result.pos_, center);
		LOG(INFO) << "center: " << center << std::endl;
		RT::translateInPlace(-center, sys_result.pos_);
		RT::getCenter(sys_result.pos_, center);
		LOG(INFO) << "center: " << center << std::endl;
		sys_result.saveObj(output + prefix + std::to_string(i) + "_center.obj");

#pragma omp parallel for 
		for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
		{
			for (int j = 0; j < 3; j++)
			{
				data_raw(i, iter_vertex * 3 + j) = sys_result.pos_[iter_vertex](j);
			}
		}
	}
	vecD meanFace;
	NRICP::centerize(data_raw, meanFace);
	Eigen::BDCSVD <Eigen::MatrixXd> svd(data_raw, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto& eigVectors = svd.matrixV().leftCols(150);
	LOG(INFO) << "eigVectors rows,cols:" << eigVectors.rows() << " " << eigVectors.cols() << endl;
	std::ofstream fout(pca + "eigs.txt");
	fout << eigVectors.rows() << " " << eigVectors.cols() << endl;
	fout << svd.singularValues().transpose().leftCols(150) / sqrt(150 - 1) << endl;
	fout << eigVectors << endl;
	fout.flush();
	fout.close();
	std::ofstream fout_mean(pca + "mean.txt");
	fout_mean << meanFace << std::endl;
	fout_mean.close();
}

void TESTFUNCTION::testMoveCSToBasis()
{
	std::string cur_root = "D:/data/0603_01/";
	std::string marker = cur_root + "fwh_part_part.cons";
	intVec src_point, dst_point;
	int num = FILEIO::getPair(marker, src_point, dst_point);
	int num_iter = 80;
	for (int i = 0; i < num_iter; i++)
	{
		MeshCompress src(cur_root + "/cs/cs_" + std::to_string(i) + ".obj");
		MeshCompress dst(cur_root + "/clean/part_" + std::to_string(i) + ".obj");		
		float3Vec dst_pos, src_pos;
		src.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float scale = RT::getScale(src_pos, dst_pos);
		LOG(INFO) << "scale: " << scale << std::endl;
		//RT::scaleInPlace(scale, src.pos_);
		//get slice for scaled pos
		dst.getSlice(dst_point, dst_pos);
		src.getSlice(src_point, src_pos);
		float3E translate;
		RT::getTranslate(src_pos, dst_pos, translate);
		LOG(INFO) << "translate: " << translate.transpose() << std::endl;
		RT::translateInPlace(translate, src.pos_);
		src.saveObj(cur_root + "cs_rt/cs_" + std::to_string(i) + ".obj");	
	}
}

void TESTFUNCTION::testRectifyDeepReconstruction()
{
	MeshSysFinder fwh;
	JsonHelper::initData("D:/data/0528_00/fwh_sys/", "config.json", fwh);
	cstr result = "D:/data/0604_00/deepRecon/";
	cstr cur_root = "D:/data/server_pack/bfm_fwh/";
	Tensor src_tensor;
	JsonHelper::initData(cur_root, "config.json", src_tensor);
	MeshCompress mean_fwh("D:/data/0603_00/fwh_pca/basis_0.obj");
	MeshCompress src_bfm("D:/data/0603_01/cs_rt/cs_0.obj");
	fwh.getSysPosLapInPlace(src_bfm.pos_);
	src_bfm.saveObj(result + "cs_0_sys.obj");
	intVec fwh_id = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_id.txt");
	float3Vec bfm_pos, fwh_pos;
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	//float scale = RT::getScale(bfm_pos, fwh_pos);
	//RT::scaleInPlace(scale, src_bfm.pos_);
	//get slice for scaled pos
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	float3E translate;
	RT::getTranslate(bfm_pos, fwh_pos, translate);
	//set x to zero
	LOG(INFO) << "translate: " << translate.transpose() << std::endl;
	RT::translateInPlace(translate, src_bfm.pos_);
	src_bfm.saveObj(result + "cs_0_trans.obj");
	int weight = 0;
	floatVec reg(src_tensor.n_id_ - 1, weight);
	floatVec coef_roi;
	intVec all_roi = FILEIO::loadIntDynamic("D:/data/0603_00/round_fwh_2.txt");
	//intVec second_drag = FILEIO::readIntDynamic("D:/data/0603_01/fwh_drag_s.txt");
	intVec second_drag = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_id.txt");
	fwh.getSysIdsInPlace(second_drag);
	intVec fwh_fix = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_fix_neck_s_ear.txt");
	fwh.getSysIdsInPlace(fwh_fix);
	//intVec certain_set = { 64,66,67 };
	cstr src_path = "D:/data/0604_00/star/";
	for (int i = 1; i < 15; i++)
		//for(int i: certain_set)
	{	
		src_bfm.loadObj(src_path + "tb" + FILEIO::addZero(i, 2) + "_bfm.obj");
		src_bfm.getSlice(fwh_id, bfm_pos);
		float scale = RT::getScale(bfm_pos, fwh_pos);
		float3E translate;
		RT::getTranslate(bfm_pos, fwh_pos, translate);
		MeshCompress res = src_bfm;
		RT::translateInPlace(translate, src_bfm.pos_);
		src_bfm.saveObj(result + "cs_" + std::to_string(i) + ".obj");
		src_tensor.fitID(src_bfm.pos_, reg, all_roi, src_tensor.ev_data_, coef_roi);
		if (isnan(coef_roi[0]))
		{
			LOG(INFO) << "went wrong" << std::endl;
		}
		std::cout << "fit with roi." << std::endl;
		std::for_each(coef_roi.begin(), coef_roi.end(), [](auto c) {std::cout << c << ","; });
		cout << std::endl;
		floatVec proj_res_roi = src_tensor.interpretID(coef_roi);
		SG::safeMemcpy(res.pos_.data(), proj_res_roi.data(), proj_res_roi.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(i) + "_roi.obj");
		LaplacianDeform to_bfm;
		float3Vec deform_pos;
		to_bfm.init(res, second_drag, fwh_fix);
		for (int iter_handle : second_drag)
		{
			deform_pos.push_back(src_bfm.pos_[iter_handle]);
		}
		to_bfm.deform(deform_pos, res.pos_);
		res.saveObj(result + "proj_" + std::to_string(i) + "_lap.obj");
		fwh.getSysPosLapInPlace(res.pos_);
		res.saveObj(result + "proj_" + std::to_string(i) + "_sys.obj");
		RT::translateInPlace(-translate, res.pos_);
		res.saveObj(result + "proj_" + std::to_string(i) + "_bfm.obj");
		LOG(INFO) << "end of project process." << std::endl;
	}
}

void TESTFUNCTION::testErrorDT()
{
#if 1
	MeshCompress apple_ar("D:/data/0624_04/deform.obj");
	MeshCompress bfm("D:/data/0624_04/Neutral_BFM.obj");
	intVec apple_bfm = FILEIO::loadIntDynamic("D:/data/0624_04/corre_26.000000.txt");
	MeshCompress apple_rep = apple_ar;
#pragma omp parallel for
	for (int i = 0; i < apple_bfm.size()/2; i++)
	{
		apple_rep.pos_[apple_bfm[2 * i]] = bfm.pos_[apple_bfm[2 * i+ 1]];
	}
	cstr res = "D:/data/0624_04/";
	SG::needPath(res);
	MeshCompress diff = Metric::getError(apple_ar, apple_rep, 0.01f, 1.0);
	diff.saveVertexColor(res + "diff.obj");
#else
	json sysconfig;
	sysconfig["template_obj"] = "D:/data/0528_00/id_0_exp_0.obj";
	sysconfig["mid"] = "D:/data/0528_00/mid.txt";
	sysconfig["match"] = "D:/data/0528_00/match.txt";
	MeshSysFinder fwh_sys(sysconfig);
	MeshCompress apple_ar("D:/data/0624_01/bfm_lap_sys.obj");
	MeshCompress bfm("D:/data/0624_01/3dmm_trans.obj");
	intVec apple_bfm = FILEIO::loadIntDynamic("D:/data/0624_01/whole_bfm.txt");
	MeshCompress apple_rep = apple_ar;
#pragma omp parallel for
	for (int i = 0; i < apple_bfm.size() / 2; i++)
	{
		apple_rep.pos_[apple_bfm[2 * i]] = bfm.pos_[apple_bfm[2 * i + 1]];
	}
	cstr res = "D:/data/0624_01/";
	SG::needPath(res);
	MeshCompress diff = Metric::getError(apple_ar, apple_rep, 0.025f, 1.0);
	for (int i = 0; i < fwh_sys.match_ids_.size()/2; i++)
	{
		diff.vertex_color_[fwh_sys.match_ids_[2*i]] = diff.vertex_color_[fwh_sys.match_ids_[2 * i+1]];
	}
	diff.saveVertexColor(res + "metric_diff.obj");
#endif
}

void TESTFUNCTION::testError()
{

	MeshCompress src_0("D:/data/result/0709_star/0.obj");
	MeshCompress dst_0("D:/data/result/0709_coef/0.obj_ceres.obj");	
	MeshCompress error_0_0 = Metric::getError(src_0, dst_0, 0.01f, 30 / 0.3214);
	error_0_0.saveVertexColor("D:/data/result/0709_coef/0_13.obj");





	cstr gt = "D:/data/0611_00/star_uv/";
	cstr generate = "D:/data/0611_00/generate/";
	cstr res = "D:/data/0611_00/error/";
	cstrVec gt_generate = {
		"0_rt.obj", "out_13_rt.obj",
		"1_rt.obj", "out_14_rt.obj",
		"2_rt.obj", "out_15_rt.obj",
	};

	int n = gt_generate.size() / 2;
	for (int i = 0; i < n; i++)
	{
		MeshCompress gt_model(gt+gt_generate[2*i]);
		MeshCompress generate_model(generate + gt_generate[2 * i+1]);
		MeshCompress diff = Metric::getError(gt_model, generate_model, 0.01f, 30/0.3214);
		diff.saveVertexColor(res + gt_generate[2 * i] + gt_generate[2 * i+1]);
	}


	MeshCompress src("D:/data/0611_00/star_uv/0_rt.obj");
	MeshCompress dst_13("D:/data/0611_00/generate/out_13_rt.obj");
	MeshCompress dst_14("D:/data/0611_00/generate/out_14_rt.obj");
	MeshCompress error_0_13 = Metric::getError(src, dst_13, 0.01f);
	MeshCompress error_0_14 = Metric::getError(src, dst_14, 0.01f);
	error_0_13.saveVertexColor("D:/data/0611_00/error/0_13.obj");
	error_0_14.saveVertexColor("D:/data/0611_00/error/0_14.obj");
}

void TESTFUNCTION::batchRTGuiJie()
{
	std::string cur_root = "D:/data/0606_05/";
	std::string marker = cur_root + "fwh_gj_shrink.cons";
	intVec dst_point = FILEIO::loadIntDynamic(cur_root + "fwh_landark_sys.txt");
	intVec src_point = FILEIO::loadIntDynamic(cur_root + "landmark_sys.txt");
	dst_point.resize(42);
	src_point.resize(42);
	int num_iter = 3;
	MeshCompress template_obj(cur_root + "0.obj");
	for (int i = 0; i < num_iter; i++)
	{
		MeshCompress src(cur_root + std::to_string(i) + ".obj");
		template_obj.pos_ = src.pos_;		
		MeshCompress dst(cur_root + std::to_string(i) + "_bfm.obj");
		float3Vec dst_pos, src_pos;
		template_obj.getSlice(src_point, src_pos);
		dst.getSlice(dst_point, dst_pos);
		float scale = RT::getScale(src_pos, dst_pos);
		LOG(INFO) << "scale: " << scale << std::endl;
		RT::scaleInPlace(scale, template_obj.pos_);
		template_obj.saveObj(cur_root + std::to_string(i) + "_scale.obj");
		//get slice for scaled pos
		dst.getSlice(dst_point, dst_pos);
		template_obj.getSlice(src_point, src_pos);
		float3E translate;
		RT::getTranslate(src_pos, dst_pos, translate);
		LOG(INFO) << "translate: " << translate.transpose() << std::endl;
		RT::translateInPlace(translate, template_obj.pos_);
		template_obj.saveObj(cur_root + std::to_string(i) + "_rt.obj");
	}
}

void TESTFUNCTION::transferUV()
{
	MeshCompress with_uv("D:/data/server_pack/star/uv_template.obj");
	MeshCompress with_uv_dis = with_uv;
	intVec discard = FILEIO::loadIntDynamic("D:/data/server_pack/star_uv/skip_2.txt");
	//with_uv_dis.discard(discard);
	//with_uv_dis.saveObj("D:/data/server_pack/star/0_uv.obj");
	cstr cur_root = "D:/data/server_pack/star_uv/";
	for (int i = 4; i < 5; i++)
	{
		MeshCompress vertex(cur_root + "" + std::to_string(i) + "_all.obj");
		//MeshCompress vertex(cur_root + "" + std::to_string(i) + ".obj");
		vertex.discard(discard);
		vertex.saveObj(cur_root + "" + std::to_string(i) + ".obj");
		with_uv_dis.replaceVertexBasedData(vertex);
		with_uv_dis.update();
		with_uv_dis.saveObj(cur_root + "" + std::to_string(i) + "_uv.obj");
	}
}

void TESTFUNCTION::testSubPCA()
{
	MeshSysFinder fwh;
	JsonHelper::initData("D:/data/0528_00/fwh_sys/", "config.json", fwh);
	intVec test_sys_finder = { 0,1427,9 };
	fwh.getSysIdsInPlace(test_sys_finder);
	cstr result = "D:/data/0603_00/bfm_3nd/";
	std::string cur_root = "D:/data/server_pack/bfm_fwh/";
	Tensor src_tensor;
	JsonHelper::initData(cur_root, "config.json", src_tensor);
	//FILEIO::saveToBinary(cur_root + "id_40.bin", src_tensor.data_, sizeof(float)*120*src_tensor.template_obj_.n_vertex_);
	//FILEIO::saveToBinary(cur_root + "id_all.bin", src_tensor.data_);
	//src_tensor.saveByIdExp("D:/data/0603_00/tensor_test/");
	MeshCompress src_bfm("D:/data/0603_00/cs/cs_0.obj");
	MeshCompress mean_fwh("D:/data/0603_00/fwh_pca/basis_0.obj");
	intVec fwh_id = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_id.txt");

	float3Vec bfm_pos, fwh_pos;
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	//float scale = RT::getScale(bfm_pos, fwh_pos);
	//RT::scaleInPlace(scale, src_bfm.pos_);
	//get slice for scaled pos
	src_bfm.getSlice(fwh_id, bfm_pos);
	mean_fwh.getSlice(fwh_id, fwh_pos);
	float3E translate;
	RT::getTranslate(bfm_pos, fwh_pos, translate);
	//set x to zero
	translate[0] = 0;
	RT::translateInPlace(translate, src_bfm.pos_);
	src_bfm.saveObj(result + "cs_0_trans.obj");
	intVec all_roi = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_drag.txt");
	intVec second_drag = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_drag.txt");
	fwh.getSysIdsInPlace(second_drag);
	intVec fwh_fix = FILEIO::loadIntDynamic("D:/data/0603_00/fwh_fix_neck_s.txt");
	fwh.getSysIdsInPlace(fwh_fix);
	//intVec fwh_fix = {};
	//src_bfm.loadObj(result + "cs_a_0.obj");
	MeshCompress res = src_bfm;
	int weight = 0;
	for (int iter_rep = 0; iter_rep < 5; iter_rep++)
	{
		floatVec reg(src_tensor.n_id_ - 1, weight);
		floatVec coef_res, coef_res_var, coef_roi;
		src_tensor.fitID(src_bfm.pos_, reg, coef_res);
		src_tensor.fitID(src_bfm.pos_, reg, src_tensor.ev_data_, coef_res_var);
		src_tensor.fitID(src_bfm.pos_, reg, all_roi, src_tensor.ev_data_, coef_roi);
		floatVec proj_res = src_tensor.interpretID(coef_res);
		std::cout << "fit with no var." << std::endl;
		std::for_each(coef_res.begin(), coef_res.end(), [](auto c) {std::cout << c << ","; });
		cout << std::endl;
		std::cout << "fit with var." << std::endl;
		std::for_each(coef_res_var.begin(), coef_res_var.end(), [](auto c) {std::cout << c << ","; });
		std::cout << endl;
		std::cout << "fit with roi." << std::endl;
		std::for_each(coef_roi.begin(), coef_roi.end(), [](auto c) {std::cout << c << ","; });
		cout << std::endl;
		floatVec proj_res_var = src_tensor.interpretID(coef_res_var);
		floatVec proj_res_roi = src_tensor.interpretID(coef_roi);
		SG::safeMemcpy(res.pos_.data(), proj_res.data(), proj_res.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_.obj");
		SG::safeMemcpy(res.pos_.data(), proj_res_var.data(), proj_res.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_var_.obj");
		SG::safeMemcpy(res.pos_.data(), proj_res_roi.data(), proj_res.size() * sizeof(float));
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_roi_.obj");
		LaplacianDeform to_bfm;
		float3Vec deform_pos;
		to_bfm.init(res, second_drag, fwh_fix);
		for (int iter_handle : second_drag)
		{
			deform_pos.push_back(src_bfm.pos_[iter_handle]);
		}
		to_bfm.deform(deform_pos, res.pos_);
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_lap.obj");
		fwh.getSysPosLapInPlace(res.pos_);
		res.saveObj(result + "proj_" + std::to_string(iter_rep) + "_sys.obj");
		src_bfm = res;
		LOG(INFO) << "end of project process." << std::endl;
	}
}

void TESTFUNCTION::testIDMatch()
{
	//test
	matF female_id, male_id, test_id;
	FILEIO::loadEigenMat("D:/data/0611_00/star_uv/id.txt", female_id);
	FILEIO::loadEigenMat("D:/data/0611_00/test_res/id.txt", test_id);
	//FILEIO::loadEigenMat("D:/data/0611_00/test_res/id.txt", test_id);
	std::ofstream out_result("D:/data/0611_00/test_res/match_test_to_star.txt");
	out_result << test_id.rows() << " " <<2<< std::endl;
	for (int i = 0; i < test_id.rows(); i++)
	{
		int star_id = CalcHelper::getMinEulerDisByRow(test_id.row(i), female_id);
		LOG(INFO) << "test from: "<<i<< "-------to: " << star_id << std::endl;
		out_result << i << " " << star_id << std::endl;
	}
	out_result.close();

	for (int i = 0; i < test_id.rows(); i++)
	{
		LOG(INFO) << "test from: " << i << "-------to: " << CalcHelper::getMinEulerDisByRow(test_id.row(i), male_id) << std::endl;
	}



}

void TESTFUNCTION::testMNN()
{
	MnnModel test;
	int ret = test.init_seg_model("D:/data/0615_mnn/image_seg.mnn", 224);
	if (ret != 0)
		cout << "segmentation mnn init failed" << endl;
	ret = test.init_kp_model("D:/data/0615_mnn/keypoints.mnn", 192, 256, 7);
	if (ret != 0)
		cout << "key point detection mnn init failed" << endl;


	cv::Mat img = cv::imread("D:/code/shoePoseEstimation/src2/test_data/000415-right.jpg");
	img.convertTo(img, CV_8UC3);
	cv::Mat mask(img.rows, img.cols, CV_8UC1);
	mask.setTo(255);
	vector<float> kps, kp_vals;
	vector<int> keyPtIndices;
	static int oldNumShoes = -1;
	const bool localDebug = true;	
	test.inference(img.data, img.cols, img.rows, mask.data, kps, kp_vals, keyPtIndices);
	static char name[1000];
	static int index = 0;
	cv::Mat img_canvas = ImageUtils::drawKps(img, kps, keyPtIndices);
	cv::imshow("what", img_canvas);
	cv::waitKey(0);

	cout << "finished initialization" << endl;
}

void TESTFUNCTION::testMTCNNVideoStream(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	std::string model_path = "D:/code/MNN_FaceTrack/models/";
	MTCNN_SCOPE::FaceTracking faceTrack(model_path);
	//MTCNN detector;
	//detector.setIsMaxFace(true);
	//detector.init(model_path);
	cv::Mat frame;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return;
	}

	int frameIndex = 0;
	std::vector<int> IDs;
	std::vector<cv::Scalar> Colors;
	cv::Scalar color;
	srand((unsigned int)time(0));//初始化种子为随机值

	std::vector<MTCNN_SCOPE::Face> faces;
	for (;;) {
		if (!cap.read(frame))
		{
			break;
		}

		if (frame.empty())
		{
			continue;
		}
		int q = cv::waitKey(1);
		if (q == 27) break;


		//cv::transpose(frame, frame);
		//cv::flip(frame, frame, -1);
		cv::flip(frame, frame, 1);
		double t1 = (double)cv::getTickCount();



		if (frameIndex == 0)
		{
			faceTrack.Init(frame);
			frameIndex = 1;
		}
		else {
			faceTrack.update(frame);
		}

		cv::Mat mtcnn_test = frame.clone();
		printf("total %gms\n", ((double)cv::getTickCount() - t1) * 1000 / cv::getTickFrequency());
		printf("------------------\n");
		double diff = 0;
		std::vector<MTCNN_SCOPE::Face> faceActions = faceTrack.trackingFace;
		for (int i = 0; i < faceActions.size(); i++)
		{
			const MTCNN_SCOPE::Face &info = faceActions[i];
			cv::Rect rect;
			rect.x = info.faceBbox.bbox.xmin;
			rect.y = info.faceBbox.bbox.ymin;
			rect.width = info.faceBbox.bbox.xmax - info.faceBbox.bbox.xmin;
			rect.height = info.faceBbox.bbox.ymax - info.faceBbox.bbox.ymin;

			std::vector< MTCNN_SCOPE::FaceInfo> finalBbox;
	
			const_var->ptr_model->mtcnn_->Detect_T(mtcnn_test, finalBbox);
			cv::Rect rect_mtcnn = rect;
			if (!finalBbox.empty())
			{
				rect_mtcnn.x = finalBbox[0].bbox.xmin;
				rect_mtcnn.y = finalBbox[0].bbox.ymin;
				rect_mtcnn.width = finalBbox[0].bbox.xmax - finalBbox[0].bbox.xmin;
				rect_mtcnn.height = finalBbox[0].bbox.ymax - finalBbox[0].bbox.ymin;
			}

			bool isExist = false;
			for (int j = 0; j < IDs.size(); j++)
			{
				if (IDs[j] == info.face_id)
				{
					color = Colors[j];
					isExist = true;
					break;
				}
			}

			if (!isExist)
			{
				IDs.push_back(info.face_id);
				int r = rand() % 255 + 1;
				int g = rand() % 255 + 1;
				int b = rand() % 255 + 1;
				color = cv::Scalar(r, g, b);
				Colors.push_back(color);
			}

			rectangle(frame, rect, color, 2);

			for (int j = 0; j < 5; j++)
			{
				cv::Point p = cv::Point(info.faceBbox.landmark[j], info.faceBbox.landmark[j + 5]);
				cv::circle(frame, p, 2, color, 2);
			}
			rectangle(mtcnn_test, rect_mtcnn, color, 2);
			if (!finalBbox.empty())
			{
				for (int j = 0; j < 5; j++)
				{
					cv::Point p = cv::Point(finalBbox[0].landmark[j], finalBbox[0].landmark[j + 5]);
					cv::Point p_com = cv::Point(info.faceBbox.landmark[j], info.faceBbox.landmark[j + 5]);
					cv::circle(mtcnn_test, p, 2, color, 2);
					diff += (p - p_com).dot(p - p_com);
				}
			}
		}
		std::cout << "diff: " << diff << std::endl;
		imshow("frame", frame);
		imshow("frame_mtcnn", mtcnn_test);

	}

	IDs.clear();
	Colors.clear();
	cap.release();
	cv::destroyAllWindows();

}

void TESTFUNCTION::testMTCNNPic(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	cv::Mat frame = cv::imread("D:/data/0611_00/star_mtcnn/3.jpg");
	cv::Mat frame_back = frame.clone();
	std::vector< MTCNN_SCOPE::FaceInfo> finalBbox;
	const_var->ptr_model->mtcnn_->Detect_T(frame, finalBbox);
	cv::Rect rect_mtcnn;
	cv::Rect rect_pytorch(204.29466,155.22049, 808.39026- 204.29466, 946.738 - 155.22049);
	if (!finalBbox.empty())
	{
		rect_mtcnn.x = finalBbox[0].bbox.xmin;
		rect_mtcnn.y = finalBbox[0].bbox.ymin;
		rect_mtcnn.width = finalBbox[0].bbox.xmax - finalBbox[0].bbox.xmin;
		rect_mtcnn.height = finalBbox[0].bbox.ymax - finalBbox[0].bbox.ymin;
	}
	rectangle(frame, rect_mtcnn, cv::Scalar(255, 0, 0), 2);
	rectangle(frame, rect_pytorch, cv::Scalar(0, 0, 255), 2);
	floatVec pytorch_mtcnn = { 372.42715, 482.70642, 646.6895,  484.30368, 518.70605, 661.9095, 399.66324, 760.536,
		628.13275, 764.6365};
	cv::Mat crop_160 = frame_back(rect_pytorch);
	cv::resize(crop_160, crop_160, cv::Size(160, 160));
	cv::imwrite("D:/data/0611_00/star_mtcnn_res/3_160.jpg", crop_160);
	if (!finalBbox.empty())
	{
		for (int j = 0; j < 5; j++)
		{
			cv::Point p = cv::Point(finalBbox[0].landmark[j], finalBbox[0].landmark[j + 5]);
			cv::Point p_pytorch = cv::Point(pytorch_mtcnn[2*j], pytorch_mtcnn[2 * j+1]);
			cv::circle(frame, p, 2, cv::Scalar(255, 0, 0), 2);
			cv::circle(frame, p_pytorch, 2, cv::Scalar(0, 0, 255), 2);
		}
	}

	std::for_each(finalBbox[0].landmark, finalBbox[0].landmark+10, [](auto c) {std::cout << c << ","; });
	imshow("frame", frame);
	cv::waitKey(0);
}

void TESTFUNCTION::testArcFace(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var,
	const cstr& root, const cstr& img_name, const cstr& com_faceid)
{
	//test inference for arcface
	cv::Mat arc_test = cv::imread(root + img_name, cv::IMREAD_GRAYSCALE);
	vecF test_id, infer_id, trans_id, female_id;
	const_var->ptr_model->getFaceID(arc_test, infer_id, false);

	cv::Mat arc_test_8u;
	arc_test.convertTo(arc_test_8u, CV_8UC1);
	LOG(INFO) << int(arc_test_8u.at<unsigned char>(0, 0)) << std::endl;
	LOG(INFO) << (arc_test_8u.at<unsigned char>(0, 0)) << std::endl;
	FILEIO::saveEigenDynamic(root + img_name +"_cpp_faceid.txt", infer_id);
	test_id = infer_id;
	FILEIO::loadEigenMat(root + com_faceid, 512, 1, test_id);
	LOG(INFO) << "norm: " << (test_id - infer_id).norm() << std::endl;
	LOG(INFO) << "diff: " << (test_id - infer_id).transpose() << std::endl;
}

void TESTFUNCTION::test3dmm(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	//test inference for arcface
	cv::Mat arc_test = cv::imread("D:/data/server_pack/guijie_star/0_3dmm.png");
	vecF coef_3dmm, test_coef;
	const_var->ptr_model->deep3d_->inference(arc_test, 224, 224, coef_3dmm);
	FILEIO::saveEigenDynamic("D:/data/server_pack/guijie_star/0_3dmm_cpp.txt", coef_3dmm);
	vecF coef_3dmm_80 = coef_3dmm.head(80);
	FILEIO::loadEigenMat("D:/data/server_pack/guijie_star/0_3dmm.txt", test_coef);
	LOG(INFO) << "test_coef: " << test_coef.transpose() << std::endl;
	LOG(INFO) << "coef_3dmm: " << coef_3dmm_80.transpose() << std::endl;
	LOG(INFO) << "diff: " << (test_coef- coef_3dmm_80).transpose() << std::endl;
	LOG(INFO) << "euler dis: " << CalcHelper::getEulerDis(coef_3dmm_80, test_coef) << std::endl;
	LOG(INFO) << "cos dis: " << CalcHelper::getMinusCosDis(coef_3dmm_80, test_coef) << std::endl;
}

void TESTFUNCTION::testLandmark(const std::shared_ptr<ConstVar> const_var, std::shared_ptr<ResVar> res_var)
{
	//test inference for arcface
	cv::Mat landmark_test = cv::imread("D:/data/0620_00/unit_test.png");
	vecF heatmap_value, test_value;
	const_var->ptr_model->landmark_68_->inference(landmark_test, 256, 256, heatmap_value);

	cv::Mat res_clone = landmark_test.clone();
	for (int i = 0; i < 68; i++)
	{
		cv::Point p = cv::Point(heatmap_value[2*i], heatmap_value[2 * i+1]);
		cv::circle(res_clone, p, 2, cv::Scalar(255, 0, 0), 2);
	}
	cv::imwrite("D:/data/0622_00/unit_test.png", res_clone);
	//cv::imshow("res_clone", res_clone);
	//cv::waitKey(0);


	FILEIO::loadEigenMat("D:/data/0620_00/unit_test_raw.txt", test_value);
	vecF diff = test_value - heatmap_value;
	LOG(INFO) << diff.maxCoeff() << std::endl;
	LOG(INFO) << diff.minCoeff() << std::endl;
	FILEIO::saveEigenDynamic("D:/data/0620_00/diff.txt", test_value - heatmap_value);
	FILEIO::saveEigenDynamic("D:/data/0620_00/heatmap_value.txt", heatmap_value);

	//LOG(INFO) << "diff: " << (heatmap_value - test_value).transpose() << std::endl;
	LOG(INFO) << "euler dis: " << CalcHelper::getEulerDis(heatmap_value, test_value) << std::endl;
	LOG(INFO) << "cos dis: " << CalcHelper::getMinusCosDis(heatmap_value, test_value) << std::endl;
}

void TESTFUNCTION::mesh2tensor()
{
	int fwh_dim = 80;
	int bfm_dim = 0;
	cstr input_mesh = "D:/data/0623_01/deep3d_rt_clean/";
	cstr prefix_fwh = "pca_";
	cstr input_bfm = "D:/data/0603_00/bfm_use/";
	cstr prefix_bfm = "proj_";
	cstr pca_result = "D:/data/0623_01/deep3d_clean_tensor/";
	SG::needPath(pca_result);
	MeshCompress template_mesh("D:/data/0623_01/deep3d_rt_clean/pca_-1.obj");
	floatVec mean_raw(template_mesh.n_vertex_ * 3, 0);
	floatVec eigen_value(fwh_dim+bfm_dim, 1.0);
	FILEIO::saveToBinary(pca_result + "mean.bin", mean_raw);
	FILEIO::saveToBinary(pca_result + "eigen_value.bin", eigen_value);
	floatVec data_raw((fwh_dim + bfm_dim + 1)* template_mesh.n_vertex_ * 3);
	for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
	{
		for (int j = 0; j < 3; j++) {
			mean_raw[iter_vertex * 3 + j] = template_mesh.pos_[iter_vertex](j);
			data_raw[iter_vertex * 3 + j] = template_mesh.pos_[iter_vertex](j);
		}
	}


	int shift = template_mesh.n_vertex_ * 3;
	for (int i = 0; i < fwh_dim; i++)
	{
		MeshCompress iter_input(input_mesh + prefix_fwh + std::to_string(i) + ".obj");
#pragma omp parallel for 
		for (int iter_vertex = 0; iter_vertex < template_mesh.n_vertex_; iter_vertex++)
		{
			for (int j = 0; j < 3; j++) {
				data_raw[shift + i*shift + iter_vertex * 3 + j] = iter_input.pos_[iter_vertex](j)
					- data_raw[iter_vertex * 3 + j] ;
			}
		}
	}

	FILEIO::saveToBinary(pca_result + "pca.bin", data_raw);
}

void TESTFUNCTION::testProjection()
{
	cstr cur_root = "D:/data/0716_project_jl_1440_changePos/";

	Projection test_256(600, 600, 127.5, 127.5);
	floatVec landmark_xy;
	FILEIO::loadFixSizeEigenFormat(cur_root + "landmark_xy_256.txt", landmark_xy);
	MeshCompress fwh_head(cur_root + "bfm_lap_sys.obj");
	intVec landmark_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	int n_num = landmark_xy.size()*0.5;
#if 1
	intSet used_list = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,36,37,38,
		39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,54,55,56,57,58,59};
#else
	intSet used_list = {36,37,38,39,40,41,42,43,44,45,46,47,27,28,29,30,31,32,33,34,35,48,49,50,51,52,53,54,55,56,57,58,59 };
#endif
	for (int i = 0; i < n_num; i++)
	{
		if (used_list.count(i) && i<17)
		{
			img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
			float3E pos = fwh_head.pos_[landmark_idx[i]];
			obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
		}
		else
		{
			for (int iter = 0; iter < 3; iter++)
			{
				img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
				float3E pos = fwh_head.pos_[landmark_idx[i]];
				obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));
			}
		}
	}
	cvMatD rvec, tvec, r_matrix;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
	cv::solvePnP(obj_points, img_points, test_256.intrisic_,
		dist_coeffs, rvec, tvec);
	LOG(INFO) << "rvec: " << std::endl << rvec << std::endl;
	LOG(INFO) << "tvec: " << std::endl << tvec << std::endl;
	//get
	cv::Rodrigues(rvec, r_matrix);
	cv::Mat canvas = cv::imread(cur_root + "input_landmark68.png");
	float3Vec dst_pos;
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = fwh_head.pos_[landmark_idx[i]];
		cvMatD in_camera = r_matrix *(cvMatD(3, 1) << pos.x(), pos.y(), pos.z())+tvec;
		LOG(INFO) << "in_camera: " << in_camera << std::endl;
		cvMatD in_image = test_256.intrisic_*in_camera;
		LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		LOG(INFO) <<"img_x: "<< img_x << std::endl;
		LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2*i], landmark_xy[2 * i+1]), 2, cv::Scalar(255, 0, 0), 2);
		//green opt
		//fix z
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - test_256.cx_)*GETD(in_camera_fix, 2, 0) / test_256.fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - test_256.cy_)*GETD(in_camera_fix, 2, 0) / test_256.fy_);
		cvMatD in_image_fix = test_256.intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		if (i < 17)
		{
			dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
		}
		float img_x_fix = GETD(in_image_fix, 0, 0) / GETD(in_image_fix, 2, 0);
		float img_y_fix = GETD(in_image_fix, 1, 0) / GETD(in_image_fix, 2, 0);
		cv::circle(canvas, cv::Point2f(img_x_fix, img_y_fix), 2, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite(cur_root + "2d_3d.png", canvas);
	LaplacianDeform to_image;
	float3Vec dst_select;
	intVec land_select;
	intSet select_idx = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	for (int i : select_idx)
	{
		dst_select.push_back(dst_pos[i]);
		land_select.push_back(landmark_idx[i]);
	}
	intVec move_pos = FILEIO::loadIntDynamic("D:/data/0716_project_jl_1440_changePos/move_idx.txt");
	intSet move_set(move_pos.begin(), move_pos.end());
	intVec fix_pos;
	for (size_t i = 0; i < fwh_head.n_vertex_; i++)
	{
		if (!move_set.count(i))
		{
			fix_pos.push_back(i);
		}
	}
	to_image.init(fwh_head, land_select, fix_pos);
	to_image.deform(dst_select, fwh_head.pos_);
	fwh_head.saveObj(cur_root + "fwh_to_image.obj");
}

void TESTFUNCTION::testProjectionUsingCache()
{
	cstr cur_root = "D:/data/0716_project_jl_1929/";
	cv::FileStorage fs(cur_root + "test.yml", cv::FileStorage::READ);
	Projection test_256(2000, 2000, 127.5, 127.5);
	floatVec landmark_xy;
	FILEIO::loadFixSizeEigenFormat(cur_root + "landmark_xy_256.txt", landmark_xy);
	MeshCompress fwh_head(cur_root + "bfm_lap_sys.obj");
	intVec landmark_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	std::vector<cv::Point2d> img_points;
	std::vector<cv::Point3d> obj_points;
	int n_num = landmark_xy.size()*0.5;

	for (int i = 0; i < n_num; i++)
	{
		img_points.push_back(cv::Point2d(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
		float3E pos = fwh_head.pos_[landmark_idx[i]];
		obj_points.push_back(cv::Point3d(pos.x(), pos.y(), pos.z()));		
	}
	cvMatD rvec, tvec, r_matrix, trans_fwh_3dmm;
	fs["rvec"] >> rvec;
	fs["tvec"] >> tvec;
	fs["r_matrix"] >> r_matrix;
	fs["trans_fwh_bfm"] >> trans_fwh_3dmm;
	float3E translate = float3E(GETD(trans_fwh_3dmm, 0, 0), GETD(trans_fwh_3dmm, 1, 0), GETD(trans_fwh_3dmm, 2, 0));
	//RT::translateInPlace(translate, fwh_head.pos_);

	cv::Mat canvas = cv::imread(cur_root + "input_landmark68.png");
	float3Vec dst_pos;
	float dis = 0;
	intSet select_idx = { 1,2,3,4,5,6,7,9,10,11,12,13,14,15 };
	for (int i = 0; i < n_num; i++)
	{
		//cv
		float3E pos = fwh_head.pos_[landmark_idx[i]];
		cvMatD in_camera = r_matrix * (cvMatD(3, 1) << pos.x(), pos.y(), pos.z()) + tvec;
		LOG(INFO) << "in_camera: " << in_camera << std::endl;
		cvMatD in_image = test_256.intrisic_*in_camera;
		LOG(INFO) << "in_image: " << in_image << std::endl;
		float img_x = GETD(in_image, 0, 0) / GETD(in_image, 2, 0);
		float img_y = GETD(in_image, 1, 0) / GETD(in_image, 2, 0);
		LOG(INFO) << "img_x: " << img_x << std::endl;
		LOG(INFO) << "img_y: " << img_y << std::endl;
		//red 3d
		cv::circle(canvas, cv::Point2f(img_x, img_y), 2, cv::Scalar(0, 0, 255), 2);
		//blue 2d
		cv::circle(canvas, cv::Point2f(landmark_xy[2 * i], landmark_xy[2 * i + 1]), 2, cv::Scalar(255, 0, 0), 2);
		//green opt
		//fix z
		cvMatD in_camera_fix = in_camera;
		SETD(in_camera_fix, 0, 0, (landmark_xy[2 * i] - test_256.cx_)*GETD(in_camera_fix, 2, 0) / test_256.fx_);
		SETD(in_camera_fix, 1, 0, (landmark_xy[2 * i + 1] - test_256.cy_)*GETD(in_camera_fix, 2, 0) / test_256.fy_);
		cvMatD in_image_fix = test_256.intrisic_*in_camera_fix;
		cvMatD in_3d_fix = r_matrix.t()*(in_camera_fix - tvec);
		if (i < 17)
		{
			dst_pos.push_back(float3E(GETD(in_3d_fix, 0, 0), GETD(in_3d_fix, 1, 0), GETD(in_3d_fix, 2, 0)));
		}
		float img_x_fix = GETD(in_image_fix, 0, 0) / GETD(in_image_fix, 2, 0);
		float img_y_fix = GETD(in_image_fix, 1, 0) / GETD(in_image_fix, 2, 0);
		//cv::circle(canvas, cv::Point2f(img_x_fix, img_y_fix), 2, cv::Scalar(0, 255, 0), 2);
		if (select_idx.count(i))
		{
			dis += (float2E(img_x, img_y) - float2E(landmark_xy[2 * i], landmark_xy[2 * i + 1])).norm();
		}

	}
	LOG(INFO) << "error: " << dis / select_idx.size() << std::endl;
	cv::imwrite(cur_root + "2d_3d_image.png", canvas);
	LaplacianDeform to_image;
	float3Vec dst_select;
	intVec land_select;
	
	for (int i : select_idx)
	{
		dst_select.push_back(dst_pos[i]);
		land_select.push_back(landmark_idx[i]);
	}
	intVec move_pos = FILEIO::loadIntDynamic("D:/data/0716_project_jl_1440_changePos/move_idx.txt");
	intSet move_set(move_pos.begin(), move_pos.end());
	intVec fix_pos;
	for (size_t i = 0; i < fwh_head.n_vertex_; i++)
	{
		if (!move_set.count(i))
		{
			fix_pos.push_back(i);
		}
	}
	to_image.init(fwh_head, land_select, fix_pos);
	to_image.deform(dst_select, fwh_head.pos_);
	RT::translateInPlace(-translate, fwh_head.pos_);
	fwh_head.saveObj(cur_root + "fwh_to_image.obj");
}

void TESTFUNCTION::testRatioFace()
{
	cstr cur_root = "D:/data/0716_project_jl_1704_focal500/";
	floatVec landmark_xy;
	FILEIO::loadFixSizeEigenFormat(cur_root + "landmark_xy_256.txt", landmark_xy);
	MeshCompress fwh_head(cur_root + "fit_image.obj");
	intVec landmark_idx = FILEIO::loadIntDynamic("D:/data/server_pack/fwh_68/fwh_68_sys.txt");
	float2Vec img_points;
	float3Vec obj_points;
	int n_num = landmark_xy.size()*0.5;
	
	for (int i = 0; i < n_num; i++)
	{
		img_points.push_back(float2E(landmark_xy[2 * i], landmark_xy[2 * i + 1]));
		float3E pos = fwh_head.pos_[landmark_idx[i]];
		obj_points.push_back(float3E(pos.x(), pos.y(), pos.z()));
	}
	intVec left_eye = { 36,37,38,39,40,41 };
	intVec right_eye = { 42,43,44,45,46,47 };

	//get ratio
	float2E left_eye_pos = float2E(0, 0);
	float2E right_eye_pos = float2E(0, 0);
	float3E left_eye_3d = float3E(0, 0, 0);
	float3E right_eye_3d = float3E(0, 0, 0);
	for (int i : left_eye)
	{
		left_eye_pos += img_points[i];
		left_eye_3d += obj_points[i];
	}
	for (int i : right_eye)
	{
		right_eye_pos += img_points[i];	
		right_eye_3d += obj_points[i];
	}

	left_eye_3d = 1.0 / 6.0*left_eye_3d;
	right_eye_3d = 1.0 / 6.0*right_eye_3d;

	left_eye_pos = 1.0 / 6.0*left_eye_pos;
	right_eye_pos = 1.0 / 6.0*right_eye_pos;

	float eye_length = (left_eye_pos - right_eye_pos).norm();
	float ratio_3d_2d = (left_eye_3d.head(2) - right_eye_3d.head(2)).norm()/ eye_length;

	for (int i = 0; i < 8; i++)
	{
		float length_2d = (img_points[i] - img_points[16 - i]).norm();
		float length_3d = (obj_points[i].head(2) - obj_points[16 - i].head(2)).norm();

		LOG(INFO) << "length_3d: " << length_3d << std::endl;
		LOG(INFO) << "length_3d scale : " << length_2d * ratio_3d_2d << std::endl;
	}
}

void TESTFUNCTION::testLandmarkGuided()
{
	LOG(INFO) << "test dt" << std::endl;
	MeshTransfer dt_ori;
	DTGuidedLandmark dt_land;
	MeshCompress A("D:/avatar/0730_dt_test/mean_face_ori.obj");
	MeshCompress A_deform("D:/avatar/0730_dt_test/mean_face_dst_ori.obj");
	MeshCompress B("D:/avatar/0730_dt_test/delta_src_ori.obj");
	MeshCompress B_ori = B;
	MeshCompress B_land = B;
	cstr output_path = "D:/avatar/0803_df_02/";
	SG::needPath(output_path);
	intVec fix = FILEIO::loadIntDynamic("D:/avatar/guijie/fix_head_sys.txt");


	intVec right_eye_round = FILEIO::loadIntDynamic("D:/avatar/guijie/right_eye_round.txt");
	//find pair
	float3Vec dir;
	intVec up, down;
	for (int i = 0; i < right_eye_round.size(); i++)
	{
		int idx = right_eye_round[i];
		if ((A.pos_[idx] - A_deform.pos_[idx]).y() > 0)
		{
			up.push_back(idx);
		}
		else
		{
			down.push_back(idx);
		}
	}
	intVec res = FILEIO::loadIntDynamic("D:/avatar/guijie/left_up_match.txt");
	FILEIO::loadIntDynamic("D:/avatar/guijie/left_down_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_up_match.txt", res);
	FILEIO::loadIntDynamic("D:/avatar/guijie/right_down_match.txt", res);
	intVec match = MeshTools::getSrcToDstMatchKeepSign(A_deform, up, down, 2.0);
	match = CalcHelper::keepValueBiggerThan(match, -0.5);
	intVec res_eyelash_only = res;
	res.insert(res.end(), match.begin(), match.end());
	
	//dt_ori.init(A, A_deform, res, fix);
	//dt_land.init(A, A_deform, res, fix);
	dt_ori.initDynamicFix(A, A_deform, res, 0.001);
	intX2Vec face_part = 
	{ 
		{ 252,255,256,259,406,409,410,413,438,441,442,445,447,448,1337,1777,1778,1910,1917,1924,1931,1938,1943,1944,1948,1949,1955,1958,1961,1965,1967,1971,1978,1979,1982,1986,1989} 
	};
	std::vector<LandGuidedType> opt_type = { LandGuidedType::XYZ_OPT };
	dt_land.setPart(face_part, opt_type, A, A_deform, B);
	dt_land.initDynamicFix(A, A_deform, B, res_eyelash_only, 0.001);
	dt_ori.transfer(B.pos_, B_ori.pos_);
	dt_land.transfer(B.pos_, B_land.pos_);
	B_ori.saveObj(output_path + "B_ori.obj");
	B_land.saveObj(output_path + "B_land.obj");
}

void TESTFUNCTION::serverGenExp()
{
#if 1
	json exp_config = FILEIO::loadJson("D:/avatar/exp_server_config/config.json");
	exp_config["root"] = "D:/avatar/exp_server_config/";
	cstr root = "D:/avatar/0903_01/";
	//cstrVec json_vec = { "random_4.json", "eye_danfeng.json", "eye_round.json", "eye_taohua.json", "eye_xiachui.json", "eye_xingyan.json"};
	cstrVec json_vec = { "random_7.json"};
	std::shared_ptr<ExpGen> exp_ptr;
	std::shared_ptr<BsGenerate> bs_ptr;
	//bs_ptr->generateEyeTensor();
	exp_ptr.reset(new ExpGen(exp_config));
	bs_ptr.reset(new BsGenerate(exp_config));

	//while (true)
	{
		for (int i = 0; i < json_vec.size(); i++)
		{
			cstr json_name = json_vec[i];
			cstr json_raw_name = FILEIO::getFileNameWithoutExt(json_name);
			SG::needPath(root + "/" + json_raw_name);
			json json_bs = FILEIO::loadJson(root + json_name);
			MeshCompress B;
			bs_ptr->generateFace(json_bs, B);
			//TinyTool::skeletonChange();
			//MeshCompress B(root + json_raw_name + ".obj");
			exp_ptr->fixEyelash(B);
			exp_ptr->setDebug(true);
			exp_ptr->setB(B);
			exp_ptr->getExpOMP(B, root + json_raw_name + "/");
			nlohmann::json vertex_value;
			exp_ptr->getResultJson(vertex_value);
			//win need
			SG::needPath(root + json_raw_name + "_res");
			FILEIO::saveJson(root + json_raw_name + "_res/" + json_raw_name + "_delta.json", vertex_value);
			FILEIO::saveJson(root + json_raw_name + "_res/" + json_raw_name + ".json", json_bs);
			B.saveObj(root + json_raw_name + ".obj");
			exp_ptr->dumpEyelash(B);
			B.saveObj(root + json_raw_name + "_no_lash.obj");
		}
	}

#else
	cstr root = "D:/avatar/0820_eye/";
	cstrVec json_vec = { "random_4.json", "baoxia.json" };
	std::shared_ptr<ExpGen> exp_ptr;
	std::shared_ptr<BsGenerate> bs_ptr;
	exp_ptr.reset(new ExpGen);
	bs_ptr.reset(new BsGenerate);
	for (int i = 0; i < json_vec.size(); i++)
	{
		cstr json_name = json_vec[i];
		SG::needPath(root);
		json json_bs = FILEIO::loadJson(root + json_name);
		BsGenerate get_tensor;
		//get_tensor.generateTensor();
		get_tensor.generateFace(root, json_name, 0);
		//TinyTool::skeletonChange();
		ExpGen test_exp;
		test_exp.getEyeBlink(root + FILEIO::getFileNameWithoutExt(json_name) + ".obj",
			root + FILEIO::getFileNameWithoutExt(json_name) + "/", 5);
		test_exp.testExpGuided(root + FILEIO::getFileNameWithoutExt(json_name) + ".obj",
			root + FILEIO::getFileNameWithoutExt(json_name) + "/", 5);
	}
#endif
}

void TESTFUNCTION::serverOptV2()
{
	//test
	cstr root = "D:/avatar/0910_test/";
	//test for avatar generation v2
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt2_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt2_data/";
	cstrVec file_id = { "12_1","61_1","118_0" };
	
	std::shared_ptr<OptV2Gen> optV2_ptr;
	optV2_ptr.reset(new OptV2Gen(exp_config));

	//while (true)
	{
		for (int i = 0; i < file_id.size(); i++)
		{
			cstr json_name = root + file_id[i] + "_type.json";
			cstr coef_3dmm_name = root + file_id[i] + "_3dmm.txt";
			//loading 3dmm
			floatVec coef_3dmm;
			FILEIO::loadEigenMatToVector(coef_3dmm_name, coef_3dmm, false);
			SG::needPath(root + "/" + file_id[i]);
			json json_in = FILEIO::loadJson(json_name);
			json_in["coef_3dmm"] = coef_3dmm;
			json json_out;
			MeshCompress raw_type, opt_type;
			optV2_ptr->getGuijieFromType(json_in, raw_type, json_out, opt_type);

			SG::needPath(root + file_id[i] + "_res");
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_in.json", json_in);
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_out.json", json_out);

			raw_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_type.obj");
			optV2_ptr->dumpEyelash(raw_type);
			raw_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_type_no_lash.obj");

			opt_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_opt.obj");
			optV2_ptr->dumpEyelash(opt_type);
			opt_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_opt_no_lash.obj");
		}
	}
}

void TESTFUNCTION::serverOptV2FromServer()
{
	//test
	cstr root = "D:/avatar/0911_test/";
	//test for avatar generation v2
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt2_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt2_data/";
	cstrVec file_id = { "12_1_in",  "12_2_in" };

	std::shared_ptr<OptV2Gen> optV2_ptr;
	optV2_ptr.reset(new OptV2Gen(exp_config));

	//while (true)
	{
		for (int i = 0; i < file_id.size(); i++)
		{
			cstr json_name = root + file_id[i] + ".json";			
			json json_in = FILEIO::loadJson(json_name);
			json json_out;
			MeshCompress raw_type, opt_type;
			optV2_ptr->getGuijieFromType(json_in["face_type"], raw_type, json_out, opt_type);

			SG::needPath(root + file_id[i] + "_res");
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_in.json", json_in);
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_out.json", json_out);

			raw_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_type.obj");
			optV2_ptr->dumpEyelash(raw_type);
			raw_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_type_no_lash.obj");

			opt_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_opt.obj");
			optV2_ptr->dumpEyelash(opt_type);
			opt_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_opt_no_lash.obj");
		}
	}
}

void TESTFUNCTION::serverOptV3FromServer()
{
	//test
	//compare root 	
	//cstr root = "D:/avatar200923/1019_00/";
	cstr root = "D:/dota201104/1105_guijie/";
	cstrVec file_id = { 
		"0.jpg",
		"1.jpg",
		"2.jpg",
		"3.jpg",
		"4.jpg",
		"5.jpg",
		"6.jpg",
		"7.jpg",
	};

	//test for avatar generation v2
	json exp_config = FILEIO::loadJson("D:/avatar/guijie_opt3_data/config.json");
	exp_config["root"] = "D:/avatar/guijie_opt3_data/";


	std::shared_ptr<OptV3Gen> optV3_ptr;
	optV3_ptr.reset(new OptV3Gen(exp_config));

	//while (true)
	{
		for (int i = 0; i < file_id.size(); i++)
		{
			cstr json_name = root + file_id[i] + ".json";
			json json_in = FILEIO::loadJson(json_name);
			json json_out;
			MeshCompress raw_type, opt_type;
			optV3_ptr->getGuijieFromType(json_in["face_type"], raw_type, json_out, opt_type);

			SG::needPath(root + file_id[i] + "_res");
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_in.json", json_in);
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_out.json", json_out);

			raw_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_match.obj");
			optV3_ptr->dumpEyelash(raw_type);
			raw_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_match_no_lash.obj");

			opt_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_opt.obj");
			optV3_ptr->dumpEyelash(opt_type);
			opt_type.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_opt_no_lash.obj");

			MeshCompress in_att_bfm;
			optV3_ptr->getBFMMesh(optV3_ptr->att_cur_.coef_3dmm_, in_att_bfm);
			in_att_bfm.saveObj(root + file_id[i] + "_res/" + file_id[i] + "_3dmm_input.obj");
			MeshCompress match_att_bfm;
			vecD match_3dmm;
			FILEIO::loadEigenMat("D:/avatar/guijie_opt3_data/guijie_star/" + std::to_string(optV3_ptr->att_cur_.match_id_) + "_3dmm.txt", 80, 1, match_3dmm);
			optV3_ptr->getBFMMesh(match_3dmm, match_att_bfm);
			match_att_bfm.saveObj(root + file_id[i] + "_res/match_" + std::to_string(optV3_ptr->att_cur_.match_id_) + "_3dmm.obj");
		}
	}
}

void TESTFUNCTION::eyebrowType()
{
	//test
	cstr root = "D:/dota201010/1102/";
	//cstr root = "D:/dota201010/1016_seg/land_test/";

	
	//float pitch;            // 点头（Pitch）down: -; up: +  [-PI, PI]
	//float yaw;              // 摇头（Yaw）left: + ; right: - [-PI, PI]
	//float roll;             // 歪头（Roll）left: -; right: +  [-PI, PI]
	cstrVec file_id = {
		"face_parsing",
		//"f05_adjust",
	};

	//test for avatar generation v2
	json data_config = FILEIO::loadJson("D:/avatar/eyebrow_data/config.json");
	data_config["root"] = "D:/avatar/eyebrow_data/";

	std::shared_ptr<EyebrowType> eyebrow_ptr;
	eyebrow_ptr.reset(new EyebrowType(data_config));

	//while (true)
	{
		for (int i = 0; i < file_id.size(); i++)
		{
			cstr json_name = root + file_id[i] + ".json";
			//LOG(WARNING) << "begin: " << json_name << std::endl;
			json json_in = FILEIO::loadJson(json_name);
			json json_out;
			eyebrow_ptr->setDebug(true);
			eyebrow_ptr->setResultDir(root);
			//eyebrow_ptr->getEyebrowTypeOnlyLandmark(json_in["face_type"], json_out);
			eyebrow_ptr->getEyebrowType(json_in["face_type"], json_out);
			SG::needPath(root + file_id[i] + "_res");
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_in.json", json_in);
			FILEIO::saveJson(root + file_id[i] + "_res/" + file_id[i] + "_out.json", json_out);		
			//LOG(WARNING) << "end: " << json_name << std::endl;
		}
	}
}

void TESTFUNCTION::testWithServerPack()
{
	std::string root = "D:/data/server_pack/";
	JsonData json_data;
	JsonHelper::initData(root, "config.json", json_data);
	std::shared_ptr<ConstVar> ptr_const_var = std::make_shared<ConstVar>(json_data);
	std::shared_ptr<ResVar> ptr_res_var = std::make_shared<ResVar>();
	//TESTFUNCTION::testArcFace(ptr_const_var, ptr_res_var);
	TESTFUNCTION::test3dmm(ptr_const_var, ptr_res_var);
	//PREPARE::prepareData(ptr_const_var, ptr_res_var);
	//TESTFUNCTION::testRoutine(ptr_const_var, ptr_res_var);
	system("pause");
}

void TESTFUNCTION::testImageSimilarity()
{
	cstr root = "D:/avatar/eyebrow_data/img_mask/";
	cstr brow_test = "D:/dota201010/1016_seg/brow_test/";

#if 0
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(root, cstrVec{ ".jpg" }, false);
	for (int i = 0; i < 13; i++)
	{
		cv::Mat orginal_im = cv::imread(root + folder_file[i]);
		for (int j = 0; j < 13; j++)
		{
			cv::Mat frame = cv::imread(root + folder_file[j]);
			//std::cout << ImageSim::ssimWrapper(orginal_im, frame) << ",";
			//std::cout << ImageSim::matchShape(orginal_im, frame) << ",";
			std::cout << ImageSim::matchShapeImage(orginal_im, frame) << ",";
		}
		std::cout << std::endl;
	}
#else
	
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(root, cstrVec{ ".jpg" }, false);
	CGP::cstrVec test_file = FILEIO::getFolderFiles(brow_test, cstrVec{ ".png" }, false);

	//floatVec landmark_106 = json_in["landmark_106"].get<floatVec>();

	for (auto i : test_file)
	{
		double min_dist = 1000;
		int min_idx = -1;
		cv::Mat orginal_all = cv::imread(brow_test + i);
		cv::Mat orginal_part, orginal_im;
		ImageUtils::cutBasedOnImage(orginal_all, orginal_part);
		ImageUtils::putImageToCenter(orginal_part, cv::Size(256, 256), orginal_im);
		for (int j = 0; j < 13; j++)
		{
			cv::Mat frame = cv::imread(root + folder_file[j]);

			//std::cout << ImageSim::ssimWrapper(orginal_im, frame) << ",";
			double dis = ImageSim::matchShape(orginal_im, frame);
			//double dis = ImageSim::matchShapeImage(orginal_im, frame);
			//get for landmark
			if (dis < min_dist)
			{
				min_dist = dis;
				min_idx = j;
			}
			std::cout << dis << ",";
			
		}
		std::cout << std::endl;
		std::cout << i<<", min: " << min_idx<<std::endl;
	}	

#endif
}

void TESTFUNCTION::testImageSimilarityLandmark()
{
	cstr root = "D:/avatar/eyebrow_data/img_mask/";
	cstr brow_test = "D:/dota201010/1016_seg/land_test/";

#if 0
	CGP::cstrVec folder_file = FILEIO::getFolderFiles(root, cstrVec{ ".jpg" }, false);
	for (int i = 0; i < 13; i++)
	{
		cv::Mat orginal_im = cv::imread(root + folder_file[i]);
		for (int j = 0; j < 13; j++)
		{
			cv::Mat frame = cv::imread(root + folder_file[j]);
			//std::cout << ImageSim::ssimWrapper(orginal_im, frame) << ",";
			//std::cout << ImageSim::matchShape(orginal_im, frame) << ",";
			std::cout << ImageSim::matchShapeImage(orginal_im, frame) << ",";
		}
		std::cout << std::endl;
	}
#else

	CGP::cstrVec folder_file = FILEIO::getFolderFiles(root, cstrVec{ ".jpg" }, false);
	CGP::cstrVec test_file = FILEIO::getFolderFiles(brow_test, cstrVec{ ".png" }, false);

	//

	for (auto i : test_file)
	{
		double min_dist = 1000;
		int min_idx = -1;
		cv::Mat orginal_all = cv::imread(brow_test + i);
		cv::Mat orginal_part, orginal_im;
		ImageUtils::cutBasedOnImage(orginal_all, orginal_part);
		ImageUtils::putImageToCenter(orginal_part, cv::Size(256, 256), orginal_im);

		cstrVec file_name;
		FILEIO::splitString(i, file_name, '.');

		auto landmark_pos = FILEIO::loadJson(brow_test + file_name[0] + ".json");

		//LOG(INFO) << landmark_pos << std::endl;

		floatVec landmark_106 = (landmark_pos["landmark_106"].get<std::vector<floatVec>>())[0];
		landmark_106.resize(106 * 2);
		auto landmark_106_scale = CalcHelper::scaleValue(landmark_106, orginal_all.rows / 128.0);
		auto adjust_106 = ImageUtils::adjustPosRightBrow(orginal_all, landmark_106_scale);
		
		cv::Mat draw_kps = ImageUtils::drawKps(orginal_all, landmark_106_scale,1.0);
		draw_kps = ImageUtils::drawKps(draw_kps, adjust_106, 1.0, cv::Scalar(0,255,0));
			
		cv::imwrite(brow_test + file_name[0] + ".jpg", draw_kps);
		//cv::imshow("draw_kps", draw_kps);
		//adjust pos
		json rewrite;
		json landmark;
		landmark["landmark_106"] = adjust_106;
		rewrite["face_type"] = landmark;
		FILEIO::saveJson(brow_test + file_name[0] + "_adjust.json", rewrite);
	}

#endif
}

void TESTFUNCTION::testSegResult()
{
	/*
	Label list
	0: 'background'	1: 'skin'	2: 'nose'
	3: 'eye_g'	4: 'l_eye'	5: 'r_eye'
	6: 'l_brow'	7: 'r_brow'	8: 'l_ear'
	9: 'r_ear'	10: 'mouth'	11: 'u_lip'
	12: 'l_lip'	13: 'hair'	14: 'hat'
	15: 'ear_r'	16: 'neck_l'	17: 'neck'
	18: 'cloth'
	*/
	
	cstr in_txt = "D:/dota201010/1016_seg/image_out/f05.jpg_crop.jpg.txt";
	matI face_seg;

	std::ifstream in_stream(in_txt);

	float a;
	in_stream >> a;
	LOG(INFO) << "a: " << a << std::endl;


	FILEIO::loadEigenMat(in_txt, 512, 512, face_seg);
	LOG(INFO) << "face_seg:" << std::endl << face_seg << std::endl;
	cvMatU canvas(512, 512);
	canvas.setTo(0);

	ucharVec res;
	for (int y = 0; y < 512; y++)
	{
		for (int x = 0; x < 512; x++)
		{
			res.push_back(face_seg(y, x));
		}
	}

	json out_char;
	out_char["seg"] = res;
	FILEIO::saveJson("D:/dota201010/1016_seg/test.json", out_char);

	for (int i = 0; i < 18; i++)
	{
#pragma omp parallel for
		for (int y = 0; y < 512; y++)
		{
			for (int x = 0; x < 512; x++)
			{
				if (face_seg(y, x) == i)
				{
					SETU(canvas, y, x, 255);
				}
				else
				{
					SETU(canvas, y, x, 0);
				}
			}
		}
		//cv::imshow("canvas", canvas);
		cv::imwrite("D:/dota201010/1016_seg/seg_match/" + std::to_string(i) + ".png", canvas);
		LOG(INFO) << "iter: " << i << std::endl;
		//cv::waitKey(0);
	}
}

void TESTFUNCTION::turnFaceSegIntoToPic()
{
	cstr folder = "D:/dota201010/1016_seg/image_out/";
	cstr output = "D:/dota201010/1016_seg/format_seg/";
	SG::needPath(output);
	cstrVec file_name = FILEIO::getFolderFiles(folder, { ".txt" }, false);
	for (auto iter_file :file_name)
	{
		matI face_seg;
		FILEIO::loadEigenMat(folder + iter_file, 512, 512, face_seg);
		LOG(INFO) << "face_seg:" << std::endl << face_seg << std::endl;
		intVec select_tag = { 2,3,17 };
		for (int i : select_tag)
		{
			cvMatU canvas(512, 512);
			canvas.setTo(0);
#pragma omp parallel for
			for (int y = 0; y < 512; y++)
			{
				for (int x = 0; x < 512; x++)
				{
					if (face_seg(y, x) == i)
					{
						SETU(canvas, y, x, 255);
					}
					else
					{
						SETU(canvas, y, x, 0);
					}
				}
			}
			//cv::imshow("canvas", canvas);
			cv::imwrite(output + std::to_string(i)+"_"+ iter_file + ".png", canvas);
			LOG(INFO) << "iter: " << i << std::endl;
		}
	}
}

