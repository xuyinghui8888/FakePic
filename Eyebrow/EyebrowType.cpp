#include "EyebrowType.h"
#include "../Basic/MeshHeader.h"
#include "../MeshDeform/DTSim.h"
#include "../Solver/BVLSSolver.h"
#include "../OptV2/OptTypeUtils.h"
#include "../ImageSim/ImageUtils.h"
#include "../CalcFunction/CalcHelper.h"

using namespace CGP;

EyebrowType::EyebrowType(const json& config)
{
	if (is_init_ == false)
	{
		init(config);
	}
}

void EyebrowType::init(const json& config)
{
	if (is_init_ == true)
	{
		LOG(INFO) << "already init for data." << std::endl;
		return;
	}
	LOG(INFO) << "init for EyebrowType." << std::endl;
	data_root_ = config["root"].get<cstr>();
	JsonHelper::initData(data_root_ + config["eyebrow_tensor_"].get<cstr>(), "config.json", eyebrow_tensor_);

	is_init_ = true;
	LOG(INFO) << "end of init EyebrowType." << std::endl;
}

void EyebrowType::getEyebrowTypeOnlyLandmark(json& json_in, json& json_out)
{
	json_out = json_in;
	floatVec landmark_106 = json_in["landmark_106"].get<floatVec>();
	json_out["eyebrow_type"] = calculateEyebrowTypeUseLandmark(landmark_106);
}

void EyebrowType::getEyebrowType(json& json_in, json& json_out)
{
	floatVec landmark_106 = json_in["landmark_106"].get<floatVec>();
	intVec seg_info = json_in["seg"].get<intVec>();
	LOG(INFO) << "init input data" << std::endl;
	initSegmentImage(seg_info);
	LOG(INFO) << "end of init segment image" << std::endl;
	json_out.clear();
	json_out["eyebrow_type"] = calculateEyebrowTypeUseLandmark(landmark_106, seg_info);
	json_out["landmark_106_adjust"] = landmark_106_adjust_;
}

void EyebrowType::initSegmentImage(const intVec& seg)
{
	LOG(INFO) << "seg size: " << seg.size() << std::endl;
/*
	https://yuque.antfin-inc.com/zhaohaiming.pt/kb/fz6qh3
	auto res = std::minmax(seg.begin(), seg.end());
	int max_seg_idx = *res.first;
	int min_seg_idx = *res.second;

*/
	auto min_max = std::minmax_element(seg.begin(), seg.end());
	int min_seg_idx = *min_max.first;
	int max_seg_idx = *min_max.second;

	LOG(INFO) << "min_seg_idx: " << min_seg_idx << std::endl;
	LOG(INFO) << "max_seg_idx: " << max_seg_idx << std::endl;
	int n_seg = max_seg_idx + 1;
	cvMatU canvas_base(512, 512);
	canvas_base.setTo(0);

	//v0 && v1 use the same image pointer
	//canvas_seg_ = std::vector<cvMatU>(n_seg, canvas_base);
	//canvas_seg_.resize(n_seg, canvas_base);
	
	canvas_seg_.resize(n_seg);
	for (int i = 0; i < n_seg; i++)
	{
		canvas_seg_[i] = canvas_base.clone();
	}

#pragma omp parallel for
	for (int i = 0; i < seg.size(); i++)
	{
		int y = i / 512;
		int x = i % 512;
		int seg_info = seg[i];
		SETU(canvas_seg_[seg_info], y, x, 255);
	}

	if (debug_)
	{
		//visualize for image
		for (int i = 0; i < n_seg; i++)
		{
#ifdef _WIN32
			cv::imwrite(result_dir_ + std::to_string(i) + ".png", canvas_seg_[i]);
#endif
		}	
	}
}

void EyebrowType::getRightSideOfMask(const cvMatU& seg_2, const cvMatU& seg_3, cvMatU& res)
{
	cv::Rect rect_seg_2 = ImageUtils::maskBox(seg_2);
	cv::Rect rect_seg_3 = ImageUtils::maskBox(seg_3);
	bool use_2 = true;
	if (rect_seg_2.br().x < rect_seg_3.br().x)
	{
		//Ñ¡ÔñÓÒ²à×îÓÒµÄ,Èç¹ûseg_3¸üÍùÓÒ²à£¬ÔòÊ¹ÓÃseg_3
		use_2 = false;
	}
	/*
		       (260, 10)    

			    width = 250, height = 200                 
	*/
	cv::Rect right_roi = cv::Rect(260, 10, 250, 200);
	cvMatU res_raw = use_2 ? seg_2.clone() : seg_3.clone();
	res = res_raw.clone();
	res.setTo(0);
	res_raw(right_roi).copyTo(res(right_roi));

	if (debug_)
	{
#ifdef _WIN32
		//cv::imshow("res", res);
		//cv::waitKey(0);
#endif
	}
}

void EyebrowType::getCombineAndSegRight(const cvMatU& seg_2, const cvMatU& seg_3, cvMatU& res)
{
	/*
			   (260, 10)

				width = 250, height = 200
	*/
	cv::Rect right_roi = cv::Rect(260, 10, 250, 200);
	cvMatU res_raw = seg_2.clone();
	res_raw.setTo(0);
	cv::bitwise_or(seg_2, seg_3, res_raw);
	res = res_raw.clone();
	res.setTo(0);
	res_raw(right_roi).copyTo(res(right_roi));

	if (debug_)
	{
#ifdef _WIN32
		//cv::imshow("res", res);
		//cv::imshow("res_raw", res_raw);
		//cv::waitKey(0);
#endif
	}
}

int EyebrowType::calculateEyebrowTypeUseLandmark(const floatVec& landmark_106_ori)
{
	//rotation
	intVec pair = {
		80  ,81,
		82  ,83,
		0	,32,
		1	,31,
		2	,30,
		3	,29,
		4	,28,
		5	,27,
		6	,26,
		7	,25,
		8	,24,
		9	,23,
		10	,22,
		11	,21,
		12	,20,
		13	,19,
		14	,18,
		15	,17,
	};

	float3E mean_vec = float3E(0, 0, 0);
	for (int i = 0; i < pair.size() / 2; i++)
	{
		float3E left(landmark_106_ori[pair[2 * i] * 2], -landmark_106_ori[pair[2 * i] * 2 + 1], 0);
		float3E right(landmark_106_ori[pair[2 * i + 1] * 2], -landmark_106_ori[pair[2 * i + 1] * 2 + 1], 0);
		float3E vec = left - right;
		vec.normalize();
		mean_vec = mean_vec + vec;
	}
	float3E hori = float3E(1, 0, 0);
	Eigen::Quaternionf to_horizontal = Eigen::Quaternionf::FromTwoVectors(mean_vec, hori);
	Eigen::Matrix3f Rx_to_horizontal = to_horizontal.toRotationMatrix();

	//LOG(INFO) << "mean_vec:" << mean_vec.transpose() << std::endl;
	//LOG(INFO) << "mean_vec rotate:" << (Rx_to_horizontal*mean_vec).transpose() << std::endl;
	//direct rotate 


	floatVec landmark_106 = landmark_106_ori;
	for (int i = 0; i < 106; i++)
	{
		float3E ori(landmark_106_ori[2*i], landmark_106_ori[2 * i + 1], 0);
		float3E rotate_ori = Rx_to_horizontal * ori;
		landmark_106[2 * i] = rotate_ori[0];
		landmark_106[2 * i+1] = rotate_ori[1];
	}	
	
	intVec roi = {38,39,40,41,42,68,69,70,71};
	   	 
	//get raw pos
	float3Vec landmark_pos;
	for (int i: roi)
	{
		float3E temp_pos(landmark_106[i * 2], -landmark_106[i * 2 + 1], 0);
		//add for rotation
		landmark_pos.push_back(Rx_to_horizontal*temp_pos);
	}

	doubleVec landmark_xyz_min, landmark_xyz_max, mean_xyz_min, mean_xyz_max;
	MeshTools::getBoundingBox(landmark_pos, landmark_xyz_min, landmark_xyz_max);
	MeshTools::getBoundingBox(eyebrow_tensor_.template_obj_.pos_, mean_xyz_min, mean_xyz_max);
	//scale based on x
	float scale = safeDiv((mean_xyz_max[0] - mean_xyz_min[0]), (landmark_xyz_max[0] - landmark_xyz_min[0]), 0);
	RT::scaleInPlace(scale, landmark_pos);

	float3E center_landmark, center_tensor;
	MeshTools::getCenter(landmark_pos, center_landmark);
	MeshTools::getCenter(eyebrow_tensor_.template_obj_.pos_, center_tensor);
	float3E shift_landmark_to_tensor = center_tensor - center_landmark;
	RT::translateInPlace(shift_landmark_to_tensor, landmark_pos);

	if (debug_)
	{
		//MeshCompress show_pos = eyebrow_tensor_.template_obj_;
		//show_pos.pos_ = landmark_pos;
		//show_pos.saveObj("D:/avatar200923/0928_eyebrow/show.obj");
	}

	vecD dist_ratio_pass, dist_angle, dist_angle_global, first_pass;
	//LOG(WARNING) << "debug begin" << std::endl;	
	//int first_type = getTypeFromTensor(eyebrow_tensor_, landmark_pos, first_pass);
	//int dist_type = getTypeFromDist(eyebrow_tensor_, landmark_pos, {0,1,2,3,4,5,6,7,8,9,10,11,12}, dist_pass);
	int dist_type_ratio = getTypeFromDistLocalRatio(eyebrow_tensor_, landmark_pos, { 0,1,2,3,4,5,6,7,8,9,10,11,12 }, dist_ratio_pass);
	int dist_type_angle = getTypeFromDistLocalAngle(eyebrow_tensor_, landmark_pos, { 0,1,2,3,4,5,6,7,8,9,10,11,12 }, dist_angle);
	int dist_type_global_angle = getTypeFromAngle(eyebrow_tensor_, landmark_pos, { 0,1,2,3,4,5,6,7,8,9,10,11,12 }, dist_angle_global);
	//int mix_dist = getTypeFrom({ dist_ratio_pass , dist_angle, dist_angle_global });
	int mix_order_dist = getTypeFromOrder({ dist_ratio_pass , dist_angle, dist_angle_global });
	LOG(INFO) << "dist_type_ratio: " << dist_type_ratio << std::endl;
	LOG(INFO) << "dist_type_angle: " << dist_type_angle << std::endl;
	LOG(INFO) << "dist_type_global_angle: " << dist_type_global_angle << std::endl;
	LOG(INFO) << "mix_order_dist: " << mix_order_dist << std::endl;
	//get pos >0.2
	//float first_pass_thres = 0.1;
	//intVec roi_pass;
	//for (int i = 0; i < first_pass.size(); i++)
	//{
	//	if (first_pass[i] > first_pass_thres)
	//	{
	//		roi_pass.push_back(i);
	//	}
	//}

	//if (roi_pass.empty() || first_pass[first_type]>0.7)
	//{
	//	final_type =  first_type;
	//}
	//else
	//{
	//	final_type =  getTypeFromDist(eyebrow_tensor_, landmark_pos, roi_pass, first_pass);
	//}
	//LOG(WARNING) << "debug end" << std::endl;
	return dist_type_global_angle;
}

floatVec EyebrowType::adjust106Landmark(const cvMatU& right_mask, const floatVec& landmark_106)
{
	floatVec landmark_106_back = landmark_106;
	landmark_106_back.resize(106 * 2);
	auto landmark_106_scale = CalcHelper::scaleValue(landmark_106_back, right_mask.rows / 128.0);
	return ImageUtils::adjustPosRightBrow(right_mask, landmark_106_scale);
}

int EyebrowType::calculateEyebrowTypeUseLandmark(const floatVec& landmark_106_ori, const intVec& seg)
{
	/*£¨×óÓÒ°´ÕÕÈËµÄ×óÓÒ½øÐÐ£©
		Label list
		0: '±³¾°'  1: 'Æ¤·ô'  2: 'ÓÒ²àÃ¼Ã«'  3: '×ó²àÃ¼Ã«'  4: 'ÓÒ²àÑÛ¾¦'  5: '×ó²àÑÛ¾¦'
		6: 'x'  7: 'ÓÒ²à¶ú¶ä'  8: '×ó²à¶ú¶ä' 9: 'x'  10: '±Ç×Ó'  11: 'x'
		12: 'ÉÏ×ì´½'  13: 'ÏÂ×ì´½'  14: '²±×Ó' 15: 'x'  16: 'x'  17: 'Í··¢'
	*/
	landmark_106_adjust_ = landmark_106_ori;
	//check for mask on right side of image
	cvMatU right_side_mask;
	getCombineAndSegRight(canvas_seg_[2], canvas_seg_[3], right_side_mask);
	LOG(INFO) << "end of get right mask" << std::endl;
	landmark_106_adjust_ = adjust106Landmark(right_side_mask, landmark_106_ori);
	LOG(INFO) << "end of get adjust 106" << std::endl;
	return calculateEyebrowTypeUseLandmark(landmark_106_adjust_);
}

int EyebrowType::getTypeFrom(const std::vector<vecD>& dist)
{
	if (dist.empty())
	{
		return 0;
	}
	vecD res = dist[0];
	for (int i = 1; i < dist.size(); i++)
	{
		res = res.cwiseProduct(dist[i]);
	}
	int min_idx = 0;
	for (int i = 0; i < res.size(); i++)
	{
		min_idx = res[i] > res[min_idx] ? min_idx : i;
	}
	LOG(INFO) << "getTypeFrom mix: " << min_idx << std::endl;
	return min_idx;
}


int EyebrowType::getTypeFromTensor(const Tensor& tensor, const float3Vec& landmark_pos, vecD& coef)
{
	intSet weight_p = { 0,4,5};
	float add_weight = 3;
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = tensor.template_obj_.pos_.size()*3 + 1;
	int n_var = tensor.n_id_ -1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1);
	int shift = tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < tensor.template_obj_.pos_.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = i;
			float vertex_weight = 1.0;
			if (weight_p.count(vertex_id))
			{
				vertex_weight = add_weight;
			}
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim] * vertex_weight;
				B(3 * i + iter_dim) = (landmark_pos[vertex_id][iter_dim] - tensor.data_[vertex_id * 3 + iter_dim]) * vertex_weight;
			}
		}
	}
	float weight_sum_1 = 1e7;
	double sum_extra = 1.00;
	//last
	for (int j = 0; j < n_var; j++)
	{
		A(n_constrain  - 1 , j) = weight_sum_1 * 1.0;
		B(n_constrain - 1) = weight_sum_1 * sum_extra;
	}

	if (debug_)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;
	int max_idx = 0;
	for (int i = 0; i < coef.size(); i++)
	{
		max_idx = coef[i] > coef[max_idx] ? i : max_idx;
	}
	LOG(INFO) << "getTypeFromTensor: " << max_idx << std::endl;
	return max_idx;
}

int EyebrowType::getTypeFromTensor(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef)
{
	//calculate constrain: n_roi * xyz(3)  up_down_match *0.5*3  sum_coef = 1 
	int n_constrain = tensor.template_obj_.pos_.size() * 3 + 1;
	int n_var = tensor.n_id_ - 1;
	matD A(n_constrain, n_var);
	vecD B(n_constrain), lower(n_var), upper(n_var);
	A.setConstant(0);
	B.setConstant(0);
	lower.setConstant(0);
	upper.setConstant(1e-6);
	for (int i : roi)
	{
		upper[i] = 1.0;
	}

	int shift = tensor.template_obj_.n_vertex_ * 3;
	for (int i = 0; i < tensor.template_obj_.pos_.size(); i++)
	{
		//in x y z order
		for (int j = 0; j < n_var; j++)
		{
			int vertex_id = i;
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				A(3 * i + iter_dim, j) = tensor.data_[shift + shift * j + vertex_id * 3 + iter_dim];
				B(3 * i + iter_dim) = (landmark_pos[vertex_id][iter_dim] - tensor.data_[vertex_id * 3 + iter_dim]);
			}
		}
	}
	float weight_sum_1 = 1e3;
	double sum_extra = 1.00;
	//last
	for (int j = 0; j < n_var; j++)
	{
		A(n_constrain - 1, j) = weight_sum_1 * 1.0;
		B(n_constrain - 1) = weight_sum_1 * sum_extra;
	}

	if (debug_)
	{
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/A.txt", A);
		//FILEIO::saveEigenDynamic("D:/avatar/0823_test/B.txt", B.transpose());
	}

	BVLSSolver test(A, B, lower, upper);
	coef = test.solve();
	LOG(INFO) << "if converge: " << test.converged() << std::endl;
	//LOG(INFO) << "getSolution: " << coef.transpose() << std::endl;
	int max_idx = 0;
	for (int i = 0; i < coef.size(); i++)
	{
		max_idx = coef[i] > coef[max_idx] ? i : max_idx;
	}
	return max_idx;
}

int EyebrowType::getTypeFromDist(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef)
{	
	intSet weight_p = { 0,4,5};
	float add_weight = 3;
	coef.resize(tensor.n_id_ - 1);
	coef.setConstant(INT_MAX);	
	vecF landmark_vec(landmark_pos.size() * 3);
	SG::safeMemcpy(landmark_vec.data(), landmark_pos.data(), sizeof(float)*landmark_pos.size() * 3);
	for (int i : weight_p)
	{
		for (int j = 0; j < 3; j++)
		{
			landmark_vec[3 * i + j] = add_weight * landmark_vec[3 * i + j];
		}
	}

	vecF base = landmark_vec;
	SG::safeMemcpy(base.data(), &tensor.data_[0], sizeof(float)*landmark_pos.size() * 3);

	for (int i: roi)
	{
		int shift = tensor.template_obj_.n_vertex_ * 3;
		vecF roi_vec = landmark_vec;
		SG::safeMemcpy(roi_vec.data(), &tensor.data_[shift + shift * i], sizeof(float)*landmark_pos.size() * 3);
		roi_vec = base + roi_vec;
		for (int i_roi : weight_p)
		{
			for (int j = 0; j < 3; j++)
			{
				roi_vec[3 * i_roi + j] = add_weight * roi_vec[3 * i_roi + j];
			}
		}
		coef[i] = (roi_vec - landmark_vec).norm();
	}	
	
	//LOG(INFO) << "getSolution getTypeFromDist: "<<std::endl << coef << std::endl;
	int min_idx = 0;
	for (int i = 0; i < coef.size(); i++)
	{
		min_idx = coef[i] > coef[min_idx] ? min_idx : i;
	}
	LOG(INFO) << "getTypeFromDist: " << min_idx << std::endl;
	return min_idx;
}

void EyebrowType::getLocalCord(const float3Vec& landmark_pos, const vecF& actual_pos, vecF& local_pos)
{
	doubleVec xyz_min, xyz_max;
	MeshTools::getBoundingBox(landmark_pos, xyz_min, xyz_max);
	local_pos = actual_pos;
	int n_vertex = landmark_pos.size();
	for (int i = 0; i < n_vertex; i++)
	{
		local_pos[3 * i] = safeDiv(actual_pos[3 * i] - xyz_min[0], xyz_max[0] - xyz_min[0], 0);
		local_pos[3 * i + 1] = safeDiv(actual_pos[3 * i +1] - xyz_min[1], xyz_max[1] - xyz_min[1], 0);
	}
}

void EyebrowType::getLocalVec(const float3Vec& landmark_pos, float3Vec& local_vec)
{
	local_vec = landmark_pos;
	float3Vec local_pos = landmark_pos;
	doubleVec xyz_min, xyz_max;
	MeshTools::getBoundingBox(landmark_pos, xyz_min, xyz_max);
	int n_vertex = landmark_pos.size();
	for (int i = 0; i < n_vertex; i++)
	{
		local_pos[i][0] = safeDiv(landmark_pos[i][0] - xyz_min[0], xyz_max[0] - xyz_min[0], 0);
		local_pos[i][1] = safeDiv(landmark_pos[i][1] - xyz_min[1], xyz_max[1] - xyz_min[1], 0);
	}
	//get local vec
	local_vec[0] = local_pos[1] - local_pos[0];
	local_vec[1] = local_pos[2] - local_pos[1];
	local_vec[2] = local_pos[3] - local_pos[2];
	local_vec[3] = local_pos[4] - local_pos[3];
	local_vec[4] = local_pos[8] - local_pos[4];
	local_vec[5] = local_pos[7] - local_pos[8];
	local_vec[6] = local_pos[6] - local_pos[7];
	local_vec[7] = local_pos[5] - local_pos[6];
	local_vec[8] = local_pos[5] - local_pos[0];
}

void EyebrowType::getVec(const float3Vec& landmark_pos, float3Vec& local_vec)
{
	local_vec = landmark_pos;
	//get local vec
	local_vec[0] = landmark_pos[1] - landmark_pos[0];
	local_vec[1] = landmark_pos[2] - landmark_pos[1];
	local_vec[2] = landmark_pos[3] - landmark_pos[2];
	local_vec[3] = landmark_pos[4] - landmark_pos[3];
	local_vec[4] = landmark_pos[8] - landmark_pos[4];
	local_vec[5] = landmark_pos[7] - landmark_pos[8];
	local_vec[6] = landmark_pos[6] - landmark_pos[7];
	local_vec[7] = landmark_pos[5] - landmark_pos[6];
	local_vec[8] = landmark_pos[5] - landmark_pos[0];
}

void EyebrowType::getLocalCord(const vecF& landmark_pos_vec, const vecF& actual_pos, vecF& local_pos)
{
	float3Vec landmark_pos;
	for (int i = 0; i < landmark_pos_vec.size()/3; i++)
	{
		float3E land_3d = float3E(landmark_pos_vec[3 * i], landmark_pos_vec[3 * i + 1], 0);
		landmark_pos.push_back(land_3d);
	}


	doubleVec xyz_min, xyz_max;
	MeshTools::getBoundingBox(landmark_pos, xyz_min, xyz_max);
	local_pos = actual_pos;
	int n_vertex = landmark_pos.size();
	for (int i = 0; i < n_vertex; i++)
	{
		local_pos[3 * i] = safeDiv(actual_pos[3 * i] - xyz_min[0], xyz_max[0] - xyz_min[0], 0);
		local_pos[3 * i + 1] = safeDiv(actual_pos[3 * i + 1] - xyz_min[1], xyz_max[1] - xyz_min[1], 0);
	}
}



int EyebrowType::getTypeFromDistLocalRatio(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef)
{
	intSet weight_p = { 0,4,5 };
	float add_weight = 1.0;
	coef.resize(tensor.n_id_ - 1);
	coef.setConstant(INT_MAX);
	vecF landmark_vec(landmark_pos.size() * 3);
	SG::safeMemcpy(landmark_vec.data(), landmark_pos.data(), sizeof(float)*landmark_pos.size() * 3);
	vecF landmark_vec_local;
	getLocalCord(landmark_pos, landmark_vec, landmark_vec_local);
	for (int i : weight_p)
	{
		for (int j = 0; j < 3; j++)
		{
			landmark_vec_local[3 * i + j] = add_weight * landmark_vec_local[3 * i + j];
		}
	}

	vecF base = landmark_vec;
	SG::safeMemcpy(base.data(), &tensor.data_[0], sizeof(float)*landmark_pos.size() * 3);

	for (int i : roi)
	{
		int shift = tensor.template_obj_.n_vertex_ * 3;
		vecF roi_vec = landmark_vec;
		SG::safeMemcpy(roi_vec.data(), &tensor.data_[shift + shift * i], sizeof(float)*landmark_pos.size() * 3);
		roi_vec = roi_vec + base;
		//LOG(INFO) << "roi_vec:" << roi_vec.transpose() << std::endl;
		vecF roi_vec_local;
		getLocalCord(roi_vec, roi_vec, roi_vec_local);		
		//LOG(INFO) << "roi_vec_local:" << roi_vec_local.transpose() << std::endl;
		for (int i_roi : weight_p)
		{
			for (int j = 0; j < 3; j++)
			{
				roi_vec_local[3 * i_roi + j] = add_weight * roi_vec_local[3 * i_roi + j];
			}
		}
		coef[i] = (roi_vec_local - landmark_vec_local).norm();
	}

	//LOG(INFO) << "getSolution getTypeFromDistLocalRatio: " << std::endl << coef.transpose() << std::endl;
	int min_idx = 0;
	for (int i = 0; i < coef.size(); i++)
	{
		min_idx = coef[i] > coef[min_idx] ? min_idx : i;
	}
	//LOG(INFO) << "getTypeFromDistLocalRatio: " << min_idx << std::endl;
	return min_idx;
}

int EyebrowType::getTypeFromOrder(const std::vector<vecD>& dist)
{
	if (dist.empty())
	{
		return 0;
	}
	vecF res(dist[0].size());
	res.setConstant(0);

	for (int i = 0; i < dist.size(); i++)
	{
		vecF temp_order;
		CalcHelper::pairSort(dist[i], temp_order);
		res = res + temp_order;
	}
	vecF add_one = res;
	add_one.setConstant(1);

	res = res + add_one;

	vecF weight = res;
	weight.setConstant(0);
	weight[1] = 7;
	weight[4] = 7;
	weight[7] = 6;
	weight[10] = 10;
	//LOG(INFO) << "res: " <<std::endl<< res << std::endl;
	res = res + weight;
	//LOG(INFO) << "res weight: " << std::endl << res << std::endl;
	int min_idx = 0;
	for (int i = 0; i < res.size(); i++)
	{
		min_idx = res[i] > res[min_idx] ? min_idx : i;
	}
	//LOG(INFO) << "getTypeFromOrder: " << min_idx << std::endl;
	return min_idx;
}

int EyebrowType::getTypeFromDistLocalAngle(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef)
{
	//doubleVec weight_p = { 0.5,1,1,0.10,0.10,1,1,0.5,0.2 };
	doubleVec weight_p = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
	coef.resize(tensor.n_id_ - 1);
	coef.setConstant(INT_MAX);
	float3Vec landmark_angle;
	//SG::safeMemcpy(landmark_vec.data(), landmark_pos.data(), sizeof(float)*landmark_pos.size() * 3);
	getLocalVec(landmark_pos, landmark_angle);
	vecF base(landmark_pos.size() * 3);
	SG::safeMemcpy(base.data(), &tensor.data_[0], sizeof(float)*landmark_pos.size() * 3);

	for (int i : roi)
	{
		int shift = tensor.template_obj_.n_vertex_ * 3;
		vecF roi_vec = base;
		SG::safeMemcpy(roi_vec.data(), &tensor.data_[shift + shift * i], sizeof(float)*landmark_pos.size() * 3);
		roi_vec = roi_vec + base;
		float3Vec roi_angle = landmark_angle;
		float3Vec roi_pos = landmark_angle;
		SG::safeMemcpy(roi_pos.data(), roi_vec.data(), sizeof(float)*landmark_pos.size() * 3);
		getLocalVec(roi_pos, roi_angle);
		float dist_i = 0;
		for (int iter_vec = 0; iter_vec < roi_angle.size(); iter_vec++)
		{
			roi_angle[iter_vec].normalize();
			landmark_angle[iter_vec].normalize();
			dist_i += weight_p[iter_vec]*(roi_angle[iter_vec].cross(landmark_angle[iter_vec])).norm();
		}

		coef[i] = dist_i;
	}

	//LOG(INFO) << "getSolution getTypeFromDistLocalRatio: " << std::endl << coef.transpose() << std::endl;
	int min_idx = 0;
	for (int i = 0; i < coef.size(); i++)
	{
		min_idx = coef[i] > coef[min_idx] ? min_idx : i;
	}
	//LOG(INFO) << "getTypeFromDistLocalAngle: " << min_idx << std::endl;
	return min_idx;
}

int EyebrowType::getTypeFromAngle(const Tensor& tensor, const float3Vec& landmark_pos, const intVec& roi, vecD& coef)
{
	//doubleVec weight_p = { 0.5,1,1,0.10,0.10,1,1,0.5,0.2 };
	doubleVec weight_p = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
	coef.resize(tensor.n_id_ - 1);
	coef.setConstant(INT_MAX);
	float3Vec landmark_angle;
	//SG::safeMemcpy(landmark_vec.data(), landmark_pos.data(), sizeof(float)*landmark_pos.size() * 3);
	getVec(landmark_pos, landmark_angle);
	vecF base(landmark_pos.size() * 3);
	SG::safeMemcpy(base.data(), &tensor.data_[0], sizeof(float)*landmark_pos.size() * 3);

	for (int i : roi)
	{
		int shift = tensor.template_obj_.n_vertex_ * 3;
		vecF roi_vec = base;
		SG::safeMemcpy(roi_vec.data(), &tensor.data_[shift + shift * i], sizeof(float)*landmark_pos.size() * 3);
		roi_vec = roi_vec + base;
		float3Vec roi_angle = landmark_angle;
		float3Vec roi_pos = landmark_angle;
		SG::safeMemcpy(roi_pos.data(), roi_vec.data(), sizeof(float)*landmark_pos.size() * 3);
		getVec(roi_pos, roi_angle);
		float dist_i = 0;
		for (int iter_vec = 0; iter_vec < roi_angle.size(); iter_vec++)
		{
			roi_angle[iter_vec].normalize();
			landmark_angle[iter_vec].normalize();
			dist_i += weight_p[iter_vec] * (roi_angle[iter_vec].cross(landmark_angle[iter_vec])).norm();
		}

		coef[i] = dist_i;
	}

	//LOG(INFO) << "getSolution getTypeFromAngle: " << std::endl << coef.transpose() << std::endl;
	int min_idx = 0;
	for (int i = 0; i < coef.size(); i++)
	{
		min_idx = coef[i] > coef[min_idx] ? min_idx : i;
	}
	//LOG(INFO) << "getTypeFromAngle: " << min_idx << std::endl;
	return min_idx;
}

void EyebrowType::setDebug(bool is_debug)
{
	debug_ = is_debug;
}

void EyebrowType::setResultDir(const cstr& init)
{
	result_dir_ = init;
	SG::needPath(result_dir_);
}


