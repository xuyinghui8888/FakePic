#include "DTSimLandmarkGuide.h"
#include "DTUtilities.h"
#include "../Basic/MeshHeader.h"
using namespace CGP;

// from Li Hao's "Example-Based Facial Rigging"
// In transfering, the weight of each triangle is not the same:
// if the gradient of a triangle moves little in the src pair, 
// it should be emphasised to move a little in the target pair
// The weight is calculated via:
//	(1 + ||M||_F)^theta / (kappa + ||M||_F)^theta
const static double Transfer_Graident_Emhasis_kappa = 0.1;
const static double Transfer_Graident_Emhasis_theta = 2.5;

void DTGuidedLandmark::initWeight()
{
	//Transfer_Weight_Pair = 1e3;
	//Transfer_Weight_Land = 1e3;
}

void DTGuidedLandmark::setPairType(const PairType& init)
{
	pair_type_ = init;
}

//fix position are in A && A_deform
intVec DTGuidedLandmark::initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const intVec& pair_info, double thres)
{
	initWeight();
	floatVec dis(A.pos_.size(), -1);
#pragma omp parallel for
	for (int i = 0; i < A.pos_.size(); i++)
	{
		dis[i] = (A.pos_[i] - A_deform.pos_[i]).norm();
	}
	intVec fix;
	for (int i = 0; i < A.pos_.size(); i++)
	{
		if (dis[i] < thres)
		{
			fix.push_back(i);
		}
	}
	init(A, A_deform, pair_info, fix);
	return fix;
}

void DTGuidedLandmark::setDebug(bool init_info)
{
	is_debug_ = init_info;
}

intVec DTGuidedLandmark::initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intVec& pair_info, 
	double thres)
{
	initWeight();
	floatVec dis(A.pos_.size(), -1);
#pragma omp parallel for
	for (int i = 0; i < A.pos_.size(); i++)
	{
		dis[i] = (A.pos_[i] - A_deform.pos_[i]).norm();
	}
	intVec fix;
	intSet pair_set(pair_info.begin(), pair_info.end());
	for (int i = 0; i < A.pos_.size(); i++)
	{
		if (dis[i] < thres && !pair_set.count(i))
		{
			fix.push_back(i);
		}
	}
	init(A, A_deform, B, pair_info, fix);
	return fix;
}

bool DTGuidedLandmark::setupExExB()
{
	setupEaA(pair_info_);
	setupErA();
	setupE1A(A_deform_vert_pos_);
	setupErB(A_deform_vert_pos_);
	if (DTTools::hasIllegalData(E1_.valuePtr(), (int)E1_.nonZeros()))
	{
		LOG(ERROR) << "nan or inf in E1Mat!" << std::endl;
		return false;
	}

	const double w_anchor = double(Transfer_Weight_Anchor / (1e-3f + Ea_.rows()));
	const double w_reg = double(Transfer_Weight_Regularization / (1e-3f + Er_AtA_.rows()));
	const double w1 = double(Transfer_Weight_Correspond / (1e-3f + E1_.rows()));
	double w_pair = Transfer_Weight_Pair / (1e-3f + E1_.rows());
	double w_land = Transfer_Weight_Land / (1e-3f + E1_.rows());

	Ea_t_ = Ea_.transpose();
	E1_t_ = E1_.transpose();
	E1_t_ *= w1;

	E_AtA_ = E1_t_ * E1_ + Ea_t_ * Ea_ * w_anchor + Er_AtA_ * w_reg;
	Ea_Er_AtB_ = Ea_t_ * Ea_B_ * w_anchor + Er_AtB_ * w_reg;
	if (!pair_info_.empty())
	{
		E_AtA_ = E_AtA_ + Ep_t_ * Ep_* w_pair;
		Ea_Er_AtB_ = Ea_Er_AtB_ + Ep_t_ * Ep_B_*w_pair;
	}

	if (!part_info_.empty())
	{
		E_AtA_ = E_AtA_ + Ed_t_ * Ed_* w_land;
		Ea_Er_AtB_ = Ea_Er_AtB_ + Ed_t_ * Ed_B_*w_land;
	}
	solver_.compute(E_AtA_);

	is_init_ = true;
	is_check_topology_ = true;
	return true;
}

bool DTGuidedLandmark::init(const MeshCompress& A, const MeshCompress& A_deform, const intVec& pair_info, const intVec& fix)
{
	initWeight();
	if (A.safeMeshData() && A_deform.safeMeshData())
	{
		if (A.n_vertex_ != A_deform.n_vertex_ || A.n_tri_ != A_deform.n_tri_)
		{
			LOG(ERROR) << "A and A_deform have different topology." << std::endl;
			return false;
		}
		else
		{
			clear();
			A_vert_pos_ = A.pos_;
			A_deform_vert_pos_ = A_deform.pos_;
			face_tri_.resize(A.n_tri_);
			SG::safeMemcpy(face_tri_.data(), A.tri_.data(), sizeof(int)*A.n_tri_ * 3);
			anchor_ = fix;
			setupEaA();
			setupEaB(A_deform_vert_pos_);
			pair_info_ = pair_info;
			return setupExExB();
		}
	}
	else
	{
		LOG(ERROR) << "check mesh data failed. Vertex contains NAN or face idx error" << std::endl;
		return false;
	}
}

bool DTGuidedLandmark::init(const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intVec& pair_info, const intVec& fix)
{
	initWeight();
	if (A.safeMeshData() && A_deform.safeMeshData())
	{
		if (A.n_vertex_ != A_deform.n_vertex_ || A.n_tri_ != A_deform.n_tri_)
		{
			LOG(ERROR) << "A and A_deform have different topology." << std::endl;
			return false;
		}
		else
		{
			clear();
			A_vert_pos_ = A.pos_;
			A_deform_vert_pos_ = A_deform.pos_;
			B_vert_pos_ = B.pos_;
			face_tri_.resize(A.n_tri_);
			SG::safeMemcpy(face_tri_.data(), A.tri_.data(), sizeof(int)*A.n_tri_ * 3);
			anchor_ = fix;
			setupEaA();
			setupEaB(B.pos_);
			pair_info_ = pair_info;
			return setupExExB();
		}
	}
	else
	{
		LOG(ERROR) << "check mesh data failed. Vertex contains NAN or face idx error" << std::endl;
		return false;
	}
}

bool DTGuidedLandmark::transfer(const float3Vec& srcVertsDeformed, float3Vec& tarVertsDeformed)
{
	if (!is_init_)
	{
		LOG(ERROR) << "not initialized when calling transfer()" << std::endl;
		return false;
	}
	if (srcVertsDeformed.size() != A_vert_pos_.size())
	{
		LOG(ERROR) << "transfer: vertex size not matched!" << std::endl;
		return false;
	}
	if (DTTools::hasIllegalData((const float*)srcVertsDeformed.data(), (int)srcVertsDeformed.size() * 3))
	{
		LOG(ERROR) << "nan or inf in srcVertsDeformed!" << std::endl;
		return false;
	}

	// computing all energy matrices
	setupE1B(srcVertsDeformed);

	// sum all the energy terms
	E_AtB_ = E1_t_ * E1_B_;
	E_AtB_ += Ea_Er_AtB_;

	// solve
	res_vertex_pos_ = solver_.solve(E_AtB_);

	// return the value
	vertexVecToPoint(res_vertex_pos_, tarVertsDeformed);

	if (DTTools::hasIllegalData((const float*)tarVertsDeformed.data(), (int)tarVertsDeformed.size() * 3))
	{
		LOG(ERROR) << "finished transfer, but nan or inf in tarVertsDeformed!" << std::endl;
		return false;
	}
	return true;
}

void DTGuidedLandmark::clear()
{
	is_init_ = false;
	is_check_topology_ = false;
	face_tri_.clear();
	anchor_.clear();
	A_vert_pos_.clear();
}

void DTGuidedLandmark::vertexVecToPoint(const vecD& x, float3Vec& verts)const
{
	verts.resize(A_vert_pos_.size());
	for (int i = 0; i < verts.size(); i++)
		for (int k = 0; k < 3; k++)
			verts[i][k] = (float)x[k*(x.size() / 3) + i];
}

void DTGuidedLandmark::vertexPointToVec(vecD& x, const float3Vec& verts, const int3Vec& faces)const
{
	int nTotalVerts = (int)verts.size() + (int)faces.size();
	if (x.size() != nTotalVerts)
		x.resize(nTotalVerts * 3);

	x.setZero();
	for (int i = 0; i < verts.size(); i++)
		for (int k = 0; k < 3; k++)
			x[k*nTotalVerts + i] = verts[i][k];

	for (int i = 0; i < faces.size(); i++)
	{
		int row = (int)verts.size() + i;
		const int3E& f = faces[i];
		float3E v = verts[f[0]] + (verts[f[1]] - verts[f[0]]).cross(verts[f[2]] - verts[f[0]]).normalized();
		for (int k = 0; k < 3; k++)
			x[k*nTotalVerts + row] = v[k];
	}
}

void DTGuidedLandmark::setupE1A(const float3Vec& tarVerts0)
{
	if (is_E1A_init_ == true)
	{
		return;
	}
	const int nMeshVerts = (int)tarVerts0.size();
	const int nTotalVerts = nMeshVerts + (int)face_tri_.size();
	std::vector<Eigen::Triplet<double>> cooSys;

	E1_B_.resize(face_tri_.size() * 9);
	cooSys.resize(E1_B_.size() * 4);

	mat34d Ti;
	Ti.setZero();
	int4E id_vi_tar;
	float3Vec vi_tar(4);
	for (int iFace = 0; iFace < (int)face_tri_.size(); iFace++)
	{
		// face_i_tar
		DTTools::fill4VertsOfFace(iFace, face_tri_, tarVerts0, id_vi_tar, vi_tar);
		DTTools::getMatrixNamedbyT(vi_tar, Ti);

		// construct the gradient transfer matrix
		bool inValid = DTTools::hasIllegalData(Ti.data(), (int)Ti.size());
		if (inValid)
			Ti.setZero();

		// push matrix
		const int row = iFace * 9;
		DTTools::fillCooSysByMat(cooSys, row, nTotalVerts, id_vi_tar, Ti);
	}

	E1_.resize((int)E1_B_.size(), nTotalVerts * 3);
	if (cooSys.size() > 0)
		E1_.setFromTriplets(cooSys.begin(), cooSys.end());
}

void DTGuidedLandmark::fastSetupE1A(const DTGuidedLandmark& src)
{
	E1_ = src.E1_;
	E1_B_ = src.E1_B_;
	is_E1A_init_ = true;
}

void DTGuidedLandmark::fastSetupE1A(const SpMat& E1, const vecD& E1_B)
{
	E1_ = E1;
	E1_B_ = E1_B;
	is_E1A_init_ = true;
}

void DTGuidedLandmark::setupE1B(const float3Vec& srcVertsDeformed)
{
	E1_B_.resize(face_tri_.size() * 9);
	Eigen::Matrix<double, 3, 4> Si_A;
	Eigen::Matrix<double, 4, 1> Si_x[3];
	Eigen::Matrix<double, 3, 1> Si_b[3];
	Si_A.setZero();
	int4E id_vi_src0, id_vi_src1;
	float3Vec vi_src0(4), vi_src1(4);
	for (int iFace = 0; iFace < (int)face_tri_.size(); iFace++)
	{
		// face_i_src
		DTTools::fill4VertsOfFace(iFace, face_tri_, A_vert_pos_, id_vi_src0, vi_src0);
		DTTools::fill4VertsOfFace(iFace, face_tri_, srcVertsDeformed, id_vi_src1, vi_src1);

		// construct the gradient transfer matrix
		DTTools::getMatrixNamedbyT(vi_src0, Si_A);
		bool inValid = DTTools::hasIllegalData(Si_A.data(), (int)Si_A.size());
		for (int k = 0; k < 4; k++)
		{
			Si_x[0][k] = vi_src1[k][0];
			Si_x[1][k] = vi_src1[k][1];
			Si_x[2][k] = vi_src1[k][2];
		}
		if (inValid)
		{
			Si_A.setZero();
			Si_x[0].setZero();
			Si_x[1].setZero();
			Si_x[2].setZero();
		}
		Si_b[0] = Si_A * Si_x[0];
		Si_b[1] = Si_A * Si_x[1];
		Si_b[2] = Si_A * Si_x[2];

		// push matrix
		const int row = iFace * 9;
		E1_B_[row + 0] = Si_b[0][0];
		E1_B_[row + 1] = Si_b[0][1];
		E1_B_[row + 2] = Si_b[0][2];

		E1_B_[row + 3] = Si_b[1][0];
		E1_B_[row + 4] = Si_b[1][1];
		E1_B_[row + 5] = Si_b[1][2];

		E1_B_[row + 6] = Si_b[2][0];
		E1_B_[row + 7] = Si_b[2][1];
		E1_B_[row + 8] = Si_b[2][2];
	}
}

void DTGuidedLandmark::setupEaA()
{
	const int nMeshVerts = (int)A_vert_pos_.size();
	const int nTotalVerts = nMeshVerts + (int)face_tri_.size();
	Ea_.resize((int)anchor_.size() * 3, nTotalVerts * 3);

	// build matrix
	for (int i = 0; i < anchor_.size(); i++)
	{
		for (int k = 0; k < 3; k++)
			Ea_.insert(i * 3 + k, anchor_[i] + nTotalVerts * k) = 1;
	}
	Ea_.finalize();
}

void DTGuidedLandmark::setupEaB(const float3Vec& tarVerts0)
{
	Ea_B_.resize((int)anchor_.size() * 3);
	Ea_B_.setZero();

	// build matrix
	for (int i = 0; i < anchor_.size(); i++)
	{
		for (int k = 0; k < 3; k++)
			Ea_B_[i * 3 + k] = tarVerts0[anchor_[i]][k];
	}
}

void DTGuidedLandmark::setupErA()
{
	const int nMeshVerts = (int)A_vert_pos_.size();
	const int nTotalVerts = nMeshVerts + (int)face_tri_.size();
	Er_AtA_.resize(nTotalVerts * 3, nTotalVerts * 3);
	Er_AtA_.reserve(nTotalVerts * 3);
	for (int row = 0; row < Er_AtA_.rows(); row++)
		Er_AtA_.insert(row, row) = 1;
	Er_AtA_.finalize();
}

void DTGuidedLandmark::setupEaA(const intVec& pairInfo)
{
	if (pairInfo.empty() || pairInfo.size() % 2 == 1)
	{
		LOG(WARNING) << "pairInfo empty or error, NOT adding for pair info." << std::endl;
		return;
	}
	int n_vertex = A_vert_pos_.size();
	int n_total_var = n_vertex + face_tri_.size();
	tipDVec A_trip;
	
	int n_pair = pairInfo.size() / 2;
	int n_row = 3*n_pair;
	//set b
	Ep_B_.resize(n_row);
	if (B_vert_pos_.empty())
	{
		LOG(ERROR) << "B_vert_pos_ is empty." << std::endl;
	}
	for (int i = 0; i < n_pair; i++)
	{
		int cur_idx = 3 * i;
		int src_idx = pairInfo[2 * i];
		int dst_idx = pairInfo[2 * i + 1];
		float3E A_src_dst = A_vert_pos_[src_idx] - A_vert_pos_[dst_idx];
		float3E B_src_dst = B_vert_pos_[src_idx] - B_vert_pos_[dst_idx];
		Eigen::Quaternionf out = Eigen::Quaternionf::FromTwoVectors(A_src_dst, B_src_dst);
		Eigen::Matrix3f Rx = out.toRotationMatrix();
		float scale_dis = safeDiv(B_src_dst.norm(), A_src_dst.norm(), 0);
		scale_dis = scale_dis > 1.5 ? 1 : scale_dis;
		//scale_dis = scale_dis<0.1 ? 0 : scale_dis;
		float3E rotate_dis = scale_dis* Rx * (A_deform_vert_pos_[src_idx] - A_deform_vert_pos_[dst_idx]);	
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			A_trip.push_back(tripD(cur_idx + iter_dim, src_idx + iter_dim * n_total_var, 1));
			A_trip.push_back(tripD(cur_idx + iter_dim, dst_idx + iter_dim * n_total_var, -1));
			if (pair_type_ == PairType::PAIR_DIS)
			{
				Ep_B_(cur_idx + iter_dim) = rotate_dis[iter_dim];
			}
			else if (pair_type_ == PairType::PAIR_ZERO)
			{
				Ep_B_(cur_idx + iter_dim) = 0;
			}
			else
			{
				LOG(ERROR) << "solution type not defined." << std::endl;
			}
		}
	}
	Ep_.resize(n_row, n_total_var * 3);
	if (!A_trip.empty())
	{
		Ep_.setFromTriplets(A_trip.begin(), A_trip.end());
	}
	Ep_t_ = Ep_.transpose();
}

void DTGuidedLandmark::setupErB(const float3Vec& tarVerts0)
{
	const int nMeshVerts = (int)A_vert_pos_.size();
	const int nTotalVerts = nMeshVerts + (int)face_tri_.size();
	Er_AtB_.resize(nTotalVerts * 3);
	Er_AtB_.setZero();
	for (int iVert = 0; iVert < nMeshVerts; iVert++)
	{
		for (int k = 0; k < 3; k++)
			Er_AtB_[iVert + k * nTotalVerts] = tarVerts0[iVert][k];
	}
}

void DTGuidedLandmark::getPartNormal(const intX2Vec& part_info, const std::vector<LandGuidedType>& type, const MeshCompress& A, std::vector<vecD>& normal)
{
	if (part_info.size() != type.size())
	{
		LOG(ERROR) << "part_info && type not same" << std::endl;
	}
	int total_fix = 0;
	for (auto iter_idx : part_info)
	{
		total_fix += iter_idx.size();
	}
	part_info_ = part_info;
	normal.resize(part_info_.size());
#pragma omp parallel for
	for (int iter_part = 0; iter_part < part_info_.size(); iter_part++)
	{
		const intVec& part_idx = part_info[iter_part];
		//test for rotation
		float3Vec A_slice, A_deform_slice, B_slice;
		A.getSlice(part_idx, A_slice);
		vecD normal_A;
		MeshTools::getNormal(A_slice, normal_A);
		normal_A.normalize();
		normal[iter_part] = normal_A;
	}
}

void DTGuidedLandmark::setPart(const intX2Vec& part_info, const std::vector<LandGuidedType>& type,
	const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intX2Vec& xy_part_info,
	const std::vector<vecD>& A_normal_fix, const std::vector<vecD>& B_normal_fix)
{
	if (part_info.size() != type.size())
	{
		LOG(ERROR) << "part_info && type not same" << std::endl;
	}
	int total_fix = 0;
	for (auto iter_idx : part_info)
	{
		total_fix += iter_idx.size();
	}
	Ed_B_.resize(total_fix*3);
	tipDVec A_trip;
	part_info_ = part_info;
	int n_vertex = A.n_vertex_;
	int n_total_var = n_vertex + A.n_tri_;
	int n_row = 0;

	std::vector<vecD> A_normal = A_normal_fix;
	std::vector<vecD> B_normal = B_normal_fix;
	if (A_normal.empty())
	{
		getPartNormal(part_info, type, A, A_normal);
	}

	if (B_normal.empty())
	{
		getPartNormal(part_info, type, B, B_normal);
	}

	for (int iter_part = 0; iter_part < part_info_.size(); iter_part++)
	{
		const intVec& part_idx = part_info[iter_part];
		//test for rotation
		float3Vec A_slice, A_deform_slice, B_slice;
		A.getSlice(part_idx, A_slice);
		A_deform.getSlice(part_idx, A_deform_slice);
		B.getSlice(part_idx, B_slice);
		vecD z_pos;
		z_pos.resize(3);
		z_pos.setConstant(0);
		z_pos(2) = 1;
		A_normal[iter_part].normalize();
		B_normal[iter_part].normalize();
		if (is_debug_)
		{
			LOG(INFO) << "normal: " << A_normal[iter_part].transpose() << std::endl;
			LOG(INFO) << "normal_B: " << B_normal[iter_part].transpose() << std::endl;
		}
		Eigen::Quaterniond out = Eigen::Quaterniond::FromTwoVectors(A_normal[iter_part], B_normal[iter_part]);
		Eigen::Matrix3d Rx = out.toRotationMatrix();
		Eigen::Quaterniond out_A = Eigen::Quaterniond::FromTwoVectors(A_normal[iter_part], z_pos);
		Eigen::Quaterniond out_B = Eigen::Quaterniond::FromTwoVectors(B_normal[iter_part], z_pos);
		mat3d R_A = out_A.toRotationMatrix();
		mat3d R_B = out_B.toRotationMatrix();
		if (type[iter_part] == LandGuidedType::XY_NO_ROTATE)
		{
			R_A = mat3d::Identity();
			R_B = mat3d::Identity();
		}
		if (type[iter_part] == LandGuidedType::XY_ADD_PLANE_ROTATE && !xy_part_info.empty()
			&& !xy_part_info[iter_part].empty())
		{
			MeshCompress rot_A = A;
			MeshCompress rot_B = B;
			RT::getRotationInPlace(R_A.cast<float>(), rot_A.pos_);
			RT::getRotationInPlace(R_B.cast<float>(), rot_B.pos_);
			//Ôö¼Ó¶îÍârotate xy plane
			int src_idx = xy_part_info[iter_part][0];
			int dst_idx = xy_part_info[iter_part][1];
			float3E A_src_dst = R_A.cast<float>()*(rot_A.pos_[src_idx] - rot_A.pos_[dst_idx]);
			float3E B_src_dst = R_B.cast<float>()*(rot_B.pos_[src_idx] - rot_B.pos_[dst_idx]);
			A_src_dst.z() = 0;
			B_src_dst.z() = 0;
			Eigen::Quaternionf out = Eigen::Quaternionf::FromTwoVectors(A_src_dst, B_src_dst);
			Eigen::Matrix3f Rxy = out.toRotationMatrix();
			R_A = Rxy.cast<double>() * R_A;
		}
		//
		//rot_A.saveObj("D:/avatar/0803_df_00/" + std::to_string(iter_part) + "_A_rot.obj");
		//rot_B.saveObj("D:/avatar/0803_df_00/" + std::to_string(iter_part) + "_B_rot.obj");
		if (is_debug_)
		{
			//rot_A = A;
			//rot_B = B;
			//RT::getRotationInPlace(R_A.cast<float>(), rot_A.pos_);
			//RT::getRotationInPlace(R_B.cast<float>(), rot_B.pos_);
			LOG(INFO) << "normal: " << (Rx*A_normal[iter_part]).transpose() << std::endl;
			LOG(INFO) << "out:" << out.x() << out.y() << out.z() << out.w() << std::endl;
		}

		doubleVec A_xyz_min, A_xyz_max, B_xyz_min, B_xyz_max, scale;
		A.getBoundingBox(part_idx, R_A.cast<float>(), A_xyz_min, A_xyz_max);
		B.getBoundingBox(part_idx, R_B.cast<float>(), B_xyz_min, B_xyz_max);
		scale.resize(3);
		//scale B/A
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			scale[iter_dim] = safeDiv(B_xyz_max[iter_dim] - B_xyz_min[iter_dim], A_xyz_max[iter_dim] - A_xyz_min[iter_dim], 0);
		}

		//calculate normal
		int n_size = part_idx.size();
		for (int i = 0; i < n_size; i++)
		{
			int idx_i = part_idx[i];
			float3E aim_dis = A_deform.pos_[idx_i] - A.pos_[idx_i];
			aim_dis = R_A.cast<float>() * aim_dis;
			if (type[iter_part] == LandGuidedType::XY_OPT_NO_SCALE)
			{
				scale = doubleVec(3, 1.0);
			}
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				aim_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim];
			}

			aim_dis = (R_B.transpose()).cast<float>() * aim_dis;
			//
			if (type[iter_part] == LandGuidedType::XY_OPT || type[iter_part] == LandGuidedType::XY_NO_ROTATE
				|| type[iter_part] == LandGuidedType::XY_OPT_NO_SCALE)
			{
				//fix z
				aim_dis[2] = A_deform.pos_[idx_i].z() - A.pos_[idx_i].z();
			}
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				aim_dis[iter_dim] = aim_dis[iter_dim] + B.pos_[idx_i][iter_dim];
				A_trip.push_back(tripD(n_row + iter_dim, idx_i + n_total_var * iter_dim, 1.0));
				Ed_B_[n_row / 3 * 3 + iter_dim] = aim_dis[iter_dim];				
			}
			n_row += 3;		
		}
	}
	Ed_.resize(n_row, n_total_var * 3);
	Ed_.setFromTriplets(A_trip.begin(), A_trip.end());
	Ed_t_ = Ed_.transpose();
}

