#include "DTSim.h"
#include "DTUtilities.h"
using namespace CGP;
#if 0
const static double Transfer_Weight_Smoothness = 1e-3;
const static double Transfer_Weight_Identity = 1e-6;
const static double Transfer_Weight_Correspond = 1.0;
const static double Transfer_Weight_Anchor = 1e8;
const static double Transfer_Weight_Regularization = 1e-8;
const static double Transfer_Weight_Pair = 1.0;
#else
const static double Transfer_Weight_Correspond = 1.0;
const static double Transfer_Weight_Anchor = 1e8;
const static double Transfer_Weight_Regularization = 1e-8;
const static double Transfer_Weight_Pair = 1e3;
#endif
//const static double Transfer_Weight_Regularization = 0;

// from Li Hao's "Example-Based Facial Rigging"
// In transfering, the weight of each triangle is not the same:
// if the gradient of a triangle moves little in the src pair, 
// it should be emphasised to move a little in the target pair
// The weight is calculated via:
//	(1 + ||M||_F)^theta / (kappa + ||M||_F)^theta
const static double Transfer_Graident_Emhasis_kappa = 0.1;
const static double Transfer_Graident_Emhasis_theta = 2.5;

intVec MeshTransfer::initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const intVec& v, double thres)
{
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
	init(A, A_deform, v, fix);
	return fix;
}

intVec MeshTransfer::initDynamicFix(const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, const intVec& v, double thres)
{
	floatVec dis(A.pos_.size(), -1);
	floatVec dis_A_B = dis;
#pragma omp parallel for
	for (int i = 0; i < A.pos_.size(); i++)
	{
		dis[i] = (A.pos_[i] - A_deform.pos_[i]).norm();
		dis_A_B[i] = (A.pos_[i] - B.pos_[i]).norm();
	}
	intVec fix;
	for (int i = 0; i < A.pos_.size(); i++)
	{
		if (dis[i] < thres && dis_A_B[i] < thres)
		{
			fix.push_back(i);
		}
	}
	init(A, A_deform, v, fix);
	return fix;
}

bool MeshTransfer::init(const MeshCompress& A, const MeshCompress& A_deform, const intVec& pairInfo, const intVec& fix)
{
	if (A.safeMeshData() && A_deform.safeMeshData())
	{
		if (A.n_vertex_ != A_deform.n_vertex_ || A.n_tri_ != A_deform.n_tri_)
		{
			LOG(ERROR) << "A and A_deform have different topology." << std::endl;
			return false;
		}
		else
		{
			int3Vec temp_tri;
			temp_tri.resize(A.n_tri_);
			SG::safeMemcpy(temp_tri.data(), A.tri_.data(), sizeof(int)*A.n_tri_ * 3);
			init(A.n_tri_, temp_tri.data(), A.n_vertex_, A.pos_.data(), A_deform.pos_.data(), pairInfo, fix);
			return true;
		}
	}
	else
	{
		LOG(ERROR) << "check mesh data failed. Vertex contains NAN or face idx error" << std::endl;
		return false;
	}
}

bool MeshTransfer::init(int nTriangles, const int3E* pTriangles, int nVertices,
	const float3E* pSrcVertices0, const float3E* pTarVertices0, const intVec& pairInfo, const intVec& fix_points)
{
	clear();
	if (DTTools::hasIllegalData((const float*)pSrcVertices0, nVertices * 3))
	{
		LOG(ERROR) << "nan or inf in input pSrcVertices0" << std::endl;
		return false;
	}
	if (DTTools::hasIllegalData((const float*)pTarVertices0, nVertices * 3))
	{
		LOG(ERROR) << "nan or inf in input pTarVertices0" << std::endl;
		return false;
	}
	if (DTTools::hasIllegalTriangle(pTriangles, nTriangles))
	{
		LOG(ERROR) << "illegal or trivial triangles in pTriangles!" << std::endl;;
		return false;
	}

	A_vert_pos_.resize(nVertices);
	memcpy(A_vert_pos_.data(), pSrcVertices0, nVertices * sizeof(float3E));

	A_deform_vert_pos_.resize(nVertices);
	memcpy(A_deform_vert_pos_.data(), pTarVertices0, nVertices * sizeof(float3E));

	face_tri_.resize(nTriangles);
	for (int i = 0; i < nTriangles; i++)
		face_tri_[i] = pTriangles[i];

	anchor_ = fix_points;
	// precomputation
	//findAnchorPoints();
	setupEaA();
	setupErA();
	pair_info_ = pairInfo;
	setupEaA(pairInfo, A_deform_vert_pos_);
	setupE1A(A_deform_vert_pos_);
	setupErB(A_deform_vert_pos_);
	setupEaB(A_deform_vert_pos_);

	if (DTTools::hasIllegalData(E1_.valuePtr(), (int)E1_.nonZeros()))
	{
		LOG(ERROR) << "nan or inf in E1Mat!" << std::endl;
		return false;
	}

	const double w_anchor = double(Transfer_Weight_Anchor / (1e-3f + Ea_.rows()));
	const double w_reg = double(Transfer_Weight_Regularization / (1e-3f + Er_AtA_.rows()));
	const double w1 = double(Transfer_Weight_Correspond / (1e-3f + E1_.rows()));
	double w_pair = Transfer_Weight_Pair / (1e-3f + E1_.rows());

	Ea_t_ = Ea_.transpose();
	E1_t_ = E1_.transpose();
	E1_t_ *= w1;

	E_AtA_ = E1_t_ * E1_ + Ea_t_ * Ea_ * w_anchor + Er_AtA_ * w_reg;
	Ea_Er_AtB_ = Ea_t_ * Ea_B_ * w_anchor + Er_AtB_ * w_reg;
	if (!pairInfo.empty())
	{
		E_AtA_ = E_AtA_ + Ep_t_ * Ep_* w_pair;
		Ea_Er_AtB_ = Ea_Er_AtB_ + Ep_t_ * Ep_B_*w_pair;
	}

	solver_.compute(E_AtA_);

	is_init_ = true;
	is_check_topology_ = true;
	return true;
}

bool MeshTransfer::transfer(const float3Vec& srcVertsDeformed, float3Vec& tarVertsDeformed)
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

void MeshTransfer::clear()
{
	is_init_ = false;
	is_check_topology_ = false;
	face_tri_.clear();
	anchor_.clear();
	A_vert_pos_.clear();
}

void MeshTransfer::vertexVecToPoint(const vecD& x, float3Vec& verts)const
{
	verts.resize(A_vert_pos_.size());
	for (int i = 0; i < verts.size(); i++)
		for (int k = 0; k < 3; k++)
			verts[i][k] = (float)x[k*(x.size() / 3) + i];
}

void MeshTransfer::vertexPointToVec(vecD& x, const float3Vec& verts, const int3Vec& faces)const
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

void MeshTransfer::setupE1A(const float3Vec& tarVerts0)
{
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

void MeshTransfer::setupE1B(const float3Vec& srcVertsDeformed)
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

void MeshTransfer::setupEaA()
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

void MeshTransfer::setupEaB(const float3Vec& tarVerts0)
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

void MeshTransfer::setupErA()
{
	const int nMeshVerts = (int)A_vert_pos_.size();
	const int nTotalVerts = nMeshVerts + (int)face_tri_.size();
	Er_AtA_.resize(nTotalVerts * 3, nTotalVerts * 3);
	Er_AtA_.reserve(nTotalVerts * 3);
	for (int row = 0; row < Er_AtA_.rows(); row++)
		Er_AtA_.insert(row, row) = 1;
	Er_AtA_.finalize();
}

void MeshTransfer::setupEaA(const intVec& pairInfo, const float3Vec& tarVerts0)
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
	int n_row = 3 * n_pair;
	//set b
	Ep_B_.resize(n_row);
	for (int i = 0; i < n_pair; i++)
	{
		int cur_idx = 3 * i;
		int src_idx = pairInfo[2 * i];
		int dst_idx = pairInfo[2 * i + 1];
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			A_trip.push_back(tripD(cur_idx + iter_dim, src_idx + iter_dim * n_total_var, 1));
			A_trip.push_back(tripD(cur_idx + iter_dim, dst_idx + iter_dim * n_total_var, -1));
			Ep_B_(cur_idx + iter_dim) = (A_deform_vert_pos_[src_idx] - A_deform_vert_pos_[dst_idx])[iter_dim];
		}
	}
	Ep_.resize(n_row, n_total_var * 3);
	if (!A_trip.empty())
	{
		Ep_.setFromTriplets(A_trip.begin(), A_trip.end());
	}
	Ep_t_ = Ep_.transpose();
}

void MeshTransfer::setupErB(const float3Vec& tarVerts0)
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