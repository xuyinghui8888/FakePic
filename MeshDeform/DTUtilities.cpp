#include "DTUtilities.h"
#include "../Basic/MeshHeader.h"

using namespace CGP;

void DTTools::fill4VertsOfFace(int id_f, const int3Vec& faces, const float3Vec& verts, 
	int4E& id_v, float3Vec& v)
{
	int3E f = faces[id_f];
	for (int k = 0; k < 3; k++)
	{
		id_v[k] = f[k];
		v[k] = verts[id_v[k]];
	}
	v[3] = v[0] + (v[1] - v[0]).cross(v[2] - v[0]).normalized();
	id_v[3] = (int)verts.size() + id_f;
}

void DTTools::getMatrixNamedbyT(const float3Vec& v, Eigen::Matrix<double, 3, 4>& A)
{
	mat3f v_res;
	getV(v, v_res);
	mat3f V = v_res.inverse();

	// The matrix T is in block diag style:
	// | A 0 0 |
	// | 0 A 0 |
	// | 0 0 A |
	// where each A is a 3x4 matrix
	A(0, 0) = -V(0, 0) - V(1, 0) - V(2, 0);
	A(0, 1) = V(0, 0);
	A(0, 2) = V(1, 0);
	A(0, 3) = V(2, 0);
	A(1, 0) = -V(0, 1) - V(1, 1) - V(2, 1);
	A(1, 1) = V(0, 1);
	A(1, 2) = V(1, 1);
	A(1, 3) = V(2, 1);
	A(2, 0) = -V(0, 2) - V(1, 2) - V(2, 2);
	A(2, 1) = V(0, 2);
	A(2, 2) = V(1, 2);
	A(2, 3) = V(2, 2);
}

void DTTools::getV(const float3Vec& v, mat3f& res)
{
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			res(y, x) = v[x + 1][y] - v[0][y];
		}
	}
}

void DTTools::fillCooSysByMat(tipDVec& cooSys, int row, int nTotalVerts, const int4E& id, const mat34d& T)
{
	// The matrix T is in block diag style:
	// | A 0 0 |
	// | 0 A 0 |
	// | 0 0 A |
	// where each A is a 3x4 matrix
	const static int nBlocks = 3;
	const static int nPoints = 4;
	const static int nCoords = 3;
	int pos = row * 4;
	for (int iBlock = 0; iBlock < nBlocks; iBlock++)
	{
		const int yb = iBlock * nCoords;
		for (int y = 0; y < nCoords; y++)
		{
			for (int x = 0; x < nPoints; x++)
			{
				const int col = nTotalVerts * iBlock + id[x];
				cooSys[pos++] = Eigen::Triplet<double>(row + yb + y, col, T(y, x));
			}
		}
	} // end for iBlock
}

void DTTools::setPartDelta(const intX2Vec& part_raw, const intVec& A_land, const intVec& B_land, 
	const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, float3Vec& res,
	bool is_debug, const cstr& result_root)
{
	res = float3Vec(A_land.size(), float3E(0,0,0));
	//A != B
	int A_vertex = A.n_vertex_;
	int B_vertex = B.n_vertex_;
	if (A_vertex != A_deform.n_vertex_)
	{
		LOG(ERROR) << "A_vertex && A_deform size not match." << std::endl;
	}
		
	//mapping to 
	intX2Vec part_raw_A = part_raw;
	intX2Vec part_raw_B = part_raw;

	for (int iter_part = 0; iter_part < part_raw.size(); iter_part++)
	{
		for (int i = 0; i < part_raw[iter_part].size(); i++)
		{
			part_raw_A[iter_part][i] = A_land[part_raw[iter_part][i]];
			part_raw_B[iter_part][i] = B_land[part_raw[iter_part][i]];
		}
	}
	   		
	for (int iter_part = 0; iter_part < part_raw.size(); iter_part++)
	{
		const intVec& part_idx_A = part_raw_A[iter_part];
		const intVec& part_idx_B = part_raw_B[iter_part];
		
		//test for rotation
		float3Vec A_slice, A_deform_slice, B_slice;
		A.getSlice(part_idx_A, A_slice);
		A_deform.getSlice(part_idx_A, A_deform_slice);
		B.getSlice(part_idx_B, B_slice);
		vecD normal, normal_deform, normal_B, z_pos;
		z_pos.resize(3);
		z_pos(2) = 1;
		MeshTools::getNormal(A_slice, normal);
		MeshTools::getNormal(A_deform_slice, normal_deform);
		MeshTools::getNormal(B_slice, normal_B);
		normal.normalize();
		normal_deform.normalize();
		normal_B.normalize();
		LOG(INFO) << "normal: " << normal.transpose() << std::endl;
		LOG(INFO) << "normal_deform: " << normal_deform.transpose() << std::endl;
		LOG(INFO) << "normal_B: " << normal_B.transpose() << std::endl;
		Eigen::Quaterniond out = Eigen::Quaterniond::FromTwoVectors(normal, normal_B);
		Eigen::Matrix3d Rx = out.toRotationMatrix();
		Eigen::Quaterniond out_A = Eigen::Quaterniond::FromTwoVectors(normal, z_pos);
		Eigen::Quaterniond out_B = Eigen::Quaterniond::FromTwoVectors(normal_B, z_pos);
		Eigen::Matrix3d R_A = out_A.toRotationMatrix();
		Eigen::Matrix3d R_B = out_B.toRotationMatrix();
		MeshCompress rot_A = A;
		MeshCompress rot_B = B;
		RT::getRotationInPlace(R_A.cast<float>(), rot_A.pos_);
		RT::getRotationInPlace(R_B.cast<float>(), rot_B.pos_);
		//

		if (is_debug)
		{
			rot_A.saveObj(result_root + std::to_string(iter_part) + " _rot_A.obj");
			rot_B.saveObj(result_root + std::to_string(iter_part) + "_rot_B.obj");
		}

		LOG(INFO) << "normal: " << (Rx*normal).transpose() << std::endl;
		LOG(INFO) << "out:" << out.x() << out.y() << out.z() << out.w() << std::endl;
		doubleVec A_xyz_min, A_xyz_max, B_xyz_min, B_xyz_max, scale;
		
		A.getBoundingBox(part_idx_A, R_A.cast<float>(), A_xyz_min, A_xyz_max);
		B.getBoundingBox(part_idx_B, R_B.cast<float>(), B_xyz_min, B_xyz_max);

		scale.resize(3);
		//scale B/A
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			scale[iter_dim] = safeDiv(B_xyz_max[iter_dim] - B_xyz_min[iter_dim], A_xyz_max[iter_dim] - A_xyz_min[iter_dim], 0);
		}

		//calculate normal
		int n_size = part_idx_A.size();
		for (int i = 0; i < n_size; i++)
		{
			int idx_i_A = part_idx_A[i];
			int idx_i_B = part_idx_B[i];
			float3E aim_dis = A_deform.pos_[idx_i_A] - A.pos_[idx_i_A];
			LOG(INFO) << "aim_dis_ori: " << aim_dis.transpose() << std::endl;
			aim_dis = R_A.cast<float>() * aim_dis;
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				aim_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim];
			}
			aim_dis = (R_B.transpose()).cast<float>() * aim_dis;
			float3E pos_B_aim = aim_dis + B.pos_[idx_i_B];
			int land_order = part_raw[iter_part][i];
			LOG(INFO) << "land_order: " << land_order << std::endl;
			LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
			LOG(INFO) << "land ori: " << B.pos_[idx_i_B].transpose() << std::endl;
			LOG(INFO) << "land deform: " << pos_B_aim.transpose() << std::endl;
			res[land_order] = pos_B_aim;
		}
	}
}

void DTTools::setPartOri(const intX2Vec& part_raw, const intVec& A_land, const intVec& B_land,
	const MeshCompress& A, const MeshCompress& A_deform, const MeshCompress& B, float3Vec& res,
	bool is_debug, const cstr& result_root)
{
	res = float3Vec(A_land.size(), float3E(0, 0, 0));
	//A != B
	int A_vertex = A.n_vertex_;
	int B_vertex = B.n_vertex_;
	if (A_vertex != A_deform.n_vertex_)
	{
		LOG(ERROR) << "A_vertex && A_deform size not match." << std::endl;
	}

	//mapping to 
	intX2Vec part_raw_A = part_raw;
	intX2Vec part_raw_B = part_raw;

	for (int iter_part = 0; iter_part < part_raw.size(); iter_part++)
	{
		for (int i = 0; i < part_raw[iter_part].size(); i++)
		{
			part_raw_A[iter_part][i] = A_land[part_raw[iter_part][i]];
			part_raw_B[iter_part][i] = B_land[part_raw[iter_part][i]];
		}
	}

	for (int iter_part = 0; iter_part < part_raw.size(); iter_part++)
	{
		const intVec& part_idx_A = part_raw_A[iter_part];
		const intVec& part_idx_B = part_raw_B[iter_part];

		//test for rotation
		float3Vec A_slice, A_deform_slice, B_slice;
		A.getSlice(part_idx_A, A_slice);
		A_deform.getSlice(part_idx_A, A_deform_slice);
		B.getSlice(part_idx_B, B_slice);

		float3E A_center, A_defomr_center, B_center;
		RT::getCenter(A_slice, A_center);
		RT::getCenter(B_slice, B_center);
		RT::getCenter(A_deform_slice, A_defomr_center);

		vecD normal, normal_deform, normal_B, z_pos;
		z_pos.resize(3);
		z_pos(2) = 1;
		MeshTools::getNormal(A_slice, normal);
		MeshTools::getNormal(A_deform_slice, normal_deform);
		MeshTools::getNormal(B_slice, normal_B);
		normal.normalize();
		normal_deform.normalize();
		normal_B.normalize();
		LOG(INFO) << "normal: " << normal.transpose() << std::endl;
		LOG(INFO) << "normal_deform: " << normal_deform.transpose() << std::endl;
		LOG(INFO) << "normal_B: " << normal_B.transpose() << std::endl;
		Eigen::Quaterniond out = Eigen::Quaterniond::FromTwoVectors(normal, normal_B);
		Eigen::Matrix3d Rx = out.toRotationMatrix();
		Eigen::Quaterniond out_A = Eigen::Quaterniond::FromTwoVectors(normal, z_pos);
		Eigen::Quaterniond out_B = Eigen::Quaterniond::FromTwoVectors(normal_B, z_pos);
		Eigen::Matrix3d R_A = out_A.toRotationMatrix();
		Eigen::Matrix3d R_B = out_B.toRotationMatrix();

		R_A = mat3d::Identity();
		R_B = mat3d::Identity();
		MeshCompress rot_A = A;
		MeshCompress rot_B = B;
		RT::getRotationInPlace(R_A.cast<float>(), rot_A.pos_);
		RT::getRotationInPlace(R_B.cast<float>(), rot_B.pos_);

		if (is_debug)
		{
			rot_A.saveObj(result_root + std::to_string(iter_part) + " _rot_A.obj");
			rot_B.saveObj(result_root + std::to_string(iter_part) + "_rot_B.obj");
		}

		LOG(INFO) << "normal: " << (Rx*normal).transpose() << std::endl;
		LOG(INFO) << "out:" << out.x() << out.y() << out.z() << out.w() << std::endl;
		doubleVec A_xyz_min, A_xyz_max, B_xyz_min, B_xyz_max, scale;

		A.getBoundingBox(part_idx_A, R_A.cast<float>(), A_xyz_min, A_xyz_max);
		B.getBoundingBox(part_idx_B, R_B.cast<float>(), B_xyz_min, B_xyz_max);

		scale.resize(3);
		//scale B/A
		for (int iter_dim = 0; iter_dim < 3; iter_dim++)
		{
			scale[iter_dim] = safeDiv(B_xyz_max[iter_dim] - B_xyz_min[iter_dim], A_xyz_max[iter_dim] - A_xyz_min[iter_dim], 0);
		}

		//calculate normal
		int n_size = part_idx_A.size();
		for (int i = 0; i < n_size; i++)
		{
			int idx_i_A = part_idx_A[i];
			int idx_i_B = part_idx_B[i];
			float3E aim_dis = A_deform.pos_[idx_i_A] - A_center;
			LOG(INFO) << "aim_dis_ori: " << aim_dis.transpose() << std::endl;
			aim_dis = R_A.cast<float>() * aim_dis;
			for (int iter_dim = 0; iter_dim < 3; iter_dim++)
			{
				aim_dis[iter_dim] = aim_dis[iter_dim] * scale[iter_dim];
			}
			aim_dis = (R_B.transpose()).cast<float>() * aim_dis;
			float3E pos_B_aim = aim_dis + B_center;
			pos_B_aim[2] = B.pos_[idx_i_B][2];
			int land_order = part_raw[iter_part][i];
			LOG(INFO) << "land_order: " << land_order << std::endl;
			LOG(INFO) << "aim_dis: " << aim_dis.transpose() << std::endl;
			LOG(INFO) << "land ori: " << B.pos_[idx_i_B].transpose() << std::endl;
			LOG(INFO) << "land deform: " << pos_B_aim.transpose() << std::endl;
			res[land_order] = pos_B_aim;
		}
	}
}

void DTTools::fastTransGivenStructure(const SpMat& A, SpMat& At) 
{
	Eigen::VectorXi positions(At.outerSize());
	for (int i = 0; i < At.outerSize(); i++)
		positions[i] = At.outerIndexPtr()[i];
	for (int j = 0; j < A.outerSize(); ++j)
	{
		for (SpMat::InnerIterator it(A, j); it; ++it)
		{
			int i = it.index();
			int pos = positions[i]++;
			At.valuePtr()[pos] = it.value();
		}
	}
}

void DTTools::fastAtAGivenStructure(const SpMat& A, const SpMat& At, SpMat& AtA)
{
	vecD Tmp;
	Eigen::VectorXi Mark;
	Tmp.resize(AtA.innerSize());
	Mark.resize(AtA.innerSize());
	Mark.setZero();

	for (int j = 0; j < AtA.outerSize(); j++)
	{
		for (SpMat::InnerIterator it_A(A, j); it_A; ++it_A)
		{
			int k = it_A.index();
			double v_A = it_A.value();

			for (SpMat::InnerIterator it_At(At, k); it_At; ++it_At)
			{
				int i = it_At.index();
				double v_At = it_At.value();
				if (!Mark[i])
				{
					Mark[i] = 1;
					Tmp[i] = v_A * v_At;
				}
				else
					Tmp[i] += v_A * v_At;
			}//end for it_At
		}//end for it_A

		for (SpMat::InnerIterator it(AtA, j); it; ++it)
		{
			int i = it.index();
			it.valueRef() = Tmp[i];
			Mark[i] = 0;
		}
	}//end for i
}
