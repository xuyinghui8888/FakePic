#include "ShellGenerate.h"
#include "ComVector3d.h"
#include "MeshDistanceUtils.h"

using namespace CGP;
using namespace SDF;

void SHELLGEN::normalizeInPlace(CGP::doubleVec& src)
{
	if (src.size() % 3 != 0)
	{
		LOG(ERROR) << "size not fit" << std::endl;
		return;
	}
	int size = src.size() / 3;
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		double3E i_src = double3E(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
		i_src.normalize();
		src[3 * i + 0] = i_src.x();
		src[3 * i + 1] = i_src.y();
		src[3 * i + 2] = i_src.z();
	}
}

void SHELLGEN::makeFurthestMesh(doubleX2Vec& aExtXYZ, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, 
	const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode, bool is_ext, double max_height )
{
	int n = (int)aExtXYZ.size();
	aExtXYZ.clear();
	//assert(aExtXYZ.size() == 0);
	double* grad_old = new double[nXYZ * 3];
	double* grad_nor = new double[nXYZ * 3];
	double* grad_new = new double[nXYZ * 3];
	{
		double* aNorm_ = grad_nor;
		for (int i = 0; i < nXYZ * 3; i++) { aNorm_[i] = 0; }
		for (int itri = 0; itri < nTri; itri++)
		{
			unsigned int i1 = paTri[itri * 3 + 0];
			unsigned int i2 = paTri[itri * 3 + 1];
			unsigned int i3 = paTri[itri * 3 + 2];
			double un[3], area;
			Com::UnitNormalAreaTri3D(un, area, paXYZ + i1 * 3, paXYZ + i2 * 3, paXYZ + i3 * 3);
			aNorm_[i1 * 3 + 0] += un[0];  aNorm_[i1 * 3 + 1] += un[1];  aNorm_[i1 * 3 + 2] += un[2];
			aNorm_[i2 * 3 + 0] += un[0];  aNorm_[i2 * 3 + 1] += un[1];  aNorm_[i2 * 3 + 2] += un[2];
			aNorm_[i3 * 3 + 0] += un[0];  aNorm_[i3 * 3 + 1] += un[1];  aNorm_[i3 * 3 + 2] += un[2];
		}
		for (unsigned int ino = 0; ino < nXYZ; ino++) {
			double invlen = 1.0 / Com::Length3D(aNorm_ + ino * 3);
			aNorm_[ino * 3 + 0] *= invlen;
			aNorm_[ino * 3 + 1] *= invlen;
			aNorm_[ino * 3 + 2] *= invlen;
		}
	}

	double h0 = (is_ext) ? 0.0001 : -0.0001;

	double* pa_start = new double[nXYZ * 3];

	//aExtXYZ.resize(1);
	//aExtXYZ[0].resize(nXYZ * 3, 0);
	//double* pa_pos_new = aExtXYZ[0].data();
	for (unsigned int i = 0; i < nXYZ * 3; i++)
	{
		pa_start[i] = paXYZ[i] + grad_nor[i] * h0;
		//
		//pa_pos_new[i] = paXYZ[i] + grad_nor[i] * max_height;
	}

	//return;

	for (unsigned int ixyz = 0; ixyz < nXYZ; ixyz++)
	{
		SDF::UTILS::getValueGradientGrowth(pa_start + ixyz * 3, grad_nor + ixyz * 3,
			0.0001,
			nXYZ, paXYZ, nTri, paTri,
			topo,
			aNode);
	}
	for (int i = 0; i < nXYZ * 3; i++)
	{
		grad_old[i] = grad_nor[i];
	}
	double height;
	{
		aExtXYZ.resize(1);
		aExtXYZ[0].resize(nXYZ * 3, 0);
		double* pa_pos_new = aExtXYZ[0].data();

		for (unsigned int i = 0; i < nXYZ * 3; i++)
		{
			pa_pos_new[i] = pa_start[i];
		}

		for (unsigned int ixyz = 0; ixyz < nXYZ; ixyz++)
		{
			height = 0;
			//height += h0;
			double add_h = h0;
			while (fabs(height) < max_height)
			{
				double cos0 = 0;
				double cos1 = 0;
				while (cos0 < cos(3.1415*0.05) && cos1 < cos(3.1415*0.05))
				{
					SDF::UTILS::getValueGradientGrowth(pa_pos_new + ixyz * 3, grad_old + ixyz * 3,
						height,
						nXYZ, paXYZ, nTri, paTri,
						topo,
						aNode);
					double direction[3];
					for (int iter_dir = 0; iter_dir < 3; iter_dir++)
					{
						direction[iter_dir] = (pa_pos_new[ixyz * 3 + iter_dir] + grad_old[ixyz * 3 + iter_dir] * fabs(add_h) - paXYZ[ixyz * 3 + iter_dir]);
					}
					double dot0 = Com::Dot3D(grad_old + ixyz * 3, grad_nor + ixyz * 3);
					double dot1 = Com::Dot3D(direction, grad_nor + ixyz * 3);
					double len0 = Com::Length3D(grad_old + ixyz * 3);
					double len1 = Com::Length3D(grad_nor + ixyz * 3);
					double len2 = Com::Length3D(direction);
					if (len0 > 1 || len1 > 1 || len2 > 1)
					{
						//LOG(INFO) << "len0: " << len0 << std::endl;
						//LOG(INFO) << "len1: " << len1 << std::endl;
						//LOG(INFO) << "len2: " << len2 << std::endl;
					}
					cos0 = dot0 / (len0*len1);
					cos1 = dot1 / (len2*len1);
					cos1 = 1;
					//在此处设置生长最远处的条件
					if (cos0 > cos(3.1415*0.05) && cos1 > cos(3.1415*0.05))
					{
						//height += add_h;
						for (int vector_iter = 0; vector_iter < 3; vector_iter++)
						{
							pa_pos_new[ixyz * 3 + vector_iter] = pa_pos_new[ixyz * 3 + vector_iter] + grad_old[ixyz * 3 + vector_iter] * fabs(add_h);
						}
						height += add_h;
						add_h = add_h * 2.0;
					}
					else
					{
						add_h = add_h / 8.0;
						if (fabs(add_h) < 1e-8)
						{
							break;
						}
					}
				}
				if (fabs(add_h) < 1e-8)
				{
					break;
				}
				if (fabs(height) > max_height)
				{
					break;
				}

			}
			//std::cout<<height<<std::endl;
		}


	}
	delete[] grad_old;
	delete[] grad_new;
	delete[] grad_nor;
	delete[] pa_start;
}


void SHELLGEN::makeFurthestMeshNormalize(doubleX2Vec& aExtXYZ, unsigned int nXYZ, const double* paXYZ, unsigned int nTri,
	const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode, bool is_ext, double max_height)
{
	int n = (int)aExtXYZ.size();
	aExtXYZ.clear();
	//assert(aExtXYZ.size() == 0);
	doubleVec grad_old(nXYZ * 3, 0);
	doubleVec grad_nor(nXYZ * 3, 0);
	doubleVec grad_new(nXYZ * 3, 0);
	doubleVec pa_start(nXYZ * 3, 0);
	{
		double* aNorm_ = grad_nor.data();
		for (int i = 0; i < nXYZ * 3; i++) { aNorm_[i] = 0; }
		for (int itri = 0; itri < nTri; itri++)
		{
			unsigned int i1 = paTri[itri * 3 + 0];
			unsigned int i2 = paTri[itri * 3 + 1];
			unsigned int i3 = paTri[itri * 3 + 2];
			double un[3], area;
			Com::UnitNormalAreaTri3D(un, area, paXYZ + i1 * 3, paXYZ + i2 * 3, paXYZ + i3 * 3);
			aNorm_[i1 * 3 + 0] += un[0];  aNorm_[i1 * 3 + 1] += un[1];  aNorm_[i1 * 3 + 2] += un[2];
			aNorm_[i2 * 3 + 0] += un[0];  aNorm_[i2 * 3 + 1] += un[1];  aNorm_[i2 * 3 + 2] += un[2];
			aNorm_[i3 * 3 + 0] += un[0];  aNorm_[i3 * 3 + 1] += un[1];  aNorm_[i3 * 3 + 2] += un[2];
		}
		normalizeInPlace(grad_nor);
	}

	double h0 = (is_ext) ? 0.0001 : -0.0001;

	for (unsigned int i = 0; i < nXYZ * 3; i++)
	{
		pa_start[i] = paXYZ[i] + grad_nor[i] * h0;
	}


	for (unsigned int ixyz = 0; ixyz < nXYZ; ixyz++)
	{
		SDF::UTILS::getValueGradient(pa_start.data() + ixyz * 3, grad_nor.data() + ixyz * 3, 0.0001, nXYZ, paXYZ, nTri, paTri,
			topo, aNode);
	}

	normalizeInPlace(grad_nor);
	grad_old = grad_nor;

	double height;
	{
		aExtXYZ.resize(1);
		aExtXYZ[0].resize(nXYZ * 3, 0);
		double* pa_pos_new = aExtXYZ[0].data();

		for (unsigned int i = 0; i < nXYZ * 3; i++)
		{
			pa_pos_new[i] = pa_start[i];
		}

		for (unsigned int ixyz = 0; ixyz < nXYZ; ixyz++)
		{
			height = 0;
			//height += h0;
			double add_h = h0;
			while (fabs(height) < max_height)
			{
				double cos0 = 0;
				double cos1 = 0;
				while (cos0 < cos(3.1415*0.05) && cos1 < cos(3.1415*0.05))
				{
					SDF::UTILS::getValueGradient(pa_pos_new + ixyz * 3, grad_old.data() + ixyz * 3, height,
						nXYZ, paXYZ, nTri, paTri, topo, aNode);

					normalizeInPlace(grad_old);
					double direction[3];
					for (int iter_dir = 0; iter_dir < 3; iter_dir++)
					{
						direction[iter_dir] = (pa_pos_new[ixyz * 3 + iter_dir] + grad_old[ixyz * 3 + iter_dir] * fabs(add_h) - paXYZ[ixyz * 3 + iter_dir]);
					}
					double dot0 = Com::Dot3D(grad_old.data() + ixyz * 3, grad_nor.data() + ixyz * 3);
					double dot1 = Com::Dot3D(direction, grad_nor.data() + ixyz * 3);
					double len0 = Com::Length3D(grad_old.data() + ixyz * 3);
					double len1 = Com::Length3D(grad_nor.data() + ixyz * 3);
					double len2 = Com::Length3D(direction);
					if (len0 > 1 || len1 > 1 )
					{
						//LOG(INFO) << "len0: " << len0 << std::endl;
						//LOG(INFO) << "len1: " << len1 << std::endl;
						//LOG(INFO) << "len2: " << len2 << std::endl;
					}
					cos0 = dot0 / (len0*len1);
					cos1 = dot1 / (len2*len1);
					cos1 = 1;
					//在此处设置生长最远处的条件
					if (cos0 > cos(3.1415*0.05) && cos1 > cos(3.1415*0.05))
					{
						//height += add_h;
						for (int vector_iter = 0; vector_iter < 3; vector_iter++)
						{
							pa_pos_new[ixyz * 3 + vector_iter] = pa_pos_new[ixyz * 3 + vector_iter] + grad_old[ixyz * 3 + vector_iter] * fabs(add_h);
						}
						height += add_h;
						add_h = add_h * 2.0;
					}
					else
					{
						add_h = add_h / 8.0;
						if (fabs(add_h) < 1e-8)
						{
							break;
						}
					}
				}
				if (fabs(add_h) < 1e-8)
				{
					break;
				}
				if (fabs(height) > max_height)
				{
					break;
				}

			}
			//std::cout<<height<<std::endl;
		}


	}
	
}


void SHELLGEN::makeFurthestMesh(const MeshCompress& in, bool is_ext, double max_height, CGP::MeshCompress& res)
{
	std::vector<CNodeBH> aNode;
	int nXYZ = in.n_vertex_;
	int nTri = in.n_tri_;

	uintVec paTri = in.tri_;
	doubleVec paXYZ(in.n_vertex_ * 3, 0);
	for (int i = 0; i < in.n_vertex_; i++)
	{
		paXYZ[3 * i + 0] = in.pos_[i].x();
		paXYZ[3 * i + 1] = in.pos_[i].y();
		paXYZ[3 * i + 2] = in.pos_[i].z();
	}


	aNode.clear();
	aNode.resize(1);
	double cx, cy, cz, wx, wy, wz;

	cx = in.xyz_min_[0] * 0.5 + in.xyz_max_[0] * 0.5;
	cy = in.xyz_min_[1] * 0.5 + in.xyz_max_[1] * 0.5;
	cz = in.xyz_min_[2] * 0.5 + in.xyz_max_[2] * 0.5;

	wx = in.xyz_max_[0] - in.xyz_min_[0];
	wy = in.xyz_max_[1] - in.xyz_min_[1];
	wz = in.xyz_max_[2] - in.xyz_min_[2];


	aNode[0].cent_[0] = cx;
	aNode[0].cent_[1] = cy;
	aNode[0].cent_[2] = cz;
	double lmax = (wx > wy) ? wx : wy;
	lmax = (wz > lmax) ? wz : lmax;
	lmax *= 1.1;


	double shell_max = (wx > wy) ? wx : wy;
	double shell_min = (wx > wy) ? wy : wx;

	shell_max = (wz > shell_max) ? wz : shell_max;
	shell_min = (wz < shell_min) ? wz : shell_min;

	double shell_mid = wx + wy + wz - shell_min - shell_max;


	aNode[0].hw_ = lmax * 0.5;
	aNode[0].ichild_ = -1;
	for (unsigned int itri = 0; itri < in.n_tri_; itri++)
	{
		double fc[3] = { 0,0,0 };
		for (unsigned int ino = 0; ino < 3; ino++)
		{
			unsigned int ino0 = paTri[itri * 3 + ino];
			fc[0] += paXYZ[ino0 * 3 + 0];
			fc[1] += paXYZ[ino0 * 3 + 1];
			fc[2] += paXYZ[ino0 * 3 + 2];
		}
		fc[0] /= 3.0;
		fc[1] /= 3.0;
		fc[2] /= 3.0;
		SDF::UTILS::addFaceCenter(0, aNode, fc, itri, paTri.data(), paXYZ.data());
	}
	/*
	std::cout << " size node : " << aNode.size() << std::endl;
	for(unsigned int ino=0;ino<aNode.size();ino++){
		std::cout << ino << " " << aNode[ino].ichild_ << " " << aNode[ino].nTriCell << std::endl;
	}
	*/
	SDF::UTILS::setCentroidWeight(0, aNode, paTri.data(), paXYZ.data());
	

	doubleX2Vec aExtXYZ(1, paXYZ);
	doubleX2Vec aExtXYZ_Furthest_ex(0);
	doubleX2Vec aExtXYZ_Furthest_in(0);	

	CTriAryTopology tri_topo;

	//makeFurthestMesh(aExtXYZ_Furthest_ex, nXYZ, paXYZ.data(), nTri, paTri.data(), tri_topo, aNode, is_ext, max_height);
	makeFurthestMeshNormalize(aExtXYZ_Furthest_ex, nXYZ, paXYZ.data(), nTri, paTri.data(), tri_topo, aNode, is_ext, max_height);
	
	res = in;
	for (int i = 0; i < res.n_vertex_; i++)
	{
		res.pos_[i].x() = aExtXYZ_Furthest_ex[0][3 * i + 0];
		res.pos_[i].y() = aExtXYZ_Furthest_ex[0][3 * i + 1];
		res.pos_[i].z() = aExtXYZ_Furthest_ex[0][3 * i + 2];
	}
	//ex_in.saveObj("D:/dota210121/0125_cube/cube_shape_ex_test.obj");

	//makeFurthestMesh(aExtXYZ_Furthest_in, nXYZ, paXYZ.data(), nTri, paTri.data(), tri_topo, aNode, false, max_height);
}