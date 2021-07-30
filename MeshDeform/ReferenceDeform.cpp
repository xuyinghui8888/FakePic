#include "ReferenceDeform.h"
#include "DTUtilities.h"
using namespace CGP;
using namespace std;
using namespace Eigen;

int Grid::location(int3E pos) const 
{
	if (pos[0] < 0)
	{
		pos[0] = 0;
	}

	if (pos[0] > dense_[0] - 1)
	{
		pos[0] = dense_[0] - 1;
	}

	if (pos[1] < 0)
	{
		pos[1] = 0;
	}
	
	if (pos[1] > dense_[1] - 1)
	{
		pos[1] = dense_[1] - 1;
	}

	if (pos[2] < 0)
	{
		pos[2] = 0;
	}

	if (pos[2] > dense_[2] - 1)
	{
		pos[2] = dense_[2] - 1;
	}		

	return pos[0] + pos[1] * dense_[0] + pos[2] * dense_[0] * dense_[1];
}

void Grid::location(int3E& out, const int pos) const 
{
	out[2] = pos / (dense_[0] * dense_[1]);
	out[1] = (pos % (dense_[0] * dense_[1])) / dense_[0];
	out[0] = pos % dense_[0];
}

void Grid::coord(Vector3f& out, const int pos) const
{
	int3E p;
	location(p, pos);
	float3E interval = float3E(dim_[0] / (dense_[0] - 1), dim_[1] / (dense_[1] - 1), dim_[2] / (dense_[2] - 1));
	out = center_ - dim_ / 2 + float3E(p[0], p[1], p[2]).asDiagonal() * interval;
}

void Grid::getInterpWeight(float3E p, floatVec &w, intVec &pos) const 
{
	float3E tr_p = p - center_ + dim_ / 2;
	float3E interval = float3E(dim_[0] / (dense_[0] - 1), dim_[1] / (dense_[1] - 1), dim_[2] / (dense_[2] - 1));
	int3E basep = int3E(floorf(tr_p[0] / interval[0]), floorf(tr_p[1] / interval[1]), floorf(tr_p[2] / interval[2]));
	float3E base = float3E(basep[0], basep[1], basep[2]).asDiagonal() * interval;
	float3E weight = (tr_p - base).asDiagonal() * float3E(1 / interval[0], 1 / interval[1], 1 / interval[2]);
	
	float x = weight[0];
	float y = weight[1]; 
	float z = weight[2];
	
	float xx = 1 - weight[0]; 
	float yy = 1 - weight[1]; 
	float zz = 1 - weight[2];

	w.push_back(xx*yy*zz);
	pos.push_back(location(basep));
	w.push_back(x*yy*zz);
	pos.push_back(location(basep + int3E(1, 0, 0)));
	w.push_back(xx*y*zz);
	pos.push_back(location(basep + int3E(0, 1, 0)));
	w.push_back(xx*yy*z);
	pos.push_back(location(basep + int3E(0, 0, 1)));
	w.push_back(x*yy*z);
	pos.push_back(location(basep + int3E(1, 0, 1)));
	w.push_back(xx*y*z);
	pos.push_back(location(basep + int3E(0, 1, 1)));
	w.push_back(x*y*zz);
	pos.push_back(location(basep + int3E(1, 1, 0)));
	w.push_back(x*y*z);
	pos.push_back(location(basep + int3E(1, 1, 1)));
}

void Grid::getLaplaceParameter(Vector3f& out, int pos, intVec &p, floatVec &w) const 
{
	int3E cen;
	location(cen, pos);
	float3E weight = float3E(dim_[0] / (dense_[0] - 1), dim_[1] / (dense_[1] - 1), dim_[2] / (dense_[2] - 1));
	float3E& result = out;
	result = float3E(0, 0, 0);
	if (cen[0] == 0)
	{
		p.push_back(location(cen + int3E(1, 0, 0)));
		w.push_back(-1.f), result[0] -= weight[0];
	}
	else if (cen[0] == dense_[0] - 1)
	{
		p.push_back(location(cen + int3E(-1, 0, 0)));
		w.push_back(-1.f), result[0] += weight[0];
	}
	else
	{
		p.push_back(location(cen + int3E(1, 0, 0)));
		p.push_back(location(cen + int3E(-1, 0, 0)));
		w.push_back(-1.f), w.push_back(-1.f);
	}

	if (cen[1] == 0)
	{
		p.push_back(location(cen + int3E(0, 1, 0)));
		w.push_back(-1.f), result[1] -= weight[1];
	}
	else if (cen[1] == dense_[1] - 1)
	{
		p.push_back(location(cen + int3E(0, -1, 0)));
		w.push_back(-1.f), result[1] += weight[1];
	}
	else
	{
		p.push_back(location(cen + int3E(0, 1, 0)));
		p.push_back(location(cen + int3E(0, -1, 0)));
		w.push_back(-1.f), w.push_back(-1.f);
	}		

	if (cen[2] == 0)
	{
		p.push_back(location(cen + int3E(0, 0, 1)));
		w.push_back(-1.f), result[2] -= weight[2];
	}
	else if (cen[2] == dense_[2] - 1)
	{
		p.push_back(location(cen + int3E(0, 0, -1)));
		w.push_back(-1.f), result[2] += weight[2];
	}
	else
	{
		p.push_back(location(cen + int3E(0, 0, 1)));
		p.push_back(location(cen + int3E(0, 0, -1)));
		w.push_back(-1.f), w.push_back(-1.f);
	}

	p.push_back(pos);
	w.push_back(float(p.size() - 1));
}

void ReferenceDeform::init(const Eigen::Vector3f * sourceMesh1, const int mesh1VertSize, 
	const RefDefConfig & config)
{
	setDensity(Eigen::Vector3i(config.density[0], config.density[1], config.density[2]));
	setSmoothWeight(config.smooth_weight);
	init(sourceMesh1, mesh1VertSize,
		Eigen::Vector2f(config.bb_x_expand[0], config.bb_x_expand[1]),
		Eigen::Vector2f(config.bb_y_expand[0], config.bb_y_expand[1]),
		Eigen::Vector2f(config.bb_z_expand[0], config.bb_z_expand[1]));
}

void ReferenceDeform::init(const float3E * sourceMesh1, const int mesh1VertSize,
	const float2E& xScale, const float2E& yScale, const float2E& zScale)
{
	ref_A_ = float3Vec(sourceMesh1, sourceMesh1 + mesh1VertSize);
	ref_A_n_vertex_ = mesh1VertSize;
	computeBoundingBoxandScale(xScale, yScale, zScale);

	int gridSize = grid_.dense_[0] * grid_.dense_[1] * grid_.dense_[2];

	vector<Eigen::Triplet<float>> AtripletList;

	for (int i = 0; i < ref_A_n_vertex_; i++) 
	{
		vector<float> w;
		vector<int> pos;
		grid_.getInterpWeight(ref_A_[i], w, pos);
		for (int j = 0; j < w.size(); j++)
		{
			AtripletList.push_back(Eigen::Triplet<float>(i, pos[j], w[j]));
		}
	}
	grid_lap_.resize(gridSize);
	for (int i = 0; i < gridSize; i++) 
	{
		vector<int> pos;
		vector<float> w;
		grid_.getLaplaceParameter(grid_lap_[i], i, pos, w);
		for (int j = 0; j < pos.size(); j++) 
		{
			AtripletList.push_back(Eigen::Triplet<float>(i + ref_A_n_vertex_, pos[j], smooth_weight_*w[j]));
		}
	}

	//LDLT
	VectorXf x[3];
	sp_.resize(ref_A_n_vertex_ + gridSize, gridSize);
	sp_.setFromTriplets(AtripletList.begin(), AtripletList.end());
	solver_.compute(sp_.transpose()*sp_);
}

void ReferenceDeform::setDensity(const int3E& density)
{
	grid_.dense_ = density;
}

void ReferenceDeform::setSmoothWeight(float w)
{
	smooth_weight_ = w;
}

//计算包围盒, 放大scale, relativeLocation为mesh在包围盒的相对位置
void ReferenceDeform::computeBoundingBoxandScale(const float2E& xScale, const float2E& yScale, 
	const float2E& zScale)
{
	float3E m_min, m_max;// , r1min, r1max;
	m_min = ref_A_[0];
	m_max = ref_A_[0];
	for (int i = 1; i < ref_A_n_vertex_; i++) 
	{
		for (int j = 0; j < 3; j++)
		{
			if (m_min[j] > ref_A_[i][j])
				m_min[j] = ref_A_[i][j];
			if (m_max[j] < ref_A_[i][j])
				m_max[j] = ref_A_[i][j];
		}
	}
	/*r1min = refmesh1[0];
	r1max = refmesh1[0];
	for (int i = 1; i < m1size; i++) {
		for (int j = 0; j < 3; j++) {
			if (r1min[j] > refmesh1[i][j])
				r1min[j] = refmesh1[i][j];
			if (r1max[j] < refmesh1[i][j])
				r1max[j] = refmesh1[i][j];
		}
	}
	float3 s1center = (s1min + s1max) / 2;
	float3 s1dim = (s1max - s1min);
	float3 r1center = (r1min + r1max) / 2;
	float3 r1dim = (r1max - r1min);
	float3 scaled = float3(r1dim[0] / s1dim[0], r1dim[1] / s1dim[1], r1dim[2] / s1dim[2]);
	sourmesh1[0] = (sourmesh1[0] - s1center).asDiagonal()*scaled + r1center;
	float3 smin, smax;
	smin = sourmesh1[0];
	smax = sourmesh1[0];
	for (int i = 1; i < m1size; i++) {
		sourmesh1[i] = (sourmesh1[i] - s1center).asDiagonal()*scaled + r1center;
		for (int j = 0; j < 3; j++) {
			if (smin[j] > sourmesh1[i][j])
				smin[j] = sourmesh1[i][j];
			if (smax[j] < sourmesh1[i][j])
				smax[j] = sourmesh1[i][j];
		}
	}
	for (int i = 0; i < m2size; i++) {
		sourmesh2[i] = (sourmesh2[i] - s1center).asDiagonal()*scaled + r1center;
		for (int j = 0; j < 3; j++) {
			if (smin[j] > sourmesh2[i][j])
				smin[j] = sourmesh2[i][j];
			if (smax[j] < sourmesh2[i][j])
				smax[j] = sourmesh2[i][j];
		}
	}*/

	float3E m_center = (m_min + m_max) / 2.0f;
	float3E m_dim = m_max - m_min;
	float3E lowScale(xScale[0], yScale[0], zScale[0]);
	float3E highScale(xScale[1], yScale[1], zScale[1]);
	grid_.dim_ = m_dim.asDiagonal()*(lowScale + highScale) / 2.0f;
	float3E g_min = m_center - lowScale.asDiagonal()*m_dim / 2.0f;
	float3E g_max = m_center + highScale.asDiagonal()*m_dim / 2.0f;
	grid_.center_ = (g_min + g_max) / 2.0f;
}

void ReferenceDeform::process(Eigen::Vector3f* out, const Eigen::Vector3f* refMesh1, const Eigen::Vector3f* sourceMesh2, const int m2size) const
{
	int gridSize = grid_.dense_[0] * grid_.dense_[1] * grid_.dense_[2];
	VectorXf b[3], x0[3];
	for (int i = 0; i < 3; i++)
	{
		b[i].resize(ref_A_n_vertex_ + gridSize);
		x0[i].resize(gridSize);
	}

	for (int i = 0; i < ref_A_n_vertex_; i++) 
	{
		for (int j = 0; j < 3; j++)
		{
			b[j][i] = refMesh1[i][j];
		}
	}

	for (int i = 0; i < gridSize; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			b[j][i + ref_A_n_vertex_] = smooth_weight_ * grid_lap_[i][j];
		}
	}
	VectorXf x[3];
	for (int i = 0; i < 3; i++)
	{
		x[i] = solver_.solve(sp_.transpose() * b[i]);
	}

	// mapping
	vector<Eigen::Triplet<float>> A2tripletList;
	A2tripletList.reserve(m2size);
	for (int i = 0; i < m2size; i++) 
	{
		vector<float> w;
		vector<int> pos;
		grid_.getInterpWeight(sourceMesh2[i], w, pos);
		for (int j = 0; j < w.size(); j++)
		{
			A2tripletList.push_back(Eigen::Triplet<float>(i, pos[j], w[j]));
		}
	}

	VectorXf refb[3];
	SparseMatrix<float> A2(m2size, gridSize);
	A2.setFromTriplets(A2tripletList.begin(), A2tripletList.end());

	float3E tmp;
	for (int i = 0; i < gridSize; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			grid_.coord(tmp, i);
			x0[j][i] = tmp[j];
		}
	}

	for (int i = 0; i < 3; i++)
	{
		refb[i] = A2 * (x[i] - x0[i]);
	}

	//refmesh2 = new float3[m2size];
	for (int i = 0; i < m2size; i++) 
	{
		for (int j = 0; j < 3; j++) 
		{
			out[i][j] = refb[j][i] + sourceMesh2[i][j];
		}
	}
}