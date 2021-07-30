#include "ComVector3d.h"
#include "TriAnyTopology.h"
#include "SurfaceMeshReader.h"
#include "SpatialHashGrid3d.h"
#include "CNodeBH.h"
#include "MeshDistanceUtils.h"
#include "Util.h"

using namespace SDF;
using namespace UTILS;


void UTILS::getVolumes(double v[4], const double p[3], const double p0[3], const double p1[3], const double p2[3], const double p3[3])
{
	v[0] = Com::TetVolume3D(p, p1,p2,p3);
	v[1] = Com::TetVolume3D(p0,p ,p2,p3);
	v[2] = Com::TetVolume3D(p0,p1,p ,p3);
	v[3] = Com::TetVolume3D(p0,p1,p2,p );
}

bool UTILS::isIncludePrism(const double p[3], const double p0_0[3], const double p0_1[3], const double p0_2[3], const double p1_0[3], 
	const double p1_1[3], const double p1_2[3], const unsigned int ind[3])
{
	assert(ind[0] != ind[1] && ind[1] != ind[2] && ind[2] != ind[0]);

	if (ind[0] > ind[1] && ind[0] > ind[2])
	{
		if (isInsideTet3D(p, p0_0, p0_1, p0_2, p1_0)) return true;
		if (ind[1] > ind[2]) {
			if (isInsideTet3D(p, p0_2, p0_1, p1_1, p1_0)) return true;
			if (isInsideTet3D(p, p1_1, p1_2, p0_2, p1_0)) return true;
		}
		else 
		{
			if (isInsideTet3D(p, p0_2, p0_1, p1_2, p1_0)) return true;
			if (isInsideTet3D(p, p1_1, p1_2, p0_1, p1_0)) return true;
		}
	}
	else if (ind[1] > ind[0] && ind[1] > ind[2]) 
	{
		if (isInsideTet3D(p, p0_1, p0_2, p0_0, p1_1)) 
			return true;
		if (ind[2] > ind[0]) 
		{
			if (isInsideTet3D(p, p0_0, p0_2, p1_2, p1_1)) return true;
			if (isInsideTet3D(p, p1_2, p1_0, p0_0, p1_1)) return true;
		}
		else 
		{
			if (isInsideTet3D(p, p0_0, p0_2, p1_0, p1_1)) return true;
			if (isInsideTet3D(p, p1_2, p1_0, p0_2, p1_1)) return true;
		}
	}
	else {
		if (isInsideTet3D(p, p0_2, p0_0, p0_1, p1_2)) 
			return true;
		if (ind[0] > ind[1]) 
		{
			if (isInsideTet3D(p, p0_1, p0_0, p1_0, p1_2)) 
				return true;
			if (isInsideTet3D(p, p1_0, p1_1, p0_1, p1_2)) 
				return true;
		}
		else 
		{
			if (isInsideTet3D(p, p0_1, p0_0, p1_1, p1_2)) 
				return true;
			if (isInsideTet3D(p, p1_0, p1_1, p0_0, p1_2))
				return true;
		}
	}
	return false;
}

bool UTILS::isInsideTet3D(const double p[3], const double p0[3], const double p1[3], const double p2[3], const double p3[3])
{
	double v[4];
	getVolumes(v,p,p0,p1,p2,p3);
	double tv = v[0]+v[1]+v[2]+v[3];	
	double v_eps = 1.0e-3*tv;
	if( v[0] > -v_eps && v[1] > -v_eps && v[2] > -v_eps && v[3] > -v_eps ) 
		return true;
	return false;
}

double UTILS::volumePrism(const double p0_0[3], const double p0_1[3], const double p0_2[3], const double p1_0[3], 
	const double p1_1[3], const double p1_2[3], const unsigned int ind[3])
{
	assert( ind[0] != ind[1] && ind[1] != ind[2] && ind[2] != ind[0] );
	double v0,v1,v2;
	if(ind[0] > ind[1] && ind[0] > ind[2] )
	{
		v0 = Com::TetVolume3D(p0_0,p0_1,p0_2,p1_0);
		if( ind[1] > ind[2] )
		{
			v1 = Com::TetVolume3D(p0_2,p0_1,p1_1,p1_0);
			v2 = Com::TetVolume3D(p1_1,p1_2,p0_2,p1_0);
		}
		else
		{
			v1 = Com::TetVolume3D(p0_2,p0_1,p1_2,p1_0);
			v2 = Com::TetVolume3D(p1_1,p1_2,p0_1,p1_0);
		}
	}
	else if(ind[1] > ind[0] && ind[1] > ind[2])
	{
		v0 = Com::TetVolume3D(p0_1,p0_2,p0_0,p1_1);
		if( ind[2] > ind[0] )
		{
			v1 = Com::TetVolume3D(p0_0,p0_2,p1_2,p1_1);
			v2 = Com::TetVolume3D(p1_2,p1_0,p0_0,p1_1);
		}
		else
		{
			v1 = Com::TetVolume3D(p0_0,p0_2,p1_0,p1_1);
			v2 = Com::TetVolume3D(p1_2,p1_0,p0_2,p1_1);
		}
	}
	else
	{
		v0 = Com::TetVolume3D(p0_2,p0_0,p0_1,p1_2);
		if( ind[0] > ind[1] )
		{
			v1 = Com::TetVolume3D(p0_1,p0_0,p1_0,p1_2);
			v2 = Com::TetVolume3D(p1_0,p1_1,p0_1,p1_2);
		}
		else{
			v1 = Com::TetVolume3D(p0_1,p0_0,p1_1,p1_2);
			v2 = Com::TetVolume3D(p1_0,p1_1,p0_0,p1_2);
		}
	}
	return v0+v1+v2;
}

double UTILS::minVolumePrism(const double p0_0[3], const double p0_1[3], const double p0_2[3], const double p1_0[3], 
	const double p1_1[3], const double p1_2[3], const unsigned int ind[3])
{
	//assert( ind[0] != ind[1] && ind[1] != ind[2] && ind[2] != ind[0] );
	double v0,v1,v2;
	if(ind[0] > ind[1] && ind[0] > ind[2])
	{
		v0 = Com::TetVolume3D(p0_0,p0_1,p0_2,p1_0);
		if(ind[1] > ind[2])
		{
			v1 = Com::TetVolume3D(p0_2,p0_1,p1_1,p1_0);
			v2 = Com::TetVolume3D(p1_1,p1_2,p0_2,p1_0);
		}
		else
		{
			v1 = Com::TetVolume3D(p0_2,p0_1,p1_2,p1_0);
			v2 = Com::TetVolume3D(p1_1,p1_2,p0_1,p1_0);
		}
	}
	else if( ind[1] > ind[0] && ind[1] > ind[2] )
	{
		v0 = Com::TetVolume3D(p0_1,p0_2,p0_0,p1_1);
		if( ind[2] > ind[0] )
		{
			v1 = Com::TetVolume3D(p0_0,p0_2,p1_2,p1_1);
			v2 = Com::TetVolume3D(p1_2,p1_0,p0_0,p1_1);
		}
		else
		{
			v1 = Com::TetVolume3D(p0_0,p0_2,p1_0,p1_1);
			v2 = Com::TetVolume3D(p1_2,p1_0,p0_2,p1_1);
		}
	}
	else
	{
		v0 = Com::TetVolume3D(p0_2,p0_0,p0_1,p1_2);
		if( ind[0] > ind[1] )
		{
			v1 = Com::TetVolume3D(p0_1,p0_0,p1_0,p1_2);
			v2 = Com::TetVolume3D(p1_0,p1_1,p0_1,p1_2);
		}
		else{
			v1 = Com::TetVolume3D(p0_1,p0_0,p1_1,p1_2);
			v2 = Com::TetVolume3D(p1_0,p1_1,p0_0,p1_2);
		}
	}
	//  std::cout << " " << v0 << " " << v1 << " " << v2 << std::endl;
	double min_vol = (v0 < v1) ? v0 : v1;
	min_vol = ( v2 < min_vol ) ? v2 : min_vol;
	//reversion return abs value instead
	return fabs(min_vol);
	return min_vol;
}

void UTILS::findRangeCubic(int& icnt, double& r0, double& r1, double v0, double v1, double k0, double k1, double k2, double k3)
{
	icnt--;
	if( icnt <= 0 ) 
		return;
	double r2 = 0.5*(r0+r1);
	double v2 = k0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
	if( v0*v2 < 0 )
	{
		r1 = r2;
	}
	else
	{
		r0 = r2;
	}
	findRangeCubic(icnt,r0,r1,v0,v2,k0,k1,k2,k3);
}

void UTILS::getDistNormal(double& dist, double n[3], const double p[3], const double p0[3], const double p1[3], const double p2[3])
{
	const Com::CVector3D ba0(p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]);
	const Com::CVector3D ca0(p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]);
	Com::CVector3D n0 = Com::Cross(ba0,ca0);
	double area = n0.Length()*0.5;
	n0 *= 1.0/area;
	n[0] = n0.x;
	n[1] = n0.y;
	n[2] = n0.z;
	////
	double vol = Com::TetVolume3D(p0,p1,p2,p);
	dist = vol*3/area;
}

void UTILS::getNormalInsidePrism(double n[3], double t, const double p0_0[3], const double p0_1[3], const double p0_2[3],
  const double p1_0[3], const double p1_1[3], const double p1_2[3])
{
	/*
	const Com::CVector3D ba0(p0_1[0]-p0_0[0], p0_1[1]-p0_0[1], p0_1[2]-p0_0[2]);
	const Com::CVector3D ca0(p0_2[0]-p0_0[0], p0_2[1]-p0_0[1], p0_2[2]-p0_0[2]);
	const Com::CVector3D ba1(p1_1[0]-p1_0[0], p1_1[1]-p1_0[1], p1_1[2]-p1_0[2]);
	const Com::CVector3D ca1(p1_2[0]-p1_0[0], p1_2[1]-p1_0[1], p1_2[2]-p1_0[2]);
	const Com::CVector3D xt = (1-t)*ba0 + t*(ba1-ba0);
	const Com::CVector3D yt = (1-t)*ca0 + t*(ca1-ca0);
	Com::CVector3D nt = Com::Cross(xt,yt);
	nt.SetNormalizedVector();
	n[0] = nt.x;
	n[1] = nt.y;
	n[2] = nt.z;
	*/
	const Com::CVector3D ba0(p0_1[0]-p0_0[0], p0_1[1]-p0_0[1], p0_1[2]-p0_0[2]);
	const Com::CVector3D ca0(p0_2[0]-p0_0[0], p0_2[1]-p0_0[1], p0_2[2]-p0_0[2]);
	Com::CVector3D n0 = Com::Cross(ba0,ca0);
	//n0.SetNormalizedVector();
	n0.Normalize();
	n[0] = n0.x;
	n[1] = n0.y;
	n[2] = n0.z;
}

void UTILS::getAppDistInsidePrism(double& dist, double t, const double p[3], const double p0_0[3], const double p0_1[3],
	const double p0_2[3], const double p1_0[3], const double p1_1[3], const double p1_2[3])
{
	/*
	const Com::CVector3D ba0(p0_1[0]-p0_0[0], p0_1[1]-p0_0[1], p0_1[2]-p0_0[2]);
	const Com::CVector3D ca0(p0_2[0]-p0_0[0], p0_2[1]-p0_0[1], p0_2[2]-p0_0[2]);
	const Com::CVector3D ba1(p1_1[0]-p1_0[0], p1_1[1]-p1_0[1], p1_1[2]-p1_0[2]);
	const Com::CVector3D ca1(p1_2[0]-p1_0[0], p1_2[1]-p1_0[1], p1_2[2]-p1_0[2]);
	const Com::CVector3D xt = (1-t)*ba0 + t*(ba1-ba0);
	const Com::CVector3D yt = (1-t)*ca0 + t*(ca1-ca0);
	Com::CVector3D nt = Com::Cross(xt,yt);
	nt.SetNormalizedVector();
	Com::CVector3D n0 = Com::Cross(ba0,ca0);
	n0.SetNormalizedVector();
	const Com::CVector3D at = t*(Com::CVector3D(p1_0[0],p1_0[1],p1_0[2]) - Com::CVector3D(p0_0[0],p0_0[1],p0_0[2]));
	const Com::CVector3D bt = t*(Com::CVector3D(p1_1[0],p1_1[1],p1_1[2]) - Com::CVector3D(p0_1[0],p0_1[1],p0_1[2]));
	const Com::CVector3D ct = t*(Com::CVector3D(p1_2[0],p1_2[1],p1_2[2]) - Com::CVector3D(p0_2[0],p0_2[1],p0_2[2]));
	dist = 0;
	//  dist += Dot(at,n0+nt);
	//  dist += Dot(bt,n0+nt);
	//  dist += Dot(ct,n0+nt);
	//  dist /= 6.0;
	dist += Dot(at,n0);
	dist += Dot(bt,n0);
	dist += Dot(ct,n0);
	dist /= 3.0;  
	*/
	double vol = Com::TetVolume3D(p0_0,p0_1,p0_2,p);
	double area = Com::TriArea3D(p0_0,p0_1,p0_2);
	dist = vol*3/area;
}

double UTILS::getTotalVolumePrisms(unsigned int nTri, const unsigned int* paTri, unsigned int nXYZ, const double* paXYZ0,
  const double* paXYZ1)
{
	double tot_vol = 0;
	for(unsigned int itri=0;itri<nTri;itri++)
	{
		unsigned int i0 = paTri[itri*3+0];
		unsigned int i1 = paTri[itri*3+1];
		unsigned int i2 = paTri[itri*3+2];
		tot_vol += volumePrism(paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3, 
			paTri+itri*3);
	}
	return tot_vol;
}

double UTILS::getInterpCoeffInsideProsim(const double p[3], const double p0_0[3], const double p0_1[3], const double p0_2[3],
  const double p1_0[3], const double p1_1[3], const double p1_2[3])
{
	const Com::CVector3D ba0(p0_1[0]-p0_0[0], p0_1[1]-p0_0[1], p0_1[2]-p0_0[2]);
	const Com::CVector3D ca0(p0_2[0]-p0_0[0], p0_2[1]-p0_0[1], p0_2[2]-p0_0[2]);
	const Com::CVector3D ba1(p1_1[0]-p1_0[0], p1_1[1]-p1_0[1], p1_1[2]-p1_0[2]);
	const Com::CVector3D ca1(p1_2[0]-p1_0[0], p1_2[1]-p1_0[1], p1_2[2]-p1_0[2]);
	const Com::CVector3D x0 = ba0;
	const Com::CVector3D x1 = ba1-ba0;
	const Com::CVector3D y0 = ca0;
	const Com::CVector3D y1 = ca1-ca0;
	const Com::CVector3D n0 = Com::Cross(x0,y0);
	const Com::CVector3D n1 = Com::Cross(x1,y0) + Com::Cross(x0,y1);
	const Com::CVector3D n2 = Com::Cross(x1,y1);
	const Com::CVector3D q0 = Com::CVector3D(p0_0[0]-p[0],    p0_0[1]-p[1],    p0_0[2]-p[2]   );
	const Com::CVector3D q1 = Com::CVector3D(p1_0[0]-p0_0[0], p1_0[1]-p0_0[1], p1_0[2]-p0_0[2]);
	const double k0 = Com::Dot(n0,q0);
	const double k1 = Com::Dot(n0,q1) + Com::Dot(n1,q0);
	const double k2 = Com::Dot(n1,q1) + Com::Dot(n2,q0);
	const double k3 = Com::Dot(n2,q1);
	double r0=-0.01;
	double r1=+1.01;
	double v0 = k0 + k1*r0 + k2*r0*r0 + k3*r0*r0*r0;
	double v1 = k0 + k1*r1 + k2*r1*r1 + k3*r1*r1*r1;
	if( v0*v1 > 0 )
	{
		std::cout << "error!--> " << v0 << " " << v1 << std::endl;
		return 0;
	}
	int icnt=8;
	findRangeCubic(icnt,r0,r1,v0,v1,k0,k1,k2,k3);
	double rm = 0.5*(r0+r1);
	double vm = k0 + k1*rm + k2*rm*rm + k3*rm*rm*rm;
	//  std::cout << v0 << " " << v1 << " " << vm << std::endl;
	return rm;  
}

bool UTILS::IsInsidePrisms(unsigned int& itri_in, const double p[3], unsigned int nTri, const unsigned int* paTri, unsigned int nXYZ,
  const double* paXYZ0, const double* paXYZ1)
{
	double tot_vol = 0;

	for(unsigned int itri=0;itri<nTri;itri++)
	{
		unsigned int i0 = paTri[itri*3+0];
		unsigned int i1 = paTri[itri*3+1];
		unsigned int i2 = paTri[itri*3+2];
		bool res = isIncludePrism(p, paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3, paTri+itri*3);
		if(res)
		{
			itri_in = itri;
			return true;
			/*
			double t = GetInterpCoeffInsideProsim(p,     
			paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, 
			paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3);
			GetNormalInsidePrism(n,t,
			paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, 
			paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3);
			GetAppDistInsidePrism(dist,t,p,
			paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, 
			paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3);
			return true;
			*/
		}
	}

	return false;
}

bool UTILS::isInsidePrismsHash(unsigned int& itri_in, const double p[3], const CSpatialHashGrid3D& hash, unsigned int nTri,
  const unsigned int* paTri, unsigned int nXYZ, const double* paXYZ0, const double* paXYZ1)
{
	std::vector<unsigned int> cand; 
	hash.GetCellCand(p,cand);
	for(unsigned int icand=0;icand<cand.size();icand++)
	{
		unsigned int itri = cand[icand];
		unsigned int i0 = paTri[itri*3+0];
		unsigned int i1 = paTri[itri*3+1];
		unsigned int i2 = paTri[itri*3+2];
		bool res = isIncludePrism(p, paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3, paTri+itri*3);
		if(res)
		{
			itri_in = itri;
			return true;
		}
	}
	return false;
}

double UTILS::getMinVolumePrisms(unsigned int nTri, const unsigned int* paTri, unsigned int nXYZ, const double* paXYZ0,
  const double* paXYZ1)
{
	double min_vol = 0;
	for(unsigned int itri=0;itri<nTri;itri++)
	{
		unsigned int i0 = paTri[itri*3+0];
		unsigned int i1 = paTri[itri*3+1];
		unsigned int i2 = paTri[itri*3+2];
		double vol0 = minVolumePrism(paXYZ0+i0*3, paXYZ0+i1*3, paXYZ0+i2*3, paXYZ1+i0*3, paXYZ1+i1*3, paXYZ1+i2*3, paTri+itri*3);
		if( vol0 < 0 )
		{ 
			std::cout << " itri: " << itri << "    min vol : " << vol0 << std::endl;
		}
		if( itri == 0 || vol0 < min_vol )
		{
			min_vol = vol0;
		} 
	}
	return min_vol;
}

void UTILS::getBarnHutAppAreaDistGrad(unsigned int iNo0, const std::vector<CNodeBH>& aNode, const double p[3], const unsigned int* paTri,
  const double* paXYZ,  double height, double& tot_area, double& app_dist, double app_grad[3])
{
	const CNodeBH& no0 = aNode[iNo0];
	{
		double d0 = Com::Distance3D(no0.centroid_,p);
		double hw0 = no0.hw_;
		if( no0.weight_ < 1.0e-20 )
		{
			return;
		}
		/*
		//    std::cout << d0 << " " << height << "  " << iNo0 << " " << no0.centroid_[0] << " " << no0.centroid_[1] << " " << no0.centroid_[2] << std::endl;
		if( d0 > hw0*4 ){
			if( p[0] > no0.cent_[0]-no0.hw_ && p[0] < no0.cent_[0]+no0.hw_ &&      
				p[1] > no0.cent_[1]-no0.hw_ && p[1] < no0.cent_[1]+no0.hw_ &&
				p[2] > no0.cent_[2]-no0.hw_ && p[2] < no0.cent_[2]+no0.hw_ ){}
			else{
			double dA0 = no0.weight_;
			tot_area += dA0;
			app_dist += dA0/(d0*d0*d0);
			double tmp0 = dA0/(d0*d0*d0*d0*d0);
			app_grad[0] += tmp0*(p[0] - no0.centroid_[0]);
			app_grad[1] += tmp0*(p[1] - no0.centroid_[1]);       
			app_grad[2] += tmp0*(p[2] - no0.centroid_[2]);
			return;
			}
		}
		*/
		if( d0 > hw0*4 )
		{
		//      std::cout << iNo0 << " " << d0 << " " << hw0 << "     " << no0.centroid_[0] << " " << no0.centroid_[1] << " " << no0.centroid_[2] << std::endl;
			double dA0 = no0.weight_;
			tot_area += dA0;
			app_dist += dA0/(d0*d0*d0);
			double tmp0 = dA0/(d0*d0*d0*d0*d0);
		//      std::cout << tmp0 << " " << d0 << " " << no0.centroid_[0] << " " << no0.centroid_[1] << " " << no0.centroid_[2] << std::endl;
			app_grad[0] += tmp0*(p[0] - no0.centroid_[0]);
			app_grad[1] += tmp0*(p[1] - no0.centroid_[1]);       
			app_grad[2] += tmp0*(p[2] - no0.centroid_[2]);
			return;
		}    
	}
	if( no0.ichild_ == -1 )
	{  // no child    
		for(unsigned int itricell=0;itricell<no0.n_tri_cell_;itricell++)
		{
			double c[3] = {0,0,0};
			unsigned int itri0 = no0.aIndTriCell[itricell];
		//      std::cout << itricell << " " << itri0 << std::endl;
			unsigned int i0 = paTri[itri0*3+0];
			unsigned int i1 = paTri[itri0*3+1];
			unsigned int i2 = paTri[itri0*3+2];
			double der[3], val;
			jie::computeTriangle(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3,der,val);
			app_dist += val;
			app_grad[0] += der[0];
			app_grad[1] += der[1];
			app_grad[2] += der[2];
			{
				double v = Com::TetVolume3D(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
				double a = Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
			//        double c0 = Com::Dot3D(paXYZ+i1*3,paXYZ+i2*3)/(Com::Length3D(paXYZ+i1*3)*Com::Length3D(paXYZ+i2*3));
			//        double c1 = Com::Dot3D(paXYZ+i2*3,paXYZ+i0*3)/(Com::Length3D(paXYZ+i2*3)*Com::Length3D(paXYZ+i0*3));
			//        double c2 = Com::Dot3D(paXYZ+i0*3,paXYZ+i1*3)/(Com::Length3D(paXYZ+i0*3)*Com::Length3D(paXYZ+i1*3));
				double len = app_grad[0]*app_grad[0] + app_grad[1]*app_grad[1] + app_grad[2]*app_grad[2];
				len = sqrt(len);
			//        if( len > 1000 ){  
			//        std::cout << "  " << iNo0 << " " << len << " " << v*3/a << " " << " " << c0 << " " << c1 << " " << c2 << std::endl;
			//        }
			}
			tot_area += Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
		}
		return;
	}

	unsigned int ich0 = no0.ichild_;
	for(unsigned int i=0;i<8;i++)
	{
		getBarnHutAppAreaDistGrad(ich0+i,aNode,p, paTri,paXYZ,height,tot_area,app_dist,app_grad);
	}  
}

void UTILS::getValueGradient(const double p[], double grad[], double height, unsigned int nXYZ, const double* paXYZ,
  unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode)
{
	double app_dist = 0;
	double app_grad[3] = {0,0,0};
	double tot_area = 0;
	getBarnHutAppAreaDistGrad(0,aNode,p, paTri,paXYZ,height, tot_area, app_dist, app_grad);    
	////
	/*
	for(unsigned int itri=0;itri<nTri;itri++){
	unsigned int i0 = paTri[itri*3+0];
	unsigned int i1 = paTri[itri*3+1];
	unsigned int i2 = paTri[itri*3+2];
	double der[3], val;
	jie::computeTriangle(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3,der,val);
	app_dist += val;
	app_grad[0] += der[0];
	app_grad[1] += der[1];
	app_grad[2] += der[2];
	tot_area += Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
	}
	*/
	//  std::cout << "tot area : " << tot_area << std::endl;
	/////
	if( app_dist < 1.0e-10 )
	{
		grad[0] = 0;
		grad[1] = 0;
		grad[2] = 0;
		return;
	}
	double tmp0 = tot_area/app_dist;
	double tmp1 = pow(tmp0,1.0/3.0);  // distance
	double tmp2 = -tmp1/(3*app_dist);
	tmp2 *= tmp1; // ?
	grad[0] = tmp2*app_grad[0];
	grad[1] = tmp2*app_grad[1];       
	grad[2] = tmp2*app_grad[2];
	//std::cout << "tot area : " << tot_area << std::endl;
}


void UTILS::getValueGradient(const double p[], double grad[], double height, unsigned int nXYZ, const double* paXYZ,
  unsigned int nTri,  const unsigned int* paTri, const CTriAryTopology& topo)
{
	double app_A = 0, app_dist = 0, app_grad[3] = {0,0,0};
	const double cut_off_radius = height*3;
	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
	{
		double d = Com::Distance3D(p,paXYZ+ixyz*3);
		if( d > cut_off_radius ) 
			continue;
		double dA = 0;
		for(CTriAryTopology::CItr itr=topo.GetItrElSuP(ixyz);!itr.IsEnd();itr++)
		{
			unsigned int itri0 = *itr;      
			unsigned int i1 = paTri[itri0*3+0];
			unsigned int i2 = paTri[itri0*3+1];
			unsigned int i3 = paTri[itri0*3+2];
			double un[3], area;    
			Com::UnitNormalAreaTri3D(un,area, paXYZ+i1*3, paXYZ+i2*3, paXYZ+i3*3);  
			dA += area/3.0;
		}
		app_A += dA;
		app_dist += dA/(d*d*d);
		double tmp0 = dA/(d*d*d*d*d);
		app_grad[0] += tmp0*(p[0] - paXYZ[ixyz*3+0]);
		app_grad[1] += tmp0*(p[1] - paXYZ[ixyz*3+1]);       
		app_grad[2] += tmp0*(p[2] - paXYZ[ixyz*3+2]);
	}

	{
		double tmp0 = app_A/app_dist;
		double tmp1 = pow(tmp0,1.0/3.0);  // distance
		double tmp2 = tmp1*tmp1*tmp1*tmp1/app_A;
		grad[0] = tmp2*app_grad[0];
		grad[1] = tmp2*app_grad[1];       
		grad[2] = tmp2*app_grad[2];
	}
}

void UTILS::getNewPosGrad(double* pa_pos_new, double* pa_grad_new, double& dh, const double* pa_pos_old, const double* pa_grad_old,
  double height, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo,
  const std::vector<CNodeBH>& aNode)
{
	dh *= 2;
	for(unsigned int itr = 0; itr < 1; itr++)
	{
		dh *= 0.5;
	for(unsigned int i=0;i<nXYZ*3;i++)
	{		
		pa_pos_new[i] = pa_pos_old[i] + pa_grad_old[i]*dh;
	}
	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
	{
		getValueGradient(pa_pos_new+ixyz*3, pa_grad_new+ixyz*3, height, nXYZ,paXYZ,nTri,paTri, topo, aNode);
	}    

	double min_cos;

	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
	{
		double dot0 = Com::Dot3D(pa_grad_new+ixyz*3,  pa_grad_old+ixyz*3);
		double len0 = Com::Length3D(pa_grad_new+ixyz*3);
		double len1 = Com::Length3D(pa_grad_old+ixyz*3);
		const double cos0 = dot0/(len0*len1);
		if( ixyz == 0 || cos0 < min_cos )
		{
			min_cos = cos0;
			if(cos0<0)
			{
				for(int inverse_i = 0; inverse_i<3; inverse_i++)
				{
					double inverse_flag = 1;
					if(pa_grad_old[ixyz*3+inverse_i]<0)
						inverse_flag = -1;
					std::cout<<pa_grad_new[ixyz*3+inverse_i]<<std::endl;
					std::cout<<pa_grad_old[ixyz*3+inverse_i]<<std::endl;
					pa_grad_new[ixyz*3+inverse_i] = pa_grad_old[ixyz*3+inverse_i];
					std::cout<<pa_grad_new[ixyz*3+inverse_i]<<std::endl;
					std::cout<<pa_grad_old[ixyz*3+inverse_i]<<std::endl;						
				}
				std::cout<<paXYZ[ixyz*3+0]<<" "<<paXYZ[ixyz*3+1]<<" "<<paXYZ[ixyz*3+2]<<std::endl;		
			}
		}
	}

	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
	{
		double dot0 = Com::Dot3D(pa_grad_new+ixyz*3,  pa_grad_old+ixyz*3);
		double len0 = Com::Length3D(pa_grad_new+ixyz*3);
		double len1 = Com::Length3D(pa_grad_old+ixyz*3);
		const double cos0 = dot0/(len0*len1);
		if( ixyz == 0 || cos0 < min_cos )
		{
			min_cos = cos0;
			if(cos0<0)
			{
				std::cout<<"---------------------WTF------------------------"<<std::endl;
				std::cout<<pa_grad_new[ixyz*3+0]<<std::endl;
				std::cout<<pa_grad_old[ixyz*3+0]<<std::endl;
				std::cout<<pa_grad_new[ixyz*3+1]<<std::endl;
				std::cout<<pa_grad_old[ixyz*3+1]<<std::endl;
				std::cout<<pa_grad_new[ixyz*3+2]<<std::endl;
				std::cout<<pa_grad_old[ixyz*3+2]<<std::endl;
				std::cout<<paXYZ[ixyz*3+0]<<" "<<paXYZ[ixyz*3+1]<<" "<<paXYZ[ixyz*3+2]<<std::endl;			
			}
		}
	}
	std::cout << "iteration:" << itr << "  mincos:" << min_cos << std::endl;
	//break;
	if( min_cos > cos(3.1415*0.1)){
		//dh *= 2;
		break;
	}
	//if( min_cos > cos(3.1415*0.25) ) break;
	//    if( min_cos > 0 ) break;
	/*
	double diff = 0;
	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++){
		double diff0 = Com::SquareDistance3D(pa_grad_new+ixyz*3, pa_grad_old+ixyz*3);
		diff += diff0;
	}
	if( diff < 0.01 ) dh*=2;
	*/
	//    std::cout << "iteration:" << itr << "  diff:" << diff << std::endl;
	//    if( diff > cos(3.1415*0.05) ) break;
	}
}

void UTILS::setCentroidWeight(unsigned int iNo0, std::vector<CNodeBH>& aNode, const unsigned int* paTri, const double* paXYZ)
{
	CNodeBH& nbh = aNode[iNo0];
	if( nbh.ichild_ == -1 )
	{  
		// there is no child
		if( nbh.n_tri_cell_ == 0 )
		{
			nbh.weight_ = 0;
			nbh.centroid_[0] = 0;
			nbh.centroid_[1] = 0;
			nbh.centroid_[2] = 0;
			return;
		}
		double a = 0;
		double c[3] = {0,0,0};
		for(unsigned int itricell=0;itricell<nbh.n_tri_cell_;itricell++)
		{
			unsigned int itri0 = nbh.aIndTriCell[itricell];
			const unsigned int* tri = paTri+itri0*3;
			const double a0 = Com::TriArea3D(paXYZ+tri[0]*3,paXYZ+tri[1]*3,paXYZ+tri[2]*3);
			a += a0;
			const double w0 = a0/3.0;
			for(unsigned int inotri=0;inotri<3;inotri++)
			{
				unsigned int ino0 = paTri[itri0*3+inotri];
				c[0] += paXYZ[ino0*3+0]*w0;  
				c[1] += paXYZ[ino0*3+1]*w0;  
				c[2] += paXYZ[ino0*3+2]*w0;
			}
		}
		nbh.weight_ = a;
		nbh.centroid_[0] = c[0]/a;
		nbh.centroid_[1] = c[1]/a;
		nbh.centroid_[2] = c[2]/a;
		{
			double eps = 1.0e-10;
			assert( nbh.centroid_[0] > nbh.cent_[0]-nbh.hw_-eps && nbh.centroid_[0] < nbh.cent_[0]+nbh.hw_+eps );
			assert( nbh.centroid_[1] > nbh.cent_[1]-nbh.hw_-eps && nbh.centroid_[1] < nbh.cent_[1]+nbh.hw_+eps );
			assert( nbh.centroid_[2] > nbh.cent_[2]-nbh.hw_-eps && nbh.centroid_[2] < nbh.cent_[2]+nbh.hw_+eps );
		}
		return;
	}

	const unsigned int ich0 = nbh.ichild_;
	double a = 0;
	double c[3] = {0,0,0};
	for(unsigned int icell=0;icell<8;icell++)
	{
		setCentroidWeight(ich0+icell,aNode,paTri,paXYZ);
		const double w0 = aNode[ich0+icell].weight_;
		a += w0;
		c[0] += aNode[ich0+icell].centroid_[0]*w0;
		c[1] += aNode[ich0+icell].centroid_[1]*w0;
		c[2] += aNode[ich0+icell].centroid_[2]*w0;
	}
	nbh.weight_ = a;
	nbh.centroid_[0] = c[0]/a;
	nbh.centroid_[1] = c[1]/a;
	nbh.centroid_[2] = c[2]/a;

	{
		double eps = 1.0e-10;
		assert( nbh.centroid_[0] > nbh.cent_[0]-nbh.hw_-eps && nbh.centroid_[0] < nbh.cent_[0]+nbh.hw_+eps );
		assert( nbh.centroid_[1] > nbh.cent_[1]-nbh.hw_-eps && nbh.centroid_[1] < nbh.cent_[1]+nbh.hw_+eps );
		assert( nbh.centroid_[2] > nbh.cent_[2]-nbh.hw_-eps && nbh.centroid_[2] < nbh.cent_[2]+nbh.hw_+eps );   
	}
}

void UTILS::addFaceCenter(unsigned int iNo0, std::vector<CNodeBH>& aNo, const double fc[3], unsigned int indtri0, const unsigned int* paTri,
  const double* paXYZ)
{
	{
		CNodeBH& no0 = aNo[iNo0];  // if the size aNo changes, this pointer may get invalid
		if( fc[0] < no0.cent_[0]-no0.hw_ || fc[0] > no0.cent_[0]+no0.hw_ ||
			fc[1] < no0.cent_[1]-no0.hw_ || fc[1] > no0.cent_[1]+no0.hw_ ||       
			fc[2] < no0.cent_[2]-no0.hw_ || fc[2] > no0.cent_[2]+no0.hw_ )
		{
			std::cout << iNo0 << " " << std::endl;
			std::cout << fc[0] << " " << no0.cent_[0]-no0.hw_ << " " << no0.cent_[0]+no0.hw_ << std::endl;
			std::cout << fc[1] << " " << no0.cent_[1]-no0.hw_ << " " << no0.cent_[1]+no0.hw_ << std::endl;
			std::cout << fc[2] << " " << no0.cent_[2]-no0.hw_ << " " << no0.cent_[2]+no0.hw_ << std::endl;
		}
		double eps = 1.0e-10;
		assert( fc[0] > no0.cent_[0]-no0.hw_-eps && fc[0] < no0.cent_[0]+no0.hw_+eps );
		assert( fc[1] > no0.cent_[1]-no0.hw_-eps && fc[1] < no0.cent_[1]+no0.hw_+eps );
		assert( fc[2] > no0.cent_[2]-no0.hw_-eps && fc[2] < no0.cent_[2]+no0.hw_+eps );
	}
	if(aNo[iNo0].ichild_ != -1)
	{
		CNodeBH& no0 = aNo[iNo0];  // if the size aNo changes, this pointer may get invalid
		unsigned int i0 = 0;
		if( fc[0] > no0.cent_[0] ) i0 += 1;
		if( fc[1] > no0.cent_[1] ) i0 += 2;
		if( fc[2] > no0.cent_[2] ) i0 += 4;
		unsigned int ich0 = no0.ichild_+i0;
		addFaceCenter(no0.ichild_+i0,aNo,fc,indtri0,paTri,paXYZ);
		return;
	}
	if( aNo[iNo0].n_tri_cell_ < max_tri_cell )
	{
		CNodeBH& no0 = aNo[iNo0];  // if the size aNo changes, this pointer may get invalid
		no0.aIndTriCell[no0.n_tri_cell_] = indtri0;
		no0.n_tri_cell_++;
		return;
	}
	assert( aNo[iNo0].n_tri_cell_ == max_tri_cell );
	unsigned int nNo0 = aNo.size();
	aNo.resize( aNo.size() + 8 );
	CNodeBH& no0 = aNo[iNo0];  // if the size aNo changes, this pointer may get invalid 
	no0.n_tri_cell_ = 0;
	no0.ichild_ = nNo0;
	const double* cent0 = no0.cent_;
	const double hw0 = no0.hw_;
	{
		CNodeBH& nbh = aNo[no0.ichild_+0];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]-hw0*0.5; nbh.cent_[1] = cent0[1]-hw0*0.5; nbh.cent_[2] = cent0[2]-hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+1];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]+hw0*0.5; nbh.cent_[1] = cent0[1]-hw0*0.5; nbh.cent_[2] = cent0[2]-hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+2];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]-hw0*0.5; nbh.cent_[1] = cent0[1]+hw0*0.5; nbh.cent_[2] = cent0[2]-hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+3];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]+hw0*0.5; nbh.cent_[1] = cent0[1]+hw0*0.5; nbh.cent_[2] = cent0[2]-hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+4];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]-hw0*0.5; nbh.cent_[1] = cent0[1]-hw0*0.5; nbh.cent_[2] = cent0[2]+hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+5];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]+hw0*0.5; nbh.cent_[1] = cent0[1]-hw0*0.5; nbh.cent_[2] = cent0[2]+hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+6];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]-hw0*0.5; nbh.cent_[1] = cent0[1]+hw0*0.5; nbh.cent_[2] = cent0[2]+hw0*0.5;
		nbh.ichild_ = -1;
	}
	{
		CNodeBH& nbh = aNo[no0.ichild_+7];
		nbh.hw_ = no0.hw_*0.5;
		nbh.cent_[0] = cent0[0]+hw0*0.5; nbh.cent_[1] = cent0[1]+hw0*0.5; nbh.cent_[2] = cent0[2]+hw0*0.5;
		nbh.ichild_ = -1;
	}
	addFaceCenter(iNo0,aNo,fc,indtri0,paTri,paXYZ);
	for(unsigned int i=0;i<max_tri_cell;i++)
	{
		unsigned int itri0 = aNo[iNo0].aIndTriCell[i];
		aNo[iNo0].aIndTriCell[i] = 0;
		double c[3] = {0,0,0};
		for(unsigned int ino=0;ino<3;ino++)
		{
			unsigned int ino0 = paTri[itri0*3+ino];
			c[0] += paXYZ[ino0*3+0];
			c[1] += paXYZ[ino0*3+1];
			c[2] += paXYZ[ino0*3+2];
		}
		c[0] /= 3.0;
		c[1] /= 3.0;
		c[2] /= 3.0;
		addFaceCenter(iNo0,aNo,c,itri0,paTri,paXYZ);
	}
}

void UTILS::getBarnHutAppAreaDistGradIter(unsigned int iNo0, const std::vector<CNodeBH>& aNode, const double p[3],
  const unsigned int* paTri,  const double* paXYZ, double height, double& tot_area, double& app_dist, double app_grad[3])
{
	const CNodeBH& no0 = aNode[iNo0];
	{
		double d0 = Com::Distance3D(no0.centroid_,p);
		double hw0 = no0.hw_;
		if( no0.weight_ < 1.0e-20 )
		{
			return;
		}
	/*
	//    std::cout << d0 << " " << height << "  " << iNo0 << " " << no0.centroid_[0] << " " << no0.centroid_[1] << " " << no0.centroid_[2] << std::endl;
	if( d0 > hw0*4 ){
		if( p[0] > no0.cent_[0]-no0.hw_ && p[0] < no0.cent_[0]+no0.hw_ &&      
			p[1] > no0.cent_[1]-no0.hw_ && p[1] < no0.cent_[1]+no0.hw_ &&
			p[2] > no0.cent_[2]-no0.hw_ && p[2] < no0.cent_[2]+no0.hw_ ){}
		else{
		double dA0 = no0.weight_;
		tot_area += dA0;
		app_dist += dA0/(d0*d0*d0);
		double tmp0 = dA0/(d0*d0*d0*d0*d0);
		app_grad[0] += tmp0*(p[0] - no0.centroid_[0]);
		app_grad[1] += tmp0*(p[1] - no0.centroid_[1]);       
		app_grad[2] += tmp0*(p[2] - no0.centroid_[2]);
		return;
		}
	}
	*/

	// 	if(d0>fabs(height)*8.0)
	// 	{
	// 		return;
	// 	}
	//if( d0 > hw0*4 )
	//再此计算hw0系数，系数想成越高生成的曲面效果越好，在于积分区域面积增加
	//if( d0 > hw0*5 )
		if( d0 > hw0*4.0 )
		{
		//      std::cout << iNo0 << " " << d0 << " " << hw0 << "     " << no0.centroid_[0] << " " << no0.centroid_[1] << " " << no0.centroid_[2] << std::endl;
      
			if( p[0] > no0.cent_[0]-no0.hw_ && p[0] < no0.cent_[0]+no0.hw_ &&      
				p[1] > no0.cent_[1]-no0.hw_ && p[1] < no0.cent_[1]+no0.hw_ &&
				p[2] > no0.cent_[2]-no0.hw_ && p[2] < no0.cent_[2]+no0.hw_ )
			{

			}
			else
			{
				//return;
		// 			if(d0>fabs(height)*2.0)
		// 			{
		// 				return;
		// 			}

				double dA0 = no0.weight_;
				tot_area += dA0;
				app_dist += d0;
				double tmp0 = 1.0/(d0*d0*d0*d0*d0);
				//      std::cout << tmp0 << " " << d0 << " " << no0.centroid_[0] << " " << no0.centroid_[1] << " " << no0.centroid_[2] << std::endl;
				app_grad[0] += -3.0*tmp0*(p[0] - no0.centroid_[0]);
				app_grad[1] += -3.0*tmp0*(p[1] - no0.centroid_[1]);       
				app_grad[2] += -3.0*tmp0*(p[2] - no0.centroid_[2]);
				return;
			}
		}     
	}
	
	if( no0.ichild_ == -1 )
	{  // no child 
	// 	  double d0 = Com::Distance3D(no0.centroid_,p);
	// 	  if(d0>fabs(height)*2.0)
	// 	  {
	// 		  return;
	// 	  }
		for(unsigned int itricell=0;itricell<no0.n_tri_cell_;itricell++)
		{
			double c[3] = {0,0,0};
			unsigned int itri0 = no0.aIndTriCell[itricell];
		//      std::cout << itricell << " " << itri0 << std::endl;
			unsigned int i0 = paTri[itri0*3+0];
			unsigned int i1 = paTri[itri0*3+1];
			unsigned int i2 = paTri[itri0*3+2];
			double der[3], val;
			jie::computeTriangle(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3,der,val);
			app_dist += val;
		//       app_grad[0] += der[0];
		//       app_grad[1] += der[1];
		//       app_grad[2] += der[2];
			{
				double v = Com::TetVolume3D(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
				double a = Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
		//        double c0 = Com::Dot3D(paXYZ+i1*3,paXYZ+i2*3)/(Com::Length3D(paXYZ+i1*3)*Com::Length3D(paXYZ+i2*3));
		//        double c1 = Com::Dot3D(paXYZ+i2*3,paXYZ+i0*3)/(Com::Length3D(paXYZ+i2*3)*Com::Length3D(paXYZ+i0*3));
		//        double c2 = Com::Dot3D(paXYZ+i0*3,paXYZ+i1*3)/(Com::Length3D(paXYZ+i0*3)*Com::Length3D(paXYZ+i1*3));

		// 		double tmp0 = a/val;
		// 		double tmp1 = pow(tmp0,1.0/3.0);  // distance
		// 		double tmp2 = -tmp1*tmp1*tmp1*tmp1/(3*tot_area);  
				double tmp2 = 1;
				double len = app_grad[0]*app_grad[0] + app_grad[1]*app_grad[1] + app_grad[2]*app_grad[2];
				app_grad[0] += tmp2*der[0];
				app_grad[1] += tmp2*der[1];       
				app_grad[2] += tmp2*der[2];
				len = sqrt(len);
		//        if( len > 1000 ){  
		//        std::cout << "  " << iNo0 << " " << len << " " << v*3/a << " " << " " << c0 << " " << c1 << " " << c2 << std::endl;
		//        }
			}
			tot_area += Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
		}
		return;
	}
	unsigned int ich0 = no0.ichild_;
	for(unsigned int i=0;i<8;i++)
	{
		getBarnHutAppAreaDistGradIter(ich0+i,aNode,p, paTri,paXYZ,height, tot_area,app_dist,app_grad);
	}  
}

void UTILS::getValueGradientGrowth(const double p[], double grad[], double height, unsigned int nXYZ,
	const double* paXYZ, unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo,
	const std::vector<CNodeBH>& aNode)
{
	double app_dist = 0;
	double app_grad[3] = {0,0,0};
	double tot_area = 0;
	getBarnHutAppAreaDistGradIter(0,aNode,p, paTri,paXYZ,height, tot_area,app_dist,app_grad);    
	////
	/*
	for(unsigned int itri=0;itri<nTri;itri++){
	unsigned int i0 = paTri[itri*3+0];
	unsigned int i1 = paTri[itri*3+1];
	unsigned int i2 = paTri[itri*3+2];
	double der[3], val;
	jie::computeTriangle(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3,der,val);
	app_dist += val;
	app_grad[0] += der[0];
	app_grad[1] += der[1];
	app_grad[2] += der[2];
	tot_area += Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
	}
	*/
	//  std::cout << "tot area : " << tot_area << std::endl;
	/////
	if( app_dist < 1.0e-10 )
	{
		grad[0] = 0;
		grad[1] = 0;
		grad[2] = 0;
		return;
	}
	if(tot_area<0)
	{
		std::cout<<tot_area<<std::endl;
	}  
	double tmp0 = tot_area/app_dist;
	double tmp1 = pow(tmp0,1.0/3.0);  // distance
	//std::cout<<tmp1<<std::endl;
	double tmp2 = -tmp1*tmp1*tmp1*tmp1/(3*tot_area);
	//tmp2 *= tmp0; // ?
	grad[0] = tmp2*app_grad[0];
	grad[1] = tmp2*app_grad[1];       
	grad[2] = tmp2*app_grad[2];
	//std::cout << "tot area : " << tot_area << std::endl;
	//std::cout << "tot app grad: " << sqrtf(app_grad[0]*app_grad[0] + app_grad[1]*app_grad[1] + app_grad[2]*app_grad[2]) << std::endl;
	//std::cout << "tot grad: " << sqrtf(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]) << std::endl;
	//std::cout<<"grad 0 "<<grad[0]<<std::endl;
	//std::cout<<"grad 1 "<<grad[1]<<std::endl;
	//std::cout<<"grad 2 "<<grad[2]<<std::endl;
	double tot_grad = sqrtf(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]);
	double gradient_thres = pow(0.1,10);
	if(tot_grad<gradient_thres)
	{
		grad[0] = 0;
		grad[1] = 0;
		grad[2] = 0;
	} 
}

void UTILS::getValueGradientStopThres(const double p[], double grad[], double height, unsigned int nXYZ, const double* paXYZ,
	unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode)
{
	double app_dist = 0;
	double app_grad[3] = {0,0,0};
	double tot_area = 0;
	getBarnHutAppAreaDistGradIter(0,aNode,p, paTri,paXYZ,height, tot_area,app_dist,app_grad);    
	////
	/*
	for(unsigned int itri=0;itri<nTri;itri++){
	unsigned int i0 = paTri[itri*3+0];
	unsigned int i1 = paTri[itri*3+1];
	unsigned int i2 = paTri[itri*3+2];
	double der[3], val;
	jie::computeTriangle(p,paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3,der,val);
	app_dist += val;
	app_grad[0] += der[0];
	app_grad[1] += der[1];
	app_grad[2] += der[2];
	tot_area += Com::TriArea3D(paXYZ+i0*3,paXYZ+i1*3,paXYZ+i2*3);
	}
	*/
	//  std::cout << "tot area : " << tot_area << std::endl;
	/////
	if( app_dist < 1.0e-10 )
	{
		grad[0] = 0;
		grad[1] = 0;
		grad[2] = 0;
		return;
	}
	if(tot_area<0)
	{
		std::cout<<tot_area<<std::endl;
	}  
	double tmp0 = tot_area/app_dist;
	double tmp1 = pow(tmp0,1.0/3.0);  // distance
	//std::cout<<tmp1<<std::endl;
	double tmp2 = -tmp1*tmp1*tmp1*tmp1/(3*tot_area);
	//tmp2 *= tmp0; // ?
	grad[0] = tmp2*app_grad[0];
	grad[1] = tmp2*app_grad[1];       
	grad[2] = tmp2*app_grad[2];
	//std::cout << "tot area : " << tot_area << std::endl;
	//std::cout << "tot app grad: " << sqrtf(app_grad[0]*app_grad[0] + app_grad[1]*app_grad[1] + app_grad[2]*app_grad[2]) << std::endl;
	//std::cout << "tot grad: " << sqrtf(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]) << std::endl;
	//std::cout<<"grad 0 "<<grad[0]<<std::endl;
	//std::cout<<"grad 1 "<<grad[1]<<std::endl;
	//std::cout<<"grad 2 "<<grad[2]<<std::endl;
	double tot_grad = sqrtf(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]);
	//更改梯度终止项
	double gradient_thres = 1e-10;
	if(tot_grad<gradient_thres)
	{
		grad[0] = 0;
		grad[1] = 0;
		grad[2] = 0;
	}
	else
	{
		grad[0] = 1.0/tot_grad*grad[0];
		grad[1] = 1.0/tot_grad*grad[1];
		grad[2] = 1.0/tot_grad*grad[2];
	}
}

void UTILS::getPosGrad(double* pa_pos_new, double* pa_grad_new, double& dh, const double* pa_pos_old,
	const double* pa_grad_old, double height, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, const unsigned int* paTri,
	const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode)
{
	double total_h = 0;
	double termi_h = dh;
	double min_cos = 1;
	double cos0 = 0;
	double add_h = 0;

	double* pa_pos_old_c = new double[nXYZ*3];
	double* pa_grad_old_c = new double[nXYZ*3];

	for(int i = 0; i<nXYZ*3; i++)
	{
		pa_pos_old_c[i] = pa_pos_old[i];
		pa_grad_old_c[i] = pa_grad_old[i];
	}
	for(unsigned int i=0;i<nXYZ;i++)
	{		
		double cos0;
		double total_h = 0;
		dh = 2*(termi_h);
		cos0 = 0;
		add_h = 0;
		//更改限制生长条件，即生长的单位长度
		double cos_ter = cos(3.1415*0.15);
		while (fabs(total_h-termi_h)>1e-5)
		{
			cos0 = 0;
			bool terminate_flag = false;
			
			while (cos0<cos_ter)
			{
				dh *= 0.5;
				add_h = 0;
				//if(dh+total_h>termi_h+1e-8)
				if(dh+total_h>termi_h+1e-5)
				{
					dh = termi_h-total_h;
					terminate_flag = true;
				}				
				for(int i_tri = i*3;i_tri < i*3+3; i_tri++)
				{
					pa_pos_new[i_tri] = pa_pos_old_c[i_tri] + pa_grad_old_c[i_tri]*dh;
					add_h = add_h + pa_grad_old_c[i_tri]*pa_grad_old_c[i_tri];
					//std::cout<<pa_grad_old_c[i_tri]<<std::endl;
				}				
				getValueGradientStopThres(pa_pos_new+i*3, pa_grad_new+i*3,height,nXYZ,paXYZ,nTri,paTri,topo,aNode);
				double dot0 = Com::Dot3D(pa_grad_new+i*3,  pa_grad_old+i*3);
				double len0 = Com::Length3D(pa_grad_new+i*3);
				double len1 = Com::Length3D(pa_grad_old+i*3);
				cos0 = dot0/(len0*len1);
				add_h = sqrtf(add_h);
				if(terminate_flag == true)
				{
					break;
				}
				if (fabs(add_h)<1e-5)
				{
					break;
				}
				if (fabs(dh)<1e-5)
				{
					if(fabs(total_h-termi_h)<2*dh)
					{
						dh = termi_h-total_h;
					}
					else
					{
						dh = 2*dh;
					}
					cos0 = 1;
				}				
			}			
			for(int i_tri = i*3;i_tri < i*3+3; i_tri++)
			{
				pa_pos_old_c[i_tri] = pa_pos_new[i_tri];
				pa_grad_old_c[i_tri] = pa_grad_new[i_tri];
			}
			total_h = total_h + dh;	
			if(terminate_flag == true)
			{
				break;
			}
			if(fabs(add_h)<1e-5)
			{
				break;
			}
		}
		if(fabs(total_h-termi_h)>1e-5)
		{
			std::cout<<"Generate Fail"<<std::endl;
		}
	}
	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
	{
		getValueGradientStopThres(pa_pos_new+ixyz*3, pa_grad_new+ixyz*3,height,nXYZ,paXYZ,nTri,paTri,topo,aNode);
	}    
		
	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
	{
		double dot0 = Com::Dot3D(pa_grad_new+ixyz*3,  pa_grad_old+ixyz*3);
		double len0 = Com::Length3D(pa_grad_new+ixyz*3);
		double len1 = Com::Length3D(pa_grad_old+ixyz*3);
		const double cos0 = dot0/(len0*len1);
		if( ixyz == 0 || cos0 < min_cos )
		{
		min_cos = cos0;
		if(cos0<0)
		{
			//std::cout<<"---------------------WTF--------Part1----------------"<<std::endl;
			//std::cout<<pa_grad_new[ixyz*3+0]<<std::endl;
			//std::cout<<pa_grad_new[ixyz*3+1]<<std::endl;
			//std::cout<<pa_grad_new[ixyz*3+2]<<std::endl;

			//std::cout<<pa_grad_old[ixyz*3+0]<<std::endl;			
			//std::cout<<pa_grad_old[ixyz*3+1]<<std::endl;			
			//std::cout<<pa_grad_old[ixyz*3+2]<<std::endl;
			//
			//std::cout<<"Number of Tri "<<ixyz<<std::endl;

			//std::cout<<paXYZ[ixyz*3+0]<<" "<<paXYZ[ixyz*3+1]<<" "<<paXYZ[ixyz*3+2]<<std::endl;	

			//pa_grad_new[ixyz*3+0] = 0;
			//pa_grad_new[ixyz*3+1] = 0;
			//pa_grad_new[ixyz*3+2] = 0;

			//pa_grad_new[ixyz*3+0] = pa_grad_old[ixyz*3+1];
			//pa_grad_new[ixyz*3+1] = pa_grad_old[ixyz*3+2];
			//pa_grad_new[ixyz*3+2] = pa_grad_old[ixyz*3+3];
		}
		}
	}
	std::cout << "iteration:" <<  "  mincos:" << min_cos << std::endl;
	//break;
	dh = termi_h;
	delete 	[] pa_pos_old_c;
	delete  [] pa_grad_old_c;  
}
void UTILS::GetValueGradientExp(double* pa_pos_new, double* pa_grad_new, double& dh, const double* pa_pos_old, const double* pa_grad_old,
	double height, unsigned int nXYZ, const double* paXYZ, unsigned int nTri, const unsigned int* paTri, const CTriAryTopology& topo,
	const std::vector<CNodeBH>& aNode)
{
	dh *= 2;
	for(unsigned int itr=0;itr<1;itr++)
	{
		dh *= 0.5;
		for(unsigned int i=0;i<nXYZ*3;i++)
		{		
			pa_pos_new[i] = pa_pos_old[i] + pa_grad_old[i]*dh;
		}
		for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
		{
			getValueGradientStopThres(pa_pos_new+ixyz*3, pa_grad_new+ixyz*3, height, nXYZ,paXYZ,nTri,paTri, topo, aNode);
		}    
		double min_cos;

		for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
		{
			double dot0 = Com::Dot3D(pa_grad_new+ixyz*3,  pa_grad_old+ixyz*3);
			double len0 = Com::Length3D(pa_grad_new+ixyz*3);
			double len1 = Com::Length3D(pa_grad_old+ixyz*3);
			const double cos0 = dot0/(len0*len1);
			if( ixyz == 0 || cos0 < min_cos )
			{
				min_cos = cos0;
				if(cos0<0)
				{
					for(int inverse_i = 0; inverse_i<3; inverse_i++)
					{
						double inverse_flag = 1;
						if(pa_grad_old[ixyz*3+inverse_i]<0)
							inverse_flag = -1;
						std::cout<<pa_grad_new[ixyz*3+inverse_i]<<std::endl;
						std::cout<<pa_grad_old[ixyz*3+inverse_i]<<std::endl;
						pa_grad_new[ixyz*3+inverse_i] = pa_grad_old[ixyz*3+inverse_i];
						std::cout<<pa_grad_new[ixyz*3+inverse_i]<<std::endl;
						std::cout<<pa_grad_old[ixyz*3+inverse_i]<<std::endl;						
					}
					std::cout<<paXYZ[ixyz*3+0]<<" "<<paXYZ[ixyz*3+1]<<" "<<paXYZ[ixyz*3+2]<<std::endl;		
				}
			}
		}


		for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
		{
			double dot0 = Com::Dot3D(pa_grad_new+ixyz*3,  pa_grad_old+ixyz*3);
			double len0 = Com::Length3D(pa_grad_new+ixyz*3);
			double len1 = Com::Length3D(pa_grad_old+ixyz*3);
			const double cos0 = dot0/(len0*len1);
			if( ixyz == 0 || cos0 < min_cos )
			{
				min_cos = cos0;
				if(cos0<0)
				{
					std::cout<<"---------------------WTF------------------------"<<std::endl;
					std::cout<<pa_grad_new[ixyz*3+0]<<" ";
					std::cout<<pa_grad_new[ixyz*3+1]<<" ";
					std::cout<<pa_grad_new[ixyz*3+2]<<std::endl;


					std::cout<<pa_grad_old[ixyz*3+0]<<" ";			
					std::cout<<pa_grad_old[ixyz*3+1]<<" ";			
					std::cout<<pa_grad_old[ixyz*3+2]<<std::endl;

					std::cout<<paXYZ[ixyz*3+0]<<" "<<paXYZ[ixyz*3+1]<<" "<<paXYZ[ixyz*3+2]<<std::endl;			
				}
			}
		}
		std::cout << "iteration:" << itr << "  mincos:" << min_cos << std::endl;
		//break;
		if( min_cos > cos(3.1415*0.1))
		{
			//dh *= 2;
			break;
		}
	//if( min_cos > cos(3.1415*0.25) ) break;
	//    if( min_cos > 0 ) break;
	/*
	double diff = 0;
	for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++){
		double diff0 = Com::SquareDistance3D(pa_grad_new+ixyz*3, pa_grad_old+ixyz*3);
		diff += diff0;
	}
	if( diff < 0.01 ) dh*=2;
	*/
	//    std::cout << "iteration:" << itr << "  diff:" << diff << std::endl;
	//    if( diff > cos(3.1415*0.05) ) break;
	}
}
void UTILS::makeExtMesh(const double height, std::vector<double*>& aExtXYZ, unsigned int nXYZ, const double* paXYZ, unsigned int nTri,
	const unsigned int* paTri, const CTriAryTopology& topo, const std::vector<CNodeBH>& aNode, bool is_ext, int layer)
{
	int n = (int)aExtXYZ.size();
	for (int i = 0; i < n; i++)
	{
		if (NULL != aExtXYZ[i])
		{
			delete [] aExtXYZ[i];
			aExtXYZ[i] = NULL;
		}
		delete aExtXYZ[i];		
	}	
	aExtXYZ.clear();
	assert( aExtXYZ.size() == 0 );
	double* grad_old = new double [nXYZ*3];
	double* grad_new = new double [nXYZ*3];
	{
		double* aNorm_ = grad_old;
		for(unsigned int i=0;i<nXYZ*3;i++)
		{ 
			aNorm_[i] = 0; 
		}
		for(unsigned int itri=0;itri<nTri;itri++)
		{   
			unsigned int i1 = paTri[itri*3+0];
			unsigned int i2 = paTri[itri*3+1];
			unsigned int i3 = paTri[itri*3+2];
			double un[3], area;    
			Com::UnitNormalAreaTri3D(un,area, paXYZ+i1*3, paXYZ+i2*3, paXYZ+i3*3);    
			aNorm_[i1*3+0] += un[0];  aNorm_[i1*3+1] += un[1];  aNorm_[i1*3+2] += un[2];
			aNorm_[i2*3+0] += un[0];  aNorm_[i2*3+1] += un[1];  aNorm_[i2*3+2] += un[2];    
			aNorm_[i3*3+0] += un[0];  aNorm_[i3*3+1] += un[1];  aNorm_[i3*3+2] += un[2];    
		}
		for(unsigned int ino=0;ino<nXYZ;ino++){
			double invlen = 1.0/Com::Length3D(aNorm_+ino*3);
			aNorm_[ino*3+0] *= invlen;
			aNorm_[ino*3+1] *= invlen;
			aNorm_[ino*3+2] *= invlen;    
		}  
	} 

	const int ITER_NUM = layer;
	//double height_ming = ( is_ext ) ? height : -height;
	
	//double h0 = height/ITER_NUM;
	double h0 = ( is_ext ) ? height/ITER_NUM : -height/ITER_NUM;
	{
		aExtXYZ.resize(1);
		aExtXYZ[0] = new double [nXYZ*3];
		double* pa_pos_new = aExtXYZ[0];
		for(unsigned int i=0;i<nXYZ*3;i++)
		{
			pa_pos_new[i] = paXYZ[i] + grad_old[i]*h0; 
		}
		for(unsigned int ixyz=0;ixyz<nXYZ;ixyz++)
		{
			getValueGradientStopThres(pa_pos_new+ixyz*3, grad_old+ixyz*3, height, nXYZ,paXYZ,nTri,paTri, topo, aNode);
		}    
	}
	double cur_height = h0;

	//const int ITER_NUM = 1000;
	
	for(unsigned int itr=0;itr<ITER_NUM-1;itr++)
	{
		double dh = height / ITER_NUM;
		printf("===overal interation, itr = %d\n", itr);
		aExtXYZ.resize(itr+2);
		aExtXYZ[itr+1] = new double [nXYZ*3];
		getPosGrad(aExtXYZ[itr+1],grad_new,dh, aExtXYZ[itr],grad_old, height, nXYZ,paXYZ,nTri,paTri, topo, aNode);

		for(unsigned int i=0;i<nXYZ*3;i++)
		{
			grad_old[i] = grad_new[i]; 
		}
		
		if(h0>0)
		{
			h0 += dh;
		}
		else
		{
			h0 += -dh;
		}
		
		printf("dh = %.12f, height=%.6f/%.6f=\n", dh, h0, height);
		if( h0 > height )
			break;
	}  
	delete[] grad_old;
	delete[] grad_new;
}
