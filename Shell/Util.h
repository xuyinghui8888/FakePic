
#ifndef util_h_jie
#define util_h_jie

#include "Geometry.h"
#include <vector>


//compute one wedge. ASSUME v1 and v2 are not zero vector and q is not on the wedge plane
//input:  p\q\v1\v2 are in global coordinate. compute point q to wedge(p,v1,v2)
//output: derivative in global coordinate. and the scalar integral
namespace jie 
{
	void computeWedge(const Point & q, const Point & p, const Point & v1, const Point & v2, Point & derivative, double & integral);
	void computeWedgeSafe(const Point & q, const Point & p, const Point & v1, const Point & v2, Point & derivative, double & integral);
	void computeTriangle(const Point & p, const Point & p1, const Point & p2, const Point & p3, Point & derivative, double & integral);
	void computeTriangle(const double x0[3], const double x1[3], const double x2[3], const double x3[3], double derivative[3], double & integral);
	void computeTriangle(const double points[4][3], double derivative[3], double & integral);
  
	// converge data structure from blk-crs into crs
	// input : 
	// nblk == number of blks in a row/column
	// ndim == size of block (each block is ndim x ndim)
	// pbcrs_ind == crs index, size of nblk+1
	// pbcrs == array of column index, size of pbcrs_ind[nblk]
	// pbval == value list, size of size of pbcrs_ind[nblk]*ndim*ndim, each blocks has sequence of data
	// output :
	// pcrs_ind == crs index, size of nblk*ndim+1
	// pcrs == array of column index, size of pcrs_ind[nblk*ndim]
	// pval == value list, size of pcrs_ind[nblk*ndim], each row has sequence of data
	void convergeBlockCRS2CRS(const unsigned int nblk, const unsigned int ndim, const unsigned int* pbcrs_ind, const unsigned int* pbcrs, const double* pbval,
	  unsigned int* pcrs_ind, unsigned int* pcrs, double* pval);
	/*
	void getDOF (
		Point xAxis1, Point yAxis1, Point u1, Point v1,
		Point xAxis2, Point yAxis2, Point u2, Point v2,
		double * dof
		);*/

	struct ObjWriter
	{
		ObjWriter();
		void init();
		void setMTLInfo(const char * str);
		void add(const double * v, int nv, const int * f, int nf);
		void addBox(const double * a, const double * b, double width);
		void write(const char * path) const;

		int base;
		std::vector<double> vs;
		std::vector<int> fs;

	/////////////// note that vt have only one /////////////////
		void addTexCoord(const double * vt, int nvt);
		void appendTriCoord(const int * fvt, int nf);
		std::string mtlInfo;
		std::vector<double> vts;
		std::vector<int> fvts;
		char * texturePath;
	};

	void pinocchioSkel2Obj(const char * path, double width = 0.05);


	void write_mitsuba_xml(
		double fovy, 
		double nearClip, 
		double farClip,

		double eyex, 
		double eyey, 
		double eyez, 
		double centerx, 
		double centery, 
		double centerz, 
		double upx, 
		double upy, 
		double upz,

		int height,
		int width,
		double light0Dir[3],
		double light1Dir[3],
		const std::vector<ObjWriter> & objWriters,
		const std::string & fname
		);
	//std::string getLocalTimeString();
};//namespace jie
#endif //util_h_jie