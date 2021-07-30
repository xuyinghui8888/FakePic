#include <fstream>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include "util.h"
namespace jie 
{
	//compute one wedge. ASSUME v1 and v2 are not zero vector & v1 v2 are not on the same line & p is not on the wedge plane
	//input:  p\o\v1\v2 are in global coordinate. compute point p to wedge(o,v1,v2)
	//output: derivative in global coordinate. and the scalar integral
	void computeWedge(const Point & p, const Point & o, const Point & v1, const Point & v2, Point & derivative, double & integral)
	{
		//static std::ofstream fout("computeWedge.txt");
		Point x = p - o;

		Point xAxis = v1.normalize();
		Point zAxis = cross(v1,v2).normalize();
		Point yAxis = cross(zAxis, xAxis);	//assume y is normalized;

		long double u = dot(x, xAxis);
		long double v = dot(x, yAxis);
		long double w = dot(x, zAxis);
	
		if(fabs(w)<=1E-6) {
			if(w>=0)	w=1E-6;
			else		w=-1E-6;
		}

		long double q = sqrt(u*u+v*v+w*w);
		long double beta = angle(v1,v2);


		long double theta = atan(  fabs(w)  /  (  (q-u)*cot(beta/2)-v  )   );
		if(theta < 0)	theta += M_PI;
		integral = 2*theta / fabs(w);

		long double tmp = q*(q-(u*cos(beta)+v*sin(beta)));
		long double du = sin(beta) / tmp;
		long double dv = 1/q/(q-u) - cos(beta)/tmp;
		long double dw = -integral/w + (   (u*u+v*v-q*u)*sin(beta) - q*v*(1-cos(beta))   ) / (tmp*w*(q-u));

		Point xAxisInv(xAxis.x, yAxis.x, zAxis.x);
		Point yAxisInv(xAxis.y, yAxis.y, zAxis.y);
		Point zAxisInv(xAxis.z, yAxis.z, zAxis.z);
		Point duvw(du, dv, dw);

		derivative = Point(dot(duvw,xAxisInv), dot(duvw,yAxisInv), dot(duvw, zAxisInv));

		//fout << u << "\t" << v << "\t" << w << "\t" << q << "\t" << beta << "\t" << integral << "\t" << tmp << "\t" << bb << std::endl;
	}
	void computeWedgeSafe(const Point & p, const Point & o, const Point & v1, const Point & v2, Point & derivative, double & integral) 
	{
		//static std::ofstream fout("computeWedge.txt");
		Point x = p - o;

		Point xAxis = v1.normalize();
		Point zAxis = cross(v1,v2).normalize();
		Point yAxis = cross(zAxis, xAxis);	//assume y is normalized;

		long double u = dot(x, xAxis);
		long double v = dot(x, yAxis);
		long double w = dot(x, zAxis);

		if(fabs(w)<=1E-6) {
			if(w>=0)	w=1E-6;
			else		w=-1E-6;
		}

		long double q = sqrt(u*u+v*v+w*w);
		long double beta = angle(v1,v2);
		if(beta<0)
		{
			std::cout<<beta<<std::endl;
		}

		long double theta = atan(  fabs(w)  /  (  (q-u)*cot(beta/2)-v  )   );
		if(theta < 0)	theta += M_PI;
		integral = 2*theta / fabs(w);
		if(integral<0)
		{
			std::cout<<p.x<<p.y<<p.z;
		}
		long double tmp = q*(q-(u*cos(beta)+v*sin(beta)));
	
		long double du = sin(beta) / tmp;
		long double dv = 1/q/(q-u) - cos(beta)/tmp;
		long double dw = 0;
	
	
		if(fabs(w)<1e-15)
		{
			du = 0;
			dv = 0;
			dw = 0;
		}
		else
		{
			dw = -integral/w + (   (u*u+v*v-q*u)*sin(beta) - q*v*(1-cos(beta))   ) / (tmp*w*(q-u));
		}
	

		Point xAxisInv(xAxis.x, yAxis.x, zAxis.x);
		Point yAxisInv(xAxis.y, yAxis.y, zAxis.y);
		Point zAxisInv(xAxis.z, yAxis.z, zAxis.z);
		Point duvw(du, dv, dw);

		derivative = Point(dot(duvw,xAxisInv), dot(duvw,yAxisInv), dot(duvw, zAxisInv));

		//fout << u << "\t" << v << "\t" << w << "\t" << q << "\t" << beta << "\t" << integral << "\t" << tmp << "\t" << bb << std::endl;
	}
	void computeTriangle(const Point & p, const Point & p1, const Point & p2, const Point & p3, Point & derivative, double & integral) {
		derivative = Point(0,0,0);
		integral = 0;

		Point tmpDerivative;
		double tmpIntegral;

	
		computeWedgeSafe(p,p1,p2-p1,p3-p1,tmpDerivative,tmpIntegral);
		derivative = derivative + tmpDerivative;
		integral = integral + tmpIntegral;
		//std::cout<<tmpDerivative.x<<" "<<tmpDerivative.y<<" "<<tmpDerivative.z<<" "<<std::endl;

		computeWedgeSafe(p,p2,p2-p3,p2-p1,tmpDerivative,tmpIntegral);
		derivative = derivative + tmpDerivative;
		integral = integral + tmpIntegral;
		//std::cout<<tmpDerivative.x<<" "<<tmpDerivative.y<<" "<<tmpDerivative.z<<" "<<std::endl;

		computeWedgeSafe(p,p3,p2-p3,p3-p1,tmpDerivative,tmpIntegral);
		derivative = derivative - tmpDerivative;
		integral = integral - tmpIntegral;
		//std::cout<<tmpDerivative.x<<" "<<tmpDerivative.y<<" "<<tmpDerivative.z<<" "<<std::endl;
		//std::cout<<"----------------total grad-------------"<<std::endl;
		//std::cout<<derivative.x<<" "<<derivative.y<<" "<<derivative.z<<" "<<std::endl;
	}
	void computeTriangle(const double x0[3], const double x1[3], const double x2[3], const double x3[3], double derivative[3], double & integral) 
	{
		Point p0(x0);
		Point p1(x1);
		Point p2(x2);
		Point p3(x3);
		Point der;
		computeTriangle(p0,p1,p2,p3,der,integral);
		der.copyTo(derivative);
		//std::cout<<"Derive--------------"<<std::endl;
		//std::cout<<derivative[0]<<" "<<derivative[1]<<" "<<derivative[2]<<std::endl;
		//std::cout<<"Points Position------"<<std::endl;
		//std::cout<<x0[0]<<" "<<x0[1]<<" "<<x0[2]<<std::endl;
	}

	void computeTriangle(const double points[4][3], double derivative[3], double & integral){
	  computeTriangle(points[0],points[1],points[2],points[3], derivative,integral);
	}
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
	  unsigned int* pcrs_ind, unsigned int* pcrs, double* pval) 
	{
		int idx = 0, row = 0;
		for(int blockRow = 0; blockRow < nblk; blockRow ++) {
			for(int localRow = 0; localRow < ndim; localRow ++) {
				pcrs_ind[row ++] = idx;
				for(int blockId = pbcrs_ind[blockRow]; blockId < pbcrs_ind[blockRow+1]; blockId ++) {
					for(int localCol = 0; localCol < ndim; localCol ++) {
						pval[idx] = pbval[blockId*ndim*ndim+localRow*ndim+localCol];	//real value
						pcrs[idx] = (pbcrs[blockId]-1) * ndim + localCol + 1;			//real column
						idx ++;
					}
				}
			}
		}
		pcrs_ind[row ++] = idx;
	}

	ObjWriter::ObjWriter() {
		this->init();
	}
	void ObjWriter::init() {
		this->base = 1;
		this->vs.clear();
		this->fs.clear();
		this->texturePath = NULL;
	}
	void ObjWriter::setMTLInfo(const char * str) {
		this->mtlInfo = std::string(str);
	}
	void ObjWriter::add(const double * v, int nv, const int * f, int nf) {
		for(int i = 0; i < 3*nv; i ++) {
			this->vs.push_back(v[i]);
		}
		for(int i = 0; i < 3*nf; i ++) {
			this->fs.push_back(base+f[i]);
		}
		base += nv;
	}
	void ObjWriter::addBox(const double * a0, const double * b0, double width) {
		jie::Point vtx[8] = {
			jie::Point( 0.5, 0.5, 0),
			jie::Point(-0.5, 0.5, 0),
			jie::Point(-0.5,-0.5, 0),
			jie::Point( 0.5,-0.5, 0),
			jie::Point( 0.5, 0.5, 1),
			jie::Point(-0.5, 0.5, 1),
			jie::Point(-0.5,-0.5, 1),
			jie::Point( 0.5,-0.5, 1),
		};
		jie::Point a(a0), b(b0);
		jie::Point vec = b - a;

		for(int i = 0; i < 8; i ++) {
			vtx[i].z = vtx[i].z * 0.95+0.025;

			vtx[i].x *= width;
			vtx[i].y *= width;
			vtx[i].z *= vec.len();

			if(jie::sig(vec.x)==0 && jie::sig(vec.y)==0) {
				vtx[i].z += jie::myMin(a.z,b.z);
			} else {
				jie::Point z = jie::Point(0,0,1);
				jie::Point fa = cross(z, vec);
				double theta =jie::myAcos(dot(z,vec) / vec.len());
				vtx[i] = jie::rotate(vtx[i], fa, theta) + a;
			}
		}
		double v[8*3]; {
			for(int i = 0; i < 8; i ++) {
				vtx[i].copyTo(v+i*3);
			}
		}
		int f[6*2*3] = {
			0, 2, 1,
			0, 3, 2,

			4, 5, 6,
			4, 6, 7,

			2, 7, 6,
			2, 3, 7,

			1, 6, 5,
			1, 2, 6,

			0, 5, 4,
			0, 1, 5,

			3, 4, 7,
			3, 0, 4,
		};

		this->add(v, 8, f, 12);
	}
	void ObjWriter::write(const char * path) const {
		//std::ofstream fout(path);
		FILE * fp = fopen(path, "w");
		if(fp == NULL) {
			std::cout << "Error to write: " << path << std::endl;
			return;
		}
		if(this->mtlInfo.size() != 0) {
			fprintf(fp, "%s\n", this->mtlInfo.c_str());
		}
		assert(vs.size()%3==0);
		assert(fs.size()%3==0);
		int nv = vs.size() / 3;
		int nvt = vts.size() / 2;
		int nf = fs.size() / 3;
		int nfvt = fvts.size() / 3;

		for(int i = 0; i < nv; i ++) {
			const double * v = & vs[i*3];
			fprintf(fp, "v %.4f %.4f %.4f\n", v[0], v[1], v[2]);
			//fout << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
		}
		for(int i = 0; i < nvt; i ++) {
			const double * vt = & vts[i*2];
			//fout << "vt " << vt[0] << " " << vt[1] << std::endl;
			fprintf(fp, "vt %.4f %.4f\n", vt[0], vt[1]);
		}
		if(nfvt == 0) {
			for(int i = 0; i < nf; i ++) {
				const int * f = & fs[i*3];
				//fout << "f " << f[0] << " " << f[1] << " " << f[2] << std::endl;
				fprintf(fp, "f %d %d %d\n", f[0], f[1], f[2]);
			}
		} else {
			assert(nfvt == nf);
			for(int i = 0; i < nf; i ++) {
				const int * f = & fs[i*3];
				const int * fvt = & fvts[i*3];
				/*fout << "f " << 
					f[0] << "/" << fvt[0] << " " <<
					f[1] << "/" << fvt[1] << " " <<
					f[2] << "/" << fvt[2] << std::endl;
					*/
				fprintf(fp, "f %d/%d %d/%d %d/%d\n", f[0], fvt[0], f[1], fvt[1], f[2], fvt[2]);
			}
		}
		//fout.close();
		fclose(fp);
	}

	void ObjWriter::addTexCoord(const double * vt, int nvt) {
		for(int i = 0; i < 2*nvt; i ++) {
			vts.push_back(vt[i]);
		}
	}
	void ObjWriter::appendTriCoord(const int * fvt, int nf) {
		for(int i = 0; i < 3*nf; i ++) {
			this->fvts.push_back(fvt[i]);
		}
	}

	void pinocchioSkel2Obj(const char * path, double width) {
		std::ifstream fin(path);
		std::vector<double> vec;
		ObjWriter writer;
		double tmp;
		while(fin >> tmp) {
			vec.push_back(tmp);
		}
		assert(vec.size()%5 == 0);
		int n = vec.size() / 5;
		for(int i = 0; i < n; i ++) {
			assert(vec[i*5+0]==i);
		}

		for(int i = 0; i < n; i ++) {
			int p = (int)vec[i*5+4];
			if(p==-1) {
				continue;
			}
			writer.addBox(&vec[i*5+1], &vec[p*5+1], width);
		}
		fin.close();

		static char tmpPath[1024];
		strcpy(tmpPath, path);
		strcat(tmpPath, ".obj");
		writer.write(tmpPath);

		YInverse : {
			for(int i = 1; i < writer.vs.size(); i += 3) {
				writer.vs[i] = -writer.vs[i];
			}
			strcpy(tmpPath, path);
			strcat(tmpPath, "_YInversed.obj");
			writer.write(tmpPath);
		}

	}

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
		) {
		char path[128];
		sprintf(path, "mitsuba\\%s.xml", fname.c_str());
		FILE * fp = fopen(path, "w");
		if(fp == NULL) {
			printf("Error, cannot open file: %s\n", path);
			return;
		}

	fprintf(fp, "<?xml version='1.0' encoding='utf-8'?>\n");
	fprintf(fp, "\n");
	fprintf(fp, "<scene version=\"0.4.0\">\n");
	fprintf(fp, "	<integrator type=\"adaptive\">\n");
	fprintf(fp, "		<integrator type=\"irrcache\">\n");
	fprintf(fp, "			<integrator type=\"direct\"/>\n");
	fprintf(fp, "		</integrator>\n");
	fprintf(fp, "	</integrator>\n");
	fprintf(fp, "\n");
	fprintf(fp, "	<sensor type=\"perspective\">\n");
	fprintf(fp, "		<float name=\"farClip\" value=\"%.4f\"/>\n", farClip);
	fprintf(fp, "		<float name=\"fov\" value=\"%.4f\"/>\n", fovy);
	fprintf(fp, "		<string name=\"fovAxis\" value=\"y\"/>\n");
	fprintf(fp, "		<float name=\"nearClip\" value=\"%.4f\"/>\n", nearClip);
	fprintf(fp, "		<transform name=\"toWorld\">\n");
	fprintf(fp, "\n");
	fprintf(fp, "			<lookat target=\"%.4f, %.4f, %.4f\" origin=\"%.4f, %.4f, %.4f\" up=\"%.4f, %.4f, %.4f\"/>\n",
		centerx, centery, centerz, eyex, eyey, eyez, upx, upy, upz);
	fprintf(fp, "		</transform>\n");
	fprintf(fp, "\n");
	fprintf(fp, "		<sampler type=\"independent\">\n");
	fprintf(fp, "			<integer name=\"sampleCount\" value=\"64\"/>\n");
	fprintf(fp, "		</sampler>\n");
	fprintf(fp, "\n");
	fprintf(fp, "		<film type=\"hdrfilm\">\n");
	fprintf(fp, "			<integer name=\"height\" value=\"%d\"/>\n", height);
	fprintf(fp, "			<integer name=\"width\" value=\"%d\"/>\n", width);
	fprintf(fp, "\n");
	fprintf(fp, "			<rfilter type=\"gaussian\"/>\n");
	fprintf(fp, "		</film>\n");
	fprintf(fp, "	</sensor>\n");
	for(int i = 0; i < objWriters.size(); i ++) {
		fprintf(fp, "	<shape type=\"obj\">\n");
		fprintf(fp, "		<string name=\"filename\" value=\"meshes/%s_%.5d.obj\"/>\n", fname.c_str(), i);
		if(objWriters[i].texturePath != NULL) {
			fprintf(fp, "		<bsdf type=\"roughplastic\">\n");
			fprintf(fp, "			<texture name='diffuseReflectance' type='bitmap'>\n");
			fprintf(fp, "				<string name=\"filename\" value=\"%s\" />\n", objWriters[i].texturePath);
			fprintf(fp, "			</texture>\n");
			fprintf(fp, "			<rgb name='specularReflectance' value='#FFFFFF'/>\n");
			fprintf(fp, "			<float name='alpha' value='0.2' />\n");
			fprintf(fp, "		</bsdf>\n");
		} else {
			fprintf(fp, "		<bsdf type='roughplastic'>\n");
			fprintf(fp, "			<rgb name='specularReflectance' value='#808080'/>\n");
			fprintf(fp, "			<rgb name='diffuseReflectance' value='#FF0000'/>\n");
			fprintf(fp, "			<float name='alpha' value='0.3' />\n");
			fprintf(fp, "		</bsdf>\n");
		}
		fprintf(fp, "	</shape>\n");
	}
	fprintf(fp, "	<shape type='cube'>\n");
	fprintf(fp, "		<transform name='toWorld'>\n");
	fprintf(fp, "			<scale x='32' y='32' z='0.1' />\n");
	fprintf(fp, "			<translate z='-0.1' />\n");
	fprintf(fp, "		</transform>\n");
	fprintf(fp, "		<bsdf type='diffuse'>\n");
	fprintf(fp, "			<texture type='checkerboard' name='reflectance'>\n");
	fprintf(fp, "				<float name='uvscale' value='32' />\n");
	fprintf(fp, "			</texture>\n");
	fprintf(fp, "		</bsdf>\n");
	fprintf(fp, "	</shape>\n");

	fprintf(fp, "	<emitter type='sky'>\n");
	fprintf(fp, "	</emitter>\n");
	fprintf(fp, "	<emitter type='directional'>\n");
	fprintf(fp, "		<vector name='direction' x='%.8f' y='%.8f'  z='%.8f' />\n", light0Dir[0], light0Dir[1], light0Dir[2]);
	fprintf(fp, "		<spectrum name='irradiance' value='1' />\n");
	fprintf(fp, "	</emitter>\n");
	fprintf(fp, "	<emitter type='directional'>\n");
	fprintf(fp, "		<vector name='direction' x='%.8f' y='%.8f'  z='%.8f' />\n", light1Dir[0], light1Dir[1], light1Dir[2]);
	fprintf(fp, "		<spectrum name='irradiance' value='1' />\n");
	fprintf(fp, "	</emitter>\n");

	fprintf(fp, "</scene>\n");
		fclose(fp);

		for(int i = 0; i < objWriters.size(); i ++) {
			sprintf(path, "mitsuba\\meshes\\%s_%.5d.obj", fname.c_str(),i);
			objWriters[i].write(path);
		}
	}
	#if 0
	std::string getLocalTimeString() {
		static char str[1024];
	#if defined(__APPLE__) && defined(__MACH__)
	#else
		SYSTEMTIME st;
		GetLocalTime(&st);
		sprintf(str, "%d-%.2d-%.2d_%.2d-%.2d-%.2d_%.3d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
	#endif
		return std::string(str);
	}
#endif
};
