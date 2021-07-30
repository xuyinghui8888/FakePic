#ifndef __DEFINEDMACRO_H__
#define __DEFINEDMACRO_H__

#define LOGDIR "../../../log"
#define NAME2STR(name) (#name)
//cut value to b=<a=<c for safe check
#define DLIP3(a,b,c) (std::min(std::max((double)a,(double)b),(double)c)) 
//cut value to b=<a=<c for safe check
#define DLIP3INT(a,b,c) (std::min(std::max(int(a),int(b)),int(c))) 
#define DMIN(a,b) (std::min((double)(a),(double)(b))) 
#define DMINI(a,b) (std::min((int)(a),(int)(b))) 
#define DMAX(a,b) (std::max((double)(a),(double)(b))) 
//线性插值a,b中间插值t=0 为a t = 1 为b
#define LERP2D(a,b,t) (double(a) + double((b - a))*double(t))
//线性插值a,b中间插值t=0 为a t = 1 为b
#define LERP2F(a,b,t) (float(a) + float((b - a))*float(t))
#define CHECKINTRANGE(a, start, end) (int(a)>=int(start) && (int(a)<int(end)))

#define EPSCG3 1e-3
#define EPSCG6 1e-6
#define EPSCG9 1e-9
#define EPSCG18 1e-18
#define NEAR0(a) (std::abs(double(a))<EPSCG18)
#define ABSDIFFD(a,b) (std::abs(double(a-b)))
#define BACKDEPTH -10000
#define MASKCUT 50
#define CGPI 3.1415926535898
#define GETD(a, y, x) (a.at<double>(y,x))
#define SETD(a, y, x, v) (a.at<double>(y,x) = v)
#define GETU(a, y, x) (a.at<unsigned char>(y,x))
#define SETU(a, y, x, v) (a.at<unsigned char>(y,x) = v)
#define GETF(a, y, x) (a.at<float>(y,x))
#define SETF(a, y, x, v) (a.at<float>(y,x) = v)
#define GETD3(a, y, x) (a.at<cv::Vec3d>(y,x))
#define SETD3(a, y, x, v) (a.at<cv::Vec3d>(y,x) = v)
#define GETF3(a, y, x) (a.at<cv::Vec3f>(y,x))
#define SETF3(a, y, x, v) (a.at<cv::Vec3f>(y,x) = v)
#define GETU3(a, y, x) (a.at<cv::Vec3b>(y,x))
#define SETU3(a, y, x, v) (a.at<cv::Vec3b>(y,x) = v)
#define GETU4(a, y, x) (a.at<cv::Vec4b>(y,x))
#define SETU4(a, y, x, v) (a.at<cv::Vec4b>(y,x) = v)
#define EIGENIN(a, y, x) (x>=0 && x<a.cols() && y>=0 && y<a.rows())
#define EIGENOUT(a, y, x) (x<0 || x>=a.cols() ||  y<0 || y>=a.rows())
#define CVIN(a, y, x) (x>=0 && x<a.cols && y>=0 && y<a.rows)
#define CVOUT(a, y, x) (x<0 || x>=a.cols ||  y<0 || y>=a.rows)
#define IMAGEIN(h, w, y, x) (x>=0 && x<w && y>=0 && y<h)
#define IMAGEOUT(h, w, y, x) (!IMAGEIN(h, w, y, x))
#define IMAGEINPAD(h, w, y, x, pad) (x>=pad && x<w-pad && y>=pad && y<h-pad && h-pad>pad && w-pad>pad)
#define IMAGEOUTPAD(h, w, y, x, pad) (!IMAGEINPAD(h, w, y, x))
#define STLX2OUT(a, y, x) (a.empty() || a[0].empty() || y<0 || y>=a.size() ||  x<0 || x>=a[y].size())
#define STLX2IN(a,y,x) (!STLX2OUT(a,y,x))
#define NORM2(x1,y1,x2,y2) (sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))
#define GETCV(a,y,x) (a.at<std::remove_reference<decltype(*a.data)>::type>(y,x))
#define GETCVTYPE(a) (BOOST_TYPEOF(*a.data))
#define SIGN(a) ((0.0<double(a)) - (double(a)<0.0))

#include <Eigen/Eigen>
#include <set>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/filesystem.hpp>

namespace CGP
{
	typedef Eigen::Vector2i int2E;
	typedef Eigen::Vector3i int3E;
	typedef Eigen::Vector4i int4E;
	typedef Eigen::VectorXi vecI;
	typedef Eigen::Vector2f float2E;
	typedef Eigen::Vector3f float3E;
	typedef Eigen::Vector4f float4E;
	typedef Eigen::VectorXf	vecF;
	typedef Eigen::Vector2d double2E;
	typedef Eigen::Vector3d double3E;
	typedef Eigen::Vector4d double4E;
	typedef Eigen::VectorXd	vecD;

	typedef Eigen::Matrix3f mat3f;
	typedef Eigen::Matrix4f mat4f;
	typedef Eigen::Matrix3d mat3d;
	typedef Eigen::Matrix4d mat4d;
	typedef Eigen::Matrix<double, 3, 4> mat34d;
	typedef Eigen::MatrixXi	matI;
	typedef Eigen::MatrixXf	matF;
	typedef Eigen::MatrixXd	matD;

	typedef std::string cstr;

	typedef std::vector<bool> boolVec;
	typedef std::vector<float> floatVec;
	typedef std::vector<int> intVec;
	typedef std::vector<cstr> cstrVec;
	typedef std::vector<unsigned int> uintVec;
	typedef std::vector<unsigned char> ucharVec;
	typedef std::vector<char> charVec;
	typedef std::vector<double> doubleVec;


	typedef std::vector<intVec> intX2Vec;
	typedef std::vector<uintVec> uintX2Vec;
	typedef std::vector<floatVec> floatX2Vec;
	typedef std::vector<doubleVec> doubleX2Vec;
	typedef std::vector<cstrVec> cstrX2Vec;


	typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> float2Vec;
	typedef std::vector<float3E, Eigen::aligned_allocator<float3E>> float3Vec;
	typedef std::vector<vecF, Eigen::aligned_allocator<vecF>> floatXVec;
	typedef std::vector<int2E, Eigen::aligned_allocator<int2E>> int2Vec;
	typedef std::vector<int3E, Eigen::aligned_allocator<int3E>> int3Vec;
	typedef std::vector<int4E, Eigen::aligned_allocator<int4E>> int4Vec;
	typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> double2Vec;
	typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> double3Vec;
	typedef std::vector<double3Vec> double3VecX2;
	typedef std::vector<matI, Eigen::aligned_allocator<matI>> matIVec;
	typedef std::vector<matF, Eigen::aligned_allocator<matF>> matFVec;

	//enum class
	//same as opencv CV_8U serial
	enum class DataType
	{
		UNSIGNED_CHAR,  //0
		CHAR,           //1
		UNSIGNED_SHORT, //2
		SHORT,          //3
		INT,            //4
		FLOAT,          //5 
		DOUBLE,         //6
	};

	//set
	typedef std::set<int> intSet;
	typedef std::set<cstr> cstrSet;

	//map
	typedef std::map<int, int> intMap;

	//pair
	typedef std::pair<int, int> intPair;
	typedef std::vector<intPair> intPairVec;

	//combined
	typedef std::vector<intSet> intSetVec;

	//opencv
	typedef cv::Mat_<uchar> cvMatU;
	typedef cv::Mat_<cv::Vec4b> cvMat4U;
	typedef cv::Mat_<int> cvMatI;
	typedef cv::Mat_<double> cvMatD;
	typedef cv::Mat_<float> cvMatF;
	typedef cv::Mat_<cv::Vec3f> cvMatF3;
	typedef cv::Mat_<cv::Vec3d> cvMatD3;

	//matrix 
	typedef Eigen::Triplet<double> tripD;
	typedef std::vector<tripD> tipDVec;

	//solver
	typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
	typedef Eigen::SimplicialCholesky<SpMat> SpSolver;

	typedef Eigen::ColPivHouseholderQR<Eigen::MatrixXd> DenseSolver;

	//template
	template <class T1, class T2>
	bool nearValue(const T1& src, const T2& bar, double eps = EPSCG9)
	{
		if (src > bar - eps && src < bar + eps)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template <class T1, class T2, class T3>
	double safeDiv(const T1& numerator, const T2& denominator, const T3& default_value)
	{
		if (NEAR0((double)denominator))
		{
			return double(default_value);
		}
		else if (isnan((double)numerator) || isnan((double)denominator))
		{
			return double(default_value);
		}
		else
		{
			return 1.f*numerator/denominator;
		}
	}

	template<typename T, typename U>
	typename std::enable_if<std::is_same<T, U>::value, bool>::type sameType(const T& t, const U& u) {
		return true;
	}
	template<typename T, typename U>
	typename std::enable_if<!std::is_same<T, U>::value, bool>::type sameType(const T& t, const U& u) {
		return false;
	}	
}

#endif