#ifndef geometry_h
#define geometry_h

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iostream>

namespace jie {

#define M_PI		3.14159265358979323846	/* pi */
#define sqr(v) ((v)*(v))

const double eps = 1E-6;
//0 if d is near to 0. 1 if d>0. -1 if d<0
int sig(double d);
//min(max(d,a),b)
double clamp(double d, double a, double b);
//acos(clamp(d,-1,1));

static double myMin(double a, double b){    
  return (a < b)?a : b; 
}

double myAcos(double d);
//asin(clamp(d,-1,1));
double myAsin(double d);
//sqrt(max(d,0))
double mySqrt(double d);
//a's period is 2*b. to mod a into [-b,b)
double myFmod(double a, double b);
//generate a random double between 0 and 1
double randDouble();
//generate a random double between minVal and maxVal
double randDouble(double minVal, double maxVal);
//cot(d)
double cot(double d);
//normal distribution
double normal(double mu, double sigma, double x);
//Point class
class Point {
public:
	double x, y, z;
	Point(double x=0, double y=0, double z=0);
	Point(const double p[3]);
	Point(const float p[3]);
	Point(Point a, Point b);
	Point operator + (const Point & p) const;
	Point operator - (const Point & p) const;
	Point operator * (const double d) const;
	Point operator / (const double d) const;
	Point operator - () const;
	Point resize(double d) const;
	Point normalize() const;
	double len() const;
	bool operator < (const Point & p) const;
	bool operator == (const Point & p) const;
	void copyTo(double arr[3]) const;
	void copyTo(float arr[3]) const;
	void output() const;

	static Point input();
	static Point rand();
};
//Face class
class Face {
public:
	Point a, b, c;
	Face(Point a, Point b, Point c);
	Point normal() const;
	bool sameSide(Point q , Point  p) const;
	bool inFace(Point q) const;
	bool operator == (const Face & face) const;
};
//dot product
double dot(const Point & a, const Point & b);
double dot(const Point & o, const Point & a, const Point & b);
//cross product
Point cross(const Point & a, const Point & b);
Point cross(const Point & o, const Point & a, const Point & b);
//two points distance
double dis(const Point & a, const Point & b);
//point to line distance
double dis(const Point & o, const Point & a, const Point & b);
//point to face distance
double dis(const Face & f, const Point & p);
//line to line distance
double dis(const Point & a, const Point & b, const Point & c, const Point & d);
//the project point from Point o to Line (a,b)
Point pointToLine(const Point & o, const Point & a, const Point & b);
//whether three points are on the same line
bool sameLine(const Point & a, const Point & b, const Point & c);
//whether four points are on the same face
bool sameFace(const Point & a, const Point & b, const Point & c, const Point & d);
//rotate vector r across vector n by radian theta
Point rotate(Point r, Point n, double theta);
//rotate r across line(n1,n2) by radian theta
Point rotate(Point r, Point n1, Point n2, double theta);
//rotate vec to axis z, where will p going to ?
Point rotateToZ(Point p, Point vec);
//rotate axis z to vec, where will p going to ?
Point rotateFromZ(Point p, Point vec);
//whether line(a,b) and line(c,d) are parallel
bool parallel(const Point & a, const Point & b, const Point & c, const Point & d);
//whether face f and line(a,b) are parallel
bool parallel(const Face & f, const Point & a, const Point & b);
//whether face f1 and face f2 are parallel
bool parallel(const Face & f1, const Face & f2);
//whether line(a,b) and line(c,d) are perpendicular
bool perpendicular(const Point & a, const Point & b, const Point & c, const Point & d);
//whether face f and line(a,b) are perpendicular
bool perpendicular(const Face & f, const Point & a, const Point & b);
//whether face f1 and face f2 are perpendicular
bool perpendicular(const Face & f1, const Face & f2);
//the intersection point between face f and line(a,b). ASSUME f isn't parallel to (a,b)
Point intersect(const Face & f, const Point & a, const Point & b);
//the intersection point between line(a,b) and line(c,d). ASSUME ab and cd are in the same face and not parallel
Point intersect(const Point & a, const Point & b, const Point & c, const Point & d);
//the intersection line between face f1 and f2. ASSUME f1 isn't parallel to f2
void intersect(const Face & f1, const Face & f2, Point & p1, Point & p2);
//common normal of line(a,b) and line(c,d). p1 will be on (a,b), p2 will be on(c,d)
void commonNormal(const Point & a, const Point & b, const Point & c, const Point & d, Point & p1, Point & p2);
//whether o is in triangle(a,b,c), edges included. ASSUME a/b/c are not on the same line and a/b/c/d are on the same face
bool inTriangle(const Point & o, const Point & a, const Point & b, const Point & c);
//whether o is on segment(a,b); ASSUME oab is on the same line
//-1: o is inside ab
//0:  o is a or b
//1:  o is outside ab
int onSeg(const Point & o, const Point & a, const Point & b);
//Sagittal plane divide p1 and p2
Face sagittalPlane(Point p1, Point p2);
//we want to rotate segment(a1,a2) to segment(b1,b2). What is the rotation line(p1,p2) and angle ang
//return true: if we can find the answer
//assume |a1,a2| = |b1,b2|
bool angle(Point a1, Point a2, Point b1, Point b2, Point & p1, Point & p2, double & ang);
//the angle between v1 and v2
double angle(Point v1, Point v2);
//return the component of v, that parrel to axis
Point compParrel(Point v, Point axis);
//return the component of v, that perpendicular to axis
Point compOtho(Point v, Point axis);
//solve a*sin(theta)+b*cos(theta)=c
//assume it have solution.
//return theta
double solveSC(double a, double b, double c);


/*
void transpose(double mat[3][3]);
void Point2ColMat(const Point & a, const Point & b, const Point &c, double mat[3][3]);
void Point2RowMat(const Point & a, const Point & b, const Point &c, double mat[3][3]);
void mulMat(const double a[3][3], const double b[3][3], double c[3][3]);
Point mulMatPoint(const double mat[3][3], Point p);
//getMatrix from rotate vector
//we want an orthogonal matrix A, such that v1=A*u1, v2=A*u2.
//assume u1/u2 are perpendicular
//assume v1/v2 are perpendicular
void getMatrix(Point u1, Point u2, Point v1, Point v2, double A[3][3]);

void rotation3dToEulerAngles(const double mat[3][3], double & thetaZ, double & thetaY, double &thetaX);
void getEulerAngle(Point u1, Point u2, Point v1, Point v2, double & thetaZ, double & thetaY, double & thetaX);
*/
void orthoMat2Axis(Point & axis, double & angle, const double val[3][3]);
struct Mat3 {
	double val[3][3];
	Mat3();
	Mat3(
		double v00, double v01, double v02,
		double v10, double v11, double v12,
		double v20, double v21, double v22
		);
	Mat3(const double val[3][3]);
	Mat3 operator + (const Mat3 & mat) const;
	Mat3 operator - (const Mat3 & mat) const;
	Mat3 operator * (const double d) const;
	Mat3 operator * (const Mat3 & mat) const;
	Point operator * (const Point & point) const;
	void orthoMat2Axis(Point & axis, double & angle) const;

	bool operator == (const Mat3 & mat) const;
	Point operator()(int idx) const;
	double & operator() (int row, int col);
	Mat3 transpose() const;
	void copyTo(double val[3][3]);
	void output() const;
	static Mat3 eye();
	static Mat3 Point2RowMat3(const Point & a, const Point & b, const Point & c);
	static Mat3 Point2ColMat3(const Point & a, const Point & b, const Point & c);
	static Mat3 input();
	static Mat3 rand();
	//euler angles related
	static Mat3 createRotationOx(double thetaX);
	static Mat3 createRotationOy(double thetaY);
	static Mat3 createRotationOz(double thetaZ);
	static Mat3 eulerAnglesToRotation3d(double thetaZ, double thetaY, double thetaX);
	void rotation3dToEulerAngles(double & thetaZ, double & thetaY, double & thetaX);
	//other angles
	static Mat3 createRotation3dLineAngle(const Point & axis, const double & angle);
};

struct Quaternion {
	Point v;
	double s;
	Quaternion();
	Quaternion(double s, double a, double b, double c);
	Quaternion(double s, Point v);
	Quaternion operator + (const Quaternion & q) const;
	Quaternion operator - (const Quaternion & q) const;
	Quaternion operator - () const;
	Quaternion operator * (const Quaternion & q) const;
	Quaternion operator * (const double d) const;
	Quaternion operator / (const double d) const;
	Quaternion conjugate() const;
	Quaternion normalize() const;
	Point rotate(Point r);
	void decomposeRotation(Point & axis, double & angle) const;
	void decomposeRotation(double mat[3][3]) const;
	double norm() const;
	static Quaternion createRotation(Point axis, double angle);
	static Quaternion createRotation(const double mat[3][3]);
	void output();
};
double dot(const Quaternion & a, const Quaternion & b);

struct DQ {
	Quaternion real, dual;
	DQ();
	DQ(Quaternion real, Quaternion dual);
	DQ(const jie::Point & axis, const double angle, const jie::Point & translation);
	DQ(double realS, Point realV, double dualS, Point dualV);
	void decompose(jie::Point & axis, double & angle, jie::Point & translation) const;
	DQ conjugate2() const;
	DQ operator + (const DQ & dq) const;
	DQ operator - (const DQ & dq) const;
	DQ operator - () const;
	DQ operator * (const DQ & dq) const;
	DQ operator * (const double d) const;
	Point transform(Point r) const;
};
};//namespace Jie

#endif