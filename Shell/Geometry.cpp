#include "geometry.h"


namespace jie {

////////////////////// general functions ////////////////////////////

//0 if d is near to 0. 1 if d>0. -1 if d<0
int sig(double d) {
	return (d>eps) - (d<-eps);
}
//min(max(d,a),b)
double clamp(double d, double a, double b) {
	d = std::max(d, a);
	d = std::min(d, b);
	return d;
}
//acos(clamp(d,-1,1));
double myAcos(double d) {
	return acos(clamp(d,-1,1));
}
//asin(clamp(d,-1,1));
double myAsin(double d) {
	return asin(clamp(d,-1,1));
}
double mySqrt(double d) {
	return sqrt(std::max(d,0.0));
}
//a's period is 2*b. to mod a into [-b,b)
double myFmod(double a, double b) {
	int k = (int)ceil(0.5-a/(2.0*b))-1;
	return a + k * 2 * b;
}
//generate a random double between 0 and 1
double randDouble() {
	return (double)rand() / RAND_MAX;
}
//generate a random double between minVal and maxVal
double randDouble(double minVal, double maxVal) {
	return randDouble() * (maxVal-minVal) + minVal;
}
//cot(d)
double cot(double d) {
	return 1/tan(d);
}
//normal distribution
double normal(double mu, double sigma, double x) {
	static double sqrt2PI = sqrt(2*M_PI);
	return exp(-sqr(x-mu)/2/sqr(sigma)) / (sigma*sqrt2PI);
}
////////////////////// Point member		/////////////////////////////
Point::Point(double x, double y, double z) : x(x), y(y), z(z) {}
Point::Point(const double p[3]) : x(p[0]), y(p[1]), z(p[2]) {}
Point::Point(const float p[3]) : x(p[0]), y(p[1]), z(p[2]) {}
Point::Point(Point a, Point b): x(b.x-a.x), y(b.y-a.y), z(b.z-a.z) {}
Point Point::operator + (const Point & p) const {
	return Point(x+p.x, y+p.y, z+p.z);
}
Point Point::operator - (const Point & p) const {
	return Point(x-p.x, y-p.y, z-p.z);
}
Point Point::operator * (const double d) const {
	return Point(x*d, y*d, z*d);
}
Point Point::operator / (const double d) const {
	return Point(x/d, y/d, z/d);
}
Point Point::operator-() const {
	return Point(-x,-y,-z);
}
Point Point::resize(double d) const {
	d /= mySqrt(x*x+y*y+z*z);
	return Point(x*d, y*d, z*d);
}
Point Point::normalize() const {
	return resize(1);
};
double Point::len() const {
	return mySqrt(x*x+y*y+z*z);
}
bool Point::operator < (const Point & p) const {
	return sig(x-p.x)!=0 ? x<p.x : sig(y-p.y)!=0 ? y<p.y : sig(z-p.z)<0;
}
bool Point::operator == (const Point & p) const {
	return sig(x-p.x)==0 && sig(y-p.y)==0 && sig(z-p.z)==0;
}

void Point::copyTo(double arr[3]) const {
	arr[0] = x;
	arr[1] = y;
	arr[2] = z;
}
void Point::copyTo(float arr[3]) const {
	arr[0] = x;
	arr[1] = y;
	arr[2] = z;
}
void Point::output() const {
	printf("\t%.12f, %.12f, %.12f\t", x, y, z);
}
Point Point::input() {
	double x, y, z;
	scanf("%lf%lf%lf", &x, &y, &z);
	return Point(x,y,z);
}
Point Point::rand() {
	double x = randDouble();
	double y = randDouble();
	double z = randDouble();
	return Point(x,y,z);
}

/////////////////////////////Face member //////////////////////////////////
Face::Face(Point a, Point b, Point c) : a(a), b(b), c(c) {}
Point Face::normal() const {
	return cross(a, b, c);
}
bool Face::sameSide(Point q , Point  p) const {
	return sig ( dot(a - q, cross(q, b, c)) 
		* dot(a - p , cross(p, b, c)) ) > 0 ;
}
bool Face::inFace(Point q) const {
	return sameFace(a, b, c, q);
}
bool Face::operator == (const Face & face) const {
	Point fa1 = normal();
	Point fa2 = face.normal();
	if(sig(cross(fa1,fa2).len())!=0)    return false;
	return inFace(face.a);
}

////////////////////// Other functions		/////////////////////////////
//dot product
double dot(const Point & a, const Point & b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
double dot(const Point & o, const Point & a, const Point & b) {
	return dot(a-o, b-o);
}
//cross product
Point cross(const Point & a, const Point & b) {
	return Point(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
Point cross(const Point & o, const Point & a, const Point & b) {
	return cross(a-o,b-o);
}
//two points distance
double dis(const Point & a, const Point & b) {
	return mySqrt(sqr(a.x-b.x) + sqr(a.y-b.y) + sqr(a.z-b.z));
}
//point to line distance
double dis(const Point & o, const Point & a, const Point & b) {
	return cross(o, a, b).len() / dis(a,b);
}
//point to face distance
double dis(const Face & f, const Point & p) {
	Point fa = f.normal();
	return fabs(dot(fa, f.a-p) / fa.len());
}
//line to line distance
double dis(const Point & a, const Point & b, const Point & c, const Point & d) {
	Point fa = cross(b-a,d-c);
	if(fa.len() == 0) {	//the two line are parallel
		return cross(c, a, b).len() / dis(a, b);
		return dis(c, a, b);		//can also use this guy
	} else {
		return fabs(dot(c-a, fa) / fa.len());
	}
}
//the project point from Point o to Line (a,b)
Point pointToLine(const Point & o, const Point & a, const Point & b) {
	Point fa = cross(o, a, b);	//oab的垂向量
	Point vec = cross(b-a, fa);	//vec为在oab平面上，且由o指向ab的一条向量
	return o + vec.resize(fa.len()/dis(a,b));
}
//whether three points are on the same line
bool sameLine(const Point & a, const Point & b, const Point & c) {
	return sig(cross(a, b, c).len())==0;
}
//whether four points are on the same face
bool sameFace(const Point & a, const Point & b, const Point & c, const Point & d) {
	return sig(dot(b-a, cross(a, c, d))) == 0;
}
//rotate vector r across vector n by radian theta
Point rotate(Point r, Point n, double theta) {
	n = n.resize(1);
	double c=cos(theta), s=sin(theta);
	return r*c + n*dot(n,r)*(1-c) + cross(n,r)*s;
}
//rotate r across line(n1,n2) by radian theta
Point rotate(Point r, Point n1, Point n2, double theta) {
	return n1 + rotate(r-n1, n2-n1, theta);
}
//rotate vec to axis z, where will p going to ?
Point rotateToZ(Point p, Point vec) {
	vec = vec.resize(1);
	Point z(0, 0, 1);
	Point fa = cross(vec, z);
	if(sig(fa.len())==0)	return p;
	return rotate(p, fa, myAcos( dot(z,vec) ));
}
//rotate axis z to vec, where will p going to ?
Point rotateFromZ(Point p, Point vec) {
	vec = vec.resize(1);
	Point z(0, 0, 1);
	Point fa = cross(z, vec);	//only here changed
	if(sig(fa.len())==0)	return p;
	return rotate(p, fa, myAcos( dot(z,vec) ));
}
//whether line(a,b) and line(c,d) are parallel
bool parallel(const Point & a, const Point & b, const Point & c, const Point & d) {
	return sig(cross(b-a, d-c).len()) == 0;
}
//whether face f and line(a,b) are parallel
bool parallel(const Face & f, const Point & a, const Point & b) {
	return sig(dot(f.normal(), b-a))==0;
}
//whether face f1 and face f2 are parallel
bool parallel(const Face & f1, const Face & f2) {
	return sig(cross(f1.normal(),f2.normal()).len()) == 0;
}
//whether line(a,b) and line(c,d) are perpendicular
bool perpendicular(const Point & a, const Point & b, const Point & c, const Point & d) {
	return sig(dot(b-a, d-a)) == 0;
}
//whether face f and line(a,b) are perpendicular
bool perpendicular(const Face & f, const Point & a, const Point & b) {
	return sig(cross(f.normal(), b-a).len()) == 0;
}
//whether face f1 and face f2 are perpendicular
bool perpendicular(const Face & f1, const Face & f2) {
	return sig(dot(f1.normal(), f2.normal())) == 0;
}

//the intersection point between face f and line(a,b). ASSUME f isn't parallel to (a,b)
Point intersect(const Face & f, const Point & a, const Point & b) {
	Point fa = f.normal();
	double t = dot(fa,f.a-a) / dot(fa,b-a);
	return a+(b-a)*t;
}
//the intersection point between line(a,b) and line(c,d). ASSUME ab and cd are in the same face and not parallel
Point intersect(const Point & a, const Point & b, const Point & c, const Point & d) {
	Point e = d + cross(a-b, c-d);
	return intersect(Face(c,d,e), a, b);
	//or use this guy.
	Point fa = cross(c, d, e);
	double t = dot(fa, c-a) / dot(fa, b-a);
	return a+(b-a) * t;
}
//the intersection line between face f1 and f2. ASSUME f1 isn't parallel to f2
void intersect(const Face & f1, const Face & f2, Point & p1, Point & p2) {
	p1 = false==parallel(f2, f1.a, f1.b) ? intersect(f2, f1.a, f1.b) : intersect(f2, f1.b, f1.c);
	p2 = p1 + cross(f1.normal(), f2.normal());
}
//common normal of line(a,b) and line(c,d). p1 will be on (a,b), p2 will be on(c,d)
void commonNormal(const Point & a, const Point & b, const Point & c, const Point & d, Point & p1, Point & p2) {
	Point e = d + cross(a-b, c-d);
	p1 = intersect(Face(c,d,e), a, b);
	p2 = pointToLine(p1, c, d);
}
//whether o is in triangle(a,b,c), edges included. ASSUME a/b/c are not on the same line and a/b/c/d are on the same face
bool inTriangle(const Point & o, const Point & a, const Point & b, const Point & c) {
	double s = cross(a, b, c).len();
	double s1 = cross(o, a, b).len();
	double s2 = cross(o, b, c).len();
	double s3 = cross(o, a, c).len();
//	return sig(s - s1 - s2 - s3) == 0;
	return sig(s-s1-s2-s3)==0 && sig(s1)==1 && sig(s2)==1 && sig(s3)==1;//不包括边界的版本
}
//whether o is on segment(a,b); ASSUME oab is on the same line
//-1: o is inside ab
//0:  o is a or b
//1:  o is outside ab
int onSeg(const Point & o, const Point & a, const Point & b) {
	return sig(dot(o, a, b));
}
//Sagittal plane divide p1 and p2
Face sagittalPlane(Point p1, Point p2) {//返回p1和p2的中垂面
	Point c = (p1+p2)*0.5;
	Point a = c + rotateFromZ(Point(1,0,0), p2-p1);
	Point b = c + rotateFromZ(Point(0,1,0), p2-p1);
	return Face(a, b, c);
}
//we want to rotate segment(a1,a2) to segment(b1,b2). What is the rotation line(p1,p2) and angle ang
//return true: if we can find the answer
//assume |a1,a2| = |b1,b2|
bool angle(Point a1, Point a2, Point b1, Point b2, Point & p1, Point & p2, double & ang) {
	if(sig(cross(a2-a1, b2-b1).len())==0)	return false;		//平行
	
	if(sig(cross(a1-b1,a2-b2).len())==0) {		//八字形
		p1 = intersect(a1,a2,b1,b2);
		p2 = p1 + cross(a2-a1, b2-b1);
	} else {
		intersect(sagittalPlane(a1,b1),sagittalPlane(a2,b2),p1,p2);
	}
	Point a, b;
	if(dis(a1,b1)>dis(a2,b2))	a=a1,b=b1;
	else						a=a2,b=b2;
	Point o, tmp;
	commonNormal(p1,p2,a,b,o,tmp);
	Point oa=(a-o).resize(1), ob=(b-o).resize(1);
	ang = myAcos( dot(oa,ob) );
	if(dot(p2-p1,cross(oa,ob))<0)	std::swap(p1,p2);
	return true;
}
//the angle between v1 and v2
double angle(Point v1, Point v2) {
	return myAcos(dot(v1.normalize(), v2.normalize()));
}
//return the component of v, that parrel to axis
Point compParrel(Point v, Point axis) {
	axis = axis.normalize();
	return axis * dot(v,axis);
}
//return the component of v, that perpendicular to axis
Point compOtho(Point v, Point axis) {
	return v - compParrel(v,axis);
}
//solve a*sin(theta)+b*cos(theta)=c
//assume it have solution.
//return theta. (if it has multiple solution, return the one with min abs value)
double solveSC(double a, double b, double c) {
	double phi = atan2(b, a);
	double psi1 = myAsin(c / sqrt(a*a+b*b));
	double psi2 = M_PI - psi1;

	double theta1 = myFmod(psi1 - phi, M_PI);
	double theta2 = myFmod(psi2 - phi, M_PI);

	return fabs(theta1)<fabs(theta2) ? theta1 : theta2;
}

/*
void transpose(double mat[3][3]) {
	std::swap(mat[0][1], mat[1][0]);
	std::swap(mat[0][2], mat[2][0]);
	std::swap(mat[1][2], mat[2][1]);
}
void Point2ColMat(const Point & a, const Point & b, const Point &c, double mat[3][3]) {
	mat[0][0]=a.x;	mat[0][1]=b.x;	mat[0][2]=c.x;
	mat[1][0]=a.y;	mat[1][1]=b.y;	mat[1][2]=c.y;
	mat[2][0]=a.z;	mat[2][1]=b.z;	mat[2][2]=c.z;
}
void Point2RowMat(const Point & a, const Point & b, const Point &c, double mat[3][3]) {
	mat[0][0]=a.x;	mat[0][1]=a.y;	mat[0][2]=a.z;
	mat[1][0]=b.x;	mat[1][1]=b.y;	mat[1][2]=b.z;
	mat[2][0]=c.x;	mat[2][1]=c.y;	mat[2][2]=c.z;
}
void mulMat(const double a[3][3], const double b[3][3], double c[3][3]) {
	for(int i = 0; i < 3; i ++) {
		for(int j = 0; j < 3; j ++) {
			c[i][j] = 0;
			for(int k = 0; k < 3; k ++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}
Point mulMatPoint(const double mat[3][3], Point p) {
	double arr[3], res[3];
	p.copyTo(arr);

	for(int i = 0; i < 3; i ++) {
		res[i] = 0;
		for(int j = 0; j < 3; j ++) {
			res[i] += mat[i][j] * arr[j];
		}
	}
	return Point(res);
}


//getMatrix from rotate vector
//we want an orthogonal matrix A, such that v1 is parrael to A*u1, and plane(v1,v2) are parrael to plane(A*u1,A*u2)
void getMatrix(Point u1, Point u2, Point v1, Point v2, double A[3][3]) {
	u1 = u1.normalize();
	v1 = v1.normalize();

	Point u3 = cross(u1,u2).normalize();
	Point v3 = cross(v1,v2).normalize();

	u2 = cross(u3, u1);
	v2 = cross(v3, v1);

	double V[3][3], UT[3][3];
	Point2ColMat(v1, v2, v3, V);
	Point2RowMat(u1, u2, u3, UT);
	mulMat(V, UT, A);
}

void rotation3dToEulerAngles(const double mat[3][3], double & thetaZ, double & thetaY, double &thetaX) {
	double cy = hypot(mat[0][0], mat[1][0]);
	if(sig(cy) > 0) {
		thetaX = atan2( mat[2][1], mat[2][2]);
		thetaY = atan2(-mat[2][0], cy);
		thetaZ = atan2( mat[1][0], mat[0][0]);
	} else {
		thetaX = atan2(-mat[1][2], mat[1][1]);
		thetaY = atan2(-mat[2][0], cy);
		thetaZ = 0;
	}
}
void getEulerAngle(Point u1, Point u2, Point v1, Point v2, double & thetaZ, double & thetaY, double & thetaX) {
	double mat[3][3];
	getMatrix(u1, u2, v1, v2, mat);
	rotation3dToEulerAngles(mat, thetaZ, thetaY, thetaX);
}
*/
void orthoMat2Axis(Point & axis, double & angle, const double val[3][3]) {
	double c = (val[0][0]+val[1][1]+val[2][2]-1)/2;
	angle = jie::myAcos(c);
	if(sig(angle)==0) {
		angle = 0;
		axis = Point(1,0,0);
	}
  else if(sig(angle-M_PI)==0) {
		axis.x = mySqrt( (val[0][0]+1)/2 );
		axis.y = mySqrt( (val[1][1]+1)/2 );
		axis.z = mySqrt( (val[2][2]+1)/2 );

		if(sig(axis.x) != 0) {
			if(val[1][0] < 0) {
				axis.y = - axis.y;
			}
			if(val[2][0] < 0) {
				axis.z = - axis.z;
			}
		} else {
			axis.x = 0;
			if(sig(axis.y) != 0) {
				if(val[2][1] < 0) {
					axis.z = -axis.z;
				}
			} else {
				axis.y = 0;
			}
		}
	}
  else {
		axis = Point(val[2][1]-val[1][2], val[0][2]-val[2][0], val[1][0]-val[0][1]);
		axis = axis*(1.0/(2.0*sin(angle)));
	}
	if(sig(axis.len() != 0) ) {
		axis = axis.normalize();
	}
}

Mat3::Mat3() {
		memset(val, 0, sizeof(val));
	}
	Mat3::Mat3(
		double v00, double v01, double v02,
		double v10, double v11, double v12,
		double v20, double v21, double v22
		) {
		val[0][0]=v00;	val[0][1]=v01;	val[0][2]=v02;
		val[1][0]=v10;	val[1][1]=v11;	val[1][2]=v12;
		val[2][0]=v20;	val[2][1]=v21;	val[2][2]=v22;
	}
	Mat3::Mat3(const double val[3][3]) {
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				this->val[i][j] = val[i][j];
			}
		}
	}
	Mat3 Mat3::operator + (const Mat3 & mat) const {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = this->val[i][j] + mat.val[i][j];
			}
		}
		return Mat3(val);
	}
	Mat3 Mat3::operator - (const Mat3 & mat) const {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = this->val[i][j] - mat.val[i][j];
			}
		}
		return Mat3(val);
	}
	Mat3 Mat3::operator * (const double d) const {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = this->val[i][j] * d;
			}
		}
		return jie::Mat3(val);
	}
	Mat3 Mat3::operator * (const Mat3 & mat) const {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = 0;
				for(int k = 0; k < 3; k ++) {
					val[i][j] += this->val[i][k] * mat.val[k][j];
				}
			}
		}
		return Mat3(val);
	}
	Point Mat3::operator * (const Point & point) const {
		double arr[3], res[3];
		point.copyTo(arr);
		for(int i = 0; i < 3; i ++) {
			res[i] = 0;
			for(int j = 0; j < 3; j ++) {
				res[i] += val[i][j] * arr[j];
			}
		}
		return Point(res);
	}
	void Mat3::orthoMat2Axis(Point & axis, double & angle) const {
		jie::orthoMat2Axis(axis, angle, val);
	}
	bool Mat3::operator == (const Mat3 & mat) const {
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				if(sig(this->val[i][j]-mat.val[i][j]) != 0) {
					return false;
				}
			}
		}
		return true;
	}
	Point Mat3::operator()(int idx) const {
		return Point(val[idx]);
	}
	double & Mat3::operator() (int row, int col) {
		return val[row][col];
	}
	Mat3 Mat3::transpose() const {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = this->val[j][i];
			}
		}
		return Mat3(val);
	}
	void Mat3::copyTo(double val[3][3]) {
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = this->val[i][j];
			}
		}
	}
	void Mat3::output() const {
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				printf("%.6f\t", val[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	Mat3 Mat3::eye() {
		return Mat3(
			1,0,0,
			0,1,0,
			0,0,1);
	}
	Mat3 Mat3::Point2RowMat3(const Point & a, const Point & b, const Point & c) {
		return Mat3(
			a.x, a.y, a.z,
			b.x, b.y, b.z,
			c.x, c.y, c.z
			);
	}
	Mat3 Mat3::Point2ColMat3(const Point & a, const Point & b, const Point & c) {
		return Mat3(
			a.x, b.x, c.x,
			a.y, b.y, c.y,
			a.z, b.z, c.z
			);
	}
	Mat3 Mat3::input() {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				scanf("%lf", &val[i][j]);
			}
		}
		return Mat3(val);
	}
	Mat3 Mat3::rand() {
		double val[3][3];
		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				val[i][j] = jie::randDouble();
			}
		}
		return Mat3(val);
	}
	Mat3 Mat3::createRotationOx(double thetaX) {
		double s = sin(thetaX);
		double c = cos(thetaX);
		return Mat3(
			1, 0, 0,
			0, c, -s,
			0, s, c
			);
	}
	Mat3 Mat3::createRotationOy(double thetaY) {
		double s = sin(thetaY);
		double c = cos(thetaY);
		return Mat3(
			c, 0, s,
			0, 1, 0,
			-s,0, c
		);
	}
	Mat3 Mat3::createRotationOz(double thetaZ) {
		double s = sin(thetaZ);
		double c = cos(thetaZ);
		return Mat3(
			c, -s, 0,
			s,  c, 0,
			0,  0, 1
			);
	}
	Mat3 Mat3::eulerAnglesToRotation3d(double thetaZ, double thetaY, double thetaX) {
		double cz = cos(thetaZ);
		double sz = sin(thetaZ);
		double cy = cos(thetaY);
		double sy = sin(thetaY);
		double cx = cos(thetaX);
		double sx = sin(thetaX);
		return Mat3(
			cy*cz,	cz*sx*sy-cx*sz,	cx*cz*sy+sx*sz,
			cy*sz,	cx*cz+sx*sy*sz,	-cz*sx+cx*sy*sz,
			-sy,	cy*sx,			cx*cy
			);
	}
	void Mat3::rotation3dToEulerAngles(double & thetaZ, double & thetaY, double & thetaX) {
		double cy = hypot(val[0][0], val[1][0]);
		if(sig(cy) > 0) {
			thetaX = atan2( val[2][1], val[2][2]);
			thetaY = atan2(-val[2][0], cy);
			thetaZ = atan2( val[1][0], val[0][0]);
		} else {
			thetaX = atan2(-val[1][2], val[1][1]);
			thetaY = atan2(-val[2][0], cy);
			thetaZ = 0;
		}
	}
	Mat3 Mat3::createRotation3dLineAngle(const Point & axis, const double & angle) {
		double x = axis.x;
		double y = axis.y;
		double z = axis.z;
		double s = sin(angle);
		double c = cos(angle);
		return Mat3(
			c+x*x*(1-c),	x*y*(1-c)-z*s,	x*z*(1-c)+y*s,
			y*x*(1-c)+z*s,	c+y*y*(1-c),	y*z*(1-c)-x*s,
			z*x*(1-c)-y*s,	z*y*(1-c)+x*s,	c+z*z*(1-c)
			);
	}
	Quaternion::Quaternion() {
		this->s = 0;
		this->v = Point(0,0,0);
	}
	Quaternion::Quaternion(double s, double a, double b, double c) {
		this->s = s;
		this->v = Point(a,b,c);
	}
	Quaternion::Quaternion(double s, Point v) {
		this->s = s;
		this->v = v;
	}
	Quaternion Quaternion::operator + (const Quaternion & q) const {
		return Quaternion(this->s+q.s, this->v+q.v);
	}
	Quaternion Quaternion::operator - (const Quaternion & q) const {
		return Quaternion(this->s-q.s, this->v-q.v);
	}
	Quaternion Quaternion::operator - () const {
		return Quaternion(-this->s, -this->v);
	}
	Quaternion Quaternion::operator * (const Quaternion & q) const {
		return Quaternion(this->s*q.s-dot(this->v,q.v), q.v*this->s+this->v*q.s+cross(this->v,q.v));
	}
	Quaternion Quaternion::operator * (const double d) const {
		return Quaternion(this->s*d, this->v*d);
	}
	Quaternion Quaternion::operator / (const double d) const {
		return Quaternion(this->s/d, this->v/d);
	}
	Quaternion Quaternion::conjugate() const {
		return Quaternion(s, -v);
	}
	Quaternion Quaternion::normalize() const {
		return (*this) / this->norm();
	}
	Point Quaternion::rotate(Point r) {
		//return ((*this) * Quaternion(0,r) * this->conjugate()).v;
		return r + cross(v, cross(v,r)+r*s)*2;
	}
	void Quaternion::decomposeRotation(Point & axis, double & angle) const {
		if(s >= 0) {
			angle = acos(s)*2;
			if(sig(angle)==0) {
				axis = jie::Point(0,0,0);
			} else {
				axis = v / sin(angle/2);
			}
		} else {
			angle = acos(-s)*2;
			axis = (-v) / sin(angle/2);
		}
	}
	void Quaternion::decomposeRotation(double mat[3][3]) const {
		jie::Point axis;
		double angle;
		this->decomposeRotation(axis, angle);
		jie::Mat3::createRotation3dLineAngle(axis, angle).copyTo(mat);
	}
	double Quaternion::norm() const {
		return sqrt(s*s+v.x*v.x+v.y*v.y+v.z*v.z);
	}
	Quaternion Quaternion::createRotation(Point axis, double angle) {
		if(sig(axis.len()) == 0) {
			axis = jie::Point(0,0,0);
		}
		return Quaternion(cos(angle/2), axis*sin(angle/2));
	}
	Quaternion Quaternion::createRotation(const double mat[3][3]) {
		jie::Point axis;
		double angle;
		jie::orthoMat2Axis(axis, angle, mat);
		/*
		std::cout << "axis = "; axis.output();	std::cout << std::endl;
		std::cout << "angle = " << angle << std::endl;
		*/
		return createRotation(axis,angle);
	}
	void Quaternion::output() {
		printf("(%.2f, %.2f %.2f %.2f)", s, v.x, v.y, v.z);
	}
	double dot(const Quaternion & a, const Quaternion & b) {
		return a.s*b.s+a.v.x*b.v.x+a.v.y*b.v.y+a.v.z*b.v.z;
	}
	DQ::DQ() {
		this->real = Quaternion();
		this->dual = Quaternion();
	}
	DQ::DQ(Quaternion real, Quaternion dual) {
		this->real = real;
		this->dual = dual;
	}
	DQ::DQ(const jie::Point & axis, const double angle, const jie::Point & translation) {
		real = Quaternion::createRotation(axis, angle);
		dual = Quaternion(0,translation)*real/2;
	}
	DQ::DQ(double realS, Point realV, double dualS, Point dualV) {
		this->real = Quaternion(realS, realV);
		this->dual = Quaternion(dualS, dualV);
	}
	void DQ::decompose(jie::Point & axis, double & angle, jie::Point & translation) const {
		real.decomposeRotation(axis, angle);
		translation = ( (dual)*(real.conjugate())*2 ).v;
	}
	DQ DQ::conjugate2() const {
		return DQ(real.conjugate(), -dual.conjugate());
	}
	DQ DQ::operator + (const DQ & dq) const {
		return DQ(this->real+dq.real, this->dual+dq.dual);
	}
	DQ DQ::operator - (const DQ & dq) const {
		return DQ(this->real-dq.real, this->dual-dq.dual);
	}
	DQ DQ::operator - () const {
		return DQ(-real, -dual);
	}
	DQ DQ::operator * (const DQ & dq) const {
		return DQ(this->real*dq.real, this->real*dq.dual+this->dual*dq.real);
	}
	DQ DQ::operator * (const double d) const {
		return DQ(this->real*d, this->dual*d);
	}
	Point DQ::transform(Point r) const {
		DQ v(1,jie::Point(0,0,0),0,r);
		return ( (*this) * v * this->conjugate2() ).dual.v;
	}
};// namespace Jie