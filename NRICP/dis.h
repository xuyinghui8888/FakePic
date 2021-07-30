#ifndef DIS_H
#define DIS_H
#include <GTMathematics.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include "demo.h"
#include "defs.h"
#include "MyKDtree.h"

typedef gte::Vector<3, fnum> gteVec3;
typedef gte::Triangle<3, fnum> gteTri;
typedef gte::DCPQuery<fnum, gteVec3, gteTri> gteDCP;


static inline gteVec3 toGteVec3(const demo::vec3& p) {
	gteVec3 v;
	v[0] = p.x;
	v[1] = p.y;
	v[2] = p.z;
	return v;
}
static inline gteVec3 toGteVec3(pcl::PointXYZ& p) {
	gteVec3 vec3;
	vec3[0] = p.x;
	vec3[1] = p.y;
	vec3[2] = p.z;
	return vec3;
}
//data中的数据存储格式为每numsPerVertex(4 or 3,看step)个字节表示一个float，每4个float表示一个点，分别是xyz1
static inline gteVec3 toGteVec3(pcl::PolygonMesh& mesh, unsigned int index) {
	static int numsPerVertex = mesh.cloud.point_step / 4;
	float* data = (float*)&(mesh.cloud.data[0]);
	gteVec3 vec3;
	vec3[0] = data[numsPerVertex * index];
	vec3[1] = data[numsPerVertex * index + 1];
	vec3[2] = data[numsPerVertex * index + 2];
	return vec3;
}

static inline demo::vec3 fromGteVec3(gteVec3& p) {
	demo::vec3 v;
	v.x = p[0];
	v.y = p[1];
	v.z = p[2];
	return v;
}
static inline gteDCP::Result disPtToTriangle(gteVec3& v, gteVec3& v1, gteVec3& v2, gteVec3& v3) {
	gteTri tri(v1, v2, v3);
	gteDCP::Result result = gteDCP()(v, tri);
	return result;
}
static inline gteDCP::Result disPtToTriangle(demo::vec3& pt, demo::vec3& p1, demo::vec3& p2, demo::vec3& p3) {
	auto v = toGteVec3(pt);
	auto v1 = toGteVec3(p1);
	auto v2 = toGteVec3(p2);
	auto v3 = toGteVec3(p3);
	return disPtToTriangle(v, v1, v2, v3);
}
static inline gteDCP::Result disPtToTriangle(pcl::PointXYZ& pt, pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3) {
	auto v = toGteVec3(pt);
	auto v1 = toGteVec3(p1);
	auto v2 = toGteVec3(p2);
	auto v3 = toGteVec3(p3);
	return disPtToTriangle(v, v1, v2, v3);
}
static inline vec3 nearestPtInTriangle(const demo::vec3& v, const demo::vec3& v1, const demo::vec3& v2, const demo::vec3& v3) {
	gteTri tri(toGteVec3(v1), toGteVec3(v2), toGteVec3(v3));
	gteDCP::Result result = gteDCP()(toGteVec3(v), tri);
	auto& closest = result.closest;
	return demo::vec3(closest[0], closest[1], closest[2]);
}
fnum sqrDistPtToMesh(gteVec3& v, pcl::PolygonMesh& tarModel);
static inline fnum sqrDistPtToMesh(pcl::PointXYZ& p, pcl::PolygonMesh& tarModel) {
	auto v = toGteVec3(p);
	return sqrDistPtToMesh(v, tarModel);
}
static inline fnum sqrDistPtToMesh(fnum x, fnum y, fnum z, pcl::PolygonMesh& tarModel) {
	gteVec3 v;
	v[0] = x;
	v[1] = y;
	v[2] = z;
	return sqrDistPtToMesh(v, tarModel);
}
#endif //DIS_H