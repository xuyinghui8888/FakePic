#include <iostream>
#include "translate.h"
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

using namespace std;
using demo::vec3;
void saveCloudOBJ(char* name, pcl::PointCloud<pcl::PointXYZ>& cloud, pcl::PointIndices::Ptr indices) {
	FILE* file = fopen(name, "w+");
	for (auto i : indices->indices) {
		auto& pt = cloud[i];
		fprintf(file, "v %g %g %g\n", pt.x, pt.y, pt.z);
	}
	fclose(file);
}
void toPCLCloud(std::vector<demo::vec3>& meshPositions, pcl::PointCloud<pcl::PointXYZ>& cloud, int toProject) {
	cloud.clear();
	switch (toProject) {
		case 0:
			for (auto& vec : meshPositions)
				cloud.push_back(pcl::PointXYZ(vec.x, vec.y, vec.z));
			break;
		case 1:
			for (auto& vec : meshPositions)
				cloud.push_back(pcl::PointXYZ(0, vec.y, vec.z));
			break;
		case 2:
			for (auto& vec : meshPositions)
				cloud.push_back(pcl::PointXYZ(vec.x, 0, vec.z));
			break;
		case 3:
			for (auto& vec : meshPositions)
				cloud.push_back(pcl::PointXYZ(vec.x, vec.y, 0));
			break;
	}
}
void fromPCLCloud(std::vector<demo::vec3>& meshPositions, pcl::PointCloud<pcl::PointXYZ>& cloud) {
	meshPositions.resize(cloud.width);
	int pos = 0;
	for (auto& pt : cloud) {
		meshPositions[pos].x = pt.x; meshPositions[pos].y = pt.y; meshPositions[pos].z = pt.z;
		pos++;
	}
}
//return whether need to trans
bool calTransMatrix(double xFrom, double yFrom, double zFrom, double xTo, double yTo, double zTo, Eigen::Affine3f& trans) {
	trans = Eigen::Affine3f::Identity();
	Eigen::Vector3f from(xFrom, yFrom, zFrom);
	Eigen::Vector3f to(xTo, yTo, zTo);
	cout << "from,to:" << from.transpose() << "|" << to.transpose() << endl;
	from.normalize();
	to.normalize();
	auto e_v3fRot = from.cross(to);
	if (e_v3fRot.norm() > 0.001)
		e_v3fRot.normalize();
	else {
		//若无本块代码，(0,0,0)的轴会导致左右脚变化
		if (abs(to.dot(Eigen::Vector3f(0, 0, 1))) < 0.001)
			e_v3fRot = Eigen::Vector3f(0, 0, 1);
		else if (abs(to.dot(Eigen::Vector3f(1, 0, 0))) < 0.001)
			e_v3fRot = Eigen::Vector3f(1, 0, 0);
		else {
			Eigen::Vector3f e_v3fRot = Eigen::Vector3f(1, 0, 0).cross(to);
			e_v3fRot.normalize();
		}
	}
	double product = from.dot(to);
	if (product > 1) product = 1;
	else if (product <-1) product = -1;
	double fAng = acos(product);
	if (fAng < 0.003)
		return false;
	cout << "fAng, axis:" << fAng << "||" << e_v3fRot[0] << "," << e_v3fRot[1] << "," << e_v3fRot[2] << endl;
	Eigen::AngleAxisf e_angaxRot(fAng, e_v3fRot);
	trans.rotate(e_angaxRot);
	return true;
}
void rotate(Eigen::Affine3f& trans, std::vector<demo::vec3>& meshPositions) {
	pcl::PointCloud<pcl::PointXYZ> cloud;
	toPCLCloud(meshPositions, cloud);
	pcl::transformPointCloud(cloud, cloud, trans);
	fromPCLCloud(meshPositions, cloud);
}
void rotate(Eigen::Affine3f& trans, demo::vec3& v) {
	pcl::PointXYZ p(v.x, v.y, v.z);
	auto p2 = pcl::transformPoint(p, trans);
	v = vec3(p2.x, p2.y, p2.z);
}
void rotateZ(demo::vec3& vec, std::vector<demo::vec3>& meshPositions) {
	Eigen::Affine3f trans;
	auto needTrans = calTransMatrix(vec.x, vec.y, vec.z, 0, 0, 1, trans);
	if (!needTrans)
		return;
	rotate(trans, meshPositions);
}
//fix z dir, rotate around z
void rotateY(demo::vec3& vec, std::vector<demo::vec3>& meshPositions) {
	Eigen::Affine3f trans;
	auto needTrans = calTransMatrix(vec.x, vec.y, 0, 0,1, 0, trans);
	if (!needTrans)
		return;
	rotate(trans, meshPositions);
}
//fix z dir, rotate around z
void rotateY(demo::vec3& vec, demo::vec3& v) {
	Eigen::Affine3f trans;
	auto needTrans = calTransMatrix(vec.x, vec.y, 0, 0, 1, 0, trans);
	if (!needTrans)
		return;
	rotate(trans, v);
}
//toProject:0-no_project,1-x,2-y,3-z
void calPCA(std::vector<demo::vec3>& meshPositions, demo::vec3& vec, int toProject) {
	pcl::PointCloud<pcl::PointXYZ> cloud;
	toPCLCloud(meshPositions, cloud, toProject);

	Eigen::Vector4f centroid1;
	Eigen::Matrix3f covariance1;
	Eigen::Vector4f centroid2;
	Eigen::Matrix3f covariance2;

	pcl::compute3DCentroid(cloud, centroid1);

	computeCovarianceMatrixNormalized(cloud, centroid1, covariance1);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver1(covariance1, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigDx1 = eigen_solver1.eigenvectors();
	vec = demo::vec3(eigDx1.col(2)(0), eigDx1.col(2)(1), eigDx1.col(2)(2));
}