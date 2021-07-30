#ifndef MY_KDTREE_H
#define MY_KDTREE_H
#include "vec3.h"
#include "translate.h"
#include "time.h"
#include <pcl/kdtree/kdtree_flann.h>

using namespace std;
using namespace demo;

class MyKDtree {
public:
	vector<vec3>& meshPositions;
	MyKDtree(vector<vec3>& meshPositions):meshPositions(meshPositions) {
		//define KDtree for dstMesh
		time_t t1 = clock();
		toPCLCloud(meshPositions, cloud, false);
		tree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
		tree->setInputCloud(cloud.makeShared());
	}
	double nearest_sqr_dis(const vec3& pt) {
		vector<int> k_indices;
		vector<float> k_sqr_dist;
		pcl::PointXYZ p(pt.x, pt.y, pt.z);
		tree->nearestKSearch(p, 1, k_indices, k_sqr_dist);
		return k_sqr_dist[0];
	}
	void nearest_pt(const vec3& pt, vec3& ptOut) const {
		vector<int> k_indices;
		vector<float> k_sqr_dist;
		pcl::PointXYZ p(pt.x, pt.y, pt.z);
		tree->nearestKSearch(p, 1, k_indices, k_sqr_dist);
		auto& p2 = cloud[k_indices[0]];
		ptOut.x = p2.x; ptOut.y = p2.y; ptOut.z = p2.z;
	}
	////delta z contributes less
	//void nearest_pt_Z(const vec3& pt, vec3& ptOut) {
	//	vector<int> k_indices;
	//	vector<float> k_sqr_dist;
	//	pcl::PointXYZ p(pt.x, pt.y, pt.z);
	//	tree->nearestKSearch(p, 1, k_indices, k_sqr_dist);
	//	auto& p2 = cloud[k_indices[0]];
	//	ptOut.x = p2.x; ptOut.y = p2.y; ptOut.z = p2.z;
	//}
	vec3 nearest_disp(const vec3& pt) {
		vector<int> k_indices;
		vector<float> k_sqr_dist;
		pcl::PointXYZ p(pt.x, pt.y, pt.z);
		tree->nearestKSearch(p, 1, k_indices, k_sqr_dist);
		
		//cout << "pt:" << pt << endl;
		//cout << "k_indices:" << k_indices[0] << endl;
		//cout << "cloud[k_indices[0]]:" << cloud[k_indices[0]] << endl;
		auto& p2 = cloud[k_indices[0]];

		return vec3(p2.x,p2.y,p2.z) - pt;
	}
	void nearestKSearch(const Eigen::Vector3d& pt, vector<int>& k_indices, vector<float>& k_sqr_dist, int k) const {
		pcl::PointXYZ p(pt[0], pt[1], pt[2]);
		tree->nearestKSearch(p, k, k_indices, k_sqr_dist);
	}
	void nearestKSearch(const demo::vec3& pt, vector<int>& k_indices, vector<float>& k_sqr_dist, int k) const {
		pcl::PointXYZ p(pt.x, pt.y, pt.z);
		tree->nearestKSearch(p, k, k_indices, k_sqr_dist);
	}
	void nearestKSearch(const pcl::PointXYZ& p, vector<int>& k_indices, vector<float>& k_sqr_dist, int k) const {
		tree->nearestKSearch(p, k, k_indices, k_sqr_dist);
	}
	void renew(const double* xi, const double* yi, const double* zi) {
		//if (xi[0] != 0 || yi[0] != 0 || zi[0] != 0) {
		//	cout <<"old cloud:"<< cloud[0].x << " " << cloud[0].y << " " << cloud[0].z << endl;
		//}
		for (int i = 0; i < cloud.size(); i++) {
			cloud[i].x = meshPositions[i].x + xi[i];
			cloud[i].y = meshPositions[i].y + yi[i];
			cloud[i].z = meshPositions[i].z + zi[i];
		}
		//if (xi[0] != 0 || yi[0] != 0 || zi[0] != 0) {
		//	cout << "delta:" << xi[0] << " " << yi[0] << " " << zi[0] << endl;
		//	cout << "new cloud:" << cloud[0].x << " " << cloud[0].y << " " << cloud[0].z << endl;
		//}
		delete tree;
		tree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
		tree->setInputCloud(cloud.makeShared());
	}
	void radiusSearch(const vec3& pt, vector<int>& k_indices, vector<float>& k_sqr_dist, float radius) {
		pcl::PointXYZ p(pt.x, pt.y, pt.z);
		tree->radiusSearch(p, radius, k_indices, k_sqr_dist);
	}
	~MyKDtree() {
		delete tree;
	}
private:
	pcl::KdTreeFLANN<pcl::PointXYZ> *tree;
	pcl::PointCloud<pcl::PointXYZ> cloud;
};
#endif MY_KDTREE_H