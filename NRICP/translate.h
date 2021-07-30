#ifndef TRANSLATE_H
#define TRANSLATE_H
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "vec3.h"

void rotateZ(demo::vec3& vec, std::vector<demo::vec3>& meshPositions);
void rotateY(demo::vec3& vec, std::vector<demo::vec3>& meshPositions);
void rotateY(demo::vec3& vec, demo::vec3& v);
void calPCA(std::vector<demo::vec3>& meshPositions, demo::vec3& vec, int toProject);
void toPCLCloud(std::vector<demo::vec3>& meshPositions, pcl::PointCloud<pcl::PointXYZ>& cloud, int toProject = 0);
void fromPCLCloud(std::vector<demo::vec3>& meshPositions, pcl::PointCloud<pcl::PointXYZ>& cloud);

#endif //TRANSLATE_H