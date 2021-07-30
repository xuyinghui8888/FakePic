#ifndef DEMO_H
#define DEMO_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <set>
#include "vec3.h"
#include "translate.h"

#define PI_XXM 3.14159265358979
#define MAX_Z 80 //points with z > MAX_Z will not be used for registration
#define	CUT_Z 120 //cut the useless part with z > CUT_Z
#define MIN_Z 1//小于该值的dstMesh中的点不会拉走srcMesh中的点，防止srcMesh中脚底过于稀疏而把上面的点拉向脚底
#define LOCAL_MAX_NEIGHBOR 3
#define KEY_PT_HANDLE_SIZE 1

using std::vector;
using std::cout;
using std::endl;

namespace demo {
	struct DeformPara {
		int mainHandle;
		int handleRegionSize;
		int unconstrainedRegionSize;
		vec3 handleDst;
		DeformPara(int mainHandle, int handleRegionSize, int unconstrainedRegionSize, vec3 handleDst) :mainHandle(mainHandle), handleRegionSize(handleRegionSize), unconstrainedRegionSize(unconstrainedRegionSize), handleDst(handleDst) {}
	};
	struct Mesh {
		std::vector<vec3> meshPositions;
		std::vector<int> meshCells;
		std::vector<vec3> normals;
		std::vector<double> areas;
		bool leftFoot = true;

		Mesh() {}
		Mesh(Mesh& other) {
			meshPositions.insert(meshPositions.end(), other.meshPositions.begin(), other.meshPositions.end());
			meshCells.insert(meshCells.end(), other.meshCells.begin(), other.meshCells.end());
		}
		void clear() {
			meshCells.clear();
			meshPositions.clear();
			normals.clear();
			areas.clear();
		}
		void copy(Mesh& other) {
			clear();
			meshPositions.insert(meshPositions.end(), other.meshPositions.begin(), other.meshPositions.end());
			meshCells.insert(meshCells.end(), other.meshCells.begin(), other.meshCells.end());
			this->leftFoot = other.leftFoot;
		}

		void loadMesh(const char* filename, bool needNormalization = true, bool needMoveToCenterBeforeRotation = false, bool leftFoot = true) {
			std::string name(filename);
			if (name.find(".obj") != name.npos)
				loadMeshObj(name.c_str(), needNormalization, needMoveToCenterBeforeRotation, leftFoot);
			else if (name.find(".ply") != name.npos)
				loadMeshPly(name.c_str(), needNormalization, needMoveToCenterBeforeRotation, leftFoot);
			else
				cout << "Invalid filename to load:" << name << endl;
		}
		void loadMeshPly(const char* filename, bool needNormalization = true, bool needMoveToCenterBeforeRotation = false, bool leftFoot = true) {
			if (strlen(filename) < 5)
				cout << "too short filename in loadMesh in demo.h" << endl;
			if (strcmp(".ply", filename + strlen(filename) - 4) != 0)
				cout << "invalid file type in loadMesh in demo.h" << endl;
			FILE* file = fopen(filename, "r");
			if (file == NULL) {
				printf("could not open file %s\n", filename);
			}
			clear();
			char line[256];
			int numPieces = 0;
			bool startParse = false;
			while (fgets(line, sizeof(line), file)) {
				std::vector<std::string> strs;
				numPieces = splitString(std::string(line), strs, ' ');
				if (startParse) {
					if (numPieces == 4) {
						meshCells.push_back(std::stoi(strs[1]));
						meshCells.push_back(std::stoi(strs[2]));
						meshCells.push_back(std::stoi(strs[3]));
					}
					else if(numPieces == 3) {
						meshPositions.push_back(vec3(std::stof(strs[0]), std::stof(strs[1]), std::stof(strs[2])));
					}
				}
				else {
					if (strs[0] == "end_header\n") {
						startParse = true;
					}
				}
			}
			fclose(file);

			std::cout << "num pos,cell:" << meshPositions.size() << ", " << meshCells.size() << std::endl;
			if (needNormalization)
				normalizeRotation(needMoveToCenterBeforeRotation);

			this->leftFoot = leftFoot;
		}
		void loadMeshObj(const char* filename, bool needNormalization = true, bool needMoveToCenterBeforeRotation = false, bool leftFoot = true) {
			FILE* file = fopen(filename, "r");
			if (file == NULL) {
				printf("could not open file %s\n", filename);
			}
			clear();
			char line[256];
			while (fgets(line, sizeof(line), file)) {
				std::vector<std::string> strs;
				splitString(std::string(line), strs, ' ');
				if (strs.size() == 0)
					continue;
				if (strs[0] == "f") {
					meshCells.push_back(std::stoi(strs[1]) - 1);
					meshCells.push_back(std::stoi(strs[2]) - 1);
					meshCells.push_back(std::stoi(strs[3]) - 1);
				}
				else if (strs[0] == "v") {
					meshPositions.push_back(vec3(std::stof(strs[1]), std::stof(strs[2]), std::stof(strs[3])));
				}
			}
			fclose(file);

			std::cout << "num pos,cell:" << meshPositions.size() << ", " << meshCells.size() << std::endl;
			if (needNormalization)
				normalizeRotation(needMoveToCenterBeforeRotation);
			this->leftFoot = leftFoot;
		}

		void saveMeshObj(const char* name, bool toWritePoly) {
			FILE* f = fopen(name, "w+");
			if (f == NULL)
				cout << "unable to open file:" << name << endl;
			for (int j = 0; j < meshPositions.size(); j++)
				fprintf(f, "v %g %g %g\n", meshPositions[j].x, meshPositions[j].y, meshPositions[j].z);
			if (toWritePoly) {
				for (int j = 0; j < meshCells.size() / 3; j++) {
					fprintf(f, "f %d %d %d\n", 1 + meshCells[3 * j], 1 + meshCells[3 * j + 1], 1 + meshCells[3 * j + 2]);
				}
			}
			fclose(f);
		}
		void saveMeshPly(const char* name, bool toWritePoly) {
			FILE* f = fopen(name, "w+");
			if (f == NULL)
				cout << "unable to open file:" << name << endl;
			fprintf(f, "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex %zd\nproperty float x\nproperty float y\nproperty float z\nelement face %zd\nproperty list uchar int vertex_indices\nend_header\n", meshPositions.size(), meshCells.size() / 3);
			for (int j = 0; j < meshPositions.size(); j++) {
				fprintf(f, "%g %g %g\n", meshPositions[j].x, meshPositions[j].y, meshPositions[j].z);
			}
			if (toWritePoly) {
				for (int j = 0; j < meshCells.size() / 3; j++) {
					fprintf(f, "3 %d %d %d\n", meshCells[3 * j], meshCells[3 * j + 1], meshCells[3 * j + 2]);
				}
			}
			fclose(f);
		}
		void saveMesh(const char* filename, bool toWritePoly) {
			std::string name(filename);
			if (name.find(".obj") != name.npos)
				saveMeshObj(name.c_str(), toWritePoly);
			else if (name.find(".ply") != name.npos)
				saveMeshPly(name.c_str(), toWritePoly);
			else
				cout << "Invalid filename to save:" << name << endl;
		}
		void saveFinalMesh(const char* name, Mesh& meshOri, bool toWritePoly) {
			meshOri.saveMesh(name, toWritePoly);
		}

		void needNormals() {
			if (normals.size()>0)
				return;
			auto count = meshCells.size() / 3;
			normals.resize(count);
			for (auto i = 0; i < count; i++) {
				auto& p0 = meshPositions[meshCells[3 * i]];
				auto& p1 = meshPositions[meshCells[3 * i + 1]];
				auto& p2 = meshPositions[meshCells[3 * i + 2]];
				auto v1 = p1 - p0;
				auto v2 = p2 - p0;
				normals[i] = v1.cross(v2).normalize();
			}
		}
		void renewNormals() {
			normals.clear();
			needNormals();
		}
		void needAreas() {
			if (areas.size()>0)
				return;
			auto count = meshCells.size() / 3;
			areas.resize(count);
			for (auto i = 0; i < count; i++) {
				auto& p0 = meshPositions[meshCells[3 * i]];
				auto& p1 = meshPositions[meshCells[3 * i + 1]];
				auto& p2 = meshPositions[meshCells[3 * i + 2]];
				areas[i] = Tri3(p0, p1, p2).area();
			}
		}
		void renewAreas() {
			areas.clear();
			needAreas();
		}
		vec3 getCenter() {
			vec3 sum;
			for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
				sum += *ite;
			}
			return sum / (double)meshPositions.size();
		}
		vec3 getTriangleWeightedCenter() {
			size_t num = meshCells.size() / 3;
			if (num > 0) {
				needAreas();
				vec3 sum;
				double sumArea = 0;
				for (size_t i = 0; i < num; i++) {
					sum += Tri3(meshPositions[meshCells[3 * i]], meshPositions[meshCells[3 * i + 1]], meshPositions[meshCells[3 * i + 2]]).center() * areas[i];
					sumArea += areas[i];
				}
				return sum / sumArea;
			}
			else if (meshPositions.size()>0) {
				vec3 sum;
				for (auto&& p : meshPositions)
					sum += p;
				return sum / meshPositions.size();
			}
			else {
				cout << "no cells and pts in mesh!" << endl;
				return vec3(0, 0, 1);
			}
		}
		//toCut must be used when the model has already been roughly aligned
		void getAabb(vec3& min, vec3& max, bool toCut = false) {
			min = vec3(FLT_MAX, FLT_MAX, FLT_MAX), max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			const double zAnkle = 60;
			if (toCut) {
				for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
					if (ite->x < min.x&&ite->z < zAnkle) min.x = ite->x;
					if (ite->y < min.y&&ite->z < zAnkle) min.y = ite->y;
					if (ite->z < min.z&&ite->z < zAnkle) min.z = ite->z;
					if (ite->x > max.x&&ite->z < zAnkle) max.x = ite->x;
					if (ite->y > max.y) max.y = ite->y;
					if (ite->z > max.z) max.z = ite->z;
				}
			}
			else {
				for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
					if (ite->x < min.x) min.x = ite->x;
					if (ite->y < min.y) min.y = ite->y;
					if (ite->z < min.z) min.z = ite->z;
					if (ite->x > max.x) max.x = ite->x;
					if (ite->y > max.y) max.y = ite->y;
					if (ite->z > max.z) max.z = ite->z;
				}
			}
		}
		static void getAabb(vec3& min, vec3& max, vector<vec3>& positions) {
			min = vec3(FLT_MAX, FLT_MAX, FLT_MAX), max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for (auto ite = positions.begin(); ite != positions.end(); ite++) {
				if (ite->x < min.x) min.x = ite->x;
				if (ite->y < min.y) min.y = ite->y;
				if (ite->z < min.z) min.z = ite->z;
				if (ite->x > max.x) max.x = ite->x;
				if (ite->y > max.y) max.y = ite->y;
				if (ite->z > max.z) max.z = ite->z;
			}
		}
		static void getAabb(vec3& min, vec3& max, vector<vec3>& positions, vector<int>& indices) {
			min = vec3(FLT_MAX, FLT_MAX, FLT_MAX), max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for (auto idx : indices) {
				auto ite = &(positions[idx]);
				if (ite->x < min.x) min.x = ite->x;
				if (ite->y < min.y) min.y = ite->y;
				if (ite->z < min.z) min.z = ite->z;
				if (ite->x > max.x) max.x = ite->x;
				if (ite->y > max.y) max.y = ite->y;
				if (ite->z > max.z) max.z = ite->z;
			}
		}
		void getAabbWithoutLeg(vec3& min, vec3& max) {
			min = vec3(FLT_MAX, FLT_MAX, FLT_MAX), max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
				if (ite->x < min.x) min.x = ite->x;
				if (ite->y < min.y) min.y = ite->y;
				if (ite->z < min.z) min.z = ite->z;
				if (ite->x > max.x) max.x = ite->x;
				if (ite->y > max.y) max.y = ite->y;
				//if (ite->z > max.z) max.z = ite->z;
			}
			auto midY = (min.y + max.y) / 2;
			cout << "midY:" << midY << endl;
			for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
				if (ite->z > max.z&&ite->y > midY) max.z = ite->z;
			}
			cout << min << " | " << max << endl;
		}
		//for a fixed y percent, find the most left pt
		int detectKeyPointLeft(const vec3& pMin, const vec3& pMax, double percent) {
			int index = -1;
			double max = -1e5;
			auto yRef = percent*pMin.y + (1 - percent)*pMax.y;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& p = meshPositions[i];
				auto val = -abs(p.y - yRef) - p.x;
				if (val > max) {
					max = val;
					index = i;
				}
			}
			assert(index >= 0);
			cout << percent << " key point:" << meshPositions[index] << endl;
			return index;
		}
		//for a fixed y percent, find the most right pt
		int detectKeyPointRight(const vec3& pMin, const vec3& pMax, double percent) {
			int index = -1;
			double max = -1e5;
			auto yRef = percent*pMin.y + (1 - percent)*pMax.y;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& p = meshPositions[i];
				auto val = -abs(p.y - yRef) + p.x;
				if (val > max) {
					max = val;
					index = i;
				}
			}
			assert(index >= 0);
			cout << percent << " key point:" << meshPositions[index] << endl;
			return index;
		}
		//for a fixed y percent, find the most right pt
		int detectKeyPoint(const vec3& pMin, const vec3& pMax, double percentX, double percentY) {
			int index = -1;
			double max = -1e5;
			auto xRef = percentX*pMin.x + (1 - percentX)*pMax.x;
			auto yRef = percentY*pMin.y + (1 - percentY)*pMax.y;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& p = meshPositions[i];
				if (p.z > 20)
					continue;
				auto val = -abs(p.y - yRef) - abs(p.x - xRef) - p.z;
				if (val > max) {
					max = val;
					index = i;
				}
			}
			assert(index >= 0);
			return index;
		}
		vec3 normalizeZExtract() {
			vec3 pMin, pMax;
			getAabb(pMin, pMax);
			int idx1 = detectKeyPoint(pMin, pMax, 0.35, 0.25);//top right
			int idx2 = detectKeyPoint(pMin, pMax, 0.65, 0.38);//top left
			int idx3 = detectKeyPoint(pMin, pMax, 0.5, 0.90);//bottom
			vec3 p1 = meshPositions[idx1];
			vec3 p2 = meshPositions[idx2];
			vec3 p3 = meshPositions[idx3];
			vec3 v1 = p1 - p2;
			vec3 v2 = p2 - p3;
			vec3 zDir = v2.cross(v1).normalize();
			if (zDir.z < 0)
				zDir = -zDir;
			//flip
			cout << "extracted zDir:" << zDir << endl;
			rotateZ(zDir, meshPositions);
			return zDir;
		}
		vec3 normalizeZ(bool yijieRaw=false) {
			vec3 zDir(0, 0, 1);
			if(yijieRaw)
				zDir = vec3(0.122647, -0.817424, -0.562829);
			cout << "zDir:" << zDir << endl;
			rotateZ(zDir, meshPositions);
			return zDir;
		}
		//split the triangle into 3 smaller ones with the center
		void split(double thresArea) {
			for (int i = 0; i < meshCells.size() / 3; ) {
				auto idx1 = meshCells[3 * i], idx2 = meshCells[3 * i + 1], idx3 = meshCells[3 * i + 2];
				Tri3 tri3(meshPositions[idx1], meshPositions[idx2], meshPositions[idx3]);
				auto area = tri3.area();
				if (area > thresArea) {
					int index = (int)meshPositions.size();
					meshPositions.push_back(tri3.center());
					meshCells[3 * i + 2] = index;
					meshCells.push_back(idx2); meshCells.push_back(idx3); meshCells.push_back(index);
					meshCells.push_back(idx3); meshCells.push_back(idx1); meshCells.push_back(index);
					if (area / 3 <= thresArea)
						i++;
				}
				else
					i++;
			}
		}
		void getAdjF2F(std::vector<std::vector<int>>& adjF2F) {
			auto numFaces = meshCells.size() / 3;
			for (int i = 0; i < numFaces; ++i) {
				adjF2F.push_back(std::vector<int>(3, -1));
			}
			for (int i = 0; i < numFaces; i++) {
				for (int j = 0; j <numFaces; j++) {
					if (i == j)
						continue;
					bool flag[3] = { false,false,false };
					for (int k = 0; k < 3; k++) {
						for (int m = 0; m < 3; m++) {
							if (meshCells[3 * i + k] == meshCells[3 * j + m]) {
								flag[k] = true;
								break;
							}
						}
					}
					if (flag[0] && flag[1])adjF2F[i][0] = j;
					if (flag[0] && flag[2])adjF2F[i][1] = j;
					if (flag[1] && flag[2])adjF2F[i][2] = j;
				}
			}
		}
		void getAdjV2F(std::vector<std::vector<int>>& adjV2F) {
			auto numVertices = meshPositions.size();
			for (int i = 0; i < numVertices; i++) {
				adjV2F.push_back(std::vector<int>());
			}
			for (int i = 0; i < meshCells.size(); i += 3) {
				adjV2F[meshCells[i]].push_back(i);
				adjV2F[meshCells[i+1]].push_back(i);
				adjV2F[meshCells[i+2]].push_back(i);
			}
		}
		// get adjacancy list for vertices.
		void getAdj(std::vector<std::vector<int>>& adj) {
			auto numVertices = meshPositions.size();
			for (int i = 0; i < numVertices; ++i) {
				adj.push_back(std::vector<int>());
			}

			for (int i = 0; i < meshCells.size(); i += 3) {
				for (int j = 0; j < 3; ++j) {
					int a = meshCells[i + (j + 0)];
					int b = meshCells[i + ((j + 1) % 3)];
					adj[a].push_back(b);
				}
			}
		}
		void getBoundaryPts(std::vector<int>& pts) {
			std::vector<vector<int>> adjF2F;
			getAdjF2F(adjF2F);
			std::set<std::vector<int>> boundaryEdges;
			for (int i = 0; i < adjF2F.size(); i++) {
				for (int j = 0; j < 3; j++) {
					if (adjF2F[i][j] == -1) {
						std::vector<int> edge(2);
						if (j == 0) {
							edge[0] = meshCells[i * 3];
							edge[1] = meshCells[i * 3 + 1];
						}
						else if (j == 1) {
							edge[0] = meshCells[i * 3];
							edge[1] = meshCells[i * 3 + 2];
						}
						else {
							edge[0] = meshCells[i * 3 + 1];
							edge[1] = meshCells[i * 3 + 2];
						}
						boundaryEdges.insert(edge);
					}
				}
			}
			size_t numEdges = boundaryEdges.size();

			//fill pts
			pts.resize(numEdges, -1);
			auto&& begin = boundaryEdges.begin();
			pts[0] = begin->at(0);
			pts[1] = begin->at(1);
			boundaryEdges.erase(begin);
			for (size_t i = 1; i < numEdges - 1; i++) {
				bool found = false;
				for (auto ite = boundaryEdges.begin(); ite != boundaryEdges.end(); ite++) {
					auto idx1 = ite->at(0), idx2 = ite->at(1);
					if (idx1 == pts[i]) {
						pts[i + 1] = idx2;
						found = true;
					}
					else if (idx2 == pts[i]) {
						pts[i + 1] = idx1;
						found = true;
					}
					if (found) {
						boundaryEdges.erase(ite);
						break;
					}
				}
			}
			cout << "num boundary pts:" << pts.size() << endl;
		}
		void findLongestEdge(int& edgeIndex, vec3& edgeCenter, int idx1, int idx2, int idx3) {
			auto len1 = (meshPositions[idx1] - meshPositions[idx2]).sqrLength();
			auto len2 = (meshPositions[idx1] - meshPositions[idx3]).sqrLength();
			auto len3 = (meshPositions[idx2] - meshPositions[idx3]).sqrLength();
			if (len1 >= len2&&len1 >= len3) {
				edgeIndex = 0;
				edgeCenter = (meshPositions[idx1] + meshPositions[idx2])*0.5;
			}
			else if (len2>len1&&len2 >= len3) {

				edgeIndex = 1;
				edgeCenter = (meshPositions[idx1] + meshPositions[idx3])*0.5;
			}
			else {
				edgeIndex = 2;
				edgeCenter = (meshPositions[idx2] + meshPositions[idx3])*0.5;
			}
		}
		void getP3Index(int* a, int* b, int& start, int& p3Index) {
			for (int j = 0; j < 3; j++) {
				bool found = false;
				for (int i = 0; i < 3; i++) {
					if (a[i] == b[j])
						found = true;
				}
				if (!found) {
					start = j;
					p3Index = b[j];
					return;
				}
			}
			assert(false);
		}
		//更新adj对于自身的相邻关系
		void updateReverseAdj(std::vector<std::vector<int>>& adjF2F, int currentTriIndex, int newTriIndex, int otherTriIndex) {
			for (int i = 0; i < 3; i++) {
				if (adjF2F[otherTriIndex][i] == currentTriIndex) {
					adjF2F[otherTriIndex][i] = newTriIndex;
					return;
				}
			}
			assert(false);
		}
		//是否在脚趾处
		bool isHead(vec3& center, vec3& pMin, vec3& pMax) {
			double pct1 = 4.0 / 6, pct2 = 5.0 / 6;
			if (!leftFoot) {
				auto tmp = pct1;
				pct1 = pct2;
				pct2 = tmp;
			}
			auto dy = (pct2 - pct1)*(pMax.y - pMin.y);
			auto dx = pMax.x - pMin.x;
			auto x0 = (pMin.x + pMax.x)*0.5;
			auto y0 = pMin.y + (0.5*pct1 + 0.5*pct2)*(pMax.y - pMin.y);
			if (center.y - y0 > dy / dx*(center.x - x0))
				return true;
			else
				return false;
		}
		//split the longest edge and its adjacent triangle
		void split2(double thresArea, bool onlyHead) {
			vec3 pMin, pMax;
			if (onlyHead)
				getAabb(pMin, pMax);

			std::vector<std::vector<int>> adjF2F;
			getAdjF2F(adjF2F);
			while (true) {
				bool splited = false;
				auto numTri = meshCells.size() / 3;
				for (int i = 0; i <numTri;) {
					auto idx1 = meshCells[3 * i], idx2 = meshCells[3 * i + 1], idx3 = meshCells[3 * i + 2];
					Tri3 tri3(meshPositions[idx1], meshPositions[idx2], meshPositions[idx3]);
					auto area = tri3.area();

					int edgeIndex;
					vec3 edgeCenter;
					findLongestEdge(edgeIndex, edgeCenter, idx1, idx2, idx3);

					if (area > thresArea/*||tri3.deltaX()>6*/) {
						if (onlyHead && !isHead(tri3.center(), pMin, pMax)) {
							i++;
							continue;
						}
						splited = true;
						int index = (int)meshPositions.size();
						meshPositions.push_back(edgeCenter);
						auto triIndexAdj = adjF2F[i][edgeIndex];
						int startIndex, p3Index;
						getP3Index(&meshCells[3 * i], &meshCells[3 * triIndexAdj], startIndex, p3Index);

						auto cellCount = meshCells.size();
						auto num = int(meshCells.size() / 3);
						if (edgeIndex == 0) {
							meshCells[3 * i + 1] = index;//tri i
							meshCells.push_back(index); meshCells.push_back(idx2); meshCells.push_back(idx3);//tri num
							meshCells[3 * triIndexAdj] = index; meshCells[3 * triIndexAdj + 1] = idx1; meshCells[3 * triIndexAdj + 2] = p3Index;//tri triIndexAdj
							meshCells.push_back(index); meshCells.push_back(p3Index); meshCells.push_back(idx2);//tri num+1

							int adjAdj2 = adjF2F[triIndexAdj][(1 + 3 - startIndex) % 3];
							int adjAdj3 = adjF2F[triIndexAdj][(0 + 3 - startIndex) % 3];
							adjF2F.push_back(vector<int>(3)); adjF2F.push_back(vector<int>(3));
							int tmp = adjF2F[i][2];
							adjF2F[i][2] = num;
							adjF2F[num][0] = num + 1; adjF2F[num][1] = i; adjF2F[num][2] = tmp;
							adjF2F[triIndexAdj][0] = i; adjF2F[triIndexAdj][1] = num + 1; adjF2F[triIndexAdj][2] = adjAdj2;
							adjF2F[num + 1][0] = triIndexAdj; adjF2F[num + 1][1] = num; adjF2F[num + 1][2] = adjAdj3;
							updateReverseAdj(adjF2F, i, num, tmp); updateReverseAdj(adjF2F, triIndexAdj, num + 1, adjAdj3);
						}
						else if (edgeIndex == 1) {
							meshCells[3 * i + 2] = index;
							meshCells.push_back(index); meshCells.push_back(idx2); meshCells.push_back(idx3);
							meshCells[3 * triIndexAdj] = index; meshCells[3 * triIndexAdj + 1] = idx3; meshCells[3 * triIndexAdj + 2] = p3Index;
							meshCells.push_back(index); meshCells.push_back(p3Index); meshCells.push_back(idx1);

							int adjAdj2 = adjF2F[triIndexAdj][(1 + 3 - startIndex) % 3];
							int adjAdj3 = adjF2F[triIndexAdj][(0 + 3 - startIndex) % 3];
							adjF2F.push_back(vector<int>(3)); adjF2F.push_back(vector<int>(3));
							int tmp = adjF2F[i][2];
							adjF2F[i][1] = num + 1; adjF2F[i][2] = num;
							adjF2F[num][0] = i; adjF2F[num][1] = triIndexAdj; adjF2F[num][2] = tmp;
							adjF2F[triIndexAdj][0] = num; adjF2F[triIndexAdj][1] = num + 1; adjF2F[triIndexAdj][2] = adjAdj2;
							adjF2F[num + 1][0] = triIndexAdj; adjF2F[num + 1][1] = i; adjF2F[num + 1][2] = adjAdj3;
							updateReverseAdj(adjF2F, i, num, tmp); updateReverseAdj(adjF2F, triIndexAdj, num + 1, adjAdj3);
						}
						else {
							meshCells[3 * i + 2] = index;
							meshCells.push_back(index); meshCells.push_back(idx3); meshCells.push_back(idx1);
							meshCells[3 * triIndexAdj] = index; meshCells[3 * triIndexAdj + 1] = idx2; meshCells[3 * triIndexAdj + 2] = p3Index;
							meshCells.push_back(index); meshCells.push_back(p3Index); meshCells.push_back(idx3);

							int adjAdj2 = adjF2F[triIndexAdj][(1 + 3 - startIndex) % 3];
							int adjAdj3 = adjF2F[triIndexAdj][(0 + 3 - startIndex) % 3];
							adjF2F.push_back(vector<int>(3)); adjF2F.push_back(vector<int>(3));
							int tmp = adjF2F[i][1];
							adjF2F[i][1] = num;
							adjF2F[num][0] = num + 1; adjF2F[num][1] = i; adjF2F[num][2] = tmp;
							adjF2F[triIndexAdj][0] = i; adjF2F[triIndexAdj][1] = num + 1; adjF2F[triIndexAdj][2] = adjAdj2;
							adjF2F[num + 1][0] = triIndexAdj; adjF2F[num + 1][1] = num; adjF2F[num + 1][2] = adjAdj3;
							updateReverseAdj(adjF2F, i, num, tmp); updateReverseAdj(adjF2F, triIndexAdj, num + 1, adjAdj3);
						}
					}
					i++;
				}
				if (!splited)
					break;
			}
		}
		
		void refineYAndPos() {
			vec3 pMin, pMax;
			getAabb(pMin, pMax);
			vec3 pLeftUp = meshPositions[detectKeyPointLeft(pMin, pMax, 0.34)];
			vec3 pRightUp = meshPositions[detectKeyPointRight(pMin, pMax, 0.34)];
			vec3 pLeftDown = meshPositions[detectKeyPointLeft(pMin, pMax, 0.9)];
			vec3 pRightDown = meshPositions[detectKeyPointRight(pMin, pMax, 0.9)];

			vec3 pMidDown = (pLeftDown + pRightDown)*0.5;
			vec3 pMidUp;
			if (leftFoot)
				pMidUp = pLeftUp*0.4 + pRightUp*0.6;
			else
				pMidUp = pLeftUp*0.6 + pRightUp*0.4;
			vec3 dir = (pMidUp - pMidDown).normalize();
			rotateY(dir, meshPositions);
			rotateY(dir, pMidDown);
			rotateY(dir, pMidUp);
			auto deltaX = (pMidDown.x + pMidUp.x)*0.5;
			getAabb(pMin, pMax, true);
			auto midY = (pMin.y + pMax.y)*0.5;
			for (auto&& item : meshPositions) {
				item.x -= deltaX;
				//item.x -= midX;//already set in normalizeY--ps:not in normalizeY now, refineYAndPos is not automatically called.
				item.y -= midY;
				item.z -= pMin.z;
			}
		}

		//delta z in both ends, the larger is ankle side
		bool isFlipY() {
			const double percent = 0.1;
			vec3 aabbMin, aabbMax;
			getAabb(aabbMin, aabbMax);
			double topLine = (1 - percent)*aabbMax.y + percent*aabbMin.y;
			double downLine = (1 - percent)*aabbMin.y + percent*aabbMax.y;
			double zmintop = 1e10, zmaxtop = -1e10, zmindown = 1e10, zmaxdown = -1e10;
			for (auto&& p : meshPositions) {
				if (p.y > topLine) {
					if (p.z > zmaxtop)
						zmaxtop = p.z;
					if (p.z < zmintop)
						zmintop = p.z;
				}
				else if (p.y < downLine) {
					if (p.z > zmaxdown)
						zmaxdown = p.z;
					if (p.z < zmindown)
						zmindown = p.z;
				}
			}
			if (zmaxtop - zmintop > zmaxdown - zmindown)
				return true;
			else
				return false;
		}
		vec3 normalizeY() {
			vec3 yDir;
			calPCA(meshPositions, yDir, 3);
			rotateY(yDir, meshPositions);
			bool flipY = isFlipY();
			cout << "flipY:" << flipY << endl;
			if (flipY) {
				yDir = -yDir;
				for (auto&& p : meshPositions) {
					p.x = -p.x;
					p.y = -p.y;
				}
			}
			return yDir;
		}
		void normalizeRotation(bool needMoveToCenterBeforeRotation = false, bool yijieRaw=false) {
			if (needMoveToCenterBeforeRotation) {
				vec3 aabbMin, aabbMax;
				getAabb(aabbMin, aabbMax);
				vec3 aabbCenter = (aabbMin + aabbMax) / 2;
				for (auto&& pt : meshPositions) {

					pt -= aabbCenter;

				}
			}
			normalizeZ(yijieRaw);
			normalizeY();
			normalizeZExtract();
			normalizeY();
		}
		int findFrontPtIdx(double height) {
			double min = DBL_MAX;
			int index = -1;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& pt = meshPositions[i];
				if (abs(pt.z - height) > 1)
					continue;
				double tmp = 2 * abs(pt.z - height) - pt.y;
				if (tmp < min) {
					min = tmp;
					index = i;
				}
			}
			assert(index != -1);
			return index;
		}
		int findBackPtIdx(double height) {
			double min = DBL_MAX;
			int index = -1;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& pt = meshPositions[i];
				if (abs(pt.z - height) > 1)
					continue;
				double tmp = 2 * abs(pt.z - height) + pt.y;
				if (tmp < min) {
					min = tmp;
					index = i;
				}
			}
			assert(index != -1);
			return index;
		}
		//before cut, check if the max height is lower than the pos to cut
		bool isHeightLegal(double heightFront) {
			double maxHeight = 0;
			for (auto p : meshPositions) {
				if (p.z > maxHeight) {
					maxHeight = p.z;
				}
			}
			if (maxHeight < heightFront) {
				cout << "maxHeight < heightFront:" << maxHeight << ", " << heightFront << endl;
				//assert(maxHeight >= heightFront);
				return false;
			}
			return true;
		}
		//for src mesh, only once
		bool cutAnkle(double heightFront = 50, double heightBack = 20) {
			if (!isHeightLegal(heightFront))
				return false;
			int idxF = findFrontPtIdx(heightFront);
			int idxB = findBackPtIdx(heightBack);
			vec3 f = meshPositions[idxF];
			vec3 b = meshPositions[idxB];
			Mesh mesh;
			std::set<int> cutIdx;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& p = meshPositions[i];
				//part not cut
				if ((p.z - b.z)*(f.y - b.y) <= (f.z - b.z)*(p.y - b.y)) {
					mesh.meshPositions.push_back(p);
				}
				else
					cutIdx.insert(i);
			}
			for (int i = 0; i < meshCells.size() / 3; i++) {
				if (cutIdx.find(meshCells[3 * i]) == cutIdx.end()
					&& cutIdx.find(meshCells[3 * i + 1]) == cutIdx.end()
					&& cutIdx.find(meshCells[3 * i + 2]) == cutIdx.end()) {
					mesh.meshCells.push_back(meshCells[3 * i]);
					mesh.meshCells.push_back(meshCells[3 * i + 1]);
					mesh.meshCells.push_back(meshCells[3 * i + 2]);
				}
			}
			for (auto& ite = cutIdx.rbegin(); ite != cutIdx.rend(); ite++) {
				auto& item = *ite;
				for (int i = 0; i < mesh.meshCells.size(); i++) {
					if (mesh.meshCells[i] > item)
						mesh.meshCells[i]--;
				}
			}
			meshPositions.swap(mesh.meshPositions);
			meshCells.swap(mesh.meshCells);
			return true;
		}
		//only leave the ankle
		bool cutAnkleReverse(double heightFront, double heightBack) {
			int idxF = findFrontPtIdx(heightFront);
			int idxB = findBackPtIdx(heightBack);
			vec3 f = meshPositions[idxF];
			vec3 b = meshPositions[idxB];
			Mesh mesh;
			std::set<int> cutIdx;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& p = meshPositions[i];
				//part not cut
				if ((p.z - b.z)*(f.y - b.y) >= (f.z - b.z)*(p.y - b.y)) {
					mesh.meshPositions.push_back(p);
				}
				else
					cutIdx.insert(i);
			}
			for (int i = 0; i < meshCells.size() / 3; i++) {
				if (cutIdx.find(meshCells[3 * i]) == cutIdx.end()
					&& cutIdx.find(meshCells[3 * i + 1]) == cutIdx.end()
					&& cutIdx.find(meshCells[3 * i + 2]) == cutIdx.end()) {
					mesh.meshCells.push_back(meshCells[3 * i]);
					mesh.meshCells.push_back(meshCells[3 * i + 1]);
					mesh.meshCells.push_back(meshCells[3 * i + 2]);
				}
			}
			for (auto& ite = cutIdx.rbegin(); ite != cutIdx.rend(); ite++) {
				auto& item = *ite;
				for (int i = 0; i < mesh.meshCells.size(); i++) {
					if (mesh.meshCells[i] > item)
						mesh.meshCells[i]--;
				}
			}
			meshPositions.swap(mesh.meshPositions);
			meshCells.swap(mesh.meshCells);
			return true;
		}
		//for dst mesh
		bool cutZ(double height) {
			if (!isHeightLegal(height))
				return false;
			Mesh mesh;
			std::set<int> cutIdx;
			for (int i = 0; i < meshPositions.size(); i++) {
				auto& p = meshPositions[i];
				//part not cut
				if (p.z<height) {
					mesh.meshPositions.push_back(p);
				}
				else
					cutIdx.insert(i);
			}
			for (int i = 0; i < meshCells.size() / 3; i++) {
				if (cutIdx.find(meshCells[3 * i]) == cutIdx.end()
					&& cutIdx.find(meshCells[3 * i + 1]) == cutIdx.end()
					&& cutIdx.find(meshCells[3 * i + 2]) == cutIdx.end()) {
					mesh.meshCells.push_back(meshCells[3 * i]);
					mesh.meshCells.push_back(meshCells[3 * i + 1]);
					mesh.meshCells.push_back(meshCells[3 * i + 2]);
				}
			}
			for (auto& ite = cutIdx.rbegin(); ite != cutIdx.rend(); ite++) {
				auto& item = *ite;
				for (int i = 0; i < mesh.meshCells.size(); i++) {
					if (mesh.meshCells[i] > item)
						mesh.meshCells[i]--;
				}
			}
			meshPositions.swap(mesh.meshPositions);
			meshCells.swap(mesh.meshCells);
			return true;
		}
		//before each deformation, translate and scale src to fit dst
		void normalizeScaleTranslation(Mesh& dst) {
			vec3 min, max;
			normalizeScaleTranslation(dst, min, max);
		}
		void normalizeScaleTranslation(Mesh& dst, vec3& min, vec3& max) {
			getAabbWithoutLeg(min, max);
			cout << "min,max:" << min << "|" << max << endl;
			vec3 anchorSrc(0, (max.y + min.y) *0.5, min.z);
			vec3 scaleSrc = max - min;

			dst.getAabbWithoutLeg(min, max);
			cout << "min,max:" << min << "|" << max << endl;
			vec3 anchorDst(0, (max.y + min.y) *0.5, min.z);
			vec3 scaleDst = max - min;
			cout << "anchorSrc,anchorDst:" << anchorSrc << "|" << anchorDst << endl;
			vec3 scale = scaleDst / scaleSrc;
			cout << "scale:" << scale << endl;

			for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
				*ite = (*ite - anchorSrc)*scale;
			}
			for (auto ite = dst.meshPositions.begin(); ite != dst.meshPositions.end(); ite++) {
				*ite -= anchorDst;
			}
		}
		void normalizeScale(Mesh& dst) {
			vec3 min, max;
			normalizeScale(dst, min, max);
		}
		void normalizeScale(Mesh& dst, vec3& min, vec3& max) {
			getAabbWithoutLeg(min, max);
			vec3 scaleSrc = max - min;

			dst.getAabbWithoutLeg(min, max);
			vec3 scaleDst = max - min;
			vec3 scale = scaleDst / scaleSrc;
			cout << "scale" << scale << endl;
			for (auto ite = meshPositions.begin(); ite != meshPositions.end(); ite++) {
				*ite = (*ite)*scale;
			}
		}
		//keep the integer part of sqrt, fast edition, tested to be the same as (int)sqrt(x)
		int mySqrt(int x, int max)
		{
			int lo = 0, hi = max + 1, mi;
			while (lo<hi)
			{
				mi = (lo + hi) >> 1;
				if (mi*mi + mi - 1<x) lo = mi + 1;
				else if (mi*mi + mi - 1>x) hi = mi;
				else return mi;
			}
			//if not found
			if (mi*mi + mi - 1<x) return mi;  //(mi-1)        (mi)(x here)(mi+1)
			else return mi - 1;       //(mi-1)(x here)(mi)        (mi+1)
		}
		//two ramdom weights with Barycentric coordinates
		vec3 sampleOne(int cellIdx) {
			auto& p0 = meshPositions[meshCells[3 * cellIdx]];
			auto& p1 = meshPositions[meshCells[3 * cellIdx + 1]];
			auto& p2 = meshPositions[meshCells[3 * cellIdx + 2]];
			double r1 = (double)rand() / RAND_MAX;
			double r2 = (double)rand() / RAND_MAX;
			if (r1 < r2) {
				return p0*r1 + p1* (r2 - r1) + p2*(1 - r2);
			}
			else
				return p0*r2 + p1* (r1 - r2) + p2*(1 - r1);
		}
		//treat a triangle as half a box, two ramdom weights with edges, if p+q>0.5, use its mirror in the other triangle
		vec3 sampleOne(int cellIdx, double progress) {
			static const int PIECES = 32;
			auto& p0 = meshPositions[meshCells[3 * cellIdx]];
			auto& p1 = meshPositions[meshCells[3 * cellIdx + 1]];
			auto& p2 = meshPositions[meshCells[3 * cellIdx + 2]];
			auto v1 = p1 - p0;
			auto v2 = p2 - p1;
			int intProgressTwice = 2 * int(progress*(PIECES*(PIECES + 1) / 2 - 1));
			int i1 = mySqrt(intProgressTwice, PIECES * 2);
			int i2 = ((intProgressTwice - i1*i1 - i1) >> 1);
			double r1 = (i1 + 0.666666) / PIECES, r2 = (i2 + 0.333333) / PIECES;
			return v1*r1 + v2*r2 + p0;
		}
		//average sample numSample pts on the surface, put them into sampled
		void avgSample(Mesh& sampled, int numSample) {
			if (meshCells.size() == 0) {
				cout << "meshCells ==0 in avgSample" << endl;
				exit(1);
			}
			needAreas();
			vector<double> vecArea;
			double sum = 0;
			for (int i = 0; i < meshCells.size() / 3; i++) {
				sum += areas[i];
				vecArea.push_back(sum);
			}
			double step = sum / numSample;
			int current = 0;
			sampled.meshPositions.resize(numSample);
			for (int i = 0; i < numSample; i++) {
				while (vecArea[current] < step*i) {
					current++;
				}
				sampled.meshPositions.push_back(sampleOne(current, (vecArea[current] - step*i) / (areas[current])));
			}
		}
		//average sample numSample pts on the surface, put them into sampled
		void avgSampleWithIndices(Mesh& sampled, int numSample, vector<int>& sampleIndices) {
			if (meshCells.size() == 0) {
				cout << "meshCells ==0 in avgSampleWithIndices" << endl;
				exit(1);
			}
			needAreas();
			vector<double> vecArea;
			double sum = 0;
			for (int i = 0; i < meshCells.size() / 3; i++) {
				sum += areas[i];
				vecArea.push_back(sum);
			}
			double step = sum / numSample;
			int current = 0;
			sampled.meshPositions.resize(numSample);
			sampleIndices.resize(numSample);
			for (int i = 0; i < numSample; i++) {
				while (vecArea[current] < step*i) {
					current++;
				}
				sampled.meshPositions[i]=sampleOne(current, (vecArea[current] - step*i) / (areas[current]));
				sampleIndices[i] = current;
			}
		}
	};
};
#endif //DEMO_H