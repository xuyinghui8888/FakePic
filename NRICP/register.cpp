//ceres relies on glog, need to add preprocessor GOOGLE_GLOG_DLL_DECL= to use static glog properly
#include "../FileIO/FileIO.h"
#include "register.h"
#include "residual.h"
#include <boost/program_options.hpp>
#include <io.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <time.h>
using namespace std;
using namespace Eigen;
using namespace demo;
using namespace boost;
using namespace NRICP;

#define WITH_ANKLE 1

//program options
double lamda = 0.1;
int base = 0, endNum = 0, iteCount = 0, numSamples = 1000000;

//const values
const char* tplName = "left9.obj";
const char* markerNameSrc = "markerLeft9.txt";
const char* workingDir = "D:/work/footdata/";

void readMarkers(const char* markerName, int* keyPtIdx) {
	int numMarkers = 7;
	FILE* file = fopen(markerName, "r");
	if (!file) {
		cout << "missing file:" << markerName << endl;
		exit(0);
	}
	char line[200];
	for (int i = 0; i < numMarkers; i++) {
		fgets(line, 200, file);
		stringstream ss(line);
		ss >> keyPtIdx[i];
	}
	fclose(file);
}

//脚后跟下部
static int detectKeyPoint12(Mesh& mesh) {
	int index = -1;
	double max = -1e5;
	for (int i = 0; i < mesh.meshPositions.size(); i++) {
		auto& p = mesh.meshPositions[i];
		if (p.z > 40)continue;
		if (p.y > 20)continue;
		auto val = -p.y - 3 * p.z;
		if (val > max) {
			max = val;
			index = i;
		}
	}
	assert(index >= 0);
	cout << "key point 12 heel down:" << mesh.meshPositions[index] << endl;
	return index;
}
void parseParas(int argc, char** argv) {
	program_options::options_description opts("demo options");  //核心类
	opts.add_options()  //增加程序选项
		("help", "just a help info")
		("lamda", program_options::value<double>(), "lamda")
		("base", program_options::value<int>(), "start index")
		("endNum", program_options::value<int>(), "end number of model")
		("numEigs", program_options::value<int>(), "numEigs")
		("numSamples", program_options::value<int>(), "numSamples")
		("iteCount", program_options::value<int>(), "iterations for matching");
	program_options::variables_map vm;  //选项存储map容器
	try {
		program_options::store(program_options::parse_command_line(argc, argv, opts), vm);  //解析存储
	}
	catch (...) {
		cout << "illegal parameter exists!" << endl;
		exit(1);
	}

	if (vm.count("help")) {
		cout << opts << endl;
	}
	if (vm.count("base")) {
		base = vm["base"].as<int>();
	}
	if (vm.count("endNum")) {
		endNum = vm["endNum"].as<int>();
	}
	if (vm.count("lamda")) {
		lamda = vm["lamda"].as<double>();
	}
	if (vm.count("iteCount")) {
		iteCount = vm["iteCount"].as<int>();
	}
	if (vm.count("numSamples")) {
		numSamples = vm["numSamples"].as<int>();
	}
	cout << "lamda:" << lamda << endl;
	cout << "iteCount:" << iteCount << endl;
	cout << "numSamples:" << numSamples << endl;
	cout << "base:" << base << endl;
	cout << "endNum:" << endNum << endl;
}
void NRICP::reg(const std::string& srcName, const std::string& markerName, const std::string& dstName, 
	const std::vector<int>& roi, const std::string& saveNames, int numIte, vector<Mesh*>& meshes)
{
	Mesh src, dst, sampled;
	src.loadMesh(srcName.c_str(), false);
	dst.loadMesh(dstName.c_str(), false);
	
	std::vector<vec3> fix_pos = src.meshPositions;
	
	//sampled.loadMesh(dstName.c_str(), false);
	int numMarkers;
	vector<int> sampleIndices;
	for (int i = 0; i < dst.meshPositions.size(); i++)
	{
		sampleIndices.push_back(i);
	}
	dst.avgSampleWithIndices(sampled, numSamples, sampleIndices);
	std::vector<std::vector<int>> adj;
	src.getAdj(adj);
	auto numPts = src.meshPositions.size();
	std::vector<int> keyPtIdxSrc, keyPtIdxDst;
	numMarkers = FILEIO::getPair(markerName, keyPtIdxSrc, keyPtIdxDst);
	for (int i = 0; i < numMarkers; i++) {
		cout << src.meshPositions[keyPtIdxSrc[i]] << " -> " << dst.meshPositions[keyPtIdxDst[i]] << endl;
	}

	MyKDtree dstTree(sampled.meshPositions);
	double *xi = new double[numPts*NUM_FREEDOM];
	const int lenPiece = sizeof(xi[0])*NUM_FREEDOM;
	memset(xi, 0, lenPiece);
	xi[0] = 1; xi[5] = 1; xi[10] = 1;
	for (int i = 1; i < numPts; i++) {
		memcpy(xi + NUM_FREEDOM*i, xi, lenPiece);
	}
	vector<int> boundaryPts;
	src.getBoundaryPts(boundaryPts);
	std::cout << "boundaryPTS: " << std::endl;
	for (int i = 0; i < boundaryPts.size(); i++)
	{
		std::cout << boundaryPts[i] << ",";
	}
	std::cout << endl;
	//round1:		1smooth+10marker+0height
	//round2:1data+1smooth+10marker+0height
	//round3:1data+0.1smooth+0.1marker+0height
	//round4:1data+0.01smooth		+0height
	
	std::set<int> move_set(roi.begin(), roi.end());
	const int iteLamdaCount = 4;
	for (int iteLamda = 0; iteLamda < iteLamdaCount; iteLamda++) {
		Problem problem;
		double alpha = 0, beta = 0, gamma = 0, delta = 0;
		if (iteLamda == 0) {
			alpha = 0; beta = 1; gamma = 10; delta = 0;
		}
		else if (iteLamda == 1) {
			alpha = 1; beta = 1; gamma = 10; delta = 0;
		}
		else if (iteLamda == 2) {
			alpha = 1; beta = 0.1; gamma = 0.1; delta = 0;
		}
		else {
			alpha = 1; beta = 0.01; gamma = 0; delta = 0;
		}
		//更改alpha
		if (alpha > 0) {
			for (int i = 0; i < numPts; i++) 
			{
				if (move_set.count(i))
				{
					problem.AddResidualBlock(
						DataFunctor2::Create(src.meshPositions[i], dstTree, dst, sampleIndices),
						NULL,
						xi + i * NUM_FREEDOM);
				}
				else
				{
					problem.AddResidualBlock(
						MarkerFunctor::Create(0.1, src.meshPositions[i], fix_pos[i]),
						NULL,
						xi + i * NUM_FREEDOM);
				}

			}
		}
		if (beta > 0) {
			for (int i = 0; i < numPts; i++) {
				for (int j = 0; j < adj[i].size(); j++) {
					if (i > adj[i][j])
						continue;
					problem.AddResidualBlock(
						SmoothFunctor2::Create(sqrt(beta*lamda), src.meshPositions[i], src.meshPositions[adj[i][j]]),
						NULL,
						xi + i*NUM_FREEDOM, xi + adj[i][j] * NUM_FREEDOM);
				}
			}
		}
		if (gamma>0){
			for (int i = 0; i < numMarkers; i++) {
				problem.AddResidualBlock(
					MarkerFunctor::Create(sqrt(gamma), src.meshPositions[keyPtIdxSrc[i]], dst.meshPositions[keyPtIdxDst[i]]),
					NULL,
					xi + keyPtIdxSrc[i] *NUM_FREEDOM);
			}
		}
		if (delta > 0) {
			for (auto&& i : boundaryPts) {
				problem.AddResidualBlock(
					HeightFunctor::Create(sqrt(delta), src.meshPositions[i]),
					NULL,
					xi + i*NUM_FREEDOM);
			}
		}

		Solver::Options options;
		options.max_num_iterations = numIte;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		options.use_nonmonotonic_steps = true;
		options.update_state_every_iteration = true;

		Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << "\n";

		//save resulting mesh
		Mesh *newMesh=new Mesh();
		newMesh->copy(src);
		for (int i = 0; i < newMesh->meshPositions.size(); i++) {
			auto v = transformPt(xi + i*NUM_FREEDOM, newMesh->meshPositions[i]);
			newMesh->meshPositions[i] = vec3(v[0], v[1], v[2]);
		}
		std::string save_iter_name = saveNames + std::to_string(iteLamda) + ".obj";
		newMesh->saveMesh(save_iter_name.c_str(), true);
		if (iteLamda == iteLamdaCount - 1)
#pragma omp critical(a)
		{
			meshes.push_back(newMesh);
		}
		//no need to release ExponentialResidual and NumericDiffCostFunction, else error
		for (auto&& ite : options.callbacks)
			delete ite;
	}
	delete[] xi;
}

void NRICP::reg(const std::string& srcName, const std::string& markerName, const std::string& dstName,
	const std::vector<int>& roi, const std::vector<int>& boundaryPts, const std::string& saveNames, int numIte, vector<Mesh*>& meshes)
{
	Mesh src, dst, sampled;
	src.loadMesh(srcName.c_str(), false);
	dst.loadMesh(dstName.c_str(), false);

	std::vector<vec3> fix_pos = src.meshPositions;

	//sampled.loadMesh(dstName.c_str(), false);
	int numMarkers;
	vector<int> sampleIndices;
	for (int i = 0; i < dst.meshPositions.size(); i++)
	{
		sampleIndices.push_back(i);
	}
	dst.avgSampleWithIndices(sampled, numSamples, sampleIndices);
	std::vector<std::vector<int>> adj;
	src.getAdj(adj);
	auto numPts = src.meshPositions.size();
	std::vector<int> keyPtIdxSrc, keyPtIdxDst;
	numMarkers = FILEIO::getPair(markerName, keyPtIdxSrc, keyPtIdxDst);
	for (int i = 0; i < numMarkers; i++) {
		cout << src.meshPositions[keyPtIdxSrc[i]] << " -> " << dst.meshPositions[keyPtIdxDst[i]] << endl;
	}

	MyKDtree dstTree(sampled.meshPositions);
	double *xi = new double[numPts*NUM_FREEDOM];
	const int lenPiece = sizeof(xi[0])*NUM_FREEDOM;
	memset(xi, 0, lenPiece);
	xi[0] = 1; xi[5] = 1; xi[10] = 1;
	for (int i = 1; i < numPts; i++) {
		memcpy(xi + NUM_FREEDOM * i, xi, lenPiece);
	}

	//round1:		1smooth+10marker+0height
	//round2:1data+1smooth+10marker+0height
	//round3:1data+0.1smooth+0.1marker+0height
	//round4:1data+0.01smooth		+0height

	std::set<int> move_set(roi.begin(), roi.end());
	const int iteLamdaCount = 4;
	for (int iteLamda = 0; iteLamda < iteLamdaCount; iteLamda++) {
		Problem problem;
		double alpha = 0, beta = 0, gamma = 0, delta = 0;
		if (iteLamda == 0) {
			alpha = 0; beta = 1; gamma = 10; delta = 0;
		}
		else if (iteLamda == 1) {
			alpha = 1; beta = 1; gamma = 10; delta = 0;
		}
		else if (iteLamda == 2) {
			alpha = 1; beta = 0.1; gamma = 0.1; delta = 0;
		}
		else {
			alpha = 1; beta = 0.01; gamma = 0; delta = 0;
		}
		//更改alpha
		if (alpha > 0) {
			for (int i = 0; i < numPts; i++)
			{
				if (move_set.count(i))
				{
					problem.AddResidualBlock(
						DataFunctor2::Create(src.meshPositions[i], dstTree, dst, sampleIndices),
						NULL,
						xi + i * NUM_FREEDOM);
				}
				else
				{
					problem.AddResidualBlock(
						MarkerFunctor::Create(0.1, src.meshPositions[i], fix_pos[i]),
						NULL,
						xi + i * NUM_FREEDOM);
				}

			}
		}
		if (beta > 0) {
			for (int i = 0; i < numPts; i++) {
				for (int j = 0; j < adj[i].size(); j++) {
					if (i > adj[i][j])
						continue;
					problem.AddResidualBlock(
						SmoothFunctor2::Create(sqrt(beta*lamda), src.meshPositions[i], src.meshPositions[adj[i][j]]),
						NULL,
						xi + i * NUM_FREEDOM, xi + adj[i][j] * NUM_FREEDOM);
				}
			}
		}
		if (gamma > 0) {
			for (int i = 0; i < numMarkers; i++) {
				problem.AddResidualBlock(
					MarkerFunctor::Create(sqrt(gamma), src.meshPositions[keyPtIdxSrc[i]], dst.meshPositions[keyPtIdxDst[i]]),
					NULL,
					xi + keyPtIdxSrc[i] * NUM_FREEDOM);
			}
		}
		if (delta > 0) {
			for (auto&& i : boundaryPts) {
				problem.AddResidualBlock(
					HeightFunctor::Create(sqrt(delta), src.meshPositions[i]),
					NULL,
					xi + i * NUM_FREEDOM);
			}
		}

		Solver::Options options;
		options.max_num_iterations = numIte;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		options.use_nonmonotonic_steps = true;
		options.update_state_every_iteration = true;

		Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << "\n";

		//save resulting mesh
		Mesh *newMesh = new Mesh();
		newMesh->copy(src);
		for (int i = 0; i < newMesh->meshPositions.size(); i++) {
			auto v = transformPt(xi + i * NUM_FREEDOM, newMesh->meshPositions[i]);
			newMesh->meshPositions[i] = vec3(v[0], v[1], v[2]);
		}
		std::string save_iter_name = saveNames + std::to_string(iteLamda) + ".obj";
		newMesh->saveMesh(save_iter_name.c_str(), true);
		if (iteLamda == iteLamdaCount - 1)
#pragma omp critical(a)
		{
			meshes.push_back(newMesh);
		}
		//no need to release ExponentialResidual and NumericDiffCostFunction, else error
		for (auto&& ite : options.callbacks)
			delete ite;
	}
	delete[] xi;
}

void NRICP::reg(const std::string& srcName, const std::string& dstName, 
	const std::vector<int>& keyPtIdxSrc, const std::vector<int>& keyPtIdxDst,
	const std::vector<int>& roi, const std::string& saveNames, int numIte, vector<Mesh*>& meshes)
{
	Mesh src, dst, sampled;
	src.loadMesh(srcName.c_str(), false);
	dst.loadMesh(dstName.c_str(), false);

	std::vector<vec3> fix_pos = src.meshPositions;

	//sampled.loadMesh(dstName.c_str(), false);
	vector<int> sampleIndices;
	for (int i = 0; i < dst.meshPositions.size(); i++)
	{
		sampleIndices.push_back(i);
	}
	dst.avgSampleWithIndices(sampled, numSamples, sampleIndices);
	std::vector<std::vector<int>> adj;
	src.getAdj(adj);
	auto numPts = src.meshPositions.size();
	int numMarkers = keyPtIdxDst.size();
	for (int i = 0; i < numMarkers; i++) {
		cout << src.meshPositions[keyPtIdxSrc[i]] << " -> " << dst.meshPositions[keyPtIdxDst[i]] << endl;
	}

	MyKDtree dstTree(sampled.meshPositions);
	double *xi = new double[numPts*NUM_FREEDOM];
	const int lenPiece = sizeof(xi[0])*NUM_FREEDOM;
	memset(xi, 0, lenPiece);
	xi[0] = 1; xi[5] = 1; xi[10] = 1;
	for (int i = 1; i < numPts; i++) {
		memcpy(xi + NUM_FREEDOM * i, xi, lenPiece);
	}
	vector<int> boundaryPts;
	src.getBoundaryPts(boundaryPts);
	//round1:		1smooth+10marker+0height
	//round2:1data+1smooth+10marker+0height
	//round3:1data+0.1smooth+0.1marker+0height
	//round4:1data+0.01smooth		+0height

	std::set<int> move_set(roi.begin(), roi.end());
	const int iteLamdaCount = 4;
	for (int iteLamda = 0; iteLamda < iteLamdaCount; iteLamda++) {
		Problem problem;
		double alpha = 0, beta = 0, gamma = 0, delta = 0;
		if (iteLamda == 0) {
			alpha = 0; beta = 1; gamma = 10; delta = 0;
		}
		else if (iteLamda == 1) {
			alpha = 1; beta = 1; gamma = 10; delta = 0;
		}
		else if (iteLamda == 2) {
			alpha = 1; beta = 0.1; gamma = 0.1; delta = 0;
		}
		else {
			alpha = 1; beta = 0.01; gamma = 0; delta = 0;
		}
		//更改alpha
		if (alpha > 0) {
			for (int i = 0; i < numPts; i++)
			{
				if (move_set.count(i))
				{
					problem.AddResidualBlock(
						DataFunctor2::Create(src.meshPositions[i], dstTree, dst, sampleIndices),
						NULL,
						xi + i * NUM_FREEDOM);
				}
				else
				{
					problem.AddResidualBlock(
						MarkerFunctor::Create(0.1, src.meshPositions[i], fix_pos[i]),
						NULL,
						xi + i * NUM_FREEDOM);
				}

			}
		}
		if (beta > 0) {
			for (int i = 0; i < numPts; i++) {
				for (int j = 0; j < adj[i].size(); j++) {
					if (i > adj[i][j])
						continue;
					problem.AddResidualBlock(
						SmoothFunctor2::Create(sqrt(beta*lamda), src.meshPositions[i], src.meshPositions[adj[i][j]]),
						NULL,
						xi + i * NUM_FREEDOM, xi + adj[i][j] * NUM_FREEDOM);
				}
			}
		}
		if (gamma > 0) {
			for (int i = 0; i < numMarkers; i++) {
				problem.AddResidualBlock(
					MarkerFunctor::Create(sqrt(gamma), src.meshPositions[keyPtIdxSrc[i]], dst.meshPositions[keyPtIdxDst[i]]),
					NULL,
					xi + keyPtIdxSrc[i] * NUM_FREEDOM);
			}
		}
		if (delta > 0) {
			for (auto&& i : boundaryPts) {
				problem.AddResidualBlock(
					HeightFunctor::Create(sqrt(delta), src.meshPositions[i]),
					NULL,
					xi + i * NUM_FREEDOM);
			}
		}

		Solver::Options options;
		options.max_num_iterations = numIte;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		options.use_nonmonotonic_steps = true;
		options.update_state_every_iteration = true;

		Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << "\n";

		//save resulting mesh
		Mesh *newMesh = new Mesh();
		newMesh->copy(src);
		for (int i = 0; i < newMesh->meshPositions.size(); i++) {
			auto v = transformPt(xi + i * NUM_FREEDOM, newMesh->meshPositions[i]);
			newMesh->meshPositions[i] = vec3(v[0], v[1], v[2]);
		}
		std::string save_iter_name = saveNames + std::to_string(iteLamda) + ".obj";
		newMesh->saveMesh(save_iter_name.c_str(), true);
		if (iteLamda == iteLamdaCount - 1)
#pragma omp critical(a)
		{
			meshes.push_back(newMesh);
		}
		//no need to release ExponentialResidual and NumericDiffCostFunction, else error
		for (auto&& ite : options.callbacks)
			delete ite;
	}
	delete[] xi;
}


void writeMat(const char* name, MatrixXd m) {
	ofstream fout(name);
	fout << m.rows() << " " << m.cols() << endl;
	fout << m << endl;
	fout.close();
}
void NRICP::centerize(MatrixXd& input) {
	Eigen::MatrixXd meanVec = input.colwise().mean();
	Eigen::RowVectorXd meanVecRow(Eigen::RowVectorXd::Map(meanVec.data(), input.cols()));
	input.rowwise() -= meanVecRow;
}
void NRICP::centerize(MatrixXd& input, VectorXd& meanVec)
{
	meanVec = input.colwise().mean();
	Eigen::RowVectorXd meanVecRow(Eigen::RowVectorXd::Map(meanVec.data(), input.cols()));
	input.rowwise() -= meanVecRow;
}
void calCovarianceMat(MatrixXd& input, MatrixXd& covMat) {
	Eigen::MatrixXd meanVec = input.colwise().mean();
	Eigen::RowVectorXd meanVecRow(Eigen::RowVectorXd::Map(meanVec.data(), input.cols()));

	Eigen::MatrixXd zeroMeanMat = input;
	zeroMeanMat.rowwise() -= meanVecRow;
	if (input.rows() == 1)
		covMat = (zeroMeanMat.adjoint()*zeroMeanMat) / double(input.rows());
	else
		covMat = (zeroMeanMat.adjoint()*zeroMeanMat) / double(input.rows() - 1);
}
void calPCA(vector<Mesh*>& meshes) {
	auto numPoints = meshes[0]->meshPositions.size();
	int cols = numPoints * 3;
	int cols3 = cols / 3;
	auto rows = meshes.size();
	MatrixXd m(rows, cols3 * 3);
	cout << "rows,cols:" << m.rows() << " " << m.cols() << endl;
	{
		Mesh mor;
		mor.meshPositions.resize(numPoints);
		for (auto& mesh : meshes) {
			for (int i = 0; i < mor.meshPositions.size(); i++) {
				mor.meshPositions[i] += mesh->meshPositions[i];
			}
		}
		for (int i = 0; i < mor.meshPositions.size(); i++) {
			mor.meshPositions[i] /= meshes.size();
		}
		for (auto& mesh : meshes) {
			for (int i = 0; i < mor.meshPositions.size(); i++) {
				mesh->meshPositions[i] -= mor.meshPositions[i];
			}
		}
		mor.meshCells.insert(mor.meshCells.end(), meshes[0]->meshCells.begin(), meshes[0]->meshCells.end());
#ifdef WITH_ANKLE
		mor.saveMesh("mor0730.obj", true);
#else
		mor.saveMesh("morNoAnkle0730.obj", true);
#endif
	}
	for (int i = 0; i < rows; i++) {
		auto& positions = meshes[i]->meshPositions;
		for (int j = 0; j < cols3; j++) {
			m(i, j * 3) = positions[j].x;
			m(i, j * 3 + 1) = positions[j].y;
			m(i, j * 3 + 2) = positions[j].z;
		}
	}
	//writeMat("m.txt", m);
	//exit(0);
	centerize(m);
	cout << "finished generating matrix m, press any key." << endl;

	auto dims = meshes.size() - 1;
	for (int i = 0; i < dims; i++)
		delete meshes[i];
	cout << "deleted meshes, check the memory and press any key to continue." << endl;
	Eigen::BDCSVD <MatrixXd> svd(m, ComputeThinU | ComputeThinV);
	auto& eigVectors = svd.matrixV().leftCols(dims);
	cout << "eigVectors rows,cols:" << eigVectors.rows() << " " << eigVectors.cols() << endl;

#ifdef WITH_ANKLE
	ofstream fout("eigs0730.txt");
#else
	ofstream fout("eigs0730NoAnkle.txt");
#endif
	fout << eigVectors.rows() << " " << eigVectors.cols() << endl;
	fout << svd.singularValues().transpose().leftCols(dims) / sqrt(rows - 1) << endl;
	fout << eigVectors << endl;
	fout.flush();
	fout.close();
#ifdef WITH_ANKLE
	ofstream fout2("eigs_40.txt");
#else
	ofstream fout2("eigsNoAnkle_40.txt");
#endif
	fout2 << eigVectors.rows() << " " << 40 << endl;
	fout2 << svd.singularValues().transpose().leftCols(40) / sqrt(rows - 1) << endl;
	fout2 << eigVectors.leftCols(40) << endl;
	fout2.flush();
	fout2.close();
}
void test_nricp() {
	MatrixXd m(3, 5);
	//m << 0,1,2,
	//	0,0.1,0.2,
	//	0, 0.9, 0.6,
	//	0.2, 3, 2.2,
	//	0,0.3,0.5;
	m << 1, 2, 3, 4, 5,
		11, 22, 33, 44, 55,
		123, 321, 33, 21, 1;
	MatrixXd n = m;// .transpose();
	centerize(m);
	centerize(n);
	cout << m << endl << endl;
	JacobiSVD<MatrixXd> svd(n, ComputeThinU | ComputeThinV);
	auto& u = svd.matrixU();
	cout << u << endl << endl;
	auto& v = svd.matrixV();
	cout << v << endl << endl;
	cout << "singlular values:" << svd.singularValues().transpose() / sqrt(m.rows() - 1) << endl;
	cout << "-------" << endl;

	MatrixXd covMat;
	calCovarianceMat(m, covMat);
	JacobiSVD<MatrixXd> svd2(covMat, ComputeThinU | ComputeThinV);
	auto& u2 = svd2.matrixU();
	cout << u2 << endl << endl;
	auto& v2 = svd2.matrixV();
	cout << v2 << endl << endl;
	cout << "singlular values:" << svd2.singularValues().transpose() << endl;
	cout << "-------" << endl;

	VectorXd paras = svd.solve(m.col(0));
	cout << paras << endl << endl;
	paras = svd.solve(m.col(1));
	cout << paras << endl << endl;
	paras = svd.solve(m.col(2));
	cout << paras << endl << endl;
	getchar();
	exit(0);
}

//..\..\x64\Release\ellipse_approximation.exe --lamda=0.3 --iteCount=50 --base=231 --endNum=231
int ori_main(int argc, char** argv) {
	parseParas(argc, argv);
	vector<Mesh*> meshes;
	time_t start = clock();
#pragma omp parallel for num_threads(6)
	for (int index = base; index <= endNum; index++) {
		char srcName[200], markerName[200], dstName[200], saveNames[5][200];
		strcpy(srcName, tplName);

		sprintf(markerName, "%s%d.marker.txt", workingDir, index);
		sprintf(dstName, "%s%d.left.ply", workingDir, index);
		for (int i = 0; i < 5; i++) {
			sprintf(saveNames[i], "%s%d.left.resultDis%d_%.3f.obj", workingDir, index,i, lamda);
		}
		cout << "index---------------:" << index << ", " << srcName << ", " << dstName << endl;

		if (_access(srcName, 0) == -1) {
			printf("src not found, skip\n");
			continue;
		}
		if (_access(dstName, 0) == -1) {
			printf("dst not found, skip\n");
			continue;
		}
		//reg(srcName, markerName, dstName, saveNames,iteCount, meshes);
	}
	cout << "finished registration:" << (clock() - start) / CLOCKS_PER_SEC << endl;
	calPCA(meshes);
	cout << "finished PCA:" << (clock() - start) / CLOCKS_PER_SEC << endl;
	getchar();
	return 0;
}

