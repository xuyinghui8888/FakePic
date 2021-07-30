#include "Skeleton.h"
#include "../Basic/MeshHeader.h"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace CGP;

//Using a recursive function to iterate all joints' parents' transforms.
doubleVec Skeleton::eulerToWorld(doubleX2Vec bf, doubleX2Vec pos, doubleVec result, int joint_idx) {
	int parent_idx;
	std::vector<double> tmp =
	{
		result[0] * pos[joint_idx][0] + result[1] * pos[joint_idx][1] + result[2] * pos[joint_idx][2],
		result[0] * pos[joint_idx][3] + result[1] * pos[joint_idx][4] + result[2] * pos[joint_idx][5],
		result[0] * pos[joint_idx][6] + result[1] * pos[joint_idx][7] + result[2] * pos[joint_idx][8]
	};

	//If joint's index equals to 0 means the function has iterate to the root.
	//Then the recursive function stops.
	if (joint_idx == 0) {
		return tmp;
	}
	else {
		parent_idx = bf[joint_idx][1];
		tmp[0] += bf[parent_idx][2];
		tmp[1] += bf[parent_idx][3];
		tmp[2] += bf[parent_idx][4];
		tmp = eulerToWorld(bf, pos, tmp, parent_idx); //Iterating back to parents'.
		return tmp;
	}
}

double3E Skeleton::eulerToWorld(doubleX2Vec bf, std::vector<mat3f> pos, double3E result, int joint_idx) {
	int parent_idx;
	double3E tmp = pos[joint_idx].cast<double>() * result;
	//If joint's index equals to 0 means the function has iterate to the root.
	//Then the recursive function stops.
	if (joint_idx == 0) {
		return tmp;
	}
	else {
		parent_idx = bf[joint_idx][1];
		tmp[0] += bf[parent_idx][2];
		tmp[1] += bf[parent_idx][3];
		tmp[2] += bf[parent_idx][4];
		tmp = eulerToWorld(bf, pos, tmp, parent_idx); //Iterating back to parents'.
		return tmp;
	}
}

doubleVec Skeleton::locToWorld(doubleX2Vec bf, doubleVec world, int joint_idx)
{
	int parent_idx;
	doubleX2Vec bf_world;

	//If joint's index equals to 0 means the function has iterate to the root.
	//Then the recursive function stops.
	if (joint_idx == 0) {
		return world;
	}
	else {
		parent_idx = bf[joint_idx][1];
		world[1] += bf[parent_idx][2];
		world[2] += bf[parent_idx][3];
		world[3] += bf[parent_idx][4];
		world = locToWorld(bf, world, parent_idx); //Iterating back to parents'.
		return world;
	}
}

void Skeleton::scaleLocToWorld(doubleVec& scale_div, doubleVec& scale, const intVec& parent_node, int joint_idx, double& res)
{	
	int cur_idx = joint_idx;
	int parent_idx = parent_node[cur_idx];
	if (cur_idx == 0 && parent_idx == 0)
	{
		res = scale[0];
		return;
	}
	
	res = scale[parent_idx] * scale_div[cur_idx];
}

doubleVec Skeleton::rotationMatrix(float radX, float radY, float radZ)
{
	float cX = cos(radX / 180 * M_PI);
	float sX = sin(radX / 180 * M_PI);

	float cY = cos(radY / 180 * M_PI);
	float sY = sin(radY / 180 * M_PI);

	float cZ = cos(radZ / 180 * M_PI);
	float sZ = sin(radZ / 180 * M_PI);

	std::vector<double> matrix3f =
	{
		cZ*cY, cZ*sY*sX - sZ * cY, cZ*sY*cX + sZ * sX,
		sZ*cY, sZ*sY*sX + cZ * cX, sZ*sY*cX - cZ * sX,
		-sY, cY*sX, cY*cX
	};

	return matrix3f;
}

void Skeleton::rotationMatrixEigen(float radX, float radY, float radZ, Eigen::Matrix3f& rotation_matrix)
{	
	Eigen::AngleAxisf xAngle(radX / 180.0 * M_PI, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf yAngle(radY / 180.0 * M_PI,  Eigen::Vector3f::UnitY());
	Eigen::AngleAxisf zAngle(radZ / 180.0 * M_PI,  Eigen::Vector3f::UnitZ());
	Eigen::Quaternion<float> q = xAngle * yAngle * zAngle;
	rotation_matrix = q.matrix();
	LOG(INFO) << "rotation_matrix: " << rotation_matrix << std::endl;
	vecF ea_012 = rotation_matrix.eulerAngles(0, 1, 2)*180.f/ M_PI;
	vecF ea_210 = rotation_matrix.eulerAngles(2, 1, 0)*180.f / M_PI;
	LOG(INFO) << "ea_012: " << ea_012 << std::endl;
	//in xyz format
	LOG(INFO) << "ea_210: " << ea_210.reverse() << std::endl;
	Eigen::AngleAxisf zyxAngle(rotation_matrix);
	
}

//3X3 matrix multiplication
doubleVec Skeleton::mat33Mul(doubleVec mat1, doubleVec mat2)
{
	std::vector<double> mat_mul = { 0,0,0,0,0,0,0,0,0 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			mat_mul[i * 3 + j] = mat1[i * 3 + 0] * mat2[0 + j]
				+ mat1[i * 3 + 1] * mat2[3 + j]
				+ mat1[i * 3 + 2] * mat2[6 + j];
		}
	}

	return mat_mul;
}

void Skeleton::unitTest()
{
	cstr obj_filename = "D:/code/LinearBlendSkinning/taobao/maya.obj";
	cstr bf_input = "D:/code/LinearBlendSkinning/taobao/taobao_ori.bf";
	cstr weis_input = "D:/code/LinearBlendSkinning/taobao/weight.dmat";
	cstr weis_input_normal = "D:/code/LinearBlendSkinning/taobao/weight_normal.dmat";
	cstr pos_input = "D:/code/LinearBlendSkinning/taobao/pose_test.dmat";
	intVec discard = FILEIO::loadIntDynamic("D:/code/LinearBlendSkinning/taobao/skip_2.txt");
	cstr output_path = "D:/avatar/0804_00_skeleton/";
	SG::needPath(output_path);
	//pose data rotation
	doubleX2Vec bf_data, pos_data, weight_data, bf_world, weight_normal, scale_data, pose_world;
	bf_data = readDynamic(bf_input);
	pos_data = readDynamic(pos_input);
	weight_data = readDynamicTrans(weis_input);
	weight_normal = readDynamic(weis_input_normal);
	pose_world = readDynamic("D:/code/LinearBlendSkinning/taobao/pose_world.dmat");
	scale_data = readDynamic("D:/code/LinearBlendSkinning/taobao/scale.txt");

	int bf_rows = bf_data.size();
	int bf_cols = bf_data[0].size();

	float3Vec local_pos(bf_rows);
	intVec parent_node(bf_rows, -1);
	std::vector<mat3f> local_trans(bf_rows);
	std::vector<mat3f> local_trans_inverse(bf_rows);
	//root 位置错误
	mat3f temp;
	rotationMatrixEigen(60, 20, 18, temp);
	for (int i = 0; i < bf_rows; i++)
	{
		parent_node[i] = bf_data[i][1];
		local_pos[i] = float3E(bf_data[i][2], bf_data[i][3], bf_data[i][4]);
		rotationMatrixEigen(pos_data[0][3*i+0], pos_data[0][3 * i + 1], pos_data[0][3 * i + 2], local_trans[i]);
	}

	//test for rest pose https://veeenu.github.io/blog/implementing-skeletal-animation/
	//http://what-when-how.com/advanced-methods-in-computer-graphics/skeletal-animation-advanced-methods-in-computer-graphics-part-2/

	Eigen::Isometry3f T0 = Eigen::Isometry3f::Identity();
	T0.rotate(local_trans[0]);
	T0.pretranslate(float3E(0,	131.074, - 2.257));
	
	Eigen::Isometry3f T1 = Eigen::Isometry3f::Identity();
	T1.rotate(local_trans[1]);
	T1.pretranslate(float3E(9.123,	0,	0));

	LOG(INFO) << (T0.inverse() * T1).rotation() << std::endl;
	LOG(INFO) << (T0.inverse() * T1).translation() << std::endl;


	LOG(INFO) << (T0 * T1).rotation() << std::endl;
	LOG(INFO) << (T0 * T1).translation() << std::endl;




	float3E res;
	locToWorld(local_pos, parent_node, local_trans, 0, res);
	LOG(INFO) << "res: " << res.transpose() << std::endl;
	locToWorld(local_pos, parent_node, local_trans, 1, res);
	LOG(INFO) << "res: " << res.transpose() << std::endl;
	locToWorld(local_pos, parent_node, local_trans, 2, res);
	LOG(INFO) << "res: " << res.transpose() << std::endl;




	MeshCompress obj(obj_filename);

	for (int i = 0; i < bf_rows; i++) {
		doubleVec tmp = { bf_data[i][0], bf_data[i][2], bf_data[i][3], bf_data[i][4] };
		doubleVec temp = locToWorld(bf_data, tmp, i);
		bf_world.push_back(temp);
	}

	for (int i = 0; i < bf_rows; i++)
	{
		bf_world[i][1] = pose_world[0][i*3+0] * 100;
		bf_world[i][2] = pose_world[0][i * 3 + 1] * 100;
		bf_world[i][3] = pose_world[0][i * 3 + 2] * 100;
	}

	for (int pos_num = 0; pos_num < pos_data.size(); pos_num++)
	{
		std::vector<mat3f> pos_matrix;
		for (int j = 0; j < pos_data[pos_num].size(); j = j + 3) {
			mat3f tmp;
			rotationMatrixEigen(pos_data[pos_num][j], pos_data[pos_num][j + 1], pos_data[pos_num][j + 2], tmp);
			pos_matrix.push_back(tmp);
		}

		LOG(INFO) << "pos_matrfix: " << pos_matrix[0] << std::endl;


		//Calculating Chained rotationMatrix
		std::vector<mat3f> pos_matrix_root;
		for (int j = 0; j < pos_matrix.size(); j++) {
			mat3f pos_mat_mul = pos_matrix[j];
			int parent_idx = bf_data[j][1];
			while (parent_idx > 0) {
				pos_mat_mul = pos_mat_mul * pos_matrix[parent_idx];
				parent_idx = bf_data[parent_idx][1];
			}
			pos_matrix_root.push_back(pos_mat_mul);
		}

		doubleVec scale_root = scale_data[pos_num];

		scale_root = scale_data[pos_num];
		//Iterate all parents' transform using recursive function eulerToWorld();
		//Then finally output all joints' world coordinates.
		std::vector<double3E> pose_world;
		for (int i = 0; i < bf_rows; i++) {
			double3E tmp = double3E(bf_data[i][2], bf_data[i][3], bf_data[i][4]);
			double3E temp = eulerToWorld(bf_data, pos_matrix, tmp, i);
			pose_world.push_back(temp);
			//printf("%f %f %f\n", temp[0], temp[1], temp[2]);
		}

		LOG(INFO) << "pos_matrix_root[10]: " << pos_matrix_root[10] << std::endl;
		//V_new = sum( W_iBone * ( R_activePosetoRoot * ( V_rest - i_BoneRest'sParent) + i_BoneNew'sParent) )
		for (int i = 0; i < obj.n_vertex_; i++) 
		{
			double vertex_x = 0.0;
			double vertex_y = 0.0;
			double vertex_z = 0.0;
			float3E vertex_i = obj.pos_[i];
			float3E vertex_xyz = float3E(0, 0, 0);
			for (int j = 0; j < bf_world.size(); j++) 
			{
				float3E bf_j = float3E(bf_world[bf_data[j][1]][1], bf_world[bf_data[j][1]][2], bf_world[bf_data[j][1]][3]);
				vertex_xyz = vertex_xyz + weight_data[j][i] * (scale_root[j] * pos_matrix_root[j].cast<float>() * (vertex_i - bf_j) + bf_j);
			}
			obj.pos_[i] = vertex_xyz;
		}
		obj.saveObj(output_path + std::to_string(pos_num) + ".obj");
	}
}

void Skeleton::unitInverse()
{
	cstr obj_filename = "D:/code/LinearBlendSkinning/taobao/maya.obj";
	cstr bf_input = "D:/code/LinearBlendSkinning/taobao/taobao_ori.bf";
	cstr weis_input = "D:/code/LinearBlendSkinning/taobao/weight.dmat";
	cstr weis_input_normal = "D:/code/LinearBlendSkinning/taobao/weight_normal.dmat";
	cstr pos_input = "D:/code/LinearBlendSkinning/taobao/pose.dmat";
	intVec discard = FILEIO::loadIntDynamic("D:/code/LinearBlendSkinning/taobao/skip_2.txt");
	cstr output_path = "D:/avatar/0804_00_skeleton/";
	SG::needPath(output_path);
	//pose data rotation
	doubleX2Vec bf_data, pos_data, weight_data, bf_world, weight_normal, scale_data, pose_world, add_rotate;
	bf_data = readDynamic(bf_input);
	pos_data = readDynamic(pos_input);
	weight_data = readDynamicTrans(weis_input);
	weight_normal = readDynamic(weis_input_normal);
	pose_world = readDynamic("D:/code/LinearBlendSkinning/taobao/pose_world.dmat");
	scale_data = readDynamic("D:/code/LinearBlendSkinning/taobao/scale.txt");
	add_rotate = readDynamic("D:/code/LinearBlendSkinning/taobao/pose_add.dmat");
	int bf_rows = bf_data.size();
	int bf_cols = bf_data[0].size();

	float3Vec local_pos(bf_rows);
	intVec parent_node(bf_rows, -1);
	std::vector<mat3f> local_trans(bf_rows);
	std::vector<mat3f> local_trans_inverse(bf_rows);
	//root 位置错误
	mat3f temp;
	rotationMatrixEigen(60, 20, 18, temp);
	float3Vec rotate_xyz(bf_rows);
	for (int i = 0; i < bf_rows; i++)
	{
		parent_node[i] = bf_data[i][1];
		local_pos[i] = float3E(bf_data[i][2], bf_data[i][3], bf_data[i][4]);
		rotate_xyz[i] = float3E(pos_data[0][3 * i + 0], pos_data[0][3 * i + 1], pos_data[0][3 * i + 2]);
		rotationMatrixEigen(pos_data[0][3 * i + 0], pos_data[0][3 * i + 1], pos_data[0][3 * i + 2], local_trans[i]);
	}

	//test for rest pose https://veeenu.github.io/blog/implementing-skeletal-animation/
	//http://what-when-how.com/advanced-methods-in-computer-graphics/skeletal-animation-advanced-methods-in-computer-graphics-part-2/

	Eigen::Isometry3f T0 = Eigen::Isometry3f::Identity();
	T0.rotate(local_trans[0]);
	T0.pretranslate(float3E(0, 131.074, -2.257));

	Eigen::Isometry3f T1 = Eigen::Isometry3f::Identity();
	T1.rotate(local_trans[1]);
	T1.pretranslate(float3E(9.123, 0, 0));

	LOG(INFO) << (T0.inverse() * T1).rotation() << std::endl;
	LOG(INFO) << (T0.inverse() * T1).translation() << std::endl;


	LOG(INFO) << (T0 * T1).rotation() << std::endl;
	LOG(INFO) << (T0 * T1).translation() << std::endl;

	std::vector<Eigen::Isometry3f> local(bf_rows), world(bf_rows), bind_pos(bf_rows);
	locToWorld(local_pos, parent_node, rotate_xyz, 0, world, bind_pos, world[0], bind_pos[0]);
	LOG(INFO) << "world: " << std::endl << world[0].matrix() << std::endl;
	LOG(INFO) << "bind_pos: " << std::endl << bind_pos[0].matrix() << std::endl;
	locToWorld(local_pos, parent_node, rotate_xyz, 1, world, bind_pos, world[1], bind_pos[1]);


	LOG(INFO) << "local: " << std::endl;
	LOG(INFO) << bind_pos[1].rotation() << std::endl;
	LOG(INFO) << bind_pos[1].translation() << std::endl;
	LOG(INFO) << "world: " << std::endl;
	LOG(INFO) << world[1].rotation() << std::endl;
	LOG(INFO) << world[1].translation() << std::endl;


	float3E res;
	locToWorld(local_pos, parent_node, local_trans, 0, res);
	LOG(INFO) << "res: " << res.transpose() << std::endl;
	locToWorld(local_pos, parent_node, local_trans, 1, res);
	LOG(INFO) << "res: " << res.transpose() << std::endl;
	locToWorld(local_pos, parent_node, local_trans, 2, res);
	LOG(INFO) << "res: " << res.transpose() << std::endl;

	float3Vec trans_pos(bf_rows, float3E(0,0,0));
	//trans_pos[8] = float3E(0, 5, 0);

	MeshCompress obj(obj_filename);

	for (int i = 0; i < bf_rows; i++) {
		doubleVec tmp = { bf_data[i][0], bf_data[i][2], bf_data[i][3], bf_data[i][4] };
		doubleVec temp = locToWorld(bf_data, tmp, i);
		bf_world.push_back(temp);
	}
#if 1
	for (int i = 0; i < bf_rows; i++)
	{
		bf_world[i][1] = pose_world[0][i * 3 + 0] * 100;
		bf_world[i][2] = pose_world[0][i * 3 + 1] * 100;
		bf_world[i][3] = pose_world[0][i * 3 + 2] * 100;
	}
#endif

	for (int pos_num = 0; pos_num < pos_data.size(); pos_num++)
	{
		std::vector<mat3f> pos_matrix, rotate_add;
		for (int j = 0; j < pos_data[pos_num].size(); j = j + 3) {
			mat3f tmp;
			rotationMatrixEigen(pos_data[pos_num][j], pos_data[pos_num][j + 1], pos_data[pos_num][j + 2], tmp);
			pos_matrix.push_back(tmp);
		}

		for (int j = 0; j < add_rotate[pos_num].size(); j = j + 3) {
			mat3f tmp;
			rotationMatrixEigen(add_rotate[pos_num][j], add_rotate[pos_num][j + 1], add_rotate[pos_num][j + 2], tmp);
			rotate_add.push_back(tmp);
		}

		LOG(INFO) << "pos_matrfix: " << pos_matrix[0] << std::endl;


		//Calculating Chained rotationMatrix
		std::vector<mat3f> pos_matrix_root, rotate_add_root;
		for (int j = 0; j < pos_matrix.size(); j++) {
			mat3f pos_mat_mul = pos_matrix[j];
			int parent_idx = bf_data[j][1];
			while (parent_idx > 0) {
				pos_mat_mul = pos_mat_mul * pos_matrix[parent_idx];
				parent_idx = bf_data[parent_idx][1];
			}
			pos_matrix_root.push_back(pos_mat_mul);
		}

		for (int j = 0; j < pos_matrix.size(); j++) {
			mat3f pos_mat_mul = j == 7 ? rotate_add[j] : pos_matrix[j];
			int parent_idx = bf_data[j][1];
			while (parent_idx > 0) {
				pos_mat_mul = pos_mat_mul * pos_matrix[parent_idx];
				parent_idx = bf_data[parent_idx][1];
			}
			rotate_add_root.push_back(pos_mat_mul);
		}

		doubleVec scale_root(bf_rows, 1);
		for (int j = 0; j < bf_rows; j++)
		{
			scaleLocToWorld(scale_data[0], scale_root, parent_node, j, scale_root[j]);
		}
		//Iterate all parents' transform using recursive function eulerToWorld();
		//Then finally output all joints' world coordinates.
		std::vector<double3E> pose_world;
		for (int i = 0; i < bf_rows; i++) {
			double3E tmp = double3E(bf_data[i][2], bf_data[i][3], bf_data[i][4]);
			double3E temp = eulerToWorld(bf_data, pos_matrix, tmp, i);
			pose_world.push_back(temp);
			//printf("%f %f %f\n", temp[0], temp[1], temp[2]);
		}

		LOG(INFO) << "pos_matrix_root[10]: " << pos_matrix_root[10] << std::endl;
		//V_new = sum( W_iBone * ( R_activePosetoRoot * ( V_rest - i_BoneRest'sParent) + i_BoneNew'sParent) )
		for (int i = 0; i < obj.n_vertex_; i++)
		{
			double vertex_x = 0.0;
			double vertex_y = 0.0;
			double vertex_z = 0.0;
			float3E vertex_i = obj.pos_[i];
			float3E vertex_xyz = float3E(0, 0, 0);
			for (int j = 0; j < bf_world.size(); j++)
			{
				float3E bf_cur = float3E(bf_world[j][1], bf_world[j][2], bf_world[j][3]);
				int parent_idx = parent_node[j];
				float3E bf_parent = float3E(bf_world[parent_idx][1], bf_world[parent_idx][2], bf_world[parent_idx][3]);
				//vertex_xyz = vertex_xyz + weight_data[j][i] * (scale_root[j] * pos_matrix_root[j].inverse()* pos_matrix_root[j]* (vertex_i - bf_j) + bf_j);
				//first scale
				float3E vertex_s = scale_root[j] * (vertex_i - bf_cur) + bf_cur;
				//float3E vertex_r = pos_matrix_root[j].inverse()* pos_matrix_root[j] * rotate_add[j] * (vertex_s - bf_cur) + bf_cur;
				float3E vertex_r = rotate_add[j] * (vertex_s - bf_cur) + bf_cur;
				float3E vertex_t = vertex_r + trans_pos[j];
				vertex_xyz = vertex_xyz + weight_data[j][i] * vertex_t;
			}
			obj.pos_[i] = vertex_xyz;
		}
		obj.saveObj(output_path + std::to_string(pos_num) + ".obj");
		//带着一个很神奇的翻转
	}
}

doubleX2Vec Skeleton::readDynamic(const cstr& bf)
{
	std::fstream bf_file(bf, std::ios_base::in);
	int rows, cols;
	bf_file >> rows >> cols;
	doubleX2Vec bf_data(rows, doubleVec(cols, 0));
	if (bf_file.is_open()) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				bf_file >> bf_data[i][j];
				//printf("%f ", bf_data[i][j]);
			}
		}
	}
	bf_file.close();
	return bf_data;
}

doubleX2Vec Skeleton::readDynamicTrans(const cstr& bf)
{
	std::fstream bf_file(bf, std::ios_base::in);
	int rows, cols;
	bf_file >> rows >> cols;
	doubleX2Vec bf_data(rows, doubleVec(cols, 0));
	if (bf_file.is_open()) 
	{
		int count = 0;
		while (count < rows*cols)
		{
			int iter_rows = count % rows;
			int iter_cols = count / rows;
			bf_file >> bf_data[iter_rows][iter_cols];
			count++;
		}
	}
	bf_file.close();
	return bf_data;
}

void Skeleton::locToWorld(const float3Vec& local_pos, const intVec& parent_node, const std::vector<mat3f>& rotate, int joint_idx, float3E& res)
{
	int cur_idx = joint_idx;
	int parent_idx = parent_node[cur_idx];
	float3E world_pos = local_pos[cur_idx];

	if (cur_idx == 0 && parent_idx == 0)
	{
		res = world_pos;
		return;
	}
	mat3f rotate_iter = rotate[cur_idx];
	while (cur_idx > 0)
	{
		world_pos = rotate[parent_idx].transpose() * world_pos;
		cur_idx = parent_idx;
		parent_idx = parent_node[cur_idx];
	}
	res = local_pos[0] + float3E(world_pos[0], -world_pos[2], world_pos[1]);
}


void Skeleton::locToWorld(const float3Vec& local_pos, const intVec& parent_node, const float3Vec& rotate_xyz, int joint_idx,
	std::vector<Eigen::Isometry3f>& world_list, std::vector<Eigen::Isometry3f>& bind_list, Eigen::Isometry3f& world_matrix, Eigen::Isometry3f& bind_matrix)
{
	int cur_idx = joint_idx;
	int parent_idx = parent_node[cur_idx];
	float3E world_pos = local_pos[cur_idx];

	if (cur_idx == 0 && parent_idx == 0)
	{
		mat3f rotate;
		rotationMatrixEigen(rotate_xyz[joint_idx][0], rotate_xyz[joint_idx][1], rotate_xyz[joint_idx][2], rotate);
		world_matrix.rotate(rotate);
		world_matrix.pretranslate(local_pos[0]);
		bind_matrix = world_matrix.inverse();
		LOG(INFO) << "world_matrix: " << std::endl << world_matrix.matrix() << std::endl;
		LOG(INFO) << "bind_matrix: " << std::endl << bind_matrix.matrix() << std::endl;
		return;
	}
	Eigen::Isometry3f world_parent = world_list[parent_idx];
	mat3f rotate_iter;
	rotationMatrixEigen(rotate_xyz[joint_idx][0], rotate_xyz[joint_idx][1], rotate_xyz[joint_idx][2], rotate_iter);
	Eigen::Isometry3f world_iter;
	world_iter.rotate(rotate_iter);
	world_iter.translate(local_pos[joint_idx]);
	world_matrix = world_list[parent_idx] * world_iter;
	bind_matrix = world_matrix.inverse();
}