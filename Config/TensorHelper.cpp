#include "TensorHelper.h"
using namespace CGP;
using namespace rttr;
#include "../FileIO/FileIO.h"

void tensorHelper::rawToBinary(const json& input)
{
	cstr pca_raw = input["pca_raw"];
	cstr pca_vis = input["pca_vis"];
	cstr binary_res = input["binary_res"];
	cstr template_file = input["template"];
	bool b_vis = input["b_vis"];
	MeshCompress template_mesh(template_file);
	std::ifstream in_data(pca_raw + "mean.txt");
	for (int i = 0; i < template_mesh.n_vertex_; i++)
	{
		in_data >> template_mesh.pos_[i].x()
			>> template_mesh.pos_[i].y()
			>> template_mesh.pos_[i].z();
	}

	template_mesh.saveObj(pca_vis + "mean.obj");
	//get info
	int cols, rows;
	std::ifstream eigen_data(pca_raw + "eigs.txt");
	eigen_data >> rows >> cols;
	floatVec eigen_value(cols);
	for (int i = 0; i < cols; i++)
	{
		eigen_data >> eigen_value[i];
	}
	//LOG(INFO) << "eigen_value: " << eigen_value << std::endl;
	matF basis(rows, cols);
	for (int j = 0; j < rows; j++)
	{
		for (int i = 0; i < cols; i++)
		{
			eigen_data >> basis(j, i);
		}
	}
	if (b_vis)
	{
		for (int i_person = 0; i_person < cols; i_person++)
		{
			MeshCompress temp_mean(pca_vis + "mean.obj");
			for (int i_vertex = 0; i_vertex < rows; i_vertex++)
			{
				temp_mean.pos_[i_vertex / 3][i_vertex % 3] += basis(i_vertex, i_person);
			}
			temp_mean.saveObj(pca_vis + "basis_" + std::to_string(i_person + 1) + ".obj");
		}
	}

	int n_vertex = template_mesh.n_vertex_;
	//include mean put in first and saved in delta;
	int kept_num = 120;
	kept_num = std::min(kept_num, cols);
	floatVec res(kept_num * n_vertex * 3, 0);
	//keep main
	MeshCompress temp_mean(pca_vis + "mean.obj");
	for (int i = 0; i < n_vertex; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			res[3 * i + j] = temp_mean.pos_[i][j];
		}
	}
#pragma omp parallel for
	for (int iter_basis = 1; iter_basis < kept_num; iter_basis++)
	{
		int shift = iter_basis * 3 * n_vertex;
		for (int i = 0; i < n_vertex; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				res[3 * i + j + shift] = basis(3 * i + j, iter_basis - 1);
			}
		}
	}
	FILEIO::saveToBinary(binary_res + "eigen_value.bin", eigen_value);
	FILEIO::saveToBinary(binary_res + "mean_pca.bin", res);
}



