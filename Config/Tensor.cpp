#include "Tensor.h"
using namespace CGP;
using namespace rttr;
#include "../FileIO/FileIO.h"
RTTR_REGISTRATION
{
registration::class_<Tensor>("CGP::Tensor").constructor<>()
	.property("n_id_", &CGP::Tensor::n_id_)
	.property("n_exp_", &CGP::Tensor::n_exp_)
	.property("n_data_type_", &CGP::Tensor::n_data_type_)
	.property("tensor_file_", &CGP::Tensor::tensor_file_)
	.property("template_", &CGP::Tensor::template_)
	.property("eigen_value_", &CGP::Tensor::eigen_value_)
	;

}
void Tensor::saveByIdExp(const std::string& res_path, int weight) const
{
	int shift_size = n_vertex_ * 3;
	floatVec add_mean = data_;
/*
	std::transform(add_mean.begin() + shift_size,
		add_mean.begin() + shift_size*n_id_,
		add_mean.begin(),
		add_mean.begin() + shift_size,
		std::plus<float>());
*/
#pragma omp parallel for
	for (int iter_delta = 1; iter_delta < n_id_*n_exp_; iter_delta++)
	{
		for (int i = 0; i < weight; i++)
		{
			std::transform(data_.begin() + iter_delta * n_vertex_ * 3,
				data_.begin() + (iter_delta + 1) * n_vertex_ * 3,
				data_.begin() + iter_delta * n_vertex_ * 3,
				add_mean.begin() + iter_delta * n_vertex_ * 3,
				std::plus<float>());
		}
		std::transform(data_.begin() + iter_delta * n_vertex_ * 3,
			data_.begin() + (iter_delta + 1) * n_vertex_ * 3,
			data_.begin(),
			add_mean.begin() + iter_delta * n_vertex_ * 3,
			std::plus<float>());
	}
#pragma omp parallel for
	for (int iter_id = 0; iter_id < n_id_; iter_id++)
	{
		for (int iter_exp = 0; iter_exp < n_exp_; iter_exp++)
		{
			MeshCompress temp_obj = template_obj_;
			int shift_iter = iter_id * n_exp_ + iter_exp;
			SG::safeMemcpy(temp_obj.pos_.data(), add_mean.data()+shift_size* shift_iter, n_vertex_ * 3 * sizeof(float));
			temp_obj.saveObj(res_path + "id_" + std::to_string(iter_id) + "_exp_" + std::to_string(iter_exp) + ".obj");
		}
	}
}

void Tensor::init()
{
	//load obj
	template_obj_.loadObj(root_ + template_);
	n_vertex_ = template_obj_.n_vertex_;
	//load binary
	if (DataType(n_data_type_) == DataType::FLOAT)
	{
		FILEIO::loadFixedSizeDataFromBinary(root_ + tensor_file_, data_);
		FILEIO::loadFixedSizeDataFromBinary(root_ + eigen_value_, ev_data_);
	}
	if (data_.size() != template_obj_.n_vertex_*n_id_ * 3)
	{
		LOG(ERROR) << "loading data_.size() failed, notice the data_ should be 3x vertex." << std::endl;
	}
	if (ev_data_.size() < n_id_-1)
	{
		//ev_data_.size() >= n_id_
		LOG(ERROR) << "eigen value size is not fit." << std::endl;
	}
}

void Tensor::save(const cstr& root)
{
	SG::needPath(root);
	json config;

	template_obj_.saveObj(root + "mean.obj");

	config["template_"] = "mean.obj";
	config["n_id_"] = n_id_;
	config["n_exp_"] = 1;
	config["n_data_type_"] = 5;
	config["tensor_file_"] = "pca.bin";
	config["eigen_value_"] = "eigen_value.bin";
	FILEIO::saveJson(root + "config.json", config);
	FILEIO::saveToBinary(root + "pca.bin", data_);
	FILEIO::saveToBinary(root + "eigen_value.bin", ev_data_);	
}



