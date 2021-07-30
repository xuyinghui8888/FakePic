#include <stdexcept>
#include "SafeGuard.h"

#ifdef __linux
int memcpy_s(void* p_dst, size_t size_dst, void const* p_src, size_t size_src)
{
	memcpy(p_dst, p_src, size_dst);
	return 0;
}
#endif

int SG::safeMemcpy(void* p_dst, void const* p_src, size_t size)
{
	int r = memcpy_s(p_dst, size, p_src, size);
	if (r)
	{
		LOG(ERROR) << "memcpy_s failed." << std::endl;
	}
	return r;
}

int SG::safeMemset(void* p_dst, size_t size)
{
	memset(p_dst, 0, size);
	return 0;
}

bool SG::checkMeshDataVec(const CGP::float3Vec& data_vec)
{
	if (data_vec.empty())
	{
		return true;
	}
	int dim = data_vec[0].size();
	for (int i = 0; i < data_vec.size(); i++)
	{
		for (int iter_dim = 0; iter_dim < dim; iter_dim++)
		{
			if(!std::isfinite(data_vec[i][iter_dim]) 
				|| std::isinf(data_vec[i][iter_dim])
				|| std::isnan(data_vec[i][iter_dim]))
			{
				LOG(ERROR) << "numeric error happens at " << i << ", data is " << data_vec[i] << std::endl;
				return false;
			}
		}
	}
	return true;
}

bool SG::checkTriIdx(const CGP::uintVec& tri_idx, int max_vertex_idx)
{
	if (tri_idx.empty())
	{
		return true;
	}
	if(tri_idx.size() % 3 != 0)
	{
		LOG(ERROR) << "triangle size is not divided by 3. " << std::endl;
		return false;
	}
	int n_tri = tri_idx.size() / 3;
	for (int i = 0; i < n_tri; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (tri_idx[i * 3 + j] < 0 || tri_idx[i * 3 + j] > max_vertex_idx)
			{
				LOG(ERROR) << "tri data error" << std::endl;
				return false;
			}
			if (tri_idx[i * 3] == tri_idx[i * 3 + 1]
				|| tri_idx[i * 3] == tri_idx[i * 3 + 2]
				|| tri_idx[i * 3 + 1] == tri_idx[i * 3 + 2])
			{
				LOG(ERROR) << "tri vertex data have same vertex in triangle, tri_idx: "
					<< i << ", tri_idxs: " << tri_idx[i * 3] << ","
					<< tri_idx[i * 3 + 1] << ","
					<< tri_idx[i * 3 + 2] << std::endl;
				return false;
			}
		}
	}
	return true;
}

bool SG::checkMesh(const CGP::float3Vec& data_vec, const CGP::uintVec& tri_idx)
{
	if (data_vec.empty() || tri_idx.empty())
	{
		LOG(WARNING) << "mesh is empty, return default true." << std::endl;
		return true;
	}
	int n_vertex = data_vec.size();
	return checkMeshDataVec(data_vec) && checkTriIdx(tri_idx, n_vertex - 1);
}

bool SG::isDigits(const std::string &str)
{
	if (str == "")
	{
		return false;
	}
	return std::all_of(str.begin(), str.end(), ::isdigit);
}

bool SG::isExist(const CGP::cstr& file_name)
{
	return boost::filesystem::exists(file_name);
}

void SG::needPath(const CGP::cstr& path)
{
	if (path == "")
	{
		LOG(INFO) << "empty path." << std::endl;
		return;
	}
	if (!boost::filesystem::exists(path))
	{
		LOG(INFO) << "create: " << path << std::endl;
		!boost::filesystem::create_directories(path);
	}
	else
	{
		LOG(INFO) << "path exist. " << path << std::endl;
	}
}

std::string SG::exec(const CGP::cstr& cmd)
{
#ifndef __linux
	LOG(INFO) << "cmd: " << cmd << std::endl;
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd.c_str(), "r"), _pclose);
	if (!pipe) 
	{
		LOG(ERROR) << "popen() failed!" << std::endl;
		return "";
		//throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) 
	{
		result += buffer.data();
	}
	return result;
#else
	return "";
#endif
}