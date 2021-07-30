#ifndef __FILEIO_H__
#define __FILEIO_H__
#include "../Basic/CGPBaseHeader.h"
namespace FILEIO
{
	using json = nlohmann::json;

	enum class FILE_TYPE
	{
		IMAGE,
		MESH,
	};
	CGP::cstr head(CGP::cstr const& source, size_t const length);
	CGP::cstr tail(CGP::cstr const& source, size_t const length);
	CGP::cstr dump(CGP::cstr const& source, size_t const length);
	CGP::cstr trim(const std::string& s);
	bool isWhiteChar(char c);
	int splitString(const std::string &txt, std::vector<std::string> &strs, char ch);
	int getPair(const std::string& file_name, CGP::intVec&src_p, CGP::intVec& dst_p);
	CGP::intVec loadIntDynamic(const std::string& filename, char sep = ',');
	void loadIntDynamic(const std::string& filename, CGP::intVec& res, char sep = ',');
	CGP::floatVec loadFloatDynamic(const std::string& filename, char sep = ',');
	
	void loadBinaryC(const std::string& file_name, CGP::float3Vec& res);
	void loadBinary(const std::string& file_name, CGP::float3Vec& res);
	void saveStringBinary(std::string filepath, const std::string& src);

	CGP::cstrVec getFolderFiles(const CGP::cstr& folder, const FILE_TYPE& type, bool is_recursive = false);
	CGP::cstrVec getFolderFiles(const CGP::cstr& folder, const CGP::cstr& extension);
	CGP::cstrVec getFolderFiles(const CGP::cstr& folder, const CGP::cstr& extension);
	/*
	recursive:
	true, return aboslute path;
	false, return relative path relative to folder
	*/
	CGP::cstrVec getFolderFiles(const CGP::cstr& folder, const CGP::cstrVec& extension, bool recursive = false);

	json loadJson(const std::string& filepath);
	
	template<class T1, class T2>
	void copyFile(const T1& src, const T2& dst)
	{
		boost::filesystem::copy_file(src, dst, boost::filesystem::copy_option::overwrite_if_exists);
	}
	
	template<class T>
	void saveJson(const T& filepath, const json& res)
	{
		std::ofstream file(filepath);
		if (!file.is_open())
		{
			LOG(ERROR) << "file not open: " << filepath << std::endl;
			return;
		}
		//file.write(src.c_str(), src.size());
		file << res;
		file.close();
	}

	template<class T>
	CGP::cstr getFileNameWithoutExt(const T& path)
	{
		boost::filesystem::path src(path);
		if (boost::filesystem::is_directory(src))
		{
			return "";
		}
		else
		{
			//return src.filename().string();
			return src.stem().string();
		}
	}

	template<class T>
	void loadEigenMat(const std::string& file_name, T& res)
	{
		std::ifstream in_stream(file_name);
		if (!in_stream.is_open())
		{
			LOG(ERROR) << file_name << ", open failed." << std::endl;
			return;
		}
		//get rows cols
		int rows, cols;
		in_stream >> rows >> cols;
		res.resize(rows, cols);
		for (int iter_row = 0; iter_row < rows; iter_row++)
		{
			for (int iter_col = 0; iter_col < cols; iter_col++)
			{
				in_stream >> res(iter_row, iter_col);
			}
		}
	}

	template<class T>
	void loadEigenMat(const std::string& file_name, int rows, int cols, T& res)
	{
		std::ifstream in_stream(file_name);
		if (!in_stream.is_open())
		{
			LOG(ERROR) << file_name << ", open failed." << std::endl;
			return;
		}
		//use input rows cols
		res.resize(rows, cols);
		for (int iter_row = 0; iter_row < rows; iter_row++)
		{
			for (int iter_col = 0; iter_col < cols; iter_col++)
			{
				double num;
				in_stream >> num;
				res(iter_row, iter_col) = num;
			}
		}
	}

	template<class T>
	void loadEigenMatToVector(const std::string& file_name, T& res, bool is_app)
	{
		std::ifstream in_stream(file_name);
		if (!in_stream.is_open())
		{
			LOG(ERROR) << file_name << ", open failed." << std::endl;
			return;
		}
		//get rows cols
		int rows, cols;
		in_stream >> rows >> cols;
		int ori_size = res.size();
		if (!is_app)
		{
			ori_size = 0;
			res.resize(rows*cols);
		}
		else
		{			
			res.resize(ori_size + rows * cols);
		}

		for (int iter_row = 0; iter_row < rows; iter_row++)
		{
			for (int iter_col = 0; iter_col < cols; iter_col++)
			{
				in_stream >> res[ori_size + iter_row * cols + iter_col];
			}
		}
	}

	template<class T>
	void loadEigenMatToVector(const std::string& file_name, int rows, int cols, T& res, bool is_app)
	{
		std::ifstream in_stream(file_name);
		if (!in_stream.is_open())
		{
			LOG(ERROR) << file_name << ", open failed." << std::endl;
			return;
		}
		//get rows cols
		int ori_size = res.size();
		if (!is_app)
		{
			ori_size = 0;
			res.resize(rows*cols);
		}
		else
		{
			res.resize(ori_size + rows * cols);
		}

		for (int iter_row = 0; iter_row < rows; iter_row++)
		{
			for (int iter_col = 0; iter_col < cols; iter_col++)
			{
				in_stream >> res[ori_size + iter_row * cols + iter_col];
			}
		}
	}

	template<class T>
	CGP::cstr addZero(const T& input, int pos)
	{
		CGP::cstr s_input = std::to_string(input);
		if (s_input.length() >= pos)
		{
			return s_input;
		}
		else
		{
			CGP::cstr res = std::string(pos - s_input.length(), '0') + s_input;
			return res;
		}
	}

	template<class T>
	void loadFixedSizeDataFromBinary(const std::string& file_name, std::vector<T>& res)
	{
		std::ifstream in_stream;
		in_stream.open(file_name, std::ios::binary);
		if (!in_stream.is_open())
		{
			LOG(ERROR) << "file is not opened: " << file_name << std::endl;
			return;
		}
		in_stream.seekg(0, std::ios::end);
		int data_length = in_stream.tellg();
		if (data_length % sizeof(T) != 0)
		{
			LOG(ERROR) << "data conversion is not safe." << std::endl;
			return;
		}

		int res_length = data_length / sizeof(T);
		res.resize(res_length);

		in_stream.seekg(0, std::ios::beg);
		// read data as a block:
		in_stream.read((char*)res.data(), data_length);
		LOG(INFO) << "reading binary file: " << file_name << std::endl;
		in_stream.close();
	}

	template<class T>
	void saveToBinary(const std::string& file_name, const std::vector<T>& res, int save_size = -1)
	{
		if (save_size == -1)
		{
			save_size = res.size() * sizeof(T);
		}
		std::ofstream fout(file_name, std::ios::binary | std::ios::out);
		std::ostringstream stream;
		std::vector<char> buffer(save_size);
		SG::safeMemcpy(buffer.data(), res.data(), save_size);
		std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<char>(fout));
		fout.close();
	}

	template<class T>
	void saveToStringBinaryViaChar(const std::vector<T>& data, CGP::cstr& res)
	{
		int save_size = data.size() * sizeof(T);
		std::vector<char> buffer(save_size);
		SG::safeMemcpy(buffer.data(), data.data(), save_size);
		res = CGP::cstr(buffer.begin(), buffer.end());
	}

	template<class T>
	void saveToStringBinaryVisOss(const std::vector<T>& data, CGP::cstr& res)
	{
		int save_size = data.size() * sizeof(T);
		std::vector<char> buffer(save_size);
		SG::safeMemcpy(buffer.data(), data.data(), save_size);
		res = CGP::cstr(buffer.begin(), buffer.end());
	}

	template<class T>
	void saveDynamic(const std::string& filename, std::vector<T>& res, const  CGP::cstr& sep)
	{
		std::ofstream fout(filename);
		if (!fout.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}
		for (auto i : res)
		{
			fout << i << sep;
		}
		fout.close();
	}

	template<class T>
	void loadFixSize(const std::string& filename, std::vector<T>& res)
	{
		std::ifstream fin(filename);
		if (!fin.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}
		int num = 0;
		fin >> num;
		if (num < 0)
		{
			LOG(ERROR) << "set read in data size < 0." << std::endl;
			return;
		}
		res.resize(num);
		for (int i = 0; i<num; i++)
		{
			fin >> res[i];
		}
		fin.close();
	}

	template<class T>
	void loadFixSizeEigenFormat(const std::string& filename, std::vector<T>& res)
	{
		std::ifstream fin(filename);
		if (!fin.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}
		int rows = 0;
		int cols = 0;
		fin >> rows >> cols;
		int num = rows * cols;
		if (num < 0)
		{
			LOG(ERROR) << "set read in data size < 0." << std::endl;
			return;
		}
		res.resize(num);
		for (int i = 0; i < num; i++)
		{
			fin >> res[i];
		}
		fin.close();
	}

	template<class T1, class T2>
	void saveFixSize(const std::string& filename, std::vector<T1>& res, const T2& sep)
	{
		std::ofstream fout(filename);
		if (!fout.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}
		fout << res.size() << std::endl;
		for (auto i : res)
		{
			fout << i << sep;
		}
		fout.close();
	}

	template<class T>
	void saveEigenVecCsv(const CGP::cstr& filename, const T& res, const CGP::cstrVec& tag_init = CGP::cstrVec{})
	{
		CGP::cstrVec tag = tag_init;
		std::ofstream fout(filename);
		if (!fout.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}

		if (res.empty())
		{
			LOG(WARNING) << "saving res empty." << std::endl;
			return;
		}

		int res_size = res.size();
		int dim = res[0].size();
		if (tag.size() != res.size())
		{
			LOG(WARNING) << "tag size not fit, fill with nullptr" << std::endl;
			int ori_size = tag.size();
			tag.resize(res.size());
			for (int i = ori_size; i < res.size(); i++)
			{
				tag[i] = "null";
			}
		}
		
		int count_end = 0;
		for (int i = 0; i < res_size* dim; i++)
		{
			fout << tag[i/dim];
			count_end++;
			if (count_end != res_size * dim)
			{
				fout << ",";
			}
			else
			{
				fout << std::endl;
			}
		}

		count_end = 0;
		for (int i = 0; i < res_size; i++)
		{
			for (int iter_dim = 0; iter_dim < dim; iter_dim++)
			{
				fout << res[i][iter_dim];
				count_end++;
				if (count_end != res_size* dim)
				{
					fout << ",";
				}
				else
				{
					fout << std::endl;
				}
			}
		}
		fout.close();
	}

	template<class T>
	void saveVecToCsv(const CGP::cstr& filename, const T& res, int cols, const CGP::cstrVec& tag_init = CGP::cstrVec{})
	{
		CGP::cstrVec tag = tag_init;
		std::ofstream fout(filename);
		if (!fout.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}

		if (res.empty())
		{
			LOG(WARNING) << "saving res empty." << std::endl;
			return;
		}

		int res_size = res.size();
		if (tag.size() < cols)
		{
			LOG(WARNING) << "tag size < cols, fill with nullptr" << std::endl;
			int ori_size = tag.size();
			tag.resize(cols);
			for (int i = ori_size; i < cols; i++)
			{
				tag[i] = "null";
			}
		}
		else if (tag.size() > cols)
		{
			LOG(WARNING) << "tag size > cols, trunc size " << std::endl;
		}

		int count_end = 0;
		for (int i = 0; i < cols; i++)
		{
			fout << tag[i];
			count_end++;
			if (count_end != cols)
			{
				fout << ",";
			}
			else
			{
				fout << std::endl;
			}
		}

		if (res_size % cols != 0)
		{
			LOG(WARNING) << "has empty entries: " << std::endl;
		}

		int rows = (res_size + 1) / cols;
		count_end = 0;
		for (int i = 0; i < res_size; i++)
		{
			fout << res[i];
			count_end++;
			if (count_end % cols != 0)
			{
				fout << ",";
			}
			else
			{
				fout << std::endl;
				count_end = 0;
			}
		
		}
		fout.close();
	}

	template<class T>
	void saveEigenDynamic(const  CGP::cstr& filename, const T& res)
	{
		std::ofstream fout(filename);
		if (!fout.is_open())
		{
			LOG(ERROR) << filename << "  not opened." << std::endl;
			return;
		}
		
		if (res.cols() == 1)
		{
			fout << res.rows() << " " << res.cols() << std::endl;
			fout << res.transpose() << std::endl;
		}
		else
		{
			fout << res.rows() << " " << res.cols() << std::endl;
			fout << res << std::endl;
		}	
	
		fout.close();
	}
}
#endif