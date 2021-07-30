#include "FileIO.h"
bool FILEIO::isWhiteChar(char c) {
	if (c <= 32)
		return true;
	else
		return false;
}

std::string FILEIO::trim(const std::string& s)
{
	if (s.length() == 0)
		return s;
	size_t start = 0, end = s.length() - 1;
	while (isWhiteChar(s[start])) {
		start++;
		if (start >= s.length())
			break;
	}
	while (isWhiteChar(s[end])) {
		end--;
		if (end < 0)
			break;
	}
	if (start > end)
		return std::string("");
	else
		return s.substr(start, end - start + 1);
}

int FILEIO::splitString(const std::string &txt, std::vector<std::string> &strs, char ch)
{
	strs.clear();
	std::istringstream iss(txt);
	std::string token;
	while (std::getline(iss, token, ch))
	{	
		if (token != "")
		{
			strs.push_back(token);
		}
	}
	return strs.size();
}

CGP::intVec FILEIO::loadIntDynamic(const std::string& filename, char sep)
{	
	std::ifstream in_stream(filename);
	if (!in_stream.is_open())
	{
		LOG(ERROR) << filename << ", open failed." << std::endl;
		return {};
	}	  
	in_stream.seekg(0, std::ios::end);
	std::streampos len = in_stream.tellg();
	in_stream.seekg(0, std::ios::beg);
	std::string instr, tmp;
	instr.reserve(len);
	instr.assign((std::istreambuf_iterator<char>(in_stream)), std::istreambuf_iterator<char>());	

	std::stringstream input(instr);
	CGP::cstrVec seg_data;
	CGP::intVec res;
	while (getline(input, tmp, sep))
	{
		seg_data.push_back(tmp);
	}
	for (auto s : seg_data)
	{
		int temp_int;
		try
		{
			temp_int = stoi(s);
			res.push_back(temp_int);
		}
		catch (std::invalid_argument const &e)
		{

		}
		catch (std::out_of_range const&e)
		{

		}		
	}
	LOG(INFO) << "end reading for file: " << filename << std::endl;
	return res;
}

void FILEIO::loadIntDynamic(const std::string& filename, CGP::intVec& res, char sep)
{
	CGP::intVec res_part = loadIntDynamic(filename, sep);
	res.insert(res.end(), res_part.begin(), res_part.end());
}

CGP::floatVec FILEIO::loadFloatDynamic(const std::string& filename, char sep)
{
	std::ifstream in_stream(filename);
	if (!in_stream.is_open())
	{
		LOG(ERROR) << filename << ", open failed." << std::endl;
		return {};
	}
	in_stream.seekg(0, std::ios::end);
	std::streampos len = in_stream.tellg();
	in_stream.seekg(0, std::ios::beg);
	std::string instr, tmp;
	instr.reserve(len);
	instr.assign((std::istreambuf_iterator<char>(in_stream)), std::istreambuf_iterator<char>());

	std::stringstream input(instr);
	CGP::cstrVec seg_data;
	CGP::floatVec res;
	while (getline(input, tmp, sep))
	{
		seg_data.push_back(tmp);
	}
	for (auto s : seg_data)
	{
		float temp_float;
		try
		{
			temp_float = stof(s);
			res.push_back(temp_float);
		}
		catch (std::invalid_argument const &e)
		{

		}
		catch (std::out_of_range const&e)
		{

		}
	}
	LOG(INFO) << "end reading for file: " << filename << std::endl;
	return res;
}

int FILEIO::getPair(const std::string& file_name, CGP::intVec&src_p, CGP::intVec& dst_p)
{
	std::ifstream in_put(file_name);
	if (!in_put.is_open())
	{
		LOG(ERROR) << "file not open: " << file_name << std::endl;
		return 0;
	}
	int n_pair;
	in_put >> n_pair;
	src_p.clear();
	dst_p.clear();
	for (int i = 0; i < n_pair; i++)
	{
		int src, dst;
		in_put >> src >> dst;
		src_p.push_back(src);
		dst_p.push_back(dst);
	}
	return n_pair;
}

void FILEIO::loadBinaryC(const std::string& file_name, CGP::float3Vec& res)
{
	int length;
	std::vector<char> buffer;
	std::ifstream in_stream;
	in_stream.open(file_name, std::ios::binary);
	if (!in_stream.is_open())
	{
		LOG(ERROR) << "file is not opened: " << file_name << std::endl;
		return;
	}
	// get length of file:
	in_stream.seekg(0, std::ios::end);
	length = in_stream.tellg();
	in_stream.seekg(0, std::ios::beg);
	// allocate memory:
	buffer.resize(length);
	// read data as a block:
	in_stream.read(buffer.data(), length);
	in_stream.close();
}

void FILEIO::loadBinary(const std::string& file_name, CGP::float3Vec& res)
{
	std::ifstream input(file_name, std::ios::binary);
	// copies all data into buffer
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
}

void FILEIO::saveStringBinary(std::string filepath, const std::string& src)
{
	std::fstream file(filepath, std::ios::binary | std::ios::out);
	if (!file.is_open())
	{
		LOG(ERROR) << "file not open: " << filepath << std::endl;
		return;
	}
	//file.write(src.c_str(), src.size());
	file << src;
	file.close();
}

CGP::cstrVec FILEIO::getFolderFiles(const CGP::cstr& folder, const FILEIO::FILE_TYPE& type, bool recursive)
{
	if (type == FILE_TYPE::IMAGE)
	{
		return getFolderFiles(folder, CGP::cstrVec{ ".jpg", ".png"}, recursive);
	}
	else if (type == FILE_TYPE::MESH)
	{
		return getFolderFiles(folder, CGP::cstrVec{ ".obj"}, recursive);
	}
	else
	{
		LOG(ERROR) << "file extension undefined." << std::endl;
		return {};
	}
}

CGP::cstrVec FILEIO::getFolderFiles(const CGP::cstr& folder, const CGP::cstr& extension)
{
	return getFolderFiles(folder, CGP::cstrVec{ extension });
}

CGP::cstrVec FILEIO::getFolderFiles(const CGP::cstr& folder, const CGP::cstrVec& extension, bool recursive)
{
	CGP::cstrVec res;
	boost::filesystem::path src(folder);
	boost::filesystem::directory_iterator endIter;
	CGP::cstrSet ext_set(extension.begin(), extension.end());
	for (boost::filesystem::directory_iterator iter(src); iter != endIter; iter++)
	{
		if (boost::filesystem::is_directory(*iter)) 
		{
			if (recursive)
			{
				CGP::cstrVec res_sub = getFolderFiles(iter->path().string(), extension, recursive);
				res.insert(res.end(), res_sub.begin(), res_sub.end());
			}	
		}
		else {
			if (!recursive)
			{
				if (ext_set.count(iter->path().extension().string()))
				{
					res.push_back(iter->path().filename().string());
				}
			}
			else
			{
				if (ext_set.count(iter->path().extension().string()))
				{
					res.push_back(iter->path().string());
				}
			}	
		}
	}
	//linux windows differ in file order
	std::sort(res.begin(), res.end());
	return res;
}

nlohmann::json FILEIO::loadJson(const std::string& filepath)
{
	std::ifstream in_json(filepath);
	json j;
	if (!in_json.is_open())
	{
		LOG(ERROR) << "open file failed: " << filepath << std::endl;
		return j;
	}
	in_json >> j;
	return j;
}

CGP::cstr FILEIO::dump(CGP::cstr const& source, size_t const length)
{
	if (length >= source.size())
	{
		return {};
	}
	CGP::cstr temp = source;
	temp.resize(source.size() - length);
	return temp;
} // tail

CGP::cstr FILEIO::tail(CGP::cstr const& source, size_t const length) 
{
	if (length >= source.size()) 
	{ 
		return source; 
	}
	return source.substr(source.size() - length);
} // tail

CGP::cstr FILEIO::head(CGP::cstr const& source, size_t const length)
{
	if (length >= source.size())
	{
		return source;
	}
	return source.substr(0, length);
} // head