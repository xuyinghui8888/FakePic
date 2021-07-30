#include "MeshCompress.h"
#include "../FileIO/FileIO.h"
#include "../RT/RT.h"
using namespace CGP;

#ifdef __linux
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif
MeshCompress::MeshCompress()
{

}

MeshCompress::MeshCompress(const cstr& file_name)
{
	clear();
	loadObj(file_name);
}

MeshCompress::MeshCompress(const MeshCompress& src)
{
	pos_ = src.pos_;
	normal_ = src.normal_;
	normal_face_ = src.normal_face_;
	tri_ = src.tri_;
	tri_uv_ = src.tri_uv_;
	tex_cor_ = src.tex_cor_;
	tri_int_3_ = src.tri_int_3_;
	ori_face_ = src.ori_face_;
	vertex_tri_ = src.vertex_tri_;
	vertex_vertex_ = src.vertex_vertex_;
	material_ = src.material_;
	vertex_color_ = src.vertex_color_;
	center_ = src.center_;
	g_name_maya_ = src.g_name_maya_;
	scale_ = src.scale_;
	n_vertex_ = src.n_vertex_;
	n_tri_ = src.n_tri_;
	n_uv_ = src.n_uv_;
	xyz_min_ =  src.xyz_min_;
	xyz_max_ = src.xyz_max_;
}

void MeshCompress::loadObj(const cstr& file_name)
{
	FILE* file;
#ifdef _WIN32
	errno_t err = fopen_s(&file, file_name.c_str(), "r");
#else
	fopen_s(&file, file_name.c_str(), "r");
#endif
	if (file == NULL)
	{
		LOG(ERROR) << "could not open file " << file_name << std::endl;
		return;
	}
	clear();
	char line[256];
	while (fgets(line, sizeof(line), file)) 
	{
		cstrVec strs, strsV2;
		FILEIO::splitString(cstr(line), strs, ' ');
		if (strs.size() == 0)
			continue;
		if (strs.back() == "\n")
		{
			strs.pop_back();
		}
		if (strs[0] == "f") 
		{
			cstrX2Vec tri_uv_normal(strs.size(), cstrVec{});
			for (int iter_string = 0; iter_string < strs.size(); iter_string++)
			{
				FILEIO::splitString(strs[iter_string], tri_uv_normal[iter_string], '/');
			}
			
			for (int iter_tri = 0; iter_tri < strs.size()-3; iter_tri++)
			{
				//need split for uv/f
				tri_.push_back(std::stoi(strs[1]) - 1);
				tri_.push_back(std::stoi(strs[iter_tri + 2]) - 1);
				tri_.push_back(std::stoi(strs[iter_tri + 3]) - 1);
				if (tri_uv_normal[1].size() == 2)
				{
					//only f vertex ind/ normal ind
				}
				else if(tri_uv_normal[1].size() > 2 && SG::isDigits(tri_uv_normal[1][1]))
				{ 
					tri_uv_.push_back(std::stoi(tri_uv_normal[1][1]) - 1);
					tri_uv_.push_back(std::stoi(tri_uv_normal[iter_tri + 2][1]) - 1);
					tri_uv_.push_back(std::stoi(tri_uv_normal[iter_tri + 3][1]) - 1);
				}

			}
		}
		else if (strs[0] == "v") {
			pos_.push_back(float3E(std::stof(strs[1]), std::stof(strs[2]), std::stof(strs[3])));
		}
		else if (strs[0] == "vt")
		{
			tex_cor_.push_back(float2E(std::stof(strs[1]), std::stof(strs[2])));
		}
		else if (strs[0] == "vn")
		{
			normal_.push_back(float3E(std::stof(strs[1]), std::stof(strs[2]), std::stof(strs[3])));
		}
		else if (strs[0] == "mtllib")
		{
			material_.push_back(strs[1]);
		}
	}
	fclose(file);
	update();
	LOG(INFO) << "loading: " << file_name << ", vertex: " << n_vertex_ << ", tri: " << n_tri_ << std::endl;
}

void MeshCompress::replaceVertexBasedData(const MeshCompress& src)
{
	if (src.n_vertex_ != n_vertex_)
	{
		LOG(ERROR) << "vertex size not fit, could not locate for swap data." << std::endl;
		return;
	}
	if (src.pos_.size() != pos_.size())
	{
		LOG(ERROR) << "pos_ size not fit." << std::endl;
		return;
	}
	pos_ = src.pos_;
	if (src.normal_.size() != src.normal_.size())
	{
		LOG(ERROR) << "normal size don't fit." << std::endl;
		return;
	}
	normal_ = src.normal_;
	update();
}

void MeshCompress::loadOri(const cstr& file_name)
{
	FILE* file;
#ifdef _WIN32
	errno_t err = fopen_s(&file, file_name.c_str(), "r");
#else
	fopen_s(&file, file_name.c_str(), "r");
#endif
	if (file == NULL) {
		LOG(ERROR) << "could not open file " << file_name << std::endl;
	}
	clear();
	char line[256];
	while (fgets(line, sizeof(line), file)) {
		std::vector<cstr> strs;
		FILEIO::splitString(cstr(line), strs, ' ');
		if (strs.size() == 0)
			continue;
		if (strs[0] == "f")
		{
			intVec temp;
			intVec temp_uv;
			for (int iter_tri = 1; iter_tri < strs.size(); iter_tri++)
			{
				if (strs[iter_tri] != "\n")
				{
					cstrVec res;
					FILEIO::splitString(strs[iter_tri], res, '/');
					if (res.size() == 3)
					{
						temp_uv.push_back(std::stoi(res[1])-1);
					}
					temp.push_back(std::stoi(strs[iter_tri]) - 1);
				}
				
			}
			ori_face_uv_.push_back(temp_uv);
			ori_face_.push_back(temp);
		}
		else if (strs[0] == "v") {
			pos_.push_back(float3E(std::stof(strs[1]), std::stof(strs[2]), std::stof(strs[3])));
		}
		else if (strs[0] == "vt")
		{
			tex_cor_.push_back(float2E(std::stof(strs[1]), std::stof(strs[2])));
		}
	}
	fclose(file);
	update();
	LOG(INFO) << "loading: " << file_name << ", vertex: " << n_vertex_ << ", tri: " << n_tri_ << std::endl;
}

void MeshCompress::saveVertexColor(const cstr& file_name) const
{
	std::ofstream out_stream(file_name);
	if (!out_stream.is_open())
	{
		LOG(ERROR) << file_name << " is not opened. " << std::endl;
		return;
	}

	if (vertex_color_.size() != pos_.size())
	{
		LOG(ERROR) << "vertex color not found." << std::endl;
		return;
	}

	for (int i = 0; i < pos_.size(); i++)
	{
		out_stream << "v " <<
			pos_[i][0] << " " <<
			pos_[i][1] << " " <<
			pos_[i][2] << " " <<
			vertex_color_[i][0] << " " <<
			vertex_color_[i][1] << " " <<
			vertex_color_[i][2] << "\n";
	}
	for (int i = 0; i < tex_cor_.size(); i++)
	{
		out_stream << "vt " <<
			tex_cor_[i][0] << " " <<
			tex_cor_[i][1] << "\n";
	}

	if (tex_cor_.empty())
	{
		out_stream << "vt " <<
			0.5 << " " <<
			0.5 << "\n";
	}


	for (int i = 0; i < normal_.size(); i++)
	{
		out_stream << "vn " <<
			normal_[i][0] << " " <<
			normal_[i][1] << " " <<
			normal_[i][2] << "\n";
	}
	//f uv normal
	if (normal_.size() == n_vertex_ && tri_uv_.size() == n_tri_ * 3)
	{
		for (int i = 0; i < tri_.size() / 3; i++)
		{
			out_stream << "f " <<
				tri_[i * 3 + 0] + 1 << "/" << tri_uv_[i * 3 + 0] + 1 << "/" << tri_[i * 3 + 0] + 1 << " " <<
				tri_[i * 3 + 1] + 1 << "/" << tri_uv_[i * 3 + 1] + 1 << "/" << tri_[i * 3 + 1] + 1 << " " <<
				tri_[i * 3 + 2] + 1 << "/" << tri_uv_[i * 3 + 2] + 1 << "/" << tri_[i * 3 + 2] + 1 << "\n";
		}
	}
	else if (tri_uv_.empty())
	{
		for (int i = 0; i < tri_.size() / 3; i++)
		{
			out_stream << "f " <<
				tri_[i * 3 + 0] + 1 << "/" << 1 << "/" << tri_[i * 3 + 0] + 1 << " " <<
				tri_[i * 3 + 1] + 1 << "/" << 1 << "/" << tri_[i * 3 + 1] + 1 << " " <<
				tri_[i * 3 + 2] + 1 << "/" << 1 << "/" << tri_[i * 3 + 2] + 1 << "\n";
		}
	}
	else
	{
		for (int i = 0; i < tri_.size() / 3; i++)
		{
			out_stream << "f " <<
				tri_[i * 3 + 0] + 1 << "/" << tri_[i * 3 + 0] + 1 << " " <<
				tri_[i * 3 + 1] + 1 << "/" << tri_[i * 3 + 1] + 1 << " " <<
				tri_[i * 3 + 2] + 1 << "/" << tri_[i * 3 + 2] + 1 << "\n";
		}
	}

	out_stream.close();
	LOG(INFO) << "saving file: " << file_name << std::endl;
}

void MeshCompress::saveObj(const cstr& file_name, const cstr& mtl_name) const
{
	std::ofstream out_stream(file_name);
	if (!out_stream.is_open())
	{
		LOG(ERROR) << file_name<<" is not opened. " << std::endl;
		return;
	}
	if (mtl_name != "")
	{
		out_stream << "mtllib " << mtl_name << std::endl;
	}
	else if (g_name_maya_ != "")
	{
		out_stream << "g " << g_name_maya_ << std::endl;
	}
	else
	{
		if (!material_.empty())
		{
			for (auto i: material_)
			{
				out_stream << "mtllib " << i << std::endl;
			}
		}
	}

	for (int i = 0; i < pos_.size(); i++) 
	{
		out_stream << "v " <<
			pos_[i][0] << " " <<
			pos_[i][1] << " " <<
			pos_[i][2] << "\n";
	}
	for (int i = 0; i < tex_cor_.size(); i++) 
	{
		out_stream << "vt " <<
			tex_cor_[i][0] << " " <<
			tex_cor_[i][1] << "\n";
	}

	if (tex_cor_.empty())
	{
		out_stream << "vt " <<
			0.5 << " " <<
			0.5 << "\n";
	}


	for (int i = 0; i < normal_.size(); i++)
	{
		out_stream << "vn " <<
			normal_[i][0] << " " <<
			normal_[i][1] << " " <<
			normal_[i][2] << "\n";
	}
	//f uv normal
	if (normal_.size() == n_vertex_ && tri_uv_.size() == n_tri_*3)
	{
		for (int i = 0; i < tri_.size() / 3; i++)
		{
			out_stream << "f " <<
				tri_[i * 3 + 0] + 1 << "/" << tri_uv_[i * 3 + 0] + 1 << "/" << tri_[i * 3 + 0] + 1 << " " <<
				tri_[i * 3 + 1] + 1 << "/" << tri_uv_[i * 3 + 1] + 1 << "/" << tri_[i * 3 + 1] + 1 << " " <<
				tri_[i * 3 + 2] + 1 << "/" << tri_uv_[i * 3 + 2] + 1 << "/" << tri_[i * 3 + 2] + 1 << "\n";
		}
	}
	else if (tri_uv_.empty())
	{
		for (int i = 0; i < tri_.size() / 3; i++)
		{
			out_stream << "f " <<
				tri_[i * 3 + 0] + 1 << "/" << 1 << "/" << tri_[i * 3 + 0] + 1 << " " <<
				tri_[i * 3 + 1] + 1 << "/" << 1 << "/" << tri_[i * 3 + 1] + 1 << " " <<
				tri_[i * 3 + 2] + 1 << "/" << 1 << "/" << tri_[i * 3 + 2] + 1 << "\n";
		}
	}
	else
	{
		for (int i = 0; i < tri_.size() / 3; i++)
		{
			out_stream << "f " <<
				tri_[i * 3 + 0] + 1 << "/" << tri_[i * 3 + 0] + 1 << " " <<
				tri_[i * 3 + 1] + 1 << "/" << tri_[i * 3 + 1] + 1 << " " <<
				tri_[i * 3 + 2] + 1 << "/" << tri_[i * 3 + 2] + 1 << "\n";
		}
	}

	out_stream.close();
	LOG(INFO) << "saving file: " << file_name << std::endl;
}

void MeshCompress::saveMtl(const cstr& file_name, const cstr& texture_name) const
{
	std::ofstream out_stream(file_name);
	if (!out_stream.is_open())
	{
		LOG(ERROR) << file_name << " is not opened. " << std::endl;
		return;
	}
	out_stream << "newmtl material_0" << std::endl;
	out_stream << "map_Kd " << texture_name << std::endl;
	out_stream.close();
	LOG(INFO) << "saving file: " << file_name << std::endl;
}

void MeshCompress::saveOri(const cstr& file_name, const cstr& mtl_name) const
{
	std::ofstream out_stream(file_name);
	if (mtl_name != "")
	{
		out_stream << "mtllib " << mtl_name << std::endl;
	}
	else if (g_name_maya_ != "")
	{
		out_stream << "g " << g_name_maya_ << std::endl;
	}

	for (int i = 0; i < pos_.size(); i++)
	{
		out_stream << "v " <<
			pos_[i][0] << " " <<
			pos_[i][1] << " " <<
			pos_[i][2] << "\n";
	}
	for (int i = 0; i < tex_cor_.size(); i++)
	{
		out_stream << "vt " <<
			tex_cor_[i][0] << " " <<
			tex_cor_[i][1] << "\n";
	}
	bool use_uv = false;
	if (ori_face_uv_.size() == ori_face_.size())
	{
		use_uv = true;
	}
	for (int i = 0; i < ori_face_.size(); i++)
	{
		// http://paulbourke.net/dataformats/obj/
		out_stream << "f ";
		for (int j = 0; j < ori_face_[i].size(); j++)
		{
			if (use_uv)
			{
				out_stream << ori_face_[i][j] + 1 << "/" << ori_face_uv_[i][j] + 1 <<"/"<< ori_face_[i][j] + 1 <<" ";
			}
			else
			{
				out_stream << ori_face_[i][j] + 1 << "/" << ori_face_[i][j] + 1 << " ";
			}

		}
		out_stream << "\n";
	}
	out_stream.close();
}

void MeshCompress::clear()
{
	n_vertex_ = -1;
	n_tri_ = -1;
	n_uv_ = -1;

	pos_.clear();
	normal_.clear();
	normal_face_.clear();
	tri_.clear();
	tri_uv_.clear();
	tex_cor_.clear();
	tri_int_3_.clear();
	ori_face_.clear();
	vertex_tri_.clear();
	vertex_vertex_.clear();
	material_.clear();
}

void MeshCompress::update()
{
	n_vertex_ = pos_.size();
	n_tri_ = tri_.size() / 3;
	n_uv_ = tex_cor_.size();

	tri_int_3_.resize(n_tri_);
#pragma omp parallel for
	for (int i = 0; i < n_tri_; i++)
	{
		tri_int_3_[i].resize(3);
		for (int j = 0; j < 3; j++)
		{
			tri_int_3_[i][j] = tri_[3 * i + j];
		}
	}
	getBoundingBox();
	getVertexTopology();
	generateNormal();
}

bool MeshCompress::safeMeshData() const
{
	return SG::checkMesh(pos_, tri_);
}

void MeshCompress::getVertexTopology()
{
	vertex_tri_.resize(n_vertex_);
	vertex_vertex_.resize(n_vertex_);
	for (int i = 0; i < n_tri_; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vertex_idx = tri_[3 * i + j];
			vertex_tri_[vertex_idx].insert(i);
			for (int k = 0; k < 3; k++)
			{
				int vertex_idx_k = tri_[3 * i + k];
				if (vertex_idx != vertex_idx_k)
				{
					vertex_vertex_[vertex_idx].insert(vertex_idx_k);
				}
			}
		}
	}
}

void MeshCompress::generateNormal()
{
	normal_face_.resize(n_tri_);
	normal_.resize(n_vertex_);
	//get face
#pragma omp parallel for
	for (int i = 0; i < n_tri_; i++) {
		const float3E& p0 = pos_[tri_[3 * i]];
		const float3E& p1 = pos_[tri_[3 * i + 1]];
		const float3E& p2 = pos_[tri_[3 * i + 2]];
		float3E v1 = p1 - p0;
		float3E v2 = p2 - p0;
		normal_face_[i] = v1.cross(v2).normalized();
	}
	//get normal
#pragma omp parallel for
	for (int i = 0; i < n_vertex_; i++)
	{
		float3E sum = float3E::Zero();
		for (auto j : vertex_tri_[i])
		{
			sum = sum + normal_face_[j];
		}
		if (vertex_tri_.size() == 0)
		{
			//LOG(INFO) << "has single vertex tri." << std::endl;
			normal_[i] = float3E(0,0,1);
			LOG(INFO) << "has single vertex tri." << std::endl;
		}
		else
		{
			normal_[i] = (sum * 1.0f / vertex_tri_.size()).normalized();
		}	
	}
}

intVec MeshCompress::keepRoi(const intVec& keep_vertex)
{
	intVec discard_vertex;
	for (int i = 0; i < n_vertex_; i++)
	{
		if (std::find(keep_vertex.begin(), keep_vertex.end(), i) == keep_vertex.end())
		{
			discard_vertex.push_back(i);
		}
	}
	return discard(discard_vertex);
}

intVec MeshCompress::discard(const intX2Vec& dis_vertex_vec)
{
	intVec discard_all;
	if (dis_vertex_vec.empty())
	{
		return {};
	}
	else
	{
		discard_all = dis_vertex_vec[0];
		for (int i = 1; i < dis_vertex_vec.size(); i++)
		{
			discard_all.insert(discard_all.end(), dis_vertex_vec[i].begin(), dis_vertex_vec[i].end());
		}
	}
	return discard(discard_all);
}

intVec MeshCompress::discard(const intX2Vec& keep_vec, const intX2Vec& dis_vertex_vec)
{
	intVec discard_all;
	if (dis_vertex_vec.empty())
	{
		return {};
	}
	else
	{
		discard_all = dis_vertex_vec[0];
		for (int i = 1; i < dis_vertex_vec.size(); i++)
		{
			discard_all.insert(discard_all.end(), dis_vertex_vec[i].begin(), dis_vertex_vec[i].end());
		}
	}
	//add for opsite of keep_vec
	intSet keep_idx;
	for (int i = 0; i < keep_vec.size(); i++)
	{
		keep_idx.insert(keep_vec[i].begin(), keep_vec[i].end());
	}

	for (int i = 0; i < n_vertex_; i++)
	{
		if (!keep_idx.count(i))
		{
			discard_all.push_back(i);
		}
	}

	return discard(discard_all);
}

intVec MeshCompress::discard(const intVec& keep_vec, const intX2Vec& discard_vec)
{
	return discard(intX2Vec{ keep_vec }, discard_vec);
}

intVec MeshCompress::getDiscardMap(const intVec& dis_vertex)
{
	intVec all_part(n_vertex_, -1);
	intSet dis_vertex_set(dis_vertex.begin(), dis_vertex.end());
	int count = 0;
	for (int i = 0; i < n_vertex_; i++)
	{
		if (dis_vertex_set.count(i))
		{
			//vertex needs to discard
		}
		else
		{
			//vertex needs to keep
			all_part[i] = count;
			count++;
		}
	}
	return all_part;
}

void MeshCompress::discardMaterial()
{
	material_.clear();
}

intVec MeshCompress::discard(const intVec& dis_vertex)
{
	//copy data first
	float3Vec back_pos = pos_;
	uintVec back_tri = tri_;
	uintVec back_tri_uv = tri_uv_;
	//set mapping
	intVec all_part = getDiscardMap(dis_vertex);
	//discard
	if (!tex_cor_.empty() || !normal_.empty())
	{
		LOG(WARNING) << "TODO: due with texture coordinate or normal issue." << std::endl;
	}
	pos_.clear();
	for (int i = 0; i < n_vertex_; i++)
	{
		if (all_part[i] < -0.5)
		{
			//discard
		}
		else
		{
			pos_.push_back(back_pos[i]);
		}
	}
	tri_.clear();
	tri_uv_.clear();
	bool skip_uv = false;
	if (back_tri_uv.empty())
	{
		skip_uv = true;
	}
	for (int i = 0; i < n_tri_; i++)
	{
		//only due with triangle
		intVec map_idx(3, -1);
		intVec uv_idx(3, -1);
		bool skip = false;
		for(int j = 0; j < 3; j++)
		{
			map_idx[j] = all_part[back_tri[i * 3 + j]];
			if (map_idx[j] < -0.5)
			{
				skip = true;
			}
			if (!skip_uv)
			{
				uv_idx[j] = back_tri_uv[i * 3 + j];
			}	
		}
		if (!skip)
		{
			//used
			tri_.insert(tri_.end(), map_idx.begin(), map_idx.end());
			if (!skip_uv)
			{
				tri_uv_.insert(tri_uv_.end(), uv_idx.begin(), uv_idx.end());
			}
		}
	}
	update();
	return all_part;
}

void MeshCompress::discardOri(const intVec& dis_vertex)
{
	//copy data first
	float3Vec back_pos = pos_;
	intX2Vec back_ori_face = ori_face_;
	//set mapping
	intVec all_part = getDiscardMap(dis_vertex);
	//discard
	if (!tex_cor_.empty() || !normal_.empty())
	{
		LOG(WARNING) << "TODO: due with texture coordinate or normal issue." << std::endl;
	}
	pos_.clear();
	for (int i = 0; i < n_vertex_; i++)
	{
		if (all_part[i] < -0.5)
		{
			//discard
		}
		else
		{
			pos_.push_back(back_pos[i]);
		}
	}
	ori_face_.clear();
	for (int i = 0; i < back_ori_face.size(); i++)
	{
		//only due with triangle
		intVec map_idx(back_ori_face[i].size(), -1);
		int skip = 0;
		for (int j = 0; j < back_ori_face[i].size(); j++)
		{
			map_idx[j] = all_part[back_ori_face[i][j]];
			if (map_idx[j] < -0.5)
			{
				skip++;
			}
		}
		if (skip == 1 || skip == 2 || skip == 3)
		{
			LOG(INFO) << "something out of range" << std::endl;
		}
		if (!skip)
		{
			//used
			ori_face_.push_back(map_idx);
		}
	}
	update();
}

void MeshCompress::getSlice(const intVec& index, float3Vec& res) const
{
	res.clear();
	for (auto i: index)
	{
		if (i >= n_vertex_ || i < 0)
		{
			LOG(ERROR) << "index out of range. put 0,0,0, instead" << std::endl;
			res.push_back(float3E(0, 0, 0));
		}
		else
		{
			res.push_back(pos_[i]);
		}	
	}
}

void MeshCompress::flipYZ()
{	
#pragma omp parallel for 
	for (int i = 0; i < n_vertex_; i++)
	{
		float temp_y = pos_[i].y();
		pos_[i].y() = pos_[i].z();
		pos_[i].z() = -temp_y;
	}
	update();
}

void MeshCompress::rotateX()
{
#pragma omp parallel for 
	for (int i = 0; i < n_vertex_; i++)
	{
		float temp_y = pos_[i].y();
		pos_[i].y() = -pos_[i].z();
		pos_[i].z() = temp_y;
	}
	update();
}

void MeshCompress::getBoundingBox()
{
#pragma omp parallel for 
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		for (int i = 0; i < n_vertex_; i++)
		{
			xyz_min_[iter_dim] = DMIN(xyz_min_[iter_dim], pos_[i](iter_dim));
			xyz_max_[iter_dim] = DMAX(xyz_max_[iter_dim], pos_[i](iter_dim));
		}
	}
	//scale (longest to 1)
	double max_length = INTMAX_MIN;
	for (int iter_dim = 0; iter_dim  < 3; iter_dim ++)
	{
		max_length = DMAX(max_length, xyz_max_[iter_dim] - xyz_min_[iter_dim]);
		center_[iter_dim] = xyz_max_[iter_dim] * 0.5 + xyz_min_[iter_dim] * 0.5;
	}
	scale_ = max_length;	
}

void MeshCompress::getBoundingBox(const intVec& idx, doubleVec& xyz_min, doubleVec& xyz_max) const
{
	xyz_min = { 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX };
	xyz_max = { -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX };
#pragma omp parallel for 
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		for (int iter_idx = 0; iter_idx < idx.size(); iter_idx++)
		{
			int i = idx[iter_idx];
			xyz_min[iter_dim] = DMIN(xyz_min[iter_dim], pos_[i](iter_dim));
			xyz_max[iter_dim] = DMAX(xyz_max[iter_dim], pos_[i](iter_dim));
		}
	}	
}

void MeshCompress::getBoundingBox(const intVec& idx, const mat3f& rot, doubleVec& xyz_min, doubleVec& xyz_max) const
{
	xyz_min = { 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX };
	xyz_max = { -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX };
#pragma omp parallel for 
	for (int iter_dim = 0; iter_dim < 3; iter_dim++)
	{
		for (int iter_idx = 0; iter_idx < idx.size(); iter_idx++)
		{
			int i = idx[iter_idx];
			float3E pos_i = rot * pos_[i];
			xyz_min[iter_dim] = DMIN(xyz_min[iter_dim], pos_i(iter_dim));
			xyz_max[iter_dim] = DMAX(xyz_max[iter_dim], pos_i(iter_dim));
		}
	}
}

void MeshCompress::moveToCenter(bool is_scale)
{
	//move to center first
	if (is_scale)
	{
		RT::translateAndScaleInPlace(-center_, 1.0 / scale_, pos_);
	}
	else
	{
		RT::translateInPlace(-center_, pos_);
	}

}

void MeshCompress::moveToOri(bool is_scale)
{
	//move to center first
	if (is_scale)
	{
		RT::scaleAndTranslateInPlace(scale_, center_, pos_);
	}
	else
	{
		RT::translateInPlace(center_, pos_);
	}
}

intVec MeshCompress::getReverseSelection(const intVec& src)
{
	intSet select_set(src.begin(), src.end());
	intVec reverse_select;
	for (size_t i = 0; i < n_vertex_; i++)
	{
		if (!select_set.count(i))
		{
			reverse_select.push_back(i);
		}
	}
	return reverse_select;
}

void MeshCompress::memcpyVertex(const MeshCompress& src)
{
	if (src.n_vertex_ != n_vertex_)
	{
		LOG(ERROR) << "vertex size don't match" << std::endl;
	}
	else
	{
		SG::safeMemcpy(pos_.data(), src.pos_.data(), 3 * n_vertex_ * sizeof(float));
	}
}

void MeshCompress::setGOption(const cstr& name_vec)
{
	g_name_maya_ = name_vec;
}