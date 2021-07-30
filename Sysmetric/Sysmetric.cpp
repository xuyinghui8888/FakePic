#include "Sysmetric.h"
#include "../FileIO/FileIO.h"
#include "../MeshDeform/LaplacianDeformation.h"
using namespace CGP;
using namespace rttr;

void SysFinder::findSysBasedOnUV(const MeshCompress& src, const std::string& root)
{
	//map uv to index
	intSetVec vertex_to_uv(src.n_vertex_, intSet{});
	intSetVec uv_to_vertex(src.n_uv_, intSet{});
	for (int i = 0; i < src.tri_.size(); i++)
	{
		int vertex_ind = src.tri_[i];
		int uv_ind = src.tri_uv_[i];
		vertex_to_uv[vertex_ind].insert(uv_ind);
		uv_to_vertex[uv_ind].insert(vertex_ind);
	}
	//get uv sysmetric 0 not visited, 1 visited
	intVec visited_vertex(src.n_vertex_, 0);
	intVec uv_vertex_1_to_1(src.n_uv_, -1);
	intVec visited_uv(src.n_uv_, 0);
	//uv index
	intVec left_to_right(src.n_vertex_, -1);
	intVec right_to_left(src.n_vertex_, -1);
	intVec mid_line(src.n_vertex_, -1);
	

	//mapping uv_vertex to vertex_ind check for multiple mapping
	std::ofstream out_two(root + "two_mapping.txt");
	int n_one_vertex_multi_uv = 0;
	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (vertex_to_uv[i].size() > 1)
		{
			LOG(INFO) << "vertex idx: " << i << std::endl;
			out_two << i << ",";
			for (auto iter_uv: vertex_to_uv[i])
			{
				LOG(INFO) << "uv_pos: " << src.tex_cor_[iter_uv].transpose() << std::endl;
				visited_uv[iter_uv] = 1;
			}
			n_one_vertex_multi_uv += vertex_to_uv[i].size() - 1;
			mid_line[i] = i;
			visited_vertex[i] = 1;			
		}
		else
		{
			//int uv_index = *(vertex_to_uv[i].begin());
			//LOG(INFO) << "uv_index: " << uv_index << std::endl;
			//LOG(INFO) << "vertex_index: " << i << std::endl;
		}
	}
	if (n_one_vertex_multi_uv != src.n_uv_ - src.n_vertex_)
	{
		LOG(ERROR) << "extra uv and vertex size not match." << std::endl;
	}
	out_two.close();
	LOG(INFO) << "total vertex: " << src.n_vertex_ << std::endl;
	LOG(INFO) << "total found match: " << std::accumulate(visited_vertex.begin(), visited_vertex.end(), 0) << std::endl;
	LOG(INFO) << "total uv: " << src.n_uv_ << std::endl;
	LOG(INFO) << "total found match: " << std::accumulate(visited_uv.begin(), visited_uv.end(), 0) << std::endl;
	for (int i = 0; i < src.n_uv_; i++)
	{
		if (visited_uv[i])
		{
			//already visited
		}
		else
		{
			int vertex_ind = *(uv_to_vertex[i].begin());
			if (visited_uv[i] || visited_vertex[vertex_ind])
			{

			}
			else
			{
				if (nearValue(src.tex_cor_[i].x(), 0.5, EPSCG6))
				{
					visited_vertex[vertex_ind] = 1;
					visited_uv[i] = 1;
					mid_line[vertex_ind] = vertex_ind;
				}
				else
				{
					visited_vertex[vertex_ind] = 1;
					visited_uv[i] = 1;
					floatVec temp_dis(src.n_uv_, INT_MAX);
					for (int j = 0; j < src.n_uv_; j++)
					{
						//reverse	
						int vertex_ind_j = *(uv_to_vertex[j].begin());
						if (!visited_uv[j] && !visited_vertex[vertex_ind_j])
						{
							if (src.tex_cor_[i].x() < 0.5 && src.tex_cor_[j].x() > 0.5)
							{
								//only set >0.5
								float2E sys_j = src.tex_cor_[j];
								sys_j.x() = 1 - sys_j.x();
								temp_dis[j] = (src.tex_cor_[i] - sys_j).norm();
							}
							else if (src.tex_cor_[i].x() > 0.5 && src.tex_cor_[j].x() < 0.5)
							{
								float2E sys_j = src.tex_cor_[j];
								sys_j.x() = 1 - sys_j.x();
								temp_dis[j] = (src.tex_cor_[i] - sys_j).norm();
							}
						}
					}
					auto res = std::minmax_element(temp_dis.begin(), temp_dis.end());
					//std::cout << res.first - match_dis.begin() << std::endl;
					//std::cout << *res.first << std::endl;
					int match_j = res.first - temp_dis.begin();
					if (visited_uv[match_j])
					{
						LOG(ERROR) << "match visited uv." << std::endl;
						return;
					}
					visited_uv[match_j] = 1;
					int vertex_ind_j = *(uv_to_vertex[match_j].begin());
					if (visited_vertex[vertex_ind_j])
					{
						LOG(ERROR) << "match visited vertex." << std::endl;
						return;
					}
					visited_vertex[vertex_ind_j] = 1;
					if (src.tex_cor_[i].x() < 0.5)
					{
						//left --- i right---- j
						left_to_right[vertex_ind] = vertex_ind_j;
						right_to_left[vertex_ind_j] = vertex_ind;
					}
					else
					{
						left_to_right[vertex_ind_j] = vertex_ind;
						right_to_left[vertex_ind] = vertex_ind_j;
					}		
				}
			}			
		}
	}
	
	int n_mid = 0;
	int n_left = 0;
	int n_right = 0;
	//temp save
	std::ofstream out_mid(root + "mid.txt");
	   	  
	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (mid_line[i] > 0)
		{
			out_mid << mid_line[i] << ",";
			n_mid++;
		}
	}
	out_mid.close();

	std::ofstream out_match(root + "match.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (left_to_right[i] >= 0)
		{
			out_match << i << ",";
			out_match << left_to_right[i] << ",";
		}
	}
	out_match.close();

	std::ofstream out_left(root + "left.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (left_to_right[i] >= 0)
		{
			out_left << i << ",";
			n_left++;
		}
	}
	out_left.close();


	std::ofstream out_right(root + "right.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (right_to_left[i] >= 0)
		{
			out_right << i << ",";
			n_right++;
		}
	}
	out_right.close();
	   
	LOG(INFO) << "total vertex: " << src.n_vertex_ << std::endl;
	LOG(INFO) << "total n_mid: " << n_mid << std::endl;
	LOG(INFO) << "total n_left: " << n_left << std::endl;
	LOG(INFO) << "total n_right: " << n_right << std::endl;
	LOG(INFO) << "total found: " << n_mid + n_left + n_right << std::endl;

	LOG(INFO) << "total found match: " << std::accumulate(visited_vertex.begin(), visited_vertex.end(), 0) << std::endl;
	LOG(INFO) << "total uv: " << src.n_uv_ << std::endl;
	LOG(INFO) << "total found match: " << std::accumulate(visited_uv.begin(), visited_uv.end(), 0) << std::endl;
	std::ofstream out_missed(root + "missed.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (visited_vertex[i] == 0)
		{
			out_missed << i << ",";
		}
	}
	out_missed.close();

	//matched

}

void SysFinder::findSysBasedOnPosBiMap(const MeshCompress& src, const std::string& root)
{
	intVec visited_vertex(src.n_vertex_, 0);
	intVec left_to_right(src.n_vertex_, -1);
	intVec right_to_left(src.n_vertex_, -1);
	intVec mid_line(src.n_vertex_, -1);
	
	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (visited_vertex[i])
		{
			//already visited
		}
		else if (nearValue(src.pos_[i].x(), 0, EPSCG6))
		{
			visited_vertex[i] = 1;
			mid_line[i] = i;
		}
		else
		{
			visited_vertex[i] = 1;
			for (int j = 0; j < src.n_vertex_; j++)
			{
				//reverse
				float3E sys_j = src.pos_[j];
				sys_j.x() = -sys_j.x();
				if (!visited_vertex[j] && nearValue((src.pos_[i] - sys_j).norm(), 0, EPSCG3))
				{
					visited_vertex[j] = 1;
					if (src.pos_[i].x() * src.pos_[j].x() > 0)
					{
						LOG(ERROR) << "wrong with pos idx: " << i << ", " << j << std::endl;
						LOG(ERROR) << "pos_i:  " <<  src.pos_[i] << std::endl;
						LOG(ERROR) << "pos_j:  " <<  src.pos_[j] << std::endl;
					}
					if (src.pos_[i].x() < 0)
					{
						//left --- i right---- j
						left_to_right[i] = j;
						right_to_left[j] = i;
					}
					else
					{
						left_to_right[j] = i;
						right_to_left[i] = j;
					}
					break;
				}
			}
		}
	}

	int n_mid = 0;
	int n_left = 0;
	int n_right = 0;
	//temp save
	std::ofstream out_mid(root + "mid.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (mid_line[i] >= 0)
		{
			out_mid << mid_line[i] << ",";
			n_mid++;
		}
	}
	out_mid.close();

	std::ofstream out_match(root + "match.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (left_to_right[i] >= 0)
		{
			out_match << i << ",";
			out_match << left_to_right[i] << ",";
		}
	}
	out_match.close();

	std::ofstream out_left(root + "left.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (left_to_right[i] >= 0)
		{
			out_left << i << ",";
			n_left++;
		}
	}
	out_left.close();


	std::ofstream out_right(root + "right.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (right_to_left[i] >= 0)
		{
			out_right << i << ",";
			n_right++;
		}
	}
	out_right.close();

	LOG(INFO) << "total vertex: " << src.n_vertex_ << std::endl;
	LOG(INFO) << "total n_mid: " << n_mid << std::endl;
	LOG(INFO) << "total n_left: " << n_left << std::endl;
	LOG(INFO) << "total n_right: " << n_right << std::endl;
	LOG(INFO) << "total found: " << n_mid + n_left + n_right << std::endl;

	LOG(INFO) << "total found match: " << std::accumulate(visited_vertex.begin(), visited_vertex.end(), 0) << std::endl;

	std::ofstream out_missed(root + "missed.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (std::find(mid_line.begin(), mid_line.end(), i) == mid_line.end()
			&& std::find(left_to_right.begin(), left_to_right.end(), i) == left_to_right.end()
			&& std::find(right_to_left.begin(), right_to_left.end(), i) == right_to_left.end()
			)
		{
			out_missed << i << ",";
		}
	}
	out_missed.close();

	//matched

}

void SysFinder::findSysBasedOnPosOnly(const MeshCompress& src, const std::string& root, float mid_thres, float left_right_mapping_thres)
{
	SG::needPath(root);
	intVec visited_vertex(src.n_vertex_, 0);
	intVec left_to_right(src.n_vertex_, -1);
	intVec right_to_left(src.n_vertex_, -1);
	intVec mid_line(src.n_vertex_, -1);

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (visited_vertex[i])
		{
			//already visited
		}
		else if (nearValue(src.pos_[i].x(), 0, mid_thres))
		{
			visited_vertex[i] = 1;
			mid_line[i] = i;
		}
		else
		{
			visited_vertex[i] = 1;
			double min_dist = INT16_MAX;
			int min_idx = -1;
			for (int j = 0; j < src.n_vertex_; j++)
			{
				//reverse
				float3E sys_j = src.pos_[j];
				sys_j.x() = -sys_j.x();
				//avoid for match the same
				//if (!visited_vertex[j] && src.pos_[i].x() * src.pos_[j].x() <= 0)
				if (src.pos_[i].x() * src.pos_[j].x() <= 0)
				{
					double cur_dist = (src.pos_[i] - sys_j).norm();
					min_idx = cur_dist < min_dist ? j : min_idx;
					min_dist = DMIN(cur_dist, min_dist);
				}				
			}

			if (min_idx<-0 || min_idx>src.n_vertex_ - 1)
			{
				LOG(WARNING) << "missed: " << i << std::endl;
			}
			else
			{
				visited_vertex[min_idx] = 1;
				if (src.pos_[i].x() < 0)
				{
					//left --- i right---- j
					left_to_right[i] = min_idx;
					right_to_left[min_idx] = i;
				}
				else
				{
					left_to_right[min_idx] = i;
					right_to_left[i] = min_idx;
				}
			}
		}
	}

	int n_mid = 0;
	int n_left = 0;
	int n_right = 0;
	//temp save
	std::ofstream out_mid(root + "mid.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (mid_line[i] >= 0)
		{
			out_mid << mid_line[i] << ",";
			n_mid++;
		}
	}
	out_mid.close();

	std::ofstream out_match(root + "match.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (left_to_right[i] >= 0)
		{
			out_match << i << ",";
			out_match << left_to_right[i] << ",";
		}
	}
	out_match.close();

	std::ofstream out_left(root + "left.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (left_to_right[i] >= 0)
		{
			out_left << i << ",";
			n_left++;
		}
	}
	out_left.close();


	std::ofstream out_right(root + "right.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (right_to_left[i] >= 0)
		{
			out_right << i << ",";
			n_right++;
		}
	}
	out_right.close();

	LOG(INFO) << "total vertex: " << src.n_vertex_ << std::endl;
	LOG(INFO) << "total n_mid: " << n_mid << std::endl;
	LOG(INFO) << "total n_left: " << n_left << std::endl;
	LOG(INFO) << "total n_right: " << n_right << std::endl;
	LOG(INFO) << "total found: " << n_mid + n_left + n_right << std::endl;

	LOG(INFO) << "total found match: " << std::accumulate(visited_vertex.begin(), visited_vertex.end(), 0) << std::endl;

	std::ofstream out_missed(root + "missed.txt");

	for (int i = 0; i < src.n_vertex_; i++)
	{
		if (std::find(mid_line.begin(), mid_line.end(), i) == mid_line.end()
			&& std::find(left_to_right.begin(), left_to_right.end(), i) == left_to_right.end()
			&& std::find(right_to_left.begin(), right_to_left.end(), i) == right_to_left.end()
			)
		{
			out_missed << i << ",";
		}
	}
	out_missed.close();

	//matched

}

void SysFinder::rectifyLandmarkPos(const json& config)
{
	MeshCompress src(config["src_top"]);
	MeshCompress dst(config["dst_top"]);
	intVec src_land_id = FILEIO::loadIntDynamic(config["src_land_id"]);
	intVec dst_sys_2 = FILEIO::loadIntDynamic(config["dst_sys"]);
	intVec dst_mid_2 = FILEIO::loadIntDynamic(config["dst_mid"]);
	if (dst_sys_2.size() + dst_mid_2.size() != dst.n_vertex_)
	{
		LOG(ERROR) << "loaded: " << dst_sys_2.size() + dst_mid_2.size() << std::endl;
		LOG(ERROR) << "n_vertex_: " << dst.n_vertex_ << std::endl;
		LOG(ERROR) << "vertex sys info loading failed." << std::endl;
		return;
	}
	intVec landmark_mid = FILEIO::loadIntDynamic(config["landmark_mid"]);
	intVec landmark_match = FILEIO::loadIntDynamic(config["landmark_match"]);

	if (landmark_mid.size() + landmark_match.size() != src_land_id.size())
	{
		LOG(ERROR) << "landmarks sys info loading failed." << std::endl;
		return;
	}

	float3Vec src_landmark;
	for (int i = 0; i < src_land_id.size(); i++)
	{
		src_landmark.push_back(src.pos_[src_land_id[i]]);
	}
	intVec dst_land_id = src_land_id;

#pragma omp parallel for
	for (int i = 0; i < src_landmark.size(); i++)
	{
		floatVec match_dis(dst.n_vertex_, INT_MAX);
		for (int j = 0; j < dst.n_vertex_; j++)
		{
			match_dis[j] = (src_landmark[i] - dst.pos_[j]).norm();
		}
		auto res = std::minmax_element(match_dis.begin(), match_dis.end());
		//std::cout << res.first - match_dis.begin() << std::endl;
		//std::cout << *res.first << std::endl;
		dst_land_id[i] = res.first - match_dis.begin();
	}

	intSet dst_mid_set(dst_mid_2.begin(), dst_mid_2.end());
	for (int i = 0; i < landmark_mid.size(); i++)
	{
		int land_idx = landmark_mid[i];
		int dst_idx = dst_land_id[land_idx];
		if (!dst_mid_set.count(dst_idx))
		{
			LOG(ERROR) << "No. " << land_idx << " landmark for vertex id: " << dst_idx << " is not in the set." << std::endl;
		}

	}
	//rectify mapping maping right 2 left
	std::map<int, int> right_to_left;
	for (int i = 0; i < dst_sys_2.size()/2; i++)
	{
		int left = dst_sys_2[2 * i];
		int right = dst_sys_2[2 * i + 1];
		right_to_left[right] = left;
	}

	intVec landmark_match_refine = dst_land_id;
	for (int i = 0; i < landmark_match.size()/2; i++)
	{
		int right_landmark_idx = landmark_match[2 * i];
		int left_landmark_idx = landmark_match[2 * i + 1];
		int right_dst_idx = dst_land_id[right_landmark_idx];
		int left_dst_idx = dst_land_id[left_landmark_idx];
		//make match
		if (!right_to_left.count(right_dst_idx))
		{
			LOG(ERROR) << "left right error." << std::endl;
			LOG(ERROR) << "left_dst_idx: " << left_dst_idx << std::endl;
			LOG(ERROR) << "right_dst_idx: " << right_dst_idx << std::endl;
		}
		else
		{
			int left_refine_idx = right_to_left[right_dst_idx];
			landmark_match_refine[left_landmark_idx] = left_refine_idx;
		}
	}

	for (int i = 0; i < landmark_mid.size(); i++)
	{
		int vertex_id = landmark_match_refine[landmark_mid[i]];
		LOG(INFO) << "No landmar idx: " << landmark_mid[i] << ", " << dst.pos_[vertex_id].transpose() << std::endl;
	}

	for (int i = 0; i < landmark_match.size()/2; i++)
	{	
		int right_landmark_id = landmark_match[2 * i];
		int left_landmark_id = landmark_match[2 * i + 1];
		int right_vertex_id = landmark_match_refine[right_landmark_id];
		int left_vertex_id = landmark_match_refine[left_landmark_id];
		LOG(INFO) << "right idx: " << right_landmark_id << ", " << dst.pos_[right_vertex_id].transpose() << std::endl;
		LOG(INFO) << "left idx: " << left_landmark_id << ", " << dst.pos_[left_vertex_id].transpose() << std::endl;
	}

	FILEIO::saveDynamic(config["dst_land_id"], landmark_match_refine, ",");

}

void SysFinder::getRightToLeft(const json& config)
{
	intVec src_land_id = FILEIO::loadIntDynamic(config["src_land_id"]);
	intVec land_right_left = FILEIO::loadIntDynamic(config["landmark_match"]);
	intVec src_mid_2 = FILEIO::loadIntDynamic(config["landmark_mid"]);
	intVec dst_land_id = src_land_id;
	intVec mesh_mid = FILEIO::loadIntDynamic(config["mesh_mid"]);
	intVec mesh_sys = FILEIO::loadIntDynamic(config["mesh_sys"]);
	MeshCompress obj_template(config["template"]);
	for (size_t i = 0; i < src_land_id.size(); i++)
	{
		int idx = src_land_id[i];
		if (std::find(src_mid_2.begin(), src_mid_2.end(), i) != src_mid_2.end())
		{
			//in the middle
			if (std::find(mesh_mid.begin(), mesh_mid.end(), idx) == mesh_mid.end())
			{
				LOG(ERROR) << "mid point in landmark id but not in the mid line of mesh." << std::endl;
				LOG(ERROR) << "for landmark id: " << i << ", vertex Num: " << idx << std::endl;
			}
		}
		else if (idx < 0)
		{
			//need fix
		}
		else
		{
			//fix
			auto iter_pos = std::find(land_right_left.begin(), land_right_left.end(), i);
			int pos = iter_pos - land_right_left.begin();
			if (iter_pos == land_right_left.end())
			{
				LOG(ERROR) << "sys info not found" << std::endl;
			}
			else if (pos % 2 == 1)
			{
				LOG(WARNING) << "this is the left index, skip for now" << std::endl;
			}
			else
			{
				int right_land = land_right_left[pos];
				int left_land = land_right_left[pos + 1];
				int right_idx = src_land_id[right_land];
				auto iter_idx = std::find(mesh_sys.begin(), mesh_sys.end(), right_idx);
				int pos_idx = iter_idx - mesh_sys.begin();
				if (iter_idx == mesh_sys.end() || pos_idx % 2 == 0)
				{
					LOG(ERROR) << "sys info not found" << std::endl;
				}
				dst_land_id[left_land] = mesh_sys[pos_idx - 1];
			}
		}
	}
	FILEIO::saveDynamic(config["dst_land_id"], dst_land_id, ",");
}

RTTR_REGISTRATION
{
registration::class_<MeshSysFinder>("CGP::MeshSysFinder").constructor<>()
	.property("template_", &CGP::MeshSysFinder::template_)
	.property("mid_", &CGP::MeshSysFinder::mid_)
	.property("match_", &CGP::MeshSysFinder::match_)
	;

}
void MeshSysFinder::init()
{
	template_obj_.loadObj(root_ + template_);
	intVec landmark_mid = FILEIO::loadIntDynamic(root_ + mid_);
	mid_ids_ = intSet(landmark_mid.begin(), landmark_mid.end());
	match_ids_ = FILEIO::loadIntDynamic(root_ + match_);
	for (int i = 0; i < match_ids_.size()/2; i++)
	{
		left_ids_.insert(match_ids_[2 * i]);
		right_ids_.insert(match_ids_[2 * i+1]);
	}
	if (mid_ids_.size() + match_ids_.size() != template_obj_.n_vertex_)
	{
		LOG(WARNING) << "landmarks sys info loading: not all vertex are involved. Normal for unSym model." << std::endl;
		return;
	}
}

int MeshSysFinder::getSysId(int input)
{
	if (mid_ids_.count(input))
	{
		return input;
	}
	else
	{
		auto match_pos = std::find(match_ids_.begin(), match_ids_.end(), input);
		if (match_pos == match_ids_.end())
		{
			LOG(ERROR) << "this seems to be wrong, not found in mid && sys." << std::endl;
			return -1;
		}
		else
		{
			int pos = (match_pos - match_ids_.begin());
			if (pos % 2 == 0)
			{
				return match_ids_[pos + 1];
			}
			else
			{
				return match_ids_[pos -1];
			}
		}
	}
}

void MeshSysFinder::getSysIdsInPlace(intVec& input)
{
	intSet sys_data;
	for (int i: input)
	{
		if (mid_ids_.count(i))
		{
			sys_data.insert(i);
		}
		else
		{
			auto match_pos = std::find(match_ids_.begin(), match_ids_.end(), i);
			if (match_pos == match_ids_.end())
			{
				LOG(ERROR) << "this seems to be wrong, not found in mid && sys." << std::endl;
				return;
			}
			else
			{
				int half_pos = (match_pos - match_ids_.begin()) / 2;
				sys_data.insert(match_ids_[2 * half_pos]);
				sys_data.insert(match_ids_[2 * half_pos+1]);
			}
		}
	}
	input.clear();
	input = intVec(sys_data.begin(), sys_data.end());
}

void MeshSysFinder::getMirrorIdsInPlace(intVec& input)
{
	intVec input_back = input;
#pragma omp parallel for
	for (int i = 0; i<input.size(); i++)
	{
		input[i] = getSysId(input_back[i]);
	}
}

void MeshSysFinder::getSysPosLapInPlace(float3Vec& src) const
{
	//assume obj is sysmetric by y_z plane
	int n_vertex = template_obj_.n_vertex_;
	if (src.size() != n_vertex)
	{
		LOG(ERROR) << "input src size don't fit." << std::endl;
		return;
	}
	LaplacianDeform lap_sys;
	MeshCompress sys_result = template_obj_;
	intVec all_point(n_vertex);
	std::iota(all_point.begin(), all_point.end(), 0);
	SG::safeMemcpy(sys_result.pos_.data(), src.data(), n_vertex * 3 * sizeof(float));
	lap_sys.init(sys_result, all_point, {});
	float3Vec deform_pos;
	deform_pos.resize(n_vertex);
	float mid_line = 0;
	for (int i : mid_ids_)
	{
		mid_line += sys_result.pos_[i].x();
	}
	mid_line = mid_line * 1.0f / mid_ids_.size();
	for (int i: mid_ids_)
	{
		float3E temp_mid = sys_result.pos_[i];
		temp_mid.x() = mid_line;
		deform_pos[i] = temp_mid;
	}

	for (int j = 0; j < match_ids_.size() / 2; j++)
	{
		float3E left_point = sys_result.pos_[match_ids_[2 * j]];
		float3E right_point = sys_result.pos_[match_ids_[2 * j + 1]];
		float3E mid_way;
		float x_dis = (left_point.x() - right_point.x());
		float y_same = (left_point.y() + right_point.y())*0.5;
		float z_same = (left_point.z() + right_point.z())*0.5;
		mid_way.x() = (left_point.x() - right_point.x())*0.5 + mid_line;

		left_point.x() = mid_line + 0.5*x_dis;
		right_point.x() = mid_line - 0.5*x_dis;
		left_point.y() = y_same;
		right_point.y() = y_same;
		left_point.z() = z_same;
		right_point.z() = z_same;

		//set
		deform_pos[match_ids_[2 * j]] = left_point;
		deform_pos[match_ids_[2 * j + 1]] = right_point;
	}
	lap_sys.deform(deform_pos, src);
}

void MeshSysFinder::getSysPosAvgInPlace(float3Vec& src)
{
	//assume obj is sysmetric by y_z plane
	int n_vertex = template_obj_.n_vertex_;
	if (src.size() != n_vertex)
	{
		LOG(ERROR) << "input src size don't fit." << std::endl;
		return;
	}
	float mid_line = 0;
	for (int i : mid_ids_)
	{
		mid_line += src[i].x();
	}
	mid_line = mid_line * 1.0f / mid_ids_.size();
	for (int i : mid_ids_)
	{
		float3E temp_mid = src[i];
		temp_mid.x() = mid_line;
		src[i] = temp_mid;
	}
#pragma omp parallel for
	for (int j = 0; j < match_ids_.size() / 2; j++)
	{
		float3E left_point = src[match_ids_[2 * j]];
		float3E right_point = src[match_ids_[2 * j + 1]];
		float3E mid_way;
		float x_dis = (left_point.x() - right_point.x());
		float y_same = (left_point.y() + right_point.y())*0.5;
		float z_same = (left_point.z() + right_point.z())*0.5;
		mid_way.x() = (left_point.x() - right_point.x())*0.5 + mid_line;

		left_point.x() = mid_line + 0.5*x_dis;
		right_point.x() = mid_line - 0.5*x_dis;
		left_point.y() = y_same;
		right_point.y() = y_same;
		left_point.z() = z_same;
		right_point.z() = z_same;

		//set
		src[match_ids_[2 * j]] = left_point;
		src[match_ids_[2 * j + 1]] = right_point;
	}

}

void MeshSysFinder::keepOneSideUnchanged(const float3Vec& src, float3Vec& dst, bool is_left)
{	
#pragma omp parallel for
	for (int i = 0; i < match_ids_.size()/2; i++)
	{
		//in guijie sense
		int left = match_ids_[2 * i + 1];
		int right = match_ids_[2 * i];
		if (is_left == true)
		{
			dst[left] = src[left];
		}
		else
		{
			dst[right] = src[right];
		}
	}
}
