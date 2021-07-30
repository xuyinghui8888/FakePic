#include "Mapping.h"
#include "../FileIO/FileIO.h"
using namespace CGP;

intVec MAP::getReverseMapping(const intVec& src)
{
	auto min_max = std::minmax_element(src.begin(), src.end());
	int max_idx = min_max.first - src.begin();
	int max_value = *(min_max.second);	
	if (max_value < 1)
	{
		LOG(ERROR) << "max_value<=0" << std::endl;
		return src;
	}
	intVec res(max_value+1, -1);
	for (int i = 0; i < src.size(); i++)
	{
		int from_idx = i;
		int to_idx = src[i];
		if (to_idx < 0)
		{
			//mapping empty;
		}
		else if(to_idx>max_value)
		{
			LOG(ERROR) << "somehow to_idx>max_value";
		}
		else if(res[to_idx]>=0)
		{
			LOG(WARNING) << "doulbe mapping exists, replace cur value" << std::endl;
			res[to_idx] = from_idx;
		}
		else
		{
			res[to_idx] = from_idx;
		}
	}
	return res;
}

intX2Vec MAP::splitMapping(const intVec& src, bool xyzOrder, int size)
{
	//xyz order using mapping_0, mapping_1, mapping_2, mapping_0 order;
	if (src.size() % size != 0)
	{
		LOG(ERROR) << "split dst size not right." << std::endl;
		return {};
	}
	int group = src.size() / size;
	intX2Vec res(size, intVec(group, -1));

	if (xyzOrder)
	{		
#pragma omp parallel for
		for (int iter_group = 0; iter_group < group; iter_group++)
		{
			for (int iter_xyz = 0; iter_xyz < size; iter_xyz++)
			{
				res[iter_xyz][iter_group] = src[iter_group*size + iter_xyz];
			}
		}
	}
	else
	{
#pragma omp parallel for
		for (int iter_xyz = 0; iter_xyz < size; iter_xyz++)		
		{
			for (int iter_group = 0; iter_group < group; iter_group++)
			{
				res[iter_xyz][iter_group] = src[iter_xyz*group + iter_group];
			}
		}
	}
	return res;

}

intX2Vec MAP::passMapping(const intX2Vec& input)
{
	if (input.empty())
	{
		LOG(WARNING) << "input passMapping empty()." << std::endl;
		return {};
	}
	else if (input.size() == 1)
	{
		intVec input_reverse = getReverseMapping(input[0]);
		return { input[0], input_reverse };
	}
	else
	{
		intVec zero_to_end(input[0].size(), -1);
		//pass mapping directly
		int size = input.size();
		for (int i = 0; i < input[0].size(); i++)
		{
			int iter_map = 0;
			while (iter_map < size && input[iter_map][i] >= 0)
			{
				iter_map++;
			}
			if (iter_map == size - 1 && input[iter_map][i] >= 0)
			{
				zero_to_end[i] = input[iter_map][i];
			}
		}

		intVec end_to_zero_reverse = getReverseMapping(zero_to_end);
		return { zero_to_end , end_to_zero_reverse };
	}	
}

intVec MAP::passIdxThroughMapping(const intVec& idx, const intX2Vec& input_mapping)
{
	if (input_mapping.empty())
	{
		LOG(WARNING) << "input_mapping empty." << std::endl;
		return {idx};
	}
	else 
	{
		int size = input_mapping.size();
		intVec res(idx.size(), -1);
		for (int i = 0; i < idx.size(); i++)
		{
			int iter_map = 0;
			int iter_idx = idx[i];
			while (
				iter_map < size && 
				iter_idx>=0 && 
				iter_idx < input_mapping[iter_map].size() &&
				input_mapping[iter_map][iter_idx] >= 0
				)
			{
				iter_idx = input_mapping[iter_map][iter_idx];
				iter_map++;
			}
			if (iter_map == size)
			{
				res[i] = iter_idx;
			}
		}
		return res;
	}
}

intX2Vec MAP::singleToDoubleMap(const intVec& src_to_dst)
{
	if (src_to_dst.empty())
	{
		LOG(WARNING) << "mapping src_to_dst is empty()" << std::endl;
		return {};
	}
	intX2Vec res(2);
	for (int i = 0; i < src_to_dst.size(); i++)
	{
		int idx_src = i;
		int idx_dst = src_to_dst[i];
		if (idx_dst >= 0)
		{
			res[0].push_back(i);
			res[1].push_back(idx_dst);
		}
	}
	return res;
}

intVec MAP::discardRoiKeepOrder(const intVec& src, const intVec& roi)
{
	intVec res;
	intSet roi_set(roi.begin(), roi.end());
	for (int i : src)
	{
		if (!roi_set.count(i))
		{
			res.push_back(i);
		}
	}
	return res;
}

intX2Vec MAP::discardRoiKeepOrder(const intX2Vec& src, const intVec& reference, const intVec& roi)
{
	int size = reference.size();
	for (int i = 0; i < src.size(); i++)
	{
		if (src[i].size() != size)
		{
			LOG(ERROR) << "src size different." << std::endl;
			return {};
		}
	}
	int group = src.size();
	intX2Vec res(group);
	intSet roi_set(roi.begin(), roi.end());

	for (int i = 0; i< size; i++)
	{
		if (!roi_set.count(reference[i]))
		{
			for (int iter_group = 0; iter_group < group; iter_group++)
			{
				res[iter_group].push_back(src[iter_group][i]);
			}
		}
	}
	return res;
}

intVec MAP::getReverseRoi(const intVec& src, int n_max)
{
	if (src.empty())
	{
		intVec res(n_max, 0);
		std::iota(res.begin(), res.end(), 0);
		return res;
	}
	else
	{
		intVec res;
		auto min_max = std::minmax_element(src.begin(), src.end());
		int max_idx = min_max.first - src.begin();
		int max_value = *(min_max.second);
		if (max_value >= n_max)
		{
			LOG(ERROR) << "exist exceed n_max" << std::endl;
			return {};
		}
		else
		{
			intSet src_set(src.begin(), src.end());
			for (int i = 0; i < n_max; i++)
			{
				if (!src_set.count(i))
				{
					res.push_back(i);
				}
			}
			return res;
		}
	}
	
}

intVec MAP::getInterset(const intX2Vec& src)
{
	if (src.empty())
	{
		LOG(ERROR) << "input empty." << std::endl;
		return {};
	}

	if (src.size() == 1)
	{
		return src[0];
	}

	intVec res = src[0];
	for (int i = 1; i < src.size(); i++)
	{
		intSet res_set(res.begin(), res.end());
		res.clear();
		for (int iter_value : src[i])
		{
			if (res_set.count(iter_value))
			{
				res.push_back(iter_value);
			}
		}
	}
	return res;
}

intVec MAP::getUnionset(const intX2Vec& src)
{
	if (src.empty())
	{
		LOG(ERROR) << "input empty." << std::endl;
		return {};
	}

	if (src.size() == 1)
	{
		return src[0];
	}
	intVec res = src[0];
	for (int i = 1; i < src.size(); i++)
	{
		intVec res_temp;
		std::set_union(res.begin(), res.end(), src[1].begin(), src[1].end(), std::back_inserter(res_temp));
		res = res_temp;
	}

	intSet res_set(res.begin(), res.end());
	res.clear();
	res.assign(res_set.begin(), res_set.end());
	return res;
}

intVec MAP::getSubset(const intX2Vec& src)
{
	if (src.empty())
	{
		LOG(ERROR) << "input empty." << std::endl;
		return {};
	}

	if (src.size() == 1)
	{
		return src[0];
	}
	auto src_remove_top = src;
	src_remove_top.erase(src_remove_top.begin());
	intVec left_union = getUnionset(src_remove_top);
	
	return discardRoiKeepOrder(src[0], left_union);
}
