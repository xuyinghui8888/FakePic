#ifndef __SAFEGUARD_H__
#define __SAFEGUARD_H__
#include "CGPBaseHeader.h"
namespace SG
{
	int safeMemcpy(void* p_dst, void const* p_src, size_t size);
	int safeMemset(void* p_dst, size_t size);
	//used for checking if data is valid
	bool checkMeshDataVec(const CGP::float3Vec& data_vec);
	bool checkTriIdx(const CGP::uintVec& tri_idx, int max_vertex_idx);
	bool checkMesh(const CGP::float3Vec& data_vec, const CGP::uintVec& tri_idx);
	bool isDigits(const std::string &str);
	void needPath(const CGP::cstr& path);
	bool isExist(const CGP::cstr& file_name);
	std::string exec(const CGP::cstr& cmd);

	template<class T1, class T2>
	bool checkSameSize(const T1& src, const T2& dst)
	{
		return src.size() == dst.size();
	}

	template<class T1, class T2>
	bool checkBBox(const T1& img, T2& bbox)
	{
		int tl_x_back = bbox.tl().x;
		int tl_y_back = bbox.tl().y;
		int br_x_back = bbox.br().x;
		int br_y_back = bbox.br().y;
		int tl_x = DLIP3INT(tl_x_back, 0, img.cols - 1);
		int tl_y = DLIP3INT(tl_y_back, 0, img.rows - 1);
		int br_x = DLIP3INT(tl_x + bbox.width, 0, img.cols - 1);
		int br_y = DLIP3INT(tl_y + bbox.height, 0, img.rows - 1);
		//check for tl_x
		if (tl_x >= br_x || tl_y >= br_y)
		{
			bbox.x = 0;
			bbox.y = 0;
			bbox.width = 0;
			bbox.height = 0;
			return false;
		}
		else
		{
			bbox.x = tl_x;
			bbox.y = tl_y;
			bbox.width = br_x - tl_x;
			bbox.height = br_y - tl_y;
			return true;
		}
	}

}
#endif