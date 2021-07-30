#ifndef __BEAUTY_35_H__
#define __BEAUTY_35_H__
#include "../Basic/CGPBaseHeader.h"
#include "../Mesh/MeshCompress.h"
namespace CGP
{
	//only due with triangle mesh
	namespace Beauty35
	{
		void calculate35BasedOn68(const float3Vec& src, vecF& x_ratio, vecF& y_ratio);
		void calculate35BasedOn68(const vecF& src, vecF& x_ratio, vecF& y_ratio);
	};
}
#endif