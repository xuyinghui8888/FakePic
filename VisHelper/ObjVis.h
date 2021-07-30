#ifndef OBJ_VIS_H
#define OBJ_VIS_H
#include "../Shader/shader.h"
#include "../Mesh/MeshCompress.h"
#include "ObjDrawer.h"
#include "../Basic/CGPBaseHeader.h"


namespace CGP
{
	class ObjVis
	{
	public:
		void setFixSize(int width, int height, bool debug = false);
		cv::Mat drawMesh(const MeshCompress& mesh_data, int width, int height, bool debug);
	private:
		ObjDrawer mesh_drawer_;
		int width_ = -1;
		int height_ = -1;
		bool debug_ = false;
	};
}

#endif
