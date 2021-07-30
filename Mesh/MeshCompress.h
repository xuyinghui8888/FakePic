#ifndef __MESHCOMPRESS_H__
#define __MESHCOMPRESS_H__
#include "../Basic/CGPBaseHeader.h"
namespace CGP
{
	//only due with triangle mesh
	class MeshCompress
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		MeshCompress();
		MeshCompress(const cstr& file_name);
		//copy instructor
		MeshCompress(const MeshCompress& src);
		float3Vec pos_;
		float3Vec normal_;
		float3Vec normal_face_;
		uintVec tri_;
		uintVec tri_uv_;
		float2Vec tex_cor_;
		intX2Vec tri_int_3_;
		intX2Vec ori_face_;
		intX2Vec ori_face_uv_;
		intSetVec vertex_tri_;
		intSetVec vertex_vertex_;
		cstrVec material_;
		float3Vec vertex_color_;
		float3E center_;
		cstr g_name_maya_ = "";

		float scale_ = -1;
		int n_vertex_ = -1;
		int n_tri_ = -1;
		int n_uv_ = -1;
		
		doubleVec xyz_min_ = { 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX, 1.0 * INTMAX_MAX };
		doubleVec xyz_max_ = { -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX, -1.0 * INTMAX_MAX };
		
		//update for n_vertex_, n_tri_, tri_int_3_ && tri_int_3_
		void flipYZ();
		void rotateX();
		void replaceVertexBasedData(const MeshCompress& src);
		void update();
		void loadObj(const cstr& file_name);
		void loadOri(const cstr& file_name);
		void saveMtl(const cstr& file_name, const cstr& texture_name = "") const;
		void saveObj(const cstr& file_name, const cstr& mtl_name = "") const;
		void saveVertexColor(const cstr& file_name) const;
		void saveOri(const cstr& file_name, const cstr& mtl_name = "") const;
		void setGOption(const cstr& name_vec);
		intVec getDiscardMap(const intVec& dis_vertex);
		void discardMaterial();
		intVec discard(const intVec& dis_vertex);
		intVec discard(const intX2Vec& dis_vertex_vec);
		intVec discard(const intX2Vec& keep_vec, const intX2Vec& discard_vec);
		intVec discard(const intVec& keep_vec, const intX2Vec& discard_vec);
		//discard quad face
		void discardOri(const intVec& dis_vertex);
		intVec keepRoi(const intVec& keep_vertex);
		void clear();
		bool safeMeshData() const;
		void getBoundingBox();
		void getBoundingBox(const intVec& idx, doubleVec& xyz_min, doubleVec& xyz_max) const;
		void getBoundingBox(const intVec& idx, const mat3f& rot, doubleVec& xyz_min, doubleVec& xyz_max) const;
		void generateNormal();
		void getVertexTopology();
		void getSlice(const intVec& index, float3Vec& res) const;
		void moveToCenter(bool is_scale = false);
		void moveToOri(bool is_scale = false);
		void memcpyVertex(const MeshCompress& src);
		intVec getReverseSelection(const intVec& src);
	};
}
#endif