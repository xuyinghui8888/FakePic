#ifndef OBJ_DRAWER_H
#define OBJ_DRAWER_H
#include "../Shader/shader.h"
#include "../Basic/CGPBaseHeader.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace CGP
{
	class ObjDrawer
	{
	public:
		ObjDrawer() {}
		void init(const int width, const int height, bool debug = false) 
		{
			if (inited)
				return;
			this->width = width;
			this->height = height;
			this->debug = debug;
			initOpenGL();
			initFrameBuffer();
			initShaders();
			inited = true;
		}
		inline bool isInited() const
		{
			return inited;
		}
		void bindVertices(const GLfloat* const vertices, const int vertexCount) const;
		//pixels should be width*height*1, transform-3x4 mat as prjMat
		void drawAndGetContourPixels(const int vertexCount, const GLfloat* const transform, unsigned char* pixels) const;
		~ObjDrawer();
		void clear();
	private:
		void initOpenGL();
		void initShaders();
		void initFrameBuffer();
	private:
		GLFWwindow* window;
		int width, height;
		Shader *shader1, *shader2;
		GLuint VBO, VAO, VBO2, VAO2;
		GLuint textureID, fboID;
		bool debug;
		bool inited = false;
	};
}

#endif
