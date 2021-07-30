#include "ObjDrawer.h"
using namespace CGP;
//each line is the XYZ coordinates of a vertex. Here are two triangles with 6 vertices to fill the whole area of the screen
static const GLfloat vertices2[] = {
    -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f
};
ObjDrawer::~ObjDrawer()
{
	clear();
}
void ObjDrawer::clear()
{
	if (inited) {
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteVertexArrays(1, &VAO2);
		glDeleteBuffers(1, &VBO2);
		glDeleteTextures(1, &textureID);
		glDeleteFramebuffers(1, &fboID);
		delete shader1;
		delete shader2;
		glfwTerminate();
	}
}
//to tell OpenGL where the vertex data are, need to be executed only once in initialization
void ObjDrawer::bindVertices(const GLfloat* const vertices, const int vertexCount) const {
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0])*vertexCount * 3, vertices, GL_STATIC_DRAW);

    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices2[0]) * 18, vertices2, GL_STATIC_DRAW);

    glBindVertexArray(0);
}
//pixels should be width*height*1, transform-3x4 mat as prjMat
void ObjDrawer::drawAndGetContourPixels(const int vertexCount, const GLfloat* const transform, unsigned char* pixels) const{
    //draw the triangles with vertices
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    shader1->use();
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int transformLoc = glGetUniformLocation(shader1->program, "transform");

    glUniformMatrix4x3fv(transformLoc, 1, GL_TRUE, transform);
    glBindVertexArray(VAO);
    //glBindBuffer(GL_ARRAY_BUFFER, VBO);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(ff[0])*vertexCount * 3, ff, GL_STATIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

#if 0
    //extract the contour
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    shader2->use();
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(VAO2);
    //glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices2[0]) * 18, vertices2, GL_STATIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

#endif
    glBindVertexArray(0);

    //only to read the r-channel
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    if (debug) {
        int sum = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (pixels[i*width + j] != 0)
                    sum++;
            }
        }
        printf("%d\n", sum);

        glfwSwapBuffers(window);
    }
}
void ObjDrawer::initOpenGL() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    if (!debug)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    else
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    window = glfwCreateWindow(width, height, "", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
    glewInit();  

    glViewport(0, 0, width, height);
    glClearColor(0.f, 0.f, 0.f, 1.0f);
}
void ObjDrawer::initShaders() {
    shader1 = new Shader(1);
    shader2 = new Shader(2);
    shader2->use();
    GLint sizeLocation = glGetUniformLocation(shader2->program, "winsize");
    glUniform2i(sizeLocation, width, height);
    GLint texLoc = glGetUniformLocation(shader2->program, "sampler0");
    glUniform1i(texLoc, 0);//here, not texture id, but the index inside the bounded texture, we have only one 
}
void ObjDrawer::initFrameBuffer() {
    glGenFramebuffers(1, &fboID);
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);

    glGenTextures(1, &textureID);
	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, NULL);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("Error in checking frame buffer status\n");
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO2);
    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0); // Unbind VAO
}

