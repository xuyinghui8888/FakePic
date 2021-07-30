#ifndef POSE_SHADER_H
#define POSE_SHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//#define GLEW_STATIC //always enabled, to link glew statically in windows, no effect in IOS

#ifdef TARGET_OS_IPHONE

#include <OpenGLES/ES3/gl.h>
#include <OpenGLES/ES3/glext.h>

//for the IOS version, shader only performs mesh projection, without extracting the contour which is performed by CPU outside Class Shader
class Shader
{
public:
	GLuint program;
	Shader()
	{
        static const std::string vertexCode1 =
            "#version 300 es\n"
            "layout (location = 0) in vec3 position;\n"
            "uniform mat4x3 transform;\n"
            "void main(){\n"
            "    vec3 p = transform*vec4(position, 1.0f);\n"
            "    gl_Position = vec4(2.0f*p.x/p.z-1.0f, 2.0f*p.y/p.z-1.0f, 1.0f, 1.0f);//z:[0, 1]\n"
            "    //gl_Position = vec4((2.0f*p.x/p.z-1.0f)/100.f, (2.0f*p.y/p.z-1.0f)/100.f, 1.0f, 1.0f);//z:[0, 1]\n"
            "    //gl_Position = vec4(position.xy, 1.0f,1.0f);\n"
            "}\n";
        static const std::string fragmentCode1 =
            "#version 300 es\n"
            "precision highp float;\n"
            "out vec3 color;\n"
            "void main(){\n"
            "    color=vec3(1.f,0.f,0.f);\n"
            "}\n";


		const GLchar* vShaderCode = vertexCode1.c_str();
		const GLchar* fShaderCode = fragmentCode1.c_str();
		// 2. Compile shaders
		GLuint vertex = 0, fragment = 0;
		GLchar infoLog[512];
		// Vertex Shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		//        std::cout<<"vertex:"<<vertex<<std::endl;

		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		// Print compile errors if any
		GLint success = -1;
		glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
		//        std::cout<<"success:"<<success<<std::endl;
		if (!success)
		{
			glGetShaderInfoLog(vertex, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		// Fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		// Print compile errors if any
		glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragment, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// Shader program
		this->program = glCreateProgram();
		glAttachShader(this->program, vertex);
		glAttachShader(this->program, fragment);
		glLinkProgram(this->program);
		// Print linking errors if any
		glGetProgramiv(this->program, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(this->program, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::program::LINKING_FAILED\n" << infoLog << std::endl;
		}
		// Delete the shaders as they're linked into our program now and no longer necessery
		glDeleteShader(vertex);
		glDeleteShader(fragment);

	}
	// Uses the current shader
	void use()
	{
		glUseProgram(this->program);
	}
};
#endif //TARGET_OS_IPHONE

#ifdef _WIN32
#include <GL/glew.h>

class Shader
{
public:
	GLuint program;
	// mode=1 or 2
	// 1-to project the mesh into the image
	// 2-to extract the contour of the projection area
	Shader(const int mode)
	{
		static const std::string vertexCode1 =
			"#version 330 core\n"
			"layout (location = 0) in vec3 position;\n"
			"uniform mat4x3 transform;\n"
			"void main(){\n"
			"    vec3 p = transform*vec4(position, 1.0f);\n"
			"    gl_Position = vec4(2.0f*p.x/p.z-1.0f, 2.0f*p.y/p.z-1.0f, p.z/1000.0f, 1.0f);//z:[0, 1]\n"
			"}\n";
		static const std::string fragmentCode1 =
			"#version 330 core\n"
			"precision highp float;\n"
			"out vec3 color;\n"
			"void main(){\n"
			"    color=vec3(1.f,1.f,1.f);\n"
			"}\n";
		static const std::string vertexCode2 =
			"#version 330 core\n"
			"layout(location = 0) in vec3 position;\n"
			"out vec2 cord;\n"
			"void main() {\n"
			"	gl_Position = vec4(position, 1.0f);\n"
			"	cord = 0.5*(position.xy) + vec2(0.5f, 0.5f);//pos:[-1,1], cord[0,1]\n"
			"}\n";
		static const std::string fragmentCode2 =
			"#version 330 core\n"
			"uniform sampler2D sampler0;\n"
			"uniform ivec2 winsize;\n"
			"in vec2 cord;\n"
			"out vec3 c;\n"
			"float w = 1.0f / winsize.x;\n"
			"float h = 1.0f / winsize.y;\n"
			"//judge the pixel with shift dx and dy is > 0\n"
			"int get(int dx, int dy) {\n"
			"	float x = cord.x + dx * w;\n"
			"	float y = cord.y + dy * w;\n"
			"	return int(texture2D(sampler0, vec2(x, y)).r > 0);\n"
			"}\n"
			"void main() {\n"
			"	int sum = 0; sum += get(-1, 0); sum += get(1, 0); sum += get(0, -1); sum += get(0, 1);\n"
			"	c = vec3(int(cord.x > w&&cord.y > h&&cord.x < 1 - w && cord.y < 1 - h)*float(sum < 4&&texture2D(sampler0, cord).r>0), 0, 0);\n"
			"}\n";
		const GLchar* vShaderCode = NULL;
		const GLchar * fShaderCode = NULL;
		if (mode == 1) {
			vShaderCode = vertexCode1.c_str();
			fShaderCode = fragmentCode1.c_str();
		}
		else {
			vShaderCode = vertexCode2.c_str();
			fShaderCode = fragmentCode2.c_str();
		}

		// 2. Compile shaders
		GLuint vertex, fragment;
		GLint success;
		GLchar infoLog[512];
		// Vertex Shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		// Print compile errors if any
		glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertex, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// Fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		// Print compile errors if any
		glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragment, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// Shader program
		this->program = glCreateProgram();
		glAttachShader(this->program, vertex);
		glAttachShader(this->program, fragment);
		glLinkProgram(this->program);
		// Print linking errors if any
		glGetProgramiv(this->program, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(this->program, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::program::LINKING_FAILED\n" << infoLog << std::endl;
		}
		// Delete the shaders as they're linked into our program now and no longer necessery
		glDeleteShader(vertex);
		glDeleteShader(fragment);

	}
	// Uses the current shader
	void use() const
	{
		glUseProgram(this->program);
	}
};

#endif //_WIN32
#endif //POSE_SHADER_H
