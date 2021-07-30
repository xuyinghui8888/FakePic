#ifndef __OBJLOADER_H__
#define __OBJLOADER_H__
// OBJ_Loader.h - A Single Header OBJ Model Loader

// Iostream - STD I/O Library
#include <iostream>

// Vector - STD Vector/Array Library
#include <vector>

// String - STD String Library
#include <string>

// fStream - STD File I/O Library
#include <fstream>

// Math.h - STD math Library
#include <math.h>

// Print progress to console while loading (large models)
#define OBJL_CONSOLE_OUTPUT

// Namespace: OBJL
//
// Description: The namespace that holds eveyrthing that
//	is needed and used for the OBJ Model Loader

#include "MeshCompress.h"

namespace objl
{
	// Structure: Vector2
	//
	// Description: A 2D Vector that Holds Positional Data
	struct Vector2
	{
		// Default Constructor
		Vector2()
		{
			X = 0.0f;
			Y = 0.0f;
		}
		// Variable Set Constructor
		Vector2(float X_, float Y_)
		{
			X = X_;
			Y = Y_;
		}
		// Bool Equals Operator Overload
		bool operator==(const Vector2& other) const
		{
			return (this->X == other.X && this->Y == other.Y);
		}
		// Bool Not Equals Operator Overload
		bool operator!=(const Vector2& other) const
		{
			return !(this->X == other.X && this->Y == other.Y);
		}
		// Addition Operator Overload
		Vector2 operator+(const Vector2& right) const
		{
			return Vector2(this->X + right.X, this->Y + right.Y);
		}
		// Subtraction Operator Overload
		Vector2 operator-(const Vector2& right) const
		{
			return Vector2(this->X - right.X, this->Y - right.Y);
		}
		// Float Multiplication Operator Overload
		Vector2 operator*(const float& other) const
		{
			return Vector2(this->X *other, this->Y * other);
		}

		// Positional Variables
		float X;
		float Y;
	};

	// Structure: Vector3
	//
	// Description: A 3D Vector that Holds Positional Data
	struct Vector3
	{
		// Default Constructor
		Vector3()
		{
			X = 0.0f;
			Y = 0.0f;
			Z = 0.0f;
		}
		// Variable Set Constructor
		Vector3(float X_, float Y_, float Z_)
		{
			X = X_;
			Y = Y_;
			Z = Z_;
		}
		// Bool Equals Operator Overload
		bool operator==(const Vector3& other) const
		{
			return (this->X == other.X && this->Y == other.Y && this->Z == other.Z);
		}
		// Bool Not Equals Operator Overload
		bool operator!=(const Vector3& other) const
		{
			return !(this->X == other.X && this->Y == other.Y && this->Z == other.Z);
		}
		// Addition Operator Overload
		Vector3 operator+(const Vector3& right) const
		{
			return Vector3(this->X + right.X, this->Y + right.Y, this->Z + right.Z);
		}
		// Subtraction Operator Overload
		Vector3 operator-(const Vector3& right) const
		{
			return Vector3(this->X - right.X, this->Y - right.Y, this->Z - right.Z);
		}
		// Float Multiplication Operator Overload
		Vector3 operator*(const float& other) const
		{
			return Vector3(this->X * other, this->Y * other, this->Z * other);
		}
		// Float Division Operator Overload
		Vector3 operator/(const float& other) const
		{
			return Vector3(this->X / other, this->Y / other, this->Z / other);
		}

		// Positional Variables
		float X;
		float Y;
		float Z;
	};

	// Structure: Vertex
	//
	// Description: Model Vertex object that holds
	//	a Position, Normal, and Texture Coordinate
	struct Vertex
	{
		// Position Vector
		Vector3 Position;

		// Normal Vector
		Vector3 Normal;

		// Texture Coordinate Vector
		Vector2 TextureCoordinate;
	};

	struct Material
	{
		Material()
		{
			name = "";
			Ns = 0.0f;
			Ni = 0.0f;
			d = 0.0f;
			illum = 0;
		}

		// Material Name
		std::string name;
		// Ambient Color
		Vector3 Ka;
		// Diffuse Color
		Vector3 Kd;
		// Specular Color
		Vector3 Ks;
		// Specular Exponent
		float Ns;
		// Optical Density
		float Ni;
		// Dissolve
		float d;
		// Illumination
		int illum;
		// Ambient Texture Map
		std::string map_Ka;
		// Diffuse Texture Map
		std::string map_Kd;
		// Specular Texture Map
		std::string map_Ks;
		// Specular Hightlight Map
		std::string map_Ns;
		// Alpha Texture Map
		std::string map_d;
		// Bump Map
		std::string map_bump;
	};

	// Structure: Mesh
	//
	// Description: A Simple Mesh Object that holds
	//	a name, a vertex list, and an index list
	struct Mesh
	{
		// Default Constructor
		Mesh()
		{

		}
		// Variable Set Constructor
		Mesh(std::vector<Vertex>& _Vertices, std::vector<unsigned int>& _Indices)
		{
			Vertices = _Vertices;
			Indices = _Indices;
		}
		// Mesh Name
		std::string MeshName;
		// Vertex List
		std::vector<Vertex> Vertices;
		// Index List
		std::vector<unsigned int> Indices;

		// Material
		Material MeshMaterial;
	};

	// Namespace: Math
	//
	// Description: The namespace that holds all of the math
	//	functions need for OBJL
	namespace math
	{
		// Vector3 Cross Product
		Vector3 CrossV3(const Vector3 a, const Vector3 b);

		// Vector3 Magnitude Calculation
		float MagnitudeV3(const Vector3 in);

		// Vector3 DotProduct
		float DotV3(const Vector3 a, const Vector3 b);

		// Angle between 2 Vector3 Objects
		float AngleBetweenV3(const Vector3 a, const Vector3 b);

		// Projection Calculation of a onto b
		Vector3 ProjV3(const Vector3 a, const Vector3 b);
	}

	// Namespace: Algorithm
	//
	// Description: The namespace that holds all of the
	// Algorithms needed for OBJL
	namespace algorithm
	{
		// Vector3 Multiplication Opertor Overload
		Vector3 operator*(const float& left, const Vector3& right);

		// A test to see if P1 is on the same side as P2 of a line segment ab
		bool SameSide(Vector3 p1, Vector3 p2, Vector3 a, Vector3 b);

		// Generate a cross produect normal for a triangle
		Vector3 GenTriNormal(Vector3 t1, Vector3 t2, Vector3 t3);

		// Check to see if a Vector3 Point is within a 3 Vector3 Triangle
		bool inTriangle(Vector3 point, Vector3 tri1, Vector3 tri2, Vector3 tri3);		

		// Split a String into a string array at a given token
		void split(const std::string& in,
			std::vector<std::string>& out,
			std::string token);
		// Get tail of string after first token and possibly following spaces
		std::string tail(const std::string& in);

		// Get first token of string
		std::string firstToken(const std::string& in);		

		// Get element at given index position
		template <class T>
		inline const T & getElement(const std::vector<T> &elements, std::string &index)
		{
			int idx = std::stoi(index);
			if (idx < 0)
				idx = int(elements.size()) + idx;
			else
				idx--;
			return elements[idx];
		}
	}

	// Class: Loader
	//
	// Description: The OBJ Model Loader
	class Loader
	{
	public:
		// Default Constructor
		Loader()
		{

		}
		~Loader()
		{
			LoadedMeshes.clear();
		}

		// Load a file into the loader
		//
		// If file is loaded return true
		//
		// If the file is unable to be found
		// or unable to be loaded return false
		bool LoadFile(std::string Path);

		// Loaded Mesh Objects
		std::vector<Mesh> LoadedMeshes;
		// Loaded Vertex Objects
		std::vector<Vertex> LoadedVertices;
		// Loaded Index Positions
		std::vector<unsigned int> LoadedIndices;
		// Loaded Material Objects
		std::vector<Material> LoadedMaterials;

		CGP::MeshCompress mesh_data_;

	private:
		// Generate vertices from a list of positions, 
		//	tcoords, normals and a face line
		void GenVerticesFromRawOBJ(std::vector<Vertex>& oVerts,
			const std::vector<Vector3>& iPositions,
			const std::vector<Vector2>& iTCoords,
			const std::vector<Vector3>& iNormals,
			std::string icurline);

		// Triangulate a list of vertices into a face by printing
		//	inducies corresponding with triangles within it
		void VertexTriangluation(std::vector<unsigned int>& oIndices,
			const std::vector<Vertex>& iVerts);

		// Load Materials from .mtl file
		bool LoadMaterials(std::string path);

		//generate compressed version of all mesh parts
		void genCompressedVersion();
	};
}
#endif

