#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

namespace demo {
	static bool startsWith(const std::string &txt, std::string &prefix) {
		if (txt.size() < prefix.size())
			return false;
		for (int i = 0; i < prefix.size(); i++) {
			if (prefix[i] != txt[i])
				return false;
		}
		return true;
	}
	static bool isWhiteChar(char c) {
		if (c <= 32)
			return true;
		else
			return false;
	}
	static std::string trim(const std::string& s) {
		if (s.length() == 0)
			return s;
		size_t start = 0, end = s.length()-1;
		while (isWhiteChar(s[start])) {
			start++;
			if (start >= s.length())
				break;
		}
		while (isWhiteChar(s[end])) {
			end--;
			if (end < 0)
				break;
		}
		if (start > end)
			return std::string("");
		else
			return s.substr(start, end - start + 1);
	}
	static unsigned int splitString(const std::string &txt, std::vector<std::string> &strs, char ch)
	{
		if (txt.length() == 0)
			return 0;
		size_t pos = txt.find(ch);
		size_t initialPos = 0;
		strs.clear();

		// Decompose statement
		while (pos != std::string::npos) {
			strs.push_back(txt.substr(initialPos, pos - initialPos));
			initialPos = pos + 1;

			pos = txt.find(ch, initialPos);
		}

		//std::cout << txt << " ";
		//if (txt[0] == '\n') {
		//	for (auto ite : txt)
		//		std::cout << (int)ite <<" ";
		//	std::cout << initialPos << " " << txt.size() << " " << txt.size() - initialPos  << std::endl;
		//}
		// Add the last one
		if (initialPos < txt.length()) {
			std::string last = txt.substr(initialPos);
			if(trim(last).length()>0)
				strs.push_back(last);
		}
		//for (auto&& ite : strs)
		//	std::cout << ite << std::endl;
		return (unsigned int)strs.size();
	}
	class vec2 {
	public:
		double x, y;

		vec2(double x, double y) { this->x = x; this->y = y; }

		vec2(double v) { this->x = v; this->y = v; }

		vec2() { this->x = this->y = 0.0; }
	};

	class vec3 {
	public:
		double x, y, z;

		vec3(double x, double y, double z) { this->x = x; this->y = y; this->z = z; }

		//vec3(double v) { this->x = v; this->y = v; this->z = v; }

		vec3() { this->x = this->y = this->z = 0; }

		void operator+=(const vec3& b) { (*this) = (*this) + b;}
		void operator-=(const vec3& b) { (*this) = (*this) - b;}

		friend vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
		friend vec3 operator-(const vec3& a) { return vec3(-a.x, -a.y, -a.z); }
		friend vec3 operator+(const vec3& a, const vec3& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
		friend vec3 operator*(const double s, const vec3& a) { return vec3(s * a.x, s * a.y, s * a.z); }
		friend vec3 operator*(const vec3& a, const vec3& b) { return vec3(b.x * a.x, b.y * a.y, b.z * a.z); }
		friend vec3 operator*(const vec3& a, const double s) { return s * a; }
		void operator*=(const double s) { this->x *= s; this->y *= s; this->z *= s; }
		void operator*=(const vec3& a) { this->x *= a.x; this->y *= a.y; this->z *= a.z; }
		friend vec3 operator/(const vec3& a, const double s) { return vec3(a.x / s, a.y / s, a.z / s); }
		friend vec3 operator/(const vec3& a, const vec3& b) { return vec3(a.x / b.x, a.y / b.y, a.z / b.z); }
		void operator/=(const double s) { this->x /= s; this->y /= s; this->z /= s;}
		bool operator==(const vec3& v) { return v.x == this->x&&v.y == this->y&&v.z == this->z; }
		bool operator!=(const vec3& v) { return !(*this == v); }
		friend std::ostream& operator<<(std::ostream& out, const vec3& v) {
			out << v.x << ", " << v.y << ", " << v.z;
			return out;
		}

		static double length(const vec3& a) { return sqrt(vec3::dot(a, a)); }
		double length() { return sqrt(vec3::dot(*this, *this)); }
		static double sqrLength(const vec3& a) { return vec3::dot(a, a); }
		static double sqrLength(const vec3& a, const vec3& b) { return sqrLength(a - b); }
		double sqrLength() { return vec3::dot(*this, *this); }

		// dot product.
		static double dot(const vec3& a, const vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
		double dot(const vec3& b) { return x*b.x + y*b.y + z*b.z; }

		static double distance(const vec3& a, const vec3& b) { return length(a - b); }
		static vec3 normalize(const vec3& a) { return (a / vec3::length(a)); }
		vec3 normalize() { return *this / this->length(); }

		// cross product.
		static vec3 cross(const vec3& a, const vec3& b) { return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
		vec3 cross(const vec3& b) { return vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }

		//
		// Rotate the vector 'v' around the 'axis' for 'theta' degrees.
		// This is basically Rodrigues' rotation formula.
		//
		static vec3 rotate(const vec3& v, const double theta, const vec3& axis) {
			vec3 k = vec3::normalize(axis); // normalize for good measure.
			return v * cos(theta) + vec3::cross(k, v)* sin(theta) + (k * vec3::dot(k, v)) * (1.0f - cos(theta));
		}
		static void nearestSqrDisInTriangle(vec3& p, vec3& a, vec3& b, vec3& c) {
			double q1 = sqrLength(p, a);
			double q2 = sqrLength(p, b);
			double q3 = sqrLength(p, c);
			vec3 ab = b - a;
			vec3 ac = c - a;
			vec3 ap = p - a;
			vec3 a_norm = ab.cross(ac);
		}
	};
	class Tri3 {
	public:
		vec3 vertices[3];
		Tri3() {}
		Tri3(vec3& v1, vec3& v2, vec3& v3) {
			vertices[0] = v1;
			vertices[1] = v2;
			vertices[2] = v3;
		}
		double area() {
			return (vertices[0] - vertices[1]).cross(vertices[0] - vertices[2]).length()*0.5;
		}
		vec3 center() { return (vertices[0] + vertices[1] + vertices[2]) / 3; }
		vec3 normalOrigin() { return (vertices[1] - vertices[0]).cross(vertices[2] - vertices[1]); }
		vec3 normal() { return normalOrigin().normalize(); }
		//void split(double thresArea) {
		double deltaX() {
			double min = vertices[0].x;
			double max = vertices[0].x;
			for (int i = 1; i < 3; i++) {
				if (vertices[i].x < min)min = vertices[i].x;
				if (vertices[i].x > max)max = vertices[i].x;
			}
			return max - min;
		}
		//}
	};

};
#endif //VEC3_H