#ifndef __VEC_H__
#define __VEC_H__
#include <thrust/extrema.h>
template <typename T>
struct vec3
{
	union {
		struct { T x, y, z; };
		struct { T r, g, b; };
		T raw[3];
	};
	__host__ __device__ T& operator [] (int i) { return raw[i]; }
	__host__ __device__ const T& operator [] (int i) const { return raw[i]; }
	__host__ __device__ vec3() :x(0),y(0),z(0){}
	__host__ __device__ vec3(T a,T b, T c):x(a),y(b),z(c){}
	__host__ __device__ vec3(T a) : x(a), y(a), z(a) {}
	__host__ __device__ inline vec3<T> operator + (const vec3<T> &temp) { return vec3<T>(temp.x + x, temp.y + y, temp.z + z); }
	__host__ __device__ inline vec3<T> operator + (const T& temp) { return vec3<T>(temp + x, temp + y, temp + z); }
	__host__ __device__ inline void operator += (const vec3<T>& temp) { x += temp.x; y += temp.y; z += temp.z; }

	__host__ __device__ inline vec3<T> operator - (const vec3<T> &temp) { return vec3<T>(x - temp.x, y - temp.y, z - temp.z); }
	__host__ __device__ inline void operator -= (const vec3<T>& temp) { x -= temp.x; y -= temp.y; z -= temp.z; }

	__host__ __device__ inline vec3<T> operator * (const vec3<T>& temp) { return vec3<T>(x * temp.x, y * temp.y, z * temp.z); }
	__host__ __device__ inline vec3<T> operator * (const T& temp) { return vec3<T>(x * temp, y * temp, z * temp); }
	__host__ __device__ inline void operator *= (const vec3<T>& temp) { x *= temp.x; y *= temp.y; z *= temp.z; }

	__host__ __device__ inline vec3<T> operator / (const vec3<T>& temp) { return vec3<T>(x / temp.x, y / temp.y, z / temp.z); }
	__host__ __device__ inline vec3<T> operator / (const T& temp) { return vec3<T>(x / temp, y / temp, z / temp); }
	__host__ __device__ inline void operator /= (const vec3<T>& temp) { x /= temp.x; y /= temp.y; z /= temp.z; }
	__host__ __device__ inline void operator /= (const T& temp) { x /= temp; y /= temp; z /= temp; }

	__host__ __device__ inline vec3<T> operator ^ (const vec3<T>& temp) { return vec3<T>(y * temp.z - z * temp.y, z * temp.x - x * temp.z, x * temp.y - y * temp.x); }

	__host__ __device__ inline T Length() { return sqrt(x * x + y * y + z * z); }
	

	__host__ __device__ inline T dot(const vec3<T>& temp) { return x * temp.x + y * temp.y + z * temp.z; }

};
template <typename T>
__host__ __device__ inline vec3<T> operator * (T a, vec3<T>& temp) { return temp * a; }
template <typename T>
__host__ __device__ inline vec3<T> operator / (T a, vec3<T>& temp) { return vec3f(a / temp.x, a / temp.y, a / temp.z); }
template <typename T>
__host__ __device__ inline vec3<T> Min(vec3<T>& temp, vec3<T>& temp_) {return vec3f(fminf(temp.x, temp_.x), fminf(temp.y, temp_.y), fminf(temp.z, temp_.z));}
template <typename T>
__host__ __device__ inline vec3<T> Max(vec3<T>& temp, vec3<T>& temp_) { return vec3f(fmaxf(temp.x, temp_.x), fmaxf(temp.y, temp_.y), fmaxf(temp.z, temp_.z)); }

//定义别名
typedef vec3<int> vec3i;
typedef vec3<float> vec3f;

__host__ __device__ inline vec3f normalize(vec3f v) { return v/v.Length(); }
__host__ __device__ inline float dot(vec3f v,vec3f u) { return u.dot(v); }


template <typename T>
struct vec2
{
	union {
		struct { T x, y; };
		T raw[2];
	};
	__host__ __device__ vec2() :x(0), y(0) {}
	__host__ __device__ vec2(T a, T b) : x(a), y(b){}
	__host__ __device__ inline vec2<T> operator + (const vec2<T>& temp) { return vec2<T>(temp.x + x, temp.y + y); }
	__host__ __device__ inline vec2<T> operator + (const T& temp) { return vec2<T>(temp + x, temp + y); }
	__host__ __device__ inline vec2<T> operator - (const vec2<T>& temp) { return vec2<T>(x - temp.x, y - temp.y); }

	__host__ __device__ inline vec2<T> operator * (const vec2<T>& temp) { return vec2<T>(x * temp.x, y * temp.y); }
	__host__ __device__ inline vec2<T> operator * (const T& temp) { return vec2<T>(x * temp, y * temp); }

	__host__ __device__ inline vec2<T> operator / (const vec2<T>& temp) { return vec2<T>(x / temp.x, y / temp.y); }
	__host__ __device__ inline vec2<T> operator / (const T& temp) { return vec2<T>(x / temp, y / temp); }

	__host__ __device__ inline T Length() { return sqrt(x * x + y * y ); }
	__host__ __device__ inline T dot(const vec2<T>& temp) { return x * temp.x + y * temp.y ; }

};
template <typename T>
__host__ __device__ inline vec2<T> operator * (T a, vec2<T>& temp) { return temp * a; }
//定义别名
typedef vec2<int> vec2i;
typedef vec2<float> vec2f;


template <typename T>
struct vec4
{
	union {
		struct { T x, y, z ,w; };
		struct { T r, g, b ,a; };
		T raw[4];
	};
};
typedef vec4<int> vec4i;
typedef vec4<float> vec4f;

template <typename T>
__host__ __device__ inline T absolute_Value(T a) {
	if (a > 0) return a;
	else
	{
		return -a;
	}
}

template <typename T>
__host__ __device__ inline void SWAP(T& a, T& b) {
	T temp = b;
	b = a;
	a = temp;
}


#endif