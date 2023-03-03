#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH
#include"origin.h"

struct Triangle {
	vec3f vertex[3];
};

struct Sphere {
	vec3f position;
	float radius;

};

#endif // !GEOMETRY_CUH
