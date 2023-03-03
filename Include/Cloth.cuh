#ifndef CLOTH_CUH
#define CLOTH_CUH
#include"origin.h"
#include"Geometry.cuh"
#include<string>


class Cloth {
public :
	Cloth(vec3f* v, vec3f* N,int v_num, vec3i* i, int i_Num);
	Cloth(float* v, vec3f* N, int v_num, vec3i* i, int i_Num);
	~Cloth();
	void draw(unsigned int shader);
	void genShadow(unsigned int shader);
	vec3f* vertics;
	vec3i* faces;
	vec3f* normals;
	int vertics_Num;
	int faces_Num;
	unsigned int vertices_VBO;
	unsigned int normals_VBO;
	unsigned int VAO;
	unsigned int EBO;
};


#endif // !CLOTH_CUH
