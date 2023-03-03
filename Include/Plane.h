#pragma once
#include"origin.h"
#include<vector>
#include <glad/glad.h>
class Plane {
public:
	Plane(vec3f a, vec3f b, vec3f c, vec3f d, vec3f color);
	std::vector<vec3f> Verts;
	unsigned int VAO;
	unsigned int EBO;
	unsigned int VBO;
	unsigned int normals_VBO;
	void draw(unsigned int shader);
	void genShadow(unsigned int shader);
};

Plane::Plane(vec3f a, vec3f b, vec3f c, vec3f d,vec3f color) {
	Verts.push_back(a);
	Verts.push_back(b);
	Verts.push_back(c);
	Verts.push_back(d);
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	glGenBuffers(1, &normals_VBO);
	// 1. 绑定VAO
	glBindVertexArray(VAO);
	// 2. 把顶点数组复制到缓冲中供OpenGL使用

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f) * 4, Verts.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	std::vector<vec3f> colors(4);
	for (int i = 0; i < 4; i++) {
		colors[i] = color;
	}

	glBindBuffer(GL_ARRAY_BUFFER, normals_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f) * 4, colors.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	int faces[6] = { 0,1,2,1,2,3 };

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 6, faces, GL_STATIC_DRAW);
	// 3. 设置顶点属性指针
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	
}


void Plane::draw(unsigned int shader) {
	glUseProgram(shader);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Plane::genShadow(unsigned int shader) {
	glUseProgram(shader);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}