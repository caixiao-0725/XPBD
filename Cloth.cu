#include"Cloth.cuh"
#include <glad/glad.h>



Cloth::Cloth(vec3f* v, vec3f* N, int v_num, vec3i* i, int i_Num):vertics(v),vertics_Num(v_num),faces(i),faces_Num(i_Num),normals(N) {
	
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &vertices_VBO);
	glGenBuffers(1, &normals_VBO);
	glGenBuffers(1, &EBO);
	// 1. 绑定VAO
	glBindVertexArray(VAO);
	// 2. 把顶点数组复制到缓冲中供OpenGL使用
	
	glBindBuffer(GL_ARRAY_BUFFER, vertices_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f) * v_num, vertics, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, normals_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f) * v_num, normals, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vec3i)* faces_Num, faces, GL_STATIC_DRAW);
	// 3. 设置顶点属性指针
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}

Cloth::Cloth(float* v, vec3f* n, int v_num, vec3i* i, int i_Num) :vertics_Num(v_num), faces(i), faces_Num(i_Num), normals(n) {

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &vertices_VBO);
	glGenBuffers(1, &normals_VBO);
	glGenBuffers(1, &EBO);
	// 1. 绑定VAO
	glBindVertexArray(VAO);
	// 2. 把顶点数组复制到缓冲中供OpenGL使用

	glBindBuffer(GL_ARRAY_BUFFER, vertices_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f) * v_num, v, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, normals_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f) * v_num, normals, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vec3i) * faces_Num, faces, GL_STATIC_DRAW);
	// 3. 设置顶点属性指针
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}

void Cloth::genShadow(unsigned int shader) {
	glUseProgram(shader);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glDrawElements(GL_TRIANGLES, faces_Num * 3, GL_UNSIGNED_INT, 0);
}

void Cloth::draw(unsigned int shader) {
	glUseProgram(shader);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glDrawElements(GL_TRIANGLES, faces_Num*3, GL_UNSIGNED_INT, 0);
}

Cloth::~Cloth() {

}