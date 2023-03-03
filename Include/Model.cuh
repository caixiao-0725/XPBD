#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "origin.h"
//ģ����	
using namespace std;
class Model {
public:
	Model();
	Model(const char* filename);//����.obj�ļ�·������ģ��
	~Model();
	int nverts();//����ģ�Ͷ�������
	int nfaces();//����ģ����Ƭ����
	vec3f vert(int i);//���ص�i������
	vec3i face(int idx);//���ص�idx����Ƭ
	vector<vec3f> verts_;//���㼯��ÿ�����㶼����ά����
	vector<vec3i> faces_;//��Ƭ��
};

#endif //__MODEL_H__