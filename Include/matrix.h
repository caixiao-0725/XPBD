#ifndef __MATRIX_H__
#define __MATRIX_H__
//和鄙人写的向量库互动
#include"origin.h"

#define SIGN(a)			((a)<0?-1:1)

template <typename T>
struct matrix4 {
	union {
		struct { T a11, a12, a13 ,a14,
				   a21, a22, a23, a24, 
			 	   a31, a32, a33, a34, 
				   a41, a42, a43, a44 ; };
		T raw[16];
	};
	__host__ __device__ matrix4() :a11(0), a12(0), a13(0), a14(0), a21(0), a22(0), a23(0), a24(0), a31(0), a32(0), a33(0), a34(0), a41(0), a42(0), a43(0), a44(0) {}
	__host__ __device__ matrix4(T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o,T p) : a11(a), a12(b), a13(c), a14(d), a21(e), a22(f), a23(g), a24(h), a31(i), a32(j), a33(k), a34(l), a41(m), a42(n), a43(o), a44(p) {}
	__host__ __device__ inline matrix4<T>& operator [] (const int i) { return raw[i]; };

	__host__ __device__ inline matrix4<T> operator + (const matrix4<T>& temp) {
		matrix4<T> ret;
		for(int i=0;i<16;i++){
			ret.raw[i] = this->raw[i] + temp.raw[i];
		}
		return ret;
	}

	__host__ __device__ inline matrix4<T> operator - (const matrix4<T>& temp) {
		matrix4<T> ret;
		for (int i = 0; i < 16; i++) {
			ret.raw[i] = this->raw[i] - temp.raw[i];
		}
		return ret;
	}

	__host__ __device__ inline matrix4<T> operator * (const matrix4<T>& temp) {
		matrix4<T> ret;
		for (int i = 0; i < 4; i++) 
			for (int j = 0; j < 4; j++) 
			{
				ret.raw[i*4+j] = this->raw[4 * i] * temp.raw[j] + 
								 this->raw[4 * i + 1] * temp.raw[j + 4]+
								 this->raw[4 * i + 2] * temp.raw[j + 8] + 
								 this->raw[4 * i + 3] * temp.raw[j + 12];
			}
		return ret;
	}
	__host__ __device__ inline matrix4<T> operator * (const T temp) {
		matrix4<T> ret;
		for (int i = 0; i < 16; i++) {
			ret.raw[i] = this->raw[i] * temp;
		}
		return ret;
	}
	__host__ __device__ inline void identity() { a11 = 1; a22 = 1; a33 = 1; a44 = 1; };
	__host__ __device__ inline matrix4<T> transpose() { 
		matrix4<T> ret;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
			{
				ret.raw[i * 4 + j] = this->raw[j * 4 + i];
			}
		return ret;
	}
	
	__host__ __device__ matrix4<T> Inverse() {
		int i, icol, irow, j, k, l, ll;
		icol = 0; irow = 0;
		float big, dum, pivinv;
		int indxc[4], indxr[4], ipiv[4];
		for (j = 0; j < 4; j++) ipiv[j] = 0;
		for (i = 0; i < 4; i++) {
			big = 0.0;
			for (j = 0; j < 4; j++) {
				if (ipiv[j] != 1) {
					for (k = 0; k < 4; k++) {
						if (ipiv[k] == 0) {
							if (absolute_Value(this->raw[4 * j + k]) >= big) {
								big = absolute_Value(this->raw[4 * j + k]);
								irow = j;
								icol = k;
							}
						}
					}
				}
			}
			++(ipiv[icol]);
			if (irow != icol) {
				for (l = 0; l < 4; l++) SWAP(this->raw[irow*4+l], this->raw[icol*4+l]);
			}
			indxr[i] = irow;
			indxc[i] = icol;
			if (this->raw[icol*4+icol] == 0.0) break;
			pivinv = 1.0 / this->raw[icol*4+icol];
			this->raw[icol*4+icol] = 1.0;
			for (l = 0; l < 4; l++) this->raw[icol*4+l] *= pivinv;
			for (ll = 0; ll < 4; ll++)
				if (ll != icol) {
					dum = this->raw[ll*4+icol];
					this->raw[ll*4+icol] = 0.0;
					for (l = 0; l < 4; l++) this->raw[ll*4+l] -= this->raw[icol*4+l] * dum;
				}
		}
		for (l = 3; l >= 0; l--) {
			if (indxr[l] != indxc[l])
				for (k = 0; k < 4; k++)
					SWAP(this->raw[k*4+indxr[l]], this->raw[k*4+indxc[l]]);
		}
		return *this;
	}

};

typedef typename matrix4<float> matrix4f;

template <typename T>
__host__ __device__ inline float trace(const matrix4<T> temp) { return temp.a11 + temp.a22 + temp.a33 + temp.a44; };

template <typename T>
__host__ __device__ float det(matrix4<T> A) { 
	int n = 4;
	float ret = 1.0;
	for (int i = 0; i < n - 1; ++i)
	{//从上往下将矩阵转化为上三角矩阵
		int mark = i;
		for (int j = i + 1; j < n; ++j)
		{//查找当前列中最大的元素
			if (absolute_Value(A.raw[mark*4+i]) < absolute_Value(A.raw[j*4+i]))
			{
				mark = j;
			}
		}
		if (mark != i)
		{//如果最大元素不是当前元素
			for (int j = i; j < n; ++j) {//对两行元素进行交换
				SWAP(A.raw[i*4+j], A.raw[mark*4+j]);
				
			}
			ret *= -1.0f;
		}
		for (int j = i + 1; j < n; ++j)
		{//将后面行的第i列元素全部消去
			float tmp = A.raw[j*4+i] / A.raw[i*4+i]; //避免重复计算除数
			for (int k = i; k < n; ++k)
			{//A矩阵第i列前面都是０,不需要操作
				A.raw[j*4+k] -= A.raw[i*4+k] * tmp;
			}
		}
	}
	
	for (int i = 0; i < n; ++i)
	{
		ret *= A.raw[i*4+i];
	}
	return ret;
}

template <typename T>
__host__ __device__ inline matrix4<T> operator * (const float temp, matrix4<T> a) { return a * temp; };
__host__ __device__ inline matrix4f BuildEdge(vec3f o, vec3f a, vec3f b, vec3f c) {
	matrix4f ret;
	vec3f ao = a - o;
	vec3f bo = b - o;
	vec3f co = c - o;
	for (int i = 0; i < 3; i++) {
		ret.raw[4 * i] = ao.raw[i];
		ret.raw[4 * i + 1] = bo.raw[i];
		ret.raw[4 * i + 2] = co.raw[i];
	}
	ret.a44 = 1;
	return ret;

}
/*
template <typename T>
__host__ __device__ inline matrix4<T> BuildEdge(vec3f a, vec3f b, vec3f c) {
	matrix4<T> ret;
	for (int i = 0; i < 3; i++) {
		ret.raw[4 * i] = a.raw[i];
		ret.raw[4 * i + 1] = b.raw[i];
		ret.raw[4 * i + 2] = c.raw[i];
	}
	ret.a44 = 1;
	return ret;

}
*/

template <typename T>
struct matrix3 {
	union {
		struct {
			T   a11, a12, a13,
				a21, a22, a23,
				a31, a32, a33;
		};
		T raw[9];
	};
	__host__ __device__ matrix3() :a11(0), a12(0), a13(0), a21(0), a22(0), a23(0),  a31(0), a32(0), a33(0){}
	__host__ __device__ matrix3(T a, T b, T c, T d, T e, T f, T g, T h, T i) : a11(a), a12(b), a13(c),  a21(d), a22(e), a23(f), a31(g), a32(h), a33(i) {}
	__host__ __device__ inline matrix3<T>& operator [] (const int i) { return raw[i]; };

	__host__ __device__ inline matrix3<T> operator + (const matrix3<T>& temp) {
		matrix3<T> ret;
		for (int i = 0; i < 9; i++) {
			ret.raw[i] = this->raw[i] + temp.raw[i];
		}
		return ret;
	}

	__host__ __device__ inline matrix3<T> operator - (const matrix3<T>& temp) {
		matrix3<T> ret;
		for (int i = 0; i < 9; i++) {
			ret.raw[i] = this->raw[i] - temp.raw[i];
		}
		return ret;
	}

	__host__ __device__ inline matrix3<T> operator * (const matrix3<T>& temp) {
		matrix3<T> ret;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				ret.raw[i * 3 + j] = this->raw[3 * i] * temp.raw[j] +
					this->raw[3 * i + 1] * temp.raw[j + 3] +
					this->raw[3 * i + 2] * temp.raw[j + 6];
			}
		return ret;
	}
	__host__ __device__ inline matrix3<T> operator * (const T temp) {
		matrix3<T> ret;
		for (int i = 0; i < 9; i++) {
			ret.raw[i] = this->raw[i] * temp;
		}
		return ret;
	}
	__host__ __device__ inline void identity() { a11 = 1; a22 = 1; a33 = 1; };
	__host__ __device__ inline matrix3<T> transpose() {
		matrix3<T> ret;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				ret.raw[i * 3 + j] = this->raw[j * 3 + i];
			}
		return ret;
	}

	__host__ __device__ matrix3<T> Inverse() {
		int i, icol, irow, j, k, l, ll;
		icol = 0; irow = 0;
		float big, dum, pivinv;
		int indxc[3], indxr[3], ipiv[3];
		for (j = 0; j < 3; j++) ipiv[j] = 0;
		for (i = 0; i < 3; i++) {
			big = 0.0;
			for (j = 0; j < 3; j++) {
				if (ipiv[j] != 1) {
					for (k = 0; k < 3; k++) {
						if (ipiv[k] == 0) {
							if (absolute_Value(this->raw[3 * j + k]) >= big) {
								big = absolute_Value(this->raw[3 * j + k]);
								irow = j;
								icol = k;
							}
						}
					}
				}
			}
			++(ipiv[icol]);
			if (irow != icol) {
				for (l = 0; l < 3; l++) SWAP(this->raw[irow * 3 + l], this->raw[icol * 3 + l]);
			}
			indxr[i] = irow;
			indxc[i] = icol;
			if (this->raw[icol * 3 + icol] == 0.0) break;
			pivinv = 1.0 / this->raw[icol * 3 + icol];
			this->raw[icol * 3 + icol] = 1.0;
			for (l = 0; l < 3; l++) this->raw[icol * 3 + l] *= pivinv;
			for (ll = 0; ll < 3; ll++)
				if (ll != icol) {
					dum = this->raw[ll * 3 + icol];
					this->raw[ll * 3 + icol] = 0.0;
					for (l = 0; l < 3; l++) this->raw[ll * 3 + l] -= this->raw[icol * 3 + l] * dum;
				}
		}
		for (l = 2; l >= 0; l--) {
			if (indxr[l] != indxc[l])
				for (k = 0; k < 3; k++)
					SWAP(this->raw[k * 3 + indxr[l]], this->raw[k * 3 + indxc[l]]);
		}
		return *this;
	}

};

typedef typename matrix3<float> matrix3f;

template <typename T>
__host__ __device__ inline float trace(const matrix3<T> temp) { return temp.a11 + temp.a22 + temp.a33; };

template <typename T>
__host__ __device__ float det(matrix3<T> A) {
	int n = 3;
	float ret = 1.0;
	for (int i = 0; i < n - 1; ++i)
	{//从上往下将矩阵转化为上三角矩阵
		int mark = i;
		for (int j = i + 1; j < n; ++j)
		{//查找当前列中最大的元素
			if (absolute_Value(A.raw[mark * 3 + i]) < absolute_Value(A.raw[j * 3 + i]))
			{
				mark = j;
			}
		}
		if (mark != i)
		{//如果最大元素不是当前元素
			for (int j = i; j < n; ++j) {//对两行元素进行交换
				SWAP(A.raw[i * 3 + j], A.raw[mark * 3 + j]);

			}
			ret *= -1.0f;
		}
		for (int j = i + 1; j < n; ++j)
		{//将后面行的第i列元素全部消去
			float tmp = A.raw[j * 3 + i] / A.raw[i * 3 + i]; //避免重复计算除数
			for (int k = i; k < n; ++k)
			{//A矩阵第i列前面都是０,不需要操作
				A.raw[j * 3 + k] -= A.raw[i * 3 + k] * tmp;
			}
		}
	}

	for (int i = 0; i < n; ++i)
	{
		ret *= A.raw[i * 3 + i];
	}
	return ret;
}

template <typename T>
__host__ __device__ inline matrix3<T> operator * (const float temp, matrix3<T> a) { return a * temp; };
__host__ __device__ inline matrix3f BuildEdgeMat3(vec3f o, vec3f a, vec3f b, vec3f c) {
	matrix3f ret;
	vec3f ao = a - o;
	vec3f bo = b - o;
	vec3f co = c - o;
	for (int i = 0; i <3; i++) {
		ret.raw[3 * i] = ao.raw[i];
		ret.raw[3 * i + 1] = bo.raw[i];
		ret.raw[3 * i + 2] = co.raw[i];
	}
	return ret;

}

template <typename T>
struct matrix12 {
	T raw[12 * 12];
	__host__ __device__ matrix12() {
		for (int i = 0; i < 144; i++) raw[i] = 0;
	}
	__host__ __device__ matrix12<T> Inverse() {
		int i, icol, irow, j, k, l, ll;
		icol = 0; irow = 0;
		float big, dum, pivinv;
		int indxc[12], indxr[12], ipiv[12];
		for (j = 0; j < 12; j++) ipiv[j] = 0;
		for (i = 0; i < 12; i++) {
			big = 0.0;
			for (j = 0; j < 12; j++) {
				if (ipiv[j] != 1) {
					for (k = 0; k < 12; k++) {
						if (ipiv[k] == 0) {
							if (absolute_Value(this->raw[12 * j + k]) >= big) {
								big = absolute_Value(this->raw[12 * j + k]);
								irow = j;
								icol = k;
							}
						}
					}
				}
			}
			++(ipiv[icol]);
			if (irow != icol) {
				for (l = 0; l < 12; l++) SWAP(this->raw[irow * 12 + l], this->raw[icol * 12 + l]);
			}
			indxr[i] = irow;
			indxc[i] = icol;
			if (this->raw[icol * 12 + icol] == 0.0) break;
			pivinv = 1.0 / this->raw[icol * 12 + icol];
			this->raw[icol * 12 + icol] = 1.0;
			for (l = 0; l < 12; l++) this->raw[icol * 12 + l] *= pivinv;
			for (ll = 0; ll < 12; ll++)
				if (ll != icol) {
					dum = this->raw[ll * 12 + icol];
					this->raw[ll * 12 + icol] = 0.0;
					for (l = 0; l < 12; l++) this->raw[ll * 12 + l] -= this->raw[icol * 12 + l] * dum;
				}
		}
		for (l = 11; l >= 0; l--) {
			if (indxr[l] != indxc[l])
				for (k = 0; k < 12; k++)
					SWAP(this->raw[k * 12 + indxr[l]], this->raw[k * 12 + indxc[l]]);
		}
		return *this;
	}
	__host__ __device__ matrix12<T> operator * (const T temp) {
		matrix12<T> ret;
		for (int i = 0; i < 144; i++) {
			ret.raw[i] = this->raw[i] * temp;
		}
		return ret;
	}
	__host__ __device__ inline void identity() { 
		for (int i = 0; i < 12; i++) {
			raw[i * 12 + i] = 1;
		}
	};


	__host__ __device__ inline matrix12<T> operator + (const matrix12<T>& temp) {
		matrix12<T> ret;
		for (int i = 0; i < 144; i++) {
			ret.raw[i] = this->raw[i] + temp.raw[i];
		}
		return ret;
	}

	__host__ __device__  matrix12<T> operator - (const matrix12<T>& temp) {
		matrix12<T> ret;
		for (int i = 0; i < 144; i++) {
			ret.raw[i] = this->raw[i] - temp.raw[i];
		}
		return ret;
	}
};

typedef typename matrix12<float> matrix12f;

struct vec12f {
	float raw[12];
	__host__ __device__ vec12f() {
		for (int i = 0; i < 12; i++) {
			raw[i] = 0;
		}
	}
};

__host__ __device__ inline vec12f operator * (const matrix12f& mat, const vec12f& vec) {
	vec12f ret;
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 12; j++) {
			ret.raw[i] += mat.raw[12 * i + j] * vec.raw[j];
		}
	}
	return ret;
}

template <class TYPE> 
__device__  __host__ TYPE pythag(TYPE a, TYPE b)
{
	TYPE at = fabs(a), bt = fabs(b), ct, result;
	if (at > bt) { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}


template <class TYPE>
__device__  __host__ void SVD3(matrix3f &u, TYPE w[3], matrix3f &v)
{

	TYPE	anorm, c, f, g, h, s, scale;
	TYPE	x, y, z;
	TYPE	rv1[3];
	g = scale = anorm = 0.0; //Householder reduction to bidiagonal form.

	for (int i = 0; i < 3; i++)
	{
		int l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < 3)
		{
			for (int k = i; k < 3; k++) scale += fabsf(u.raw[k*3+i]);
			if (scale != 0)
			{
				for (int k = i; k < 3; k++)
				{
					u.raw[k*3+i] /= scale;
					s += u.raw[k*3+i] * u.raw[k*3+i];
				}
				f = u.raw[i*3+i];
				g = -sqrtf(s) * SIGN(f);
				h = f * g - s;
				u.raw[i*3+i] = f - g;
				for (int j = l; j < 3; j++)
				{
					s = 0;
					for (int k = i; k < 3; k++)	s += u.raw[k*3+i] * u.raw[k*3+j];
					f = s / h;
					for (int k = i; k < 3; k++)	u.raw[k*3+j] += f * u.raw[k*3+i];
				}
				for (int k = i; k < 3; k++)		u.raw[k*3+i] *= scale;
			}
		}
		w[i] = scale * g;

		g = s = scale = 0.0;
		if (i <= 2 && i != 2)
		{
			for (int k = l; k < 3; k++)	scale += fabsf(u.raw[i*3+k]);
			if (scale != 0)
			{
				for (int k = l; k < 3; k++)
				{
					u.raw[i*3+k] /= scale;
					s += u.raw[i*3+k] * u.raw[i*3+k];
				}
				f = u.raw[i*3+l];
				g = -sqrtf(s) * SIGN(f);
				h = f * g - s;
				u.raw[i*3+l] = f - g;
				for (int k = l; k < 3; k++) rv1[k] = u.raw[i*3+k] / h;
				for (int j = l; j < 3; j++)
				{
					s = 0;
					for (int k = l; k < 3; k++)	s += u.raw[j*3+k] * u.raw[i*3+k];
					for (int k = l; k < 3; k++)	u.raw[j*3+k] += s * rv1[k];
				}
				for (int k = l; k < 3; k++) u.raw[i*3+k] *= scale;
			}
		}
		anorm = max(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	for (int i = 2, l; i >= 0; i--) //Accumulation of right-hand transformations.
	{
		if (i < 2)
		{
			if (g != 0)
			{
				for (int j = l; j < 3; j++) //Double division to avoid possible under
					v.raw[j*3+i] = (u.raw[i*3+j] / u.raw[i*3+l]) / g;
				for (int j = l; j < 3; j++)
				{
					s = 0;
					for (int k = l; k < 3; k++)	s += u.raw[i*3+k] * v.raw[k*3+j];
					for (int k = l; k < 3; k++)	v.raw[k*3+j] += s * v.raw[k*3+i];
				}
			}
			for (int j = l; j < 3; j++)	v.raw[i*3+j] = v.raw[j*3+i] = 0.0;
		}
		v.raw[i*3+i] = 1.0;
		g = rv1[i];
		l = i;
	}

	for (int i = 2; i >= 0; i--) //Accumulation of left-hand transformations.
	{
		int l = i + 1;
		g = w[i];
		for (int j = l; j < 3; j++) u.raw[i*3+j] = 0;
		if (g != 0)
		{
			g = 1 / g;
			for (int j = l; j < 3; j++)
			{
				s = 0;
				for (int k = l; k < 3; k++)	s += u.raw[k*3+i] * u.raw[k*3+j];
				f = (s / u.raw[i*3+i]) * g;
				for (int k = i; k < 3; k++)	u.raw[k*3+j] += f * u.raw[k*3+i];
			}
			for (int j = i; j < 3; j++)		u.raw[j*3+i] *= g;
		}
		else for (int j = i; j < 3; j++)		u.raw[j*3+i] = 0.0;
		u.raw[i*3+i]++;
	}

	for (int k = 2; k >= 0; k--)				//Diagonalization of the bidiagonal form: Loop over
	{
		for (int its = 0; its < 30; its++)	//singular values, and over allowed iterations.
		{
			bool flag = true;
			int  l;
			int	 nm;
			for (l = k; l >= 0; l--)			//Test for splitting.
			{
				nm = l - 1;
				if ((TYPE)(fabs(rv1[l]) + anorm) == anorm)
				{
					flag = false;
					break;
				}
				if ((TYPE)(fabs(w[nm]) + anorm) == anorm)	break;
			}
			if (flag)
			{
				c = 0.0; //Cancellation of rv1[l], if l > 0.
				s = 1.0;
				for (int i = l; i < k + 1; i++)
				{
					f = s * rv1[i];
					rv1[i] = c * rv1[i];
					if ((TYPE)(fabs(f) + anorm) == anorm) break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;
					for (int j = 0; j < 3; j++)
					{
						y = u.raw[j*3+nm];
						z = u.raw[j*3+i];
						u.raw[j*3+nm] = y * c + z * s;
						u.raw[j*3+i] = z * c - y * s;
					}
				}
			}
			z = w[k];
			if (l == k)		// Convergence.
			{
				if (z < 0.0)	// Singular value is made nonnegative.
				{
					w[k] = -z;
					for (int j = 0; j < 3; j++) v.raw[j*3+k] = -v.raw[j*3+k];
				}
				break;
			}
			if (its == 29) { printf("Error: no convergence in 30 svdcmp iterations"); getchar(); }
			x = w[l]; //Shift from bottom 2-by-2 minor.
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = pythag(f, (TYPE)1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + fabs(g) * SIGN(f))) - h)) / x;
			c = s = 1.0; //Next QR transformation:

			for (int j = l; j <= nm; j++)
			{
				int i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;
				for (int jj = 0; jj < 3; jj++)
				{
					x = v.raw[jj*3+j];
					z = v.raw[jj*3+i];
					v.raw[jj*3+j] = x * c + z * s;
					v.raw[jj*3+i] = z * c - x * s;
				}
				z = pythag(f, h);
				w[j] = z; //Rotation can be arbitrary if z D 0.
				if (z)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				for (int jj = 0; jj < 3; jj++)
				{
					y = u.raw[jj*3+j];
					z = u.raw[jj*3+i];
					u.raw[jj*3+j] = y * c + z * s;
					u.raw[jj*3+i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
}

__device__ void Get_Rotation(matrix3f& F, matrix3f& R)
{
	matrix3f C;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C.raw[i*3+j] += F.raw[k*3+i] * F.raw[k*3+j];

	matrix3f C2;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C2.raw[i*3+j] += C.raw[i*3+k] * C.raw[j*3+k];

	float det = F.raw[0*3+0] * F.raw[1*3+1] * F.raw[2*3+2] +
		F.raw[0*3+1] * F.raw[1*3+2] * F.raw[2*3+0] +
		F.raw[1*3+0] * F.raw[2*3+1] * F.raw[0*3+2] -
		F.raw[0*3+2] * F.raw[1*3+1] * F.raw[2*3+0] -
		F.raw[0*3+1] * F.raw[1*3+0] * F.raw[2*3+2] -
		F.raw[0*3+0] * F.raw[1*3+2] * F.raw[2*3+1];

	float I_c = C.raw[0*3+0] + C.raw[1*3+1] + C.raw[2*3+2];
	float I_c2 = I_c * I_c;
	float II_c = 0.5 * (I_c2 - C2.raw[0*3+0] - C2.raw[1*3+1] - C2.raw[2*3+2]);
	float III_c = det * det;
	float k = I_c2 - 3 * II_c;

	matrix3f inv_U;
	if (k < 1e-10f)
	{
		float inv_lambda = 1 / sqrt(I_c / 3);
		inv_U.raw[0*3+0] = inv_lambda;
		inv_U.raw[1*3+1] = inv_lambda;
		inv_U.raw[2*3+2] = inv_lambda;
	}
	else
	{
		float l = I_c * (I_c * I_c - 4.5 * II_c) + 13.5 * III_c;
		float k_root = sqrt(k);
		float value = l / (k * k_root);
		if (value < -1.0) value = -1.0;
		if (value > 1.0) value = 1.0;
		float phi = acos(value);
		float lambda2 = (I_c + 2 * k_root * cos(phi / 3)) / 3.0;
		float lambda = sqrt(lambda2);

		float III_u = sqrt(III_c);
		if (det < 0)   III_u = -III_u;
		float I_u = lambda + sqrt(-lambda2 + I_c + 2 * III_u / lambda);
		float II_u = (I_u * I_u - I_c) * 0.5;

		matrix3f U;
		float inv_rate, factor;

		inv_rate = 1 / (I_u * II_u - III_u);
		factor = I_u * III_u * inv_rate;

		U.raw[0*3+0] = factor;
		U.raw[1*3+1] = factor;
		U.raw[2*3+2] = factor;

		factor = (I_u * I_u - II_u) * inv_rate;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				U.raw[i*3+j] += factor * C.raw[i*3+j] - inv_rate * C2.raw[i*3+j];

		inv_rate = 1 / III_u;
		factor = II_u * inv_rate;
		inv_U.raw[0*3+0] = factor;
		inv_U.raw[1*3+1] = factor;
		inv_U.raw[2*3+2] = factor;

		factor = -I_u * inv_rate;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				inv_U.raw[i*3+j] += factor * U.raw[i*3+j] + inv_rate * C.raw[i*3+j];
	}

	memset(&R.raw, 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				R.raw[i*3+j] += F.raw[i*3+k] * inv_U.raw[k*3+j];
}

#endif