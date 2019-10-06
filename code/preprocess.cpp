#include <opencv2/opencv.hpp>
#include <ctime>
#include <vector>
#include <cstdio>
#include <set>
#include <queue>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <OpenEXRConfig.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include "needfunction.h"
#include "juzhen.h"
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;
using namespace OPENEXR_IMF_NAMESPACE;
#define Height 568
#define Width 568

struct position
{
	int h, w;
};
struct chunk
{
	int index;
	position pos;
	position possource;
};
int ifmanage[Height][Width];
int idmat[Height][Width];
//生成0——number-1的随机数
int random(int number)
{
	srand(time(NULL));
	return rand() % number;
}
//合成纹理大小
int num = 1;
//const int Height = 2024;
//const int Width = 2024;
//重叠块大小
const int OverLapSize = 8;
//选取的块的大小
const int TileSize = 64;
//源图默认尺寸
const int DefaultSize = 200;
using point = pair<int, int>;
const int w = int((Width - OverLapSize) / (TileSize - OverLapSize));
const int h = int((Height - OverLapSize) / (TileSize - OverLapSize));
const int chuncknum = w*h;
chunk *chun=NULL;
typedef Array2D<Rgba> pixels;
pixels sourceEXR_normalmap;
pixels dest_normalmap;

const float x1 = 0.5f;
const float x2 = 0.5f;
float sigma_p1;
float sigma_p2;
const float sigma_r1 = 0.001f;
const float sigma_r2 = 0.001f;
const float eta = 1.55f;

float delta = 0.5f;

const float eps = 1e-4f;

int width, height;
float A_inv[16][16] = { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 },
{ -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0 },
{ 9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 },
{ -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1 },
{ 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 },
{ -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1 },
{ 4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 } };


//重载<
bool operator<(const point v1, const point v2)
{
	if (v1.first == v2.first)
		return v1.second < v2.second;
	return v1.first < v2.first;
}

void absdiff(const pixels &a, const pixels &b, pixels &c){
	int ah = a.height(), bh = b.height();
	int aw = a.width(), bw = b.width();
	if (ah == bh&&aw == bw)
	{
		c.resizeErase(ah, aw);
		for (int i = 0; i < ah; ++i)
		{
			for (int j = 0; j < aw; ++j)
			{
				c[i][j].r = abs(a[i][j].r - b[i][j].r);
				c[i][j].g = abs(a[i][j].g - b[i][j].g);
				c[i][j].b = abs(a[i][j].b - b[i][j].b);
			}
		}
	}
}

void sum(const pixels &a, Rgba &sum){
	int ah = a.height(), aw = a.width();
	sum.r = 0; sum.g = 0; sum.b = 0;
	for (int i = 0; i < ah; i++)
	{
		for (int j = 0; j < aw; j++)
		{
			sum.r += (a[i][j].r);
			sum.g += (a[i][j].g);
			sum.b += (a[i][j].b);
		}
	}
}
float dist(float x1, float y1, float x2, float y2) {
	return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

float clamp(float x, float a, float b) {
	if (x < a)
		return a;
	else if (x > b)
		return b;
	return x;
}

inline int mod(int x, int y) {
	return ((x % y) + y) % y;
}

inline float p(int x, int y) {
	return dest_normalmap[mod(x, width)][mod(y, height)].r;
}

inline float px(int x, int y) {
	return (dest_normalmap[mod(x + 1, width)][mod(y, height)].r - dest_normalmap[mod(x - 1, width)][mod(y, height)].r) / 2.0;
}

inline float py(int x, int y) {
	return (dest_normalmap[mod(x, width)][mod(y + 1, height)].r - dest_normalmap[mod(x, width)][mod(y - 1, height)].r) / 2.0;
}

inline float pxy(int x, int y) {
	return (p(x + 1, y + 1) - p(x + 1, y) - p(x, y + 1) + 2.0 * p(x, y) - p(x - 1, y) - p(x, y - 1) + p(x - 1, y - 1)) / 2.0;
}

inline float q(int x, int y) {
	return dest_normalmap[mod(x, width)][mod(y, height)].g;
}

inline float qx(int x, int y) {
	return (dest_normalmap[mod(x + 1, width)][mod(y, height)].g - dest_normalmap[mod(x - 1, width)][mod(y, height)].g) / 2.0;
}

inline float qy(int x, int y) {
	return (dest_normalmap[mod(x, width)][mod(y + 1, height)].g - dest_normalmap[mod(x, width)][mod(y - 1, height)].g) / 2.0;
}

inline float qxy(int x, int y) {
	return (q(x + 1, y + 1) - q(x + 1, y) - q(x, y + 1) + 2.0 * q(x, y) - q(x - 1, y) - q(x, y - 1) + q(x - 1, y - 1)) / 2.0;
}

void computeCoeff(float *alpha, const float *x) {
	memset(alpha, 0, sizeof(float) * 16);
	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 16; j++)
			alpha[i] += A_inv[i][j] * x[j];
}

Vector2f getNormal(float u, float v) {
	// Bicubic interpolation双线性三次插值
	float x = u * width;
	float y = v * height;
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = mod(x1, width);
	x2 = mod(x2, width);
	y1 = mod(y1, height);
	y2 = mod(y2, height);

	float a[16], b[16];
	float xp[16] = { p(x1, y1), p(x2, y1), p(x1, y2), p(x2, y2),
		px(x1, y1), px(x2, y1), px(x1, y2), px(x2, y2),
		py(x1, y1), py(x2, y1), py(x1, y2), py(x2, y2),
		pxy(x1, y1), pxy(x2, y1), pxy(x1, y2), pxy(x2, y2) };
	float xq[16] = { q(x1, y1), q(x2, y1), q(x1, y2), q(x2, y2),
		qx(x1, y1), qx(x2, y1), qx(x1, y2), qx(x2, y2),
		qy(x1, y1), qy(x2, y1), qy(x1, y2), qy(x2, y2),
		qxy(x1, y1), qxy(x2, y1), qxy(x1, y2), qxy(x2, y2) };

	computeCoeff(a, xp);
	computeCoeff(b, xq);

	float coeffA[4][4] = { { a[0], a[4], a[8], a[12] },
	{ a[1], a[5], a[9], a[13] },
	{ a[2], a[6], a[10], a[14] },
	{ a[3], a[7], a[11], a[15] } };
	float coeffB[4][4] = { { b[0], b[4], b[8], b[12] },
	{ b[1], b[5], b[9], b[13] },
	{ b[2], b[6], b[10], b[14] },
	{ b[3], b[7], b[11], b[15] } };

	float n1 = 0.0f, n2 = 0.0f;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			n1 += coeffA[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));
			n2 += coeffB[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));
		}


	return Vector2f(n1, n2);
}

Vector2f getNormalInt(int u1, int u2) {
	return Vector2f(p(u1, u2), q(u1, u2));
}

inline float getNx(float u, float v) {
	u /= width;
	v /= height;

	// Bicubic interpolation
	float x = u * width;
	float y = v * height;
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = mod(x1, width);
	x2 = mod(x2, width);
	y1 = mod(y1, height);
	y2 = mod(y2, height);

	float a[16], b[16];
	float xp[16] = { p(x1, y1), p(x2, y1), p(x1, y2), p(x2, y2),
		px(x1, y1), px(x2, y1), px(x1, y2), px(x2, y2),
		py(x1, y1), py(x2, y1), py(x1, y2), py(x2, y2),
		pxy(x1, y1), pxy(x2, y1), pxy(x1, y2), pxy(x2, y2) };
	float xq[16] = { q(x1, y1), q(x2, y1), q(x1, y2), q(x2, y2),
		qx(x1, y1), qx(x2, y1), qx(x1, y2), qx(x2, y2),
		qy(x1, y1), qy(x2, y1), qy(x1, y2), qy(x2, y2),
		qxy(x1, y1), qxy(x2, y1), qxy(x1, y2), qxy(x2, y2) };

	computeCoeff(a, xp);
	computeCoeff(b, xq);

	float coeffA[4][4] = { { a[0], a[4], a[8], a[12] },
	{ a[1], a[5], a[9], a[13] },
	{ a[2], a[6], a[10], a[14] },
	{ a[3], a[7], a[11], a[15] } };
	float coeffB[4][4] = { { b[0], b[4], b[8], b[12] },
	{ b[1], b[5], b[9], b[13] },
	{ b[2], b[6], b[10], b[14] },
	{ b[3], b[7], b[11], b[15] } };

	float n1 = 0.0f;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			n1 += coeffA[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));

	return n1;
}

inline float getNy(float u, float v) {
	u /= width;
	v /= height;

	// Bicubic interpolation
	float x = u * width;
	float y = v * height;
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = mod(x1, width);
	x2 = mod(x2, width);
	y1 = mod(y1, height);
	y2 = mod(y2, height);

	float a[16], b[16];
	float xp[16] = { p(x1, y1), p(x2, y1), p(x1, y2), p(x2, y2),
		px(x1, y1), px(x2, y1), px(x1, y2), px(x2, y2),
		py(x1, y1), py(x2, y1), py(x1, y2), py(x2, y2),
		pxy(x1, y1), pxy(x2, y1), pxy(x1, y2), pxy(x2, y2) };
	float xq[16] = { q(x1, y1), q(x2, y1), q(x1, y2), q(x2, y2),
		qx(x1, y1), qx(x2, y1), qx(x1, y2), qx(x2, y2),
		qy(x1, y1), qy(x2, y1), qy(x1, y2), qy(x2, y2),
		qxy(x1, y1), qxy(x2, y1), qxy(x1, y2), qxy(x2, y2) };

	computeCoeff(a, xp);
	computeCoeff(b, xq);

	float coeffA[4][4] = { { a[0], a[4], a[8], a[12] },
	{ a[1], a[5], a[9], a[13] },
	{ a[2], a[6], a[10], a[14] },
	{ a[3], a[7], a[11], a[15] } };
	float coeffB[4][4] = { { b[0], b[4], b[8], b[12] },
	{ b[1], b[5], b[9], b[13] },
	{ b[2], b[6], b[10], b[14] },
	{ b[3], b[7], b[11], b[15] } };

	float n2 = 0.0f;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			n2 += coeffB[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));

	return n2;
}

inline float getNxDx(float x, float y) {
	return (getNx(x + delta, y) - getNx(x - delta, y)) / (2.0f * delta);
}

inline float getNxDy(float x, float y) {
	return (getNx(x, y + delta) - getNx(x, y - delta)) / (2.0f * delta);
}

inline float getNxDxDx(float x, float y) {
	return (getNxDx(x + delta, y) - getNxDx(x - delta, y)) / (2.0f * delta);
}

inline float getNxDxDy(float x, float y) {
	return (getNxDy(x + delta, y) - getNxDy(x - delta, y)) / (2.0f * delta);
}

inline float getNxDyDx(float x, float y) {
	return (getNxDx(x, y + delta) - getNxDx(x, y - delta)) / (2.0f * delta);
}

inline float getNxDyDy(float x, float y) {
	return (getNxDy(x, y + delta) - getNxDy(x, y - delta)) / (2.0f * delta);
}

inline float getNyDx(float x, float y) {
	return (getNy(x + delta, y) - getNy(x - delta, y)) / (2.0f * delta);
}

inline float getNyDy(float x, float y) {
	return (getNy(x, y + delta) - getNy(x, y - delta)) / (2.0f * delta);
}

inline float getNyDxDx(float x, float y) {
	return (getNyDx(x + delta, y) - getNyDx(x - delta, y)) / (2.0f * delta);
}

inline float getNyDxDy(float x, float y) {
	return (getNyDy(x + delta, y) - getNyDy(x - delta, y)) / (2.0f * delta);
}

inline float getNyDyDx(float x, float y) {
	return (getNyDx(x, y + delta) - getNyDx(x, y - delta)) / (2.0f * delta);
}

inline float getNyDyDy(float x, float y) {
	return (getNyDy(x, y + delta) - getNyDy(x, y - delta)) / (2.0f * delta);
}

// Get normal
inline Vector2f getN(float x, float y) {
	return Vector2f(getNx(x, y), getNy(x, y));
}

// Get Jacobian matrix
inline Matrix2f getJ(float x, float y) {
	Matrix2f J;
	J.m[0][0] = getNxDx(x, y);
	J.m[0][1] = getNxDy(x, y);
	J.m[1][0] = getNyDx(x, y);
	J.m[1][1] = getNyDy(x, y);
	return J;
}

//******************************************************************************************************
inline float pa(int x, int y, pixels &son) {
	return son[mod(x, 9)][mod(y, 9)].r;
}

inline float pxa(int x, int y, pixels &son) {
	return (son[mod(x + 1, 9)][mod(y, 9)].r - son[mod(x - 1, 9)][mod(y,9)].r) / 2.0;
}

inline float pya(int x, int y, pixels &son) {
	return (son[mod(x, 9)][mod(y + 1, 9)].r - son[mod(x, 9)][mod(y - 1, 9)].r) / 2.0;
}

inline float pxya(int x, int y, pixels &son) {
	return (pa(x + 1, y + 1, son) - pa(x + 1, y, son) - pa(x, y + 1, son) + 2.0 * pa(x, y, son) - pa(x - 1, y, son) - pa(x, y - 1, son) + pa(x - 1, y - 1, son)) / 2.0;
}

inline float qa(int x, int y, pixels &son) {
	return son[mod(x, 9)][mod(y, 9)].g;
}

inline float qxa(int x, int y, pixels &son) {
	return (son[mod(x + 1,9)][mod(y, 9)].g - son[mod(x - 1, 9)][mod(y, 9)].g) / 2.0;
}

inline float qya(int x, int y, pixels &son) {
	return (son[mod(x, 9)][mod(y + 1, 9)].g - son[mod(x, 9)][mod(y - 1, 9)].g) / 2.0;
}

inline float qxya(int x, int y, pixels &son) {
	return (qa(x + 1, y + 1, son) - qa(x + 1, y, son) - qa(x, y + 1, son) + 2.0 * qa(x, y, son) - qa(x - 1, y, son) - qa(x, y - 1, son) + qa(x - 1, y - 1, son)) / 2.0;
}


Vector2f getNormala(float u, float v, pixels &son) {
	// Bicubic interpolation
	float x = u ;
	float y = v ;
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = mod(x1, width);
	x2 = mod(x2, width);
	y1 = mod(y1, height);
	y2 = mod(y2, height);

	float a[16], b[16];
	float xp[16] = { pa(x1, y1, son), pa(x2, y1, son), pa(x1, y2, son), pa(x2, y2, son),
		pxa(x1, y1, son), pxa(x2, y1, son), pxa(x1, y2, son), pxa(x2, y2, son),
		pya(x1, y1, son), pya(x2, y1, son), pya(x1, y2, son), pya(x2, y2, son),
		pxya(x1, y1, son), pxya(x2, y1, son), pxya(x1, y2, son), pxya(x2, y2, son) };
	float xq[16] = { qa(x1, y1, son), qa(x2, y1, son), qa(x1, y2, son), qa(x2, y2, son),
		qxa(x1, y1, son), qxa(x2, y1, son), qxa(x1, y2, son), qxa(x2, y2, son),
		qya(x1, y1, son), qya(x2, y1, son), qya(x1, y2, son), qya(x2, y2, son),
		qxya(x1, y1, son), qxya(x2, y1, son), qxya(x1, y2, son), qxya(x2, y2, son) };

	computeCoeff(a, xp);
	computeCoeff(b, xq);

	float coeffA[4][4] = { { a[0], a[4], a[8], a[12] },
	{ a[1], a[5], a[9], a[13] },
	{ a[2], a[6], a[10], a[14] },
	{ a[3], a[7], a[11], a[15] } };
	float coeffB[4][4] = { { b[0], b[4], b[8], b[12] },
	{ b[1], b[5], b[9], b[13] },
	{ b[2], b[6], b[10], b[14] },
	{ b[3], b[7], b[11], b[15] } };

	float n1 = 0.0f, n2 = 0.0f;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			n1 += coeffA[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));
			n2 += coeffB[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));
		}

	return Vector2f(n1, n2);
}

Vector2f getNormalInta(int u1, int u2, pixels &son) {
	return Vector2f(p(u1, u2), q(u1, u2));
}

inline float getNxa(float u, float v, pixels &son) {
	u /= width;
	v /= height;

	// Bicubic interpolation
	float x = u * width;
	float y = v * height;
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = mod(x1, width);
	x2 = mod(x2, width);
	y1 = mod(y1, height);
	y2 = mod(y2, height);

	float a[16], b[16];
	float xp[16] = { pa(x1, y1, son), pa(x2, y1, son), pa(x1, y2, son), pa(x2, y2, son),
		pxa(x1, y1, son), pxa(x2, y1, son), pxa(x1, y2, son), pxa(x2, y2, son),
		pya(x1, y1, son), pya(x2, y1, son), pya(x1, y2, son), pya(x2, y2, son),
		pxya(x1, y1, son), pxya(x2, y1, son), pxya(x1, y2, son), pxya(x2, y2, son) };
	float xq[16] = { qa(x1, y1, son), qa(x2, y1, son), qa(x1, y2, son), qa(x2, y2, son),
		qxa(x1, y1, son), qxa(x2, y1, son), qxa(x1, y2, son), qxa(x2, y2, son),
		qya(x1, y1, son), qya(x2, y1, son), qya(x1, y2, son), qya(x2, y2, son),
		qxya(x1, y1, son), qxya(x2, y1, son), qxya(x1, y2, son), qxya(x2, y2, son) };

	computeCoeff(a, xp);
	computeCoeff(b, xq);

	float coeffA[4][4] = { { a[0], a[4], a[8], a[12] },
	{ a[1], a[5], a[9], a[13] },
	{ a[2], a[6], a[10], a[14] },
	{ a[3], a[7], a[11], a[15] } };
	float coeffB[4][4] = { { b[0], b[4], b[8], b[12] },
	{ b[1], b[5], b[9], b[13] },
	{ b[2], b[6], b[10], b[14] },
	{ b[3], b[7], b[11], b[15] } };

	float n1 = 0.0f;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			n1 += coeffA[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));

	return n1;
}

inline float getNya(float u, float v, pixels &son) {
	u /= 9;
	v /= 9;

	// Bicubic interpolation
	float x = u * 9;
	float y = v * 9;
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = mod(x1, 9);
	x2 = mod(x2, 9);
	y1 = mod(y1, 9);
	y2 = mod(y2, 9);

	float a[16], b[16];
	float xp[16] = { pa(x1, y1, son), pa(x2, y1, son), pa(x1, y2, son), pa(x2, y2, son),
		pxa(x1, y1, son), pxa(x2, y1, son), pxa(x1, y2, son), pxa(x2, y2, son),
		pya(x1, y1, son), pya(x2, y1, son), pya(x1, y2, son), pya(x2, y2, son),
		pxya(x1, y1, son), pxya(x2, y1, son), pxya(x1, y2, son), pxya(x2, y2, son) };
	float xq[16] = { qa(x1, y1, son), qa(x2, y1, son), qa(x1, y2, son), qa(x2, y2, son),
		qxa(x1, y1, son), qxa(x2, y1, son), qxa(x1, y2, son), qxa(x2, y2, son),
		qya(x1, y1, son), qya(x2, y1, son), qya(x1, y2, son), qya(x2, y2, son),
		qxya(x1, y1, son), qxya(x2, y1, son), qxya(x1, y2, son), qxya(x2, y2, son) };

	computeCoeff(a, xp);
	computeCoeff(b, xq);

	float coeffA[4][4] = { { a[0], a[4], a[8], a[12] },
	{ a[1], a[5], a[9], a[13] },
	{ a[2], a[6], a[10], a[14] },
	{ a[3], a[7], a[11], a[15] } };
	float coeffB[4][4] = { { b[0], b[4], b[8], b[12] },
	{ b[1], b[5], b[9], b[13] },
	{ b[2], b[6], b[10], b[14] },
	{ b[3], b[7], b[11], b[15] } };

	float n2 = 0.0f;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			n2 += coeffB[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));

	return n2;
}

inline float getNxDxa(float x, float y,  pixels &son) {
	return (getNxa(x + delta, y, son) - getNxa(x - delta, y, son)) / (2.0f * delta);
}

inline float getNxDya(float x, float y, pixels &son) {
	return (getNxa(x, y + delta, son) - getNxa(x, y - delta, son)) / (2.0f * delta);
}



inline float getNyDxa(float x, float y, pixels &son) {
	return (getNya(x + delta, y, son) - getNya(x - delta, y, son)) / (2.0f * delta);
}

inline float getNyDya(float x, float y, pixels &son) {
	return (getNya(x, y + delta, son) - getNya(x, y - delta, son)) / (2.0f * delta);
}

// Get normal
inline Vector2f getNa(float x, float y, pixels &son) {
	return Vector2f(getNxa(x, y, son), getNya(x, y, son));
}

// Get Jacobian matrix
inline Matrix2f getJa(float x, float y,pixels &son) {
	Matrix2f J;
	J.m[0][0] = getNxDxa(x, y,son);
	J.m[0][1] = getNxDya(x, y,son);
	J.m[1][0] = getNyDxa(x, y,son);
	J.m[1][1] = getNyDya(x, y,son);
	return J;
}

//***************************************************************************************


void generateInit1(pixels &dest, pixels &source, int destSizeW, int destSizeH, int sourceSizeW, int sourceSizeH)
{
	int x, y;
	x = random(sourceSizeW - destSizeW);
	y = random(sourceSizeH - destSizeH);

	for (int i = 0; i < destSizeH; i++)
	{
		for (int j = 0; j < destSizeW; j++)
		{
			dest[i][j].r = source[y + i][x + j].r;
			dest[i][j].g = source[y + i][x + j].g;
			dest[i][j].b = source[y + i][x + j].b;
		}
	}
	chun[0].index = 0;
	chun[0].pos.h = 0;
	chun[0].pos.w = 0;
	chun[0].possource.h = y;
	chun[0].possource.w = x;
	for (int a = 0; a < TileSize; a++)
	{
		for (int b = 0; b < TileSize; b++)
		{
			idmat[a][b] = 0;
		}
	}
}

void generateInit2(Mat &dest, Mat &source, int destSizeW, int destSizeH, int sourceSizeW, int sourceSizeH)
{
	int x, y;
	x = random(sourceSizeW - destSizeW);
	y = random(sourceSizeH - destSizeH);
	Mat roi_d(dest, Rect(0, 0, destSizeW, destSizeH));
	Mat roi_s(source, Rect(x, y, destSizeW, destSizeH));
	chun[0].index = 0;
	chun[0].pos.h = 0;
	chun[0].pos.w = 0;
	chun[0].possource.h = y;
	chun[0].possource.w = x;
	for (int a = 0; a < TileSize; a++)
	{
		for (int b = 0; b < TileSize; b++)
		{
			idmat[a][b] = 0;
		}
	}
	roi_s.copyTo(roi_d);
}

void solve(set<point> & points, vector<vector<bool>> &mask, int x, int y, vector<vector<bool>> &tags)
{
	if (x < 0 || y < 0)
		return;
	mask[x][y] = true;
	tags[x][y] = true;
	int xn[][2] = { -1, 0, 0, -1 };
	for (int i = 0; i < 2; ++i)
	{
		int tmp_x = x + xn[i][0];
		int tmp_y = y + xn[i][1];
		if (tmp_x < 0 || tmp_y < 0)
			continue;
		point p(tmp_x, tmp_y);
		if (points.count(p) || tags[tmp_x][tmp_y])
			continue;
		solve(points, mask, tmp_x, tmp_y, tags);
	}
}

void generateMask1(pixels &area_1, pixels &area_2, pixels &c_area_1, pixels &c_area_2, vector<vector<bool>> &mask)
{
	vector<point> line_v;
	vector<point> line_h;
	int idx;
	half dis = 1e3;
	for (int i = 0; i < OverLapSize; ++i)
	{
		half tmp_0 = abs(area_1[0][i].r - c_area_1[0][i].r);
		half tmp_1 = abs(area_1[0][i].g - c_area_1[0][i].g);
		half tmp_2 = abs(area_1[0][i].b - c_area_1[0][i].b);

		half tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
		if (tmp < dis)
		{
			dis = tmp;
			idx = i;
		}
	}
	line_v.push_back(point(0, idx));
	for (int i = 1; i < TileSize; ++i)
	{
		int tmp_idx;
		dis = 1e3;
		for (int j = idx - 1; j <= idx + 1; ++j)
		{
			if (j < 0 || j >= OverLapSize)
				continue;
			half tmp_0 = abs(area_1[i][j].r - c_area_1[i][j].r);
			half tmp_1 = abs(area_1[i][j].g - c_area_1[i][j].g);
			half tmp_2 = abs(area_1[i][j].b - c_area_1[i][j].b);

			half tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
			if (tmp < dis)
			{
				dis = tmp;
				tmp_idx = j;
			}
		}
		idx = tmp_idx;
		line_v.push_back(point(i, idx));
	}
	dis = 1e3;
	for (int i = 0; i < OverLapSize; ++i)
	{
		half tmp_0 = abs(area_2[i][0].r - c_area_2[i][0].r);
		half tmp_1 = abs(area_2[i][0].g - c_area_2[i][0].g);
		half tmp_2 = abs(area_2[i][0].b - c_area_2[i][0].b);

		half tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
		if (tmp < dis)
		{
			dis = tmp;
			idx = i;
		}
	}
	line_h.push_back(point(idx, 0));
	for (int i = 1; i < TileSize; ++i)
	{
		int tmp_idx;
		dis = 1e3;
		for (int j = idx - 1; j <= idx + 1; ++j)
		{
			if (j < 0 || j >= OverLapSize)
				continue;
			half tmp_0 = abs(area_1[j][i].r - c_area_1[j][i].r);
			half tmp_1 = abs(area_1[j][i].g - c_area_1[j][i].g);
			half tmp_2 = abs(area_1[j][i].b - c_area_1[j][i].b);
			half tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
			if (tmp < dis)
			{
				//dis = (int)tmp;
				dis = tmp;
				tmp_idx = j;
			}
		}
		idx = tmp_idx;
		line_h.push_back(point(idx, i));
	}

	int pos_h = 0, pos_v = 0;
	for (int i = 0; i < TileSize; ++i){
		for (int j = 0; j < TileSize; ++j)
		{
			if (line_v[i].first == line_h[j].first && line_v[i].second == line_h[j].second)
			{
				pos_v = i;
				pos_h = j;
				break;
			}
		}
	}

	printf("insert: %d %d\n", pos_v, pos_h);
	set<point> points;
	points.insert(line_v[pos_v]);
	for (int i = pos_v + 1; i < TileSize; ++i)
		points.insert(line_v[i]);
	for (int i = pos_h + 1; i < TileSize; ++i)
		points.insert(line_h[i]);
	for (auto it : points);
	vector<vector<bool>> tags(TileSize, vector<bool>(TileSize, 0));
	solve(points, mask, TileSize - 1, TileSize - 1, tags);
}
void generateMask2(Mat &area_1, Mat &area_2, Mat &c_area_1, Mat &c_area_2, vector<vector<bool>> &mask)
{
	vector<point> line_v;
	vector<point> line_h;
	int idx;
	int dis = 1e10;
	for (int i = 0; i < OverLapSize; ++i)
	{
		int tmp_0 = abs(area_1.at<Vec3b>(0, i)[0] - c_area_1.at<Vec3b>(0, i)[0]);
		int tmp_1 = abs(area_1.at<Vec3b>(0, i)[1] - c_area_1.at<Vec3b>(0, i)[1]);
		int tmp_2 = abs(area_1.at<Vec3b>(0, i)[2] - c_area_1.at<Vec3b>(0, i)[2]);
		int tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
		if (tmp < dis)
		{
			dis = tmp;
			idx = i;
		}
	}
	line_v.push_back(point(0, idx));
	for (int i = 1; i < TileSize; ++i)
	{
		int tmp_idx;
		dis = 1e9;
		for (int j = idx - 1; j <= idx + 1; ++j)
		{
			if (j < 0 || j >= OverLapSize)
				continue;
			int tmp_0 = abs(area_1.at<Vec3b>(i, j)[0] - c_area_1.at<Vec3b>(i, j)[0]);
			int tmp_1 = abs(area_1.at<Vec3b>(i, j)[1] - c_area_1.at<Vec3b>(i, j)[1]);
			int tmp_2 = abs(area_1.at<Vec3b>(i, j)[2] - c_area_1.at<Vec3b>(i, j)[2]);
			int tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
			if (tmp < dis)
			{
				dis = tmp;
				tmp_idx = j;
			}
		}
		idx = tmp_idx;
		line_v.push_back(point(i, idx));
	}
	dis = 1e9;
	for (int i = 0; i < OverLapSize; ++i)
	{
		int tmp_0 = abs(area_2.at<Vec3b>(i, 0)[0] - c_area_2.at<Vec3b>(i, 0)[0]);
		int tmp_1 = abs(area_2.at<Vec3b>(i, 0)[1] - c_area_2.at<Vec3b>(i, 0)[1]);
		int tmp_2 = abs(area_2.at<Vec3b>(i, 0)[2] - c_area_2.at<Vec3b>(i, 0)[2]);
		int tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
		if (tmp < dis)
		{
			dis = tmp;
			idx = i;
		}
	}
	line_h.push_back(point(idx, 0));
	for (int i = 1; i < TileSize; ++i)
	{
		int tmp_idx;
		dis = 1e9;
		for (int j = idx - 1; j <= idx + 1; ++j)
		{
			if (j < 0 || j >= OverLapSize)
				continue;
			int tmp_0 = abs(area_2.at<Vec3b>(j, i)[0] - c_area_2.at<Vec3b>(j, i)[0]);
			int tmp_1 = abs(area_2.at<Vec3b>(j, i)[1] - c_area_2.at<Vec3b>(j, i)[1]);
			int tmp_2 = abs(area_2.at<Vec3b>(j, i)[2] - c_area_2.at<Vec3b>(j, i)[2]);
			int tmp = sqrt(tmp_0*tmp_0 + tmp_1*tmp_1 + tmp_2*tmp_2);
			if (tmp < dis)
			{
				dis = tmp;
				tmp_idx = j;
			}
		}
		idx = tmp_idx;
		line_h.push_back(point(idx, i));
	}
	int pos_h = 0, pos_v = 0;
	for (int i = 0; i<TileSize; ++i)
		for (int j = 0; j < TileSize; ++j)
		{
			if (line_v[i].first == line_h[j].first && line_v[i].second == line_h[j].second)
			{
				pos_v = i;
				pos_h = j;
				break;
			}
		}
	printf("insert: %d %d\n", pos_v, pos_h);
	set<point> points;
	points.insert(line_v[pos_v]);
	for (int i = pos_v + 1; i < TileSize; ++i)
		points.insert(line_v[i]);
	for (int i = pos_h + 1; i < TileSize; ++i)
		points.insert(line_h[i]);
	for (auto it : points);
	vector<vector<bool>> tags(TileSize, vector<bool>(TileSize, 0));
	solve(points, mask, TileSize - 1, TileSize - 1, tags);
}

void test1(pixels &dest, pixels const &source) {
	for (int i = Width*(num - 1) + TileSize - OverLapSize; i <= Width*num - TileSize; i += (TileSize - OverLapSize))
	{
		int index;
		index = i / (TileSize - OverLapSize);
		pixels area(TileSize, OverLapSize);
		for (int p = 0; p < TileSize; p++){
			for (int q = 0; q < OverLapSize; q++){
				area[p][q].r = dest[p][i + q].r;
				area[p][q].g = dest[p][i + q].g;
				area[p][q].b = dest[p][i + q].b;
			}
		}
		float dis = 1e5;
		int x, y;
		//find most similar form all source pixels 
		for (int j = 0; j < DefaultSize - TileSize; ++j)
			for (int k = 0; k < DefaultSize - TileSize; ++k)
			{
				pixels area_t(TileSize, OverLapSize);
				for (int p = 0; p < TileSize; p++){
					for (int q = 0; q < OverLapSize; q++){
						area_t[p][q].r = source[j + p][k + q].r;
						area_t[p][q].g = source[j + p][k + q].g;
						area_t[p][q].b = source[j + p][k + q].b;
					}
				}
				pixels res;
				absdiff(area, area_t, res);
				Rgba tmp_sum;
				sum(res, tmp_sum);
				auto tmp_dis = sqrt(tmp_sum.r*tmp_sum.r + tmp_sum.g*tmp_sum.g + tmp_sum.b*tmp_sum.b); //
				if (tmp_dis < dis)
				{
					dis = tmp_dis;
					x = k;
					y = j;
				}
			}
		printf("dis %f\n", dis);
		for (int p = 0; p < TileSize; p++)
		{
			for (int q = 0; q < TileSize; q++)
			{
				dest[p][i + q].r = source[y + p][x + q].r;
				dest[p][i + q].g = source[y + p][x + q].g;
				dest[p][i + q].b = source[y + p][x + q].b;
			}
		}

		chun[index].index = index;
		chun[index].pos.h = 0;
		chun[index].pos.w = i;
		chun[index].possource.h = y;
		chun[index].possource.w = x;
		for (int a = 0; a < TileSize; a++)
		{
			for (int b = 0; b < TileSize; b++)
			{
				idmat[a][b + i] = index;
			}
		}
	}

	// first column
	for (int i = Height *(num - 1) + TileSize - OverLapSize; i <= Height*num - TileSize; i += (TileSize - OverLapSize))
	{
		int index;
		index = (i / (TileSize - OverLapSize))*w;
		pixels area(OverLapSize, TileSize);
		for (int p = 0; p < OverLapSize; p++){
			for (int q = 0; q < TileSize; q++){
				area[p][q].r = dest[i + p][q].r;
				area[p][q].g = dest[i + p][q].g;
				area[p][q].b = dest[i + p][q].b;
			}
		}

		float dis = 1e5;
		int x, y;
		for (int m = 0; m <= DefaultSize - TileSize; ++m)
			for (int n = 0; n <= DefaultSize - TileSize; ++n)
			{
				pixels area_t(OverLapSize, TileSize);
				for (int p = 0; p < OverLapSize; p++){
					for (int q = 0; q < TileSize; q++){
						area_t[p][q].r = source[m + p][n + q].r;
						area_t[p][q].g = source[m + p][n + q].g;
						area_t[p][q].b = source[m + p][n + q].b;
					}
				}
				pixels res;
				absdiff(area, area_t, res);
				Rgba tmp_sum;
				sum(res, tmp_sum);
				auto tmp_dis = sqrt(tmp_sum.r*tmp_sum.r + tmp_sum.g*tmp_sum.g + tmp_sum.b*tmp_sum.b); //
				if (tmp_dis < dis)
				{
					dis = tmp_dis;
					x = n;
					y = m;
				}
			}
		printf("dis %f\n", dis);
		for (int p = 0; p < TileSize; p++)
		{
			for (int q = 0; q < TileSize; q++)
			{
				dest[i + p][q].r = source[y + p][x + q].r;
				dest[i + p][q].g = source[y + p][x + q].g;
				dest[i + p][q].b = source[y + p][x + q].b;
			}
		}

		chun[index].index = index;
		chun[index].pos.h = i;
		chun[index].pos.w = 0;
		chun[index].possource.h = y;
		chun[index].possource.w = x;
		for (int a = 0; a < TileSize; a++)
		{
			for (int b = 0; b < TileSize; b++)
			{
				idmat[a + i][b] = index;
			}
		}
	}

	// middle part
	for (int i = Height *(num - 1) + TileSize - OverLapSize; i <= Height*num - TileSize; i += (TileSize - OverLapSize))
		for (int j = Width *(num - 1) + TileSize - OverLapSize; j <= Width*num - TileSize; j += (TileSize - OverLapSize))
		{
			int index;
			index = (i / (TileSize - OverLapSize))*w + (j / (TileSize - OverLapSize));
			pixels area_1(TileSize, OverLapSize);
			for (int p = 0; p < TileSize; p++){
				for (int q = 0; q < OverLapSize; q++){
					area_1[p][q].r = dest[i + p][j + q].r;
					area_1[p][q].g = dest[i + p][j + q].g;
					area_1[p][q].b = dest[i + p][j + q].b;
				}
			}
			pixels area_2(OverLapSize, TileSize);
			for (int p = 0; p < OverLapSize; p++){
				for (int q = 0; q < TileSize; q++){
					area_2[p][q].r = dest[i + p][j + q].r;
					area_2[p][q].g = dest[i + p][j + q].g;
					area_2[p][q].b = dest[i + p][j + q].b;
				}
			}
			pixels area_0(OverLapSize, OverLapSize);
			for (int p = 0; p < OverLapSize; p++){
				for (int q = 0; q < OverLapSize; q++){
					area_0[p][q].r = dest[i + p][j + q].r;
					area_0[p][q].g = dest[i + p][j + q].g;
					area_0[p][q].b = dest[i + p][j + q].b;
				}
			}

			float dis = 1e5;
			int x, y;
			for (int m = 0; m <= DefaultSize - TileSize; ++m)
				for (int n = 0; n <= DefaultSize - TileSize; ++n)
				{
					pixels c_area_1(TileSize, OverLapSize);
					for (int p = 0; p < TileSize; p++){
						for (int q = 0; q < OverLapSize; q++){
							c_area_1[p][q].r = source[m + p][n + q].r;
							c_area_1[p][q].g = source[m + p][n + q].g;
							c_area_1[p][q].b = source[m + p][n + q].b;
						}
					}
					pixels c_area_2(OverLapSize, TileSize);
					for (int p = 0; p < OverLapSize; p++){
						for (int q = 0; q < TileSize; q++){
							c_area_2[p][q].r = source[m + p][n + q].r;
							c_area_2[p][q].g = source[m + p][n + q].g;
							c_area_2[p][q].b = source[m + p][n + q].b;
						}
					}
					pixels c_area_0(OverLapSize, OverLapSize);
					for (int p = 0; p < OverLapSize; p++){
						for (int q = 0; q < OverLapSize; q++){
							c_area_0[p][q].r = source[m + p][n + q].r;
							c_area_0[p][q].g = source[m + p][n + q].g;
							c_area_0[p][q].b = source[m + p][n + q].b;
						}
					}
					pixels res;
					Rgba tmp_sum;
					absdiff(c_area_1, area_1, res);
					sum(res, tmp_sum);
					auto tmp_dis = sqrt(tmp_sum.r*tmp_sum.r + tmp_sum.g*tmp_sum.g + tmp_sum.b*tmp_sum.b);
					absdiff(c_area_2, area_2, res);
					sum(res, tmp_sum);
					tmp_dis += sqrt(tmp_sum.r*tmp_sum.r + tmp_sum.g*tmp_sum.g + tmp_sum.b*tmp_sum.b);
					absdiff(c_area_0, area_0, res);
					sum(res, tmp_sum);
					tmp_dis -= sqrt(tmp_sum.r*tmp_sum.r + tmp_sum.g*tmp_sum.g + tmp_sum.b*tmp_sum.b);

					if (tmp_dis < dis)
					{
						dis = tmp_dis;
						x = n;
						y = m;
					}
				}
			chun[index].index = index;
			chun[index].pos.h = i;
			chun[index].pos.w = j;
			chun[index].possource.h = y;
			chun[index].possource.w = x;
			printf("dis %f\n", dis);
			//Mat c_area_1(source, Rect(x, y, OverLapSize, TileSize));
			pixels c_area_1(TileSize, OverLapSize);
			for (int p = 0; p < TileSize; p++){
				for (int q = 0; q < OverLapSize; q++){
					c_area_1[p][q].r = source[y + p][x + q].r;
					c_area_1[p][q].g = source[y + p][x + q].g;
					c_area_1[p][q].b = source[y + p][x + q].b;
				}
			}
			//Mat c_area_2(source, Rect(x, y, TileSize, OverLapSize));
			pixels c_area_2(OverLapSize, TileSize);
			for (int p = 0; p < OverLapSize; p++){
				for (int q = 0; q < TileSize; q++){
					c_area_2[p][q].r = source[y + p][x + q].r;
					c_area_2[p][q].g = source[y + p][x + q].g;
					c_area_2[p][q].b = source[y + p][x + q].b;
				}
			}
			vector<vector<bool>> mask(TileSize, vector<bool>(TileSize, 0));
			generateMask1(area_1, area_2, c_area_1, c_area_2, mask);// 计算掩码
			for (int m = 0; m<TileSize; ++m)
				for (int n = 0; n < TileSize; ++n)
				{
					if (mask[m][n]) // 位于掩码的部分更新
					{
						//copy_d.at<Vec3b>(m, n) = copy_s.at<Vec3b>(m, n);
						dest[i + m][j + n].r = source[y + m][x + n].r;
						dest[i + m][j + n].g = source[y + m][x + n].g;
						dest[i + m][j + n].b = source[y + m][x + n].b;
						idmat[i + m][j + n] = index;

					}
				}
		}
}
void test2(Mat &dest, Mat &source) {
	for (int i = Width*(num - 1) + TileSize - OverLapSize; i <= Width*num - TileSize; i += (TileSize - OverLapSize))
	{
		int index;
		index = i / (TileSize - OverLapSize);

		Mat area(dest, Rect(i, 0, OverLapSize, TileSize));
		double dis = 1e10;
		int x, y;
		for (int j = 0; j < DefaultSize - TileSize; ++j)
			for (int k = 0; k < DefaultSize - TileSize; ++k)
			{
				Mat area_t(source, Rect(k, j, OverLapSize, TileSize));
				Mat res;
				absdiff(area, area_t, res);
				auto tmp_dis = sqrt(sum(res)[0] * sum(res)[0] + sum(res)[1] * sum(res)[1] + sum(res)[2] * sum(res)[2]);
				if (tmp_dis < dis)
				{
					dis = tmp_dis;
					x = k;
					y = j;
				}
			}
		printf("dis %lf\n", dis);
		Mat copy_d(dest, Rect(i, 0, TileSize, TileSize));
		Mat copy_s(source, Rect(x, y, TileSize, TileSize));
		chun[index].index = index;
		chun[index].pos.h = 0;
		chun[index].pos.w = i;
		chun[index].possource.h = y;
		chun[index].possource.w = x;
		for (int a = 0; a < TileSize; a++)
		{
			for (int b = 0; b < TileSize; b++)
			{
				idmat[a][b + i] = index;
			}
		}
		copy_s.copyTo(copy_d);
	}

	//first column
	for (int i = Height *(num - 1) + TileSize - OverLapSize; i <= Height*num - TileSize; i += (TileSize - OverLapSize))
	{
		int index;
		index = (i / (TileSize - OverLapSize))*w;
		Mat area(dest, Rect(0, i, TileSize, OverLapSize));
		double dis = 1e10;
		int x, y;
		for (int m = 0; m <= DefaultSize - TileSize; ++m)
			for (int n = 0; n <= DefaultSize - TileSize; ++n)
			{
				Mat area_t(source, Rect(n, m, TileSize, OverLapSize));
				Mat res;
				absdiff(area, area_t, res);
				auto tmp_dis = sqrt(sum(res)[0] * sum(res)[0] + sum(res)[1] * sum(res)[1] + sum(res)[2] * sum(res)[2]);
				if (tmp_dis < dis)
				{
					dis = tmp_dis;
					x = n;
					y = m;
				}
			}
		printf("dis %lf\n", dis);
		Mat copy_d(dest, Rect(0, i, TileSize, TileSize));
		Mat copy_s(source, Rect(x, y, TileSize, TileSize));
		copy_s.copyTo(copy_d);
		chun[index].index = index;
		chun[index].pos.h = i;
		chun[index].pos.w = 0;
		chun[index].possource.h = y;
		chun[index].possource.w = x;
		for (int a = 0; a < TileSize; a++)
		{
			for (int b = 0; b < TileSize; b++)
			{
				idmat[a + i][b] = index;
			}
		}
	}

	// inside part
	for (int i = Height *(num - 1) + TileSize - OverLapSize; i <= Height*num - TileSize; i += (TileSize - OverLapSize))
		for (int j = Width *(num - 1) + TileSize - OverLapSize; j <= Width*num - TileSize; j += (TileSize - OverLapSize))
		{
			int index;
			index = (i / (TileSize - OverLapSize))*w + (j / (TileSize - OverLapSize));
			Mat area_1(dest, Rect(j, i, OverLapSize, TileSize));// horizontal part
			Mat area_2(dest, Rect(j, i, TileSize, OverLapSize));// vertical part
			Mat area_0(dest, Rect(j, i, OverLapSize, OverLapSize));// small square
			double dis = 1e10;
			int x, y;
			for (int m = 0; m <= DefaultSize - TileSize; ++m)
				for (int n = 0; n <= DefaultSize - TileSize; ++n)
				{
					Mat c_area_1(source, Rect(n, m, OverLapSize, TileSize));
					Mat c_area_2(source, Rect(n, m, TileSize, OverLapSize));
					Mat c_area_0(source, Rect(n, m, OverLapSize, OverLapSize));
					Mat res;
					absdiff(c_area_1, area_1, res);
					auto tmp_dis = sqrt(sum(res)[0] * sum(res)[0] + sum(res)[1] * sum(res)[1] + sum(res)[2] * sum(res)[2]);
					absdiff(c_area_2, area_2, res);
					tmp_dis += sqrt(sum(res)[0] * sum(res)[0] + sum(res)[1] * sum(res)[1] + sum(res)[2] * sum(res)[2]);
					absdiff(c_area_0, area_0, res);
					tmp_dis -= sqrt(sum(res)[0] * sum(res)[0] + sum(res)[1] * sum(res)[1] + sum(res)[2] * sum(res)[2]);
					if (tmp_dis < dis)
					{
						dis = tmp_dis;
						x = n;
						y = m;
					}
				}
			chun[index].index = index;
			chun[index].pos.h = i;
			chun[index].pos.w = j;
			chun[index].possource.h = y;
			chun[index].possource.w = x;
			printf("dis %lf\n", dis);
			Mat c_area_1(source, Rect(x, y, OverLapSize, TileSize));
			Mat c_area_2(source, Rect(x, y, TileSize, OverLapSize));
			vector<vector<bool>> mask(TileSize, vector<bool>(TileSize, 0));
			generateMask2(area_1, area_2, c_area_1, c_area_2, mask);// 计算掩码
			Mat copy_d(dest, Rect(j, i, TileSize, TileSize));
			Mat copy_s(source, Rect(x, y, TileSize, TileSize));
			for (int m = 0; m<TileSize; ++m)
				for (int n = 0; n < TileSize; ++n)
				{
					if (mask[m][n]) // 位于掩码的部分更新
					{
						copy_d.at<Vec3b>(m, n) = copy_s.at<Vec3b>(m, n);
						idmat[i + m][j + n] = index;

					}
				}
			//copy_s.copyTo(copy_d);
		}
}

void dealwith(int x, int y, int id)
{
	for (int i = x - 2; i <= x + 2; i++)
	{
		for (int j = y - 2; j <= y + 2; j++)
		{
			if (idmat[i][j] != id && 0 <= i && 0 <= j&&i<Height&&j<Width)
			{
				ifmanage[i][j] = 1;
			}
		}
	}
}
void findmap()
{
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			ifmanage[i][j] = 0;
		}
	}
	for (int i = 0; i < Height; i++)
	{
		//		printf("%d\n",i);
		for (int j = 0; j < Width; j++)
		{
			if (0 <= (i - 1))
			{
				if (idmat[i - 1][j] != idmat[i][j])
				{
					dealwith(i, j, idmat[i][j]);
				}
			}
			if ((i + 1)<Height)
			{
				if (idmat[i + 1][j] != idmat[i][j])
				{
					dealwith(i, j, idmat[i][j]);
				}
			}

			if (0 <= (j - 1))
			{
				if (idmat[i][j - 1] != idmat[i][j])
				{
					dealwith(i, j, idmat[i][j]);
				}
			}
			if ((j + 1)<Height)
			{
				if (idmat[i][j + 1] != idmat[i][j])
				{
					dealwith(i, j, idmat[i][j]);
				}
			}

		}

	}


}

void sampleForLinearFlakes(const char *flakesFilename, float sampling_rate) {

	FILE *fp = fopen(flakesFilename, "wb");

	int type = 1;
	fwrite(&type, sizeof(int), 1, fp);
	fwrite(&width, sizeof(int), 1, fp);
	fwrite(&height, sizeof(int), 1, fp);

	int numFlakes = height * width * sampling_rate * sampling_rate;
	fwrite(&numFlakes, sizeof(int), 1, fp);

	delta /= sampling_rate;

	float step = 1.0f / sampling_rate;
	for (float i = 0.0f; i < width; i += step) {
		for (float j = 0.0f; j < height; j += step) {
			Vector2f u0(i, j);
			Vector2f n0 = getNormal(i * 1.0f / width, j * 1.0f / height);
			Vector2f shape(step * 1.5 / sqrt(12.0f), step * 1.5 / sqrt(12.0f));
			Matrix2f J = getJ(i, j);

			fwrite(&u0[0], sizeof(float), 1, fp);
			fwrite(&u0[1], sizeof(float), 1, fp);
			fwrite(&n0[0], sizeof(float), 1, fp);
			fwrite(&n0[1], sizeof(float), 1, fp);
			fwrite(&shape[0], sizeof(float), 1, fp);
			fwrite(&shape[1], sizeof(float), 1, fp);
			fwrite(&J.m[0][0], sizeof(float), 1, fp);
			fwrite(&J.m[1][0], sizeof(float), 1, fp);
			fwrite(&J.m[0][1], sizeof(float), 1, fp);
			fwrite(&J.m[1][1], sizeof(float), 1, fp);

			float area = step * step;
			fwrite(&area, sizeof(float), 1, fp);
		}
	}
	fclose(fp);
}

void dealWithChain(const char *chainFilename, float sampling_rate) {
	FILE *fp = fopen(chainFilename, "wb");
	int height = Height;
	int width = Width;
	int tile = TileSize;
	int over = OverLapSize;
	fwrite(&height, sizeof(int), 1, fp);
	fwrite(&width, sizeof(int), 1, fp);

	int numelements;
	numelements = 0;
	for (int i = 0; i < Width; i++)
	{
		for (int j = 0; j < Height; j++)
		{
			fwrite(&ifmanage[j][i], sizeof(int), 1, fp);
			if (ifmanage[j][i])
			{
				numelements++;
			}

		}
	}
	for (int i = 0; i < Width; i++)
	{
		for (int j = 0; j < Height; j++)
		{
			fwrite(&idmat[j][i], sizeof(int), 1, fp);
		}
	}
	fwrite(&chuncknum, sizeof(int), 1, fp);
	fwrite(&tile, sizeof(int), 1, fp);
	fwrite(&over, sizeof(int), 1, fp);
	for (int i = 0; i < chuncknum; i++)
	{
		fwrite(&chun[i].index, sizeof(int), 1, fp);

		fwrite(&chun[i].pos.h, sizeof(int), 1, fp);

		fwrite(&chun[i].pos.w, sizeof(int), 1, fp);

		fwrite(&chun[i].possource.h, sizeof(int), 1, fp);

		fwrite(&chun[i].possource.w, sizeof(int), 1, fp);

	}
	float step = 1.0f / sampling_rate;
	numelements = numelements*sampling_rate*sampling_rate;
	fwrite(&numelements, sizeof(int), 1, fp);
	int numflake;
	numflake = 0;
	for (float p = 0; p < Width; p += step)
	{
		for (float q = 0; q < Height; q += step)
		{
			if (ifmanage[int(q)][int(p)] == 1)
			{
				numflake++;
				pixels pleaseberight(9,9);
				for (int a = 0; a < 9; a++)
				{
					for (int b = 0; b < 9; b++)
					{
						int u, v;
						u = a - 4 + p;
						v = b - 4 + q;
						if (0 <= u&&u < Width && 0 <= v&&v < Height)    //比yanqiling多出的步骤，边缘处理。
						{
							int id = idmat[v][u];
							u = u - chun[id].pos.w + chun[id].possource.w;
							v = v - chun[id].pos.h + chun[id].possource.h;
							pleaseberight[a][b].r = dest_normalmap[u][v].r;
							pleaseberight[a][b].g = dest_normalmap[u][v].g;
							pleaseberight[a][b].b = dest_normalmap[u][v].b;
						}
						else
						{
							int id = idmat[int(q)][int(p)];
							u = p - chun[id].pos.w + chun[id].possource.w;
							v = q - chun[id].pos.h + chun[id].possource.h;

							pleaseberight[a][b].r = dest_normalmap[u][v].r;
							pleaseberight[a][b].g = dest_normalmap[u][v].g;
							pleaseberight[a][b].b = dest_normalmap[u][v].b;
						}
					}
				}

				Vector2f u0(p, q);
				Vector2f n0 = getNormala(4 + p - int(p), 4 + q - int(q), pleaseberight);
				Vector2f shape(step * 1.5 / sqrt(12.0f), step * 1.5 / sqrt(12.0f));
				Matrix2f J = getJa(4 + p - int(p), 4 + q - int(q), pleaseberight);

				fwrite(&u0[0], sizeof(float), 1, fp);
				fwrite(&u0[1], sizeof(float), 1, fp);
				fwrite(&n0[0], sizeof(float), 1, fp);
				fwrite(&n0[1], sizeof(float), 1, fp);
				fwrite(&shape[0], sizeof(float), 1, fp);
				fwrite(&shape[1], sizeof(float), 1, fp);
				fwrite(&J.m[0][0], sizeof(float), 1, fp);
				fwrite(&J.m[1][0], sizeof(float), 1, fp);
				fwrite(&J.m[0][1], sizeof(float), 1, fp);
				fwrite(&J.m[1][1], sizeof(float), 1, fp);

				float area = step * step;
				fwrite(&area, sizeof(float), 1, fp);




			}


		}
	}
	if (numelements != numflake)
	{
		printf("the element number wrong\n");
	}
	fclose(fp);
}

//read .exr rgba file
void read_source_normalmap(const char *file_name, int &width, int &height) {
	RgbaInputFile file(file_name);
	Imath::Box2i dw = file.dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	printf("%d %d\n", width, height);
	sourceEXR_normalmap.resizeErase(height, width);
	file.setFrameBuffer(&sourceEXR_normalmap[0][0] - dw.min.x - dw.min.y * width, 1, width);
	file.readPixels(dw.min.y, dw.max.y);
}

int main(int argc,char** argv)
{
	dest_normalmap.resizeErase(Height, Width);
	chun = (chunk*)malloc(chuncknum*sizeof(chunk));
	if (argc != 6) {
		cout << "Usage: ./visualize_ndf_binning source_file_type source_filename flakes_filename chain_filename sampling_rate" << endl;
		return 0;
	}
	char SOURCE_FILETYPE[100];
	char SOURCE_FILENAME[100];
	char FLAKES_FILENAME[100];
	char CHAIN_FILENAME[100];
	strcpy(SOURCE_FILETYPE, argv[1]);
	strcpy(SOURCE_FILENAME, argv[2]);
	strcpy(FLAKES_FILENAME, argv[3]);
	strcpy(CHAIN_FILENAME, argv[4]);
	if (strcmp(SOURCE_FILETYPE,"exr")==0)
	{
		read_source_normalmap(SOURCE_FILENAME, width, height);
		generateInit1(dest_normalmap, sourceEXR_normalmap, TileSize, TileSize, DefaultSize, DefaultSize);
		test1(dest_normalmap, sourceEXR_normalmap);
		findmap();
		num++;
	}
	else
	{
		Mat source = imread(SOURCE_FILENAME, IMREAD_UNCHANGED);
		printf("channels: %d\n", source.channels());
		printf("size: %d %d\n", source.rows, source.cols);
		Mat dest(Height*num, Width*num, CV_8UC3);
		generateInit2(dest, source, TileSize, TileSize, DefaultSize, DefaultSize);
		test2(dest, source);
		findmap();
		num++;
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				dest_normalmap[i][j].r = dest.at<Vec3b>(i, j)[0];
				dest_normalmap[i][j].g = dest.at<Vec3b>(i, j)[1];
				dest_normalmap[i][j].b = dest.at<Vec3b>(i, j)[2];
			}
		}
	}
	
	sampleForLinearFlakes(FLAKES_FILENAME, atof(argv[5]));
	dealWithChain(CHAIN_FILENAME, atof(argv[5]));
	printf("precomputing finished!\n");
	getchar();
	return 0;
}
