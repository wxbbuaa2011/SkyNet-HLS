#ifndef CNN_H
#define CNN_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <cstring>

#ifdef __SDSCC__
#include "sds_lib.h"
#else
#define sds_alloc malloc
#endif

typedef float DT;

#define layer_count 19
#define check_scale 0.01

struct layer
{
	char name[8];
	int iw, ih, ic, ow, oh, oc;
	int k, s, p;
};

struct DT32
{
	DT data[32];
};

/**********utils.cpp************/
void load_fm(float* fm, layer l);
void load_weight(float *weight, int length);
void load_dwconv_weight(float *weight, layer l);
void load_pwconv_weight(float *weight, layer l);
void load_bias(float *bias, layer l);
void check(DT* result, DT* golden, int len, layer l);
void check_fm(float* fm, layer l);
void show_fm(float* fm, layer l);

void generate_fm(float* fm, layer l);
void generate_weight(float* weight, layer l);


void stitch(DT* ifm[4], DT* ofm, layer l);
void distitch(DT* ifm, DT* ofm[4], layer l);

void DWCONV3X3(DT IFM[32][42][82], DT OFM[32][42][82], DT WBUF3x3[32][3][3]);
void PWCONV1X1(DT IFM[32][42][82], DT OFM[32][42][82], DT WBUF1x1[32][32]);

/**********operations************/
void pwconv(float *ifm, float *ofm, float *weight, float *bias, int relu, layer l);
void dwconv(float *ifm, float *ofm, float *weight, float *bias, int relu, layer l);
void maxpool(float *ifm, float *ofm, layer l);
void concat(float *ifm1, float *ifm2, float *ofm, layer l1, layer l2);
void reorg(float *ifm, float *ofm, layer l);
#endif //CNN_H