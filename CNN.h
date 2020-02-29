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
void load_fm(DT* fm, layer l);
void load_weight(DT32 *weight, int length);
void load_dwconv_weight(DT *weight, layer l);
void load_pwconv_weight(DT *weight, layer l);
void load_bias(DT *bias, layer l);
void check(DT* result, DT* golden, int len, layer l);
void check_fm(DT* fm, layer l);
void show_fm(DT* fm, layer l);

void generate_fm(DT* fm, layer l);
void generate_weight(DT* weight, layer l);

/**********transform.cpp************/
void stitch(DT* ifm[4], DT* ofm, layer l);
void distitch(DT* ifm, DT* ofm[4], layer l);
void fm_DT_2_DT32(DT* in, DT32* out, layer l);
void fm_DT32_2_DT(DT32* in, DT* out, layer l);
void w_DT_2_DT32(DT* in, DT32* out, layer l);
void b_DT_2_DT32(DT* in, DT32* out, layer l);

void DWCONV3X3(DT IFM[32][42][82], DT OFM[32][42][82], DT WBUF3x3[32][3][3]);
void PWCONV1X1(DT IFM[32][42][82], DT OFM[32][42][82], DT WBUF1x1[32][32]);
void POOL(DT IFM[32][42][82], DT OFM[32][42][82]);

/**********operations************/
void pwconv(DT *ifm, DT *ofm, DT *weight, DT *bias, int relu, layer l);
void dwconv(DT *ifm, DT *ofm, DT *weight, DT *bias, int relu, layer l);
void maxpool(DT *ifm, DT *ofm, layer l);
void concat(DT *ifm1, DT *ifm2, DT *ofm, layer l1, layer l2);
void reorg(DT *ifm, DT *ofm, layer l);
#endif //CNN_H