#include "CNN.h"

layer config[layer_count] = {
{ "data",    320,160,3,  320,160,3,  0,0,0},    //data
{ "dwconv1", 320,160,3,  320,160,3,  3,1,1},    //dwconv1
{ "pwconv1", 320,160,3,  320,160,48, 1,1,0},    //pwconv1
{ "pool1",   320,160,48, 160,80,48,  2,2,0},    //pool1
{ "dwconv2", 160,80,48,  160,80,48,  3,1,1},    //dwconv2
{ "pwconv2", 160,80,48,  160,80,96,  1,1,0},    //pwconv2
{ "pool2",   160,80,96,  80,40,96,   2,2,0},    //pool2
{ "dwconv3", 80,40,96,   80,40,96,   3,1,1},    //dwconv3
{ "pwconv3", 80,40,96,   80,40,192,  1,1,0},    //pwconv3
{ "reorg",   80,40,192,  40,20,768,  2,2,0},    //reorg
{ "pool3",   80,40,192,  40,20,192,  2,2,0},    //pool3
{ "dwconv4", 40,20,192,  40,20,192,  3,1,1},    //dwconv4
{ "pwconv4", 40,20,192,  40,20,384,  1,1,0},    //pwconv4
{ "dwconv5", 40,20,384,  40,20,384,  3,1,1},    //dwconv5
{ "pwconv5", 40,20,384,  40,20,512,  1,1,0},    //pwconv5
{ "cat",     40,20,192,  40,20,1280, 0,0,0},    //concat
{ "dwconv6", 40,20,1280, 40,20,1280, 3,1,1},    //dwconv6
{ "pwconv6", 40,20,1280, 40,20,96,   1,1,0},    //pwconv6
{ "conv7",   40,20,96,   40,20,10,   1,1,0},    //conv7
};

void Load_Image(DT* img, DT IBUF[32][42][82], int Hx, int Wx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    for (int c=0; c<3; c++)
    {
        for (int h=0; h<42; h++)
        {
            for (int w=0; w<82; w++)
            {
                int ifm_index = c*643*323 + (h+h_offset)*643 + (w+w_offset);
                IBUF[c][h][w] = img[ifm_index];
            }
        }
    }
}

void Load_WBUF3x3(DT* weight, DT WBUF3x3[32][3][3])
{
    for(int c=0; c<3; c++)
    {
        for(int m=0; m<3; m++)
        {
            for(int n=0; n<3; n++)
            {
                WBUF3x3[c][m][n] = weight[c*3*3 + m*3 + n];
            }
        }
    }
}

void Export_DWCONV1(DT* ofm, DT OBUF[32][42][82], int Hx, int Wx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    for (int c=0; c<3; c++)
    {
        for (int h=1; h<=40; h++)
        {
            for (int w=1; w<=80; w++)
            {
                int ofm_index = c*323*643 + (h+h_offset)*643 + (w+w_offset);
                ofm[ofm_index] = OBUF[c][h][w];
                OBUF[c][h][w] = 0;
            }
        }
    }
}

void Load_BBUF(DT* bias, DT BBUF[32])
{
    for(int c=0; c<3; c++)
    {
        BBUF[c] = bias[c];
    }
}

DT* img[4];
DT* data_blob;
DT* dwconv1_blob;
DT* dwconv1[4];
DT* parameter;
DT* blob[2];
DT* reorg_blob;

void SkyNet_init()
{
    for(int p=0; p<4; p++)
    {
        img[p] = (DT*)sds_alloc(3*160*320*sizeof(DT));
        dwconv1[p] = (DT*)sds_alloc(3*160*320*sizeof(DT));
    }
    data_blob = (DT*)sds_alloc(3*323*623*sizeof(DT));
    dwconv1_blob = (DT*)sds_alloc(3*323*623*sizeof(DT));
    parameter = (DT*)sds_alloc(438728*sizeof(DT));
    load_weight(parameter, 438728);
}

void SkyNet_(DT* img, DT* ofm, DT* weight, DT* bias)
{
    DT FM1[32][42][82];
    DT FM2[32][42][82];
    DT FM3[32][42][82];
    DT FM4[32][42][82];

    DT WBUF3x3[32][3][3]={0};
    DT WBUF1x1[32][32]={0};
    DT BBUF[32]={0};

    Load_WBUF3x3(weight, WBUF3x3);
    Load_BBUF(bias, BBUF);

    for(int Hx=0; Hx<8; Hx++)
    {	
		for(int Wx=0; Wx<8; Wx++) 
        {
            Load_Image(img, FM1, Hx, Wx);
            DWCONV3X3(FM1, FM2, WBUF3x3, BBUF, 1);
            Export_DWCONV1(ofm, FM2, Hx, Wx);
		}
	}
}

void SkyNet()
{
    for(int p=0; p<4; p++)
        load_fm(img[p], config[0]);
    //for(int p=0; p<4; p++)
    //    generate_fm(img[p], config[0]);
    //show_fm(img[0], config[0]);
    stitch(img, data_blob);
    int weight_offset = 0;
    int bias_offset = weight_offset + config[1].oc*config[1].k*config[1].k;
    SkyNet_(data_blob, dwconv1_blob, &parameter[weight_offset], &parameter[bias_offset]);
    distitch(dwconv1_blob, dwconv1);
    //show_fm(dwconv1[0], config[0]);
    for(int p=0; p<4; p++)
        //check(dwconv1[p], img[0], config[1]);
        check_fm(dwconv1[p], config[1]);
}