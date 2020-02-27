#include "CNN.h"

layer config[layer_count] = {
{ "data",    320,160,3,  320,160,32, 0,0,0},    //data
{ "dwconv1", 320,160,32, 320,160,32, 3,1,1},    //dwconv1
{ "pwconv1", 320,160,32, 320,160,64, 1,1,0},    //pwconv1
{ "pool1",   320,160,64, 160,80,64,  2,2,0},    //pool1
{ "dwconv2", 160,80,64,  160,80,64,  3,1,1},    //dwconv2
{ "pwconv2", 160,80,64,  160,80,96,  1,1,0},    //pwconv2
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

void Load_IFM(DT* ifm, DT IBUF[32][42][82], int Hx, int Wx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    for (int c=0; c<32; c++)
    {
        for (int h=0; h<42; h++)
        {
            for (int w=0; w<82; w++)
            {
                int ifm_index = c*643*323 + (h+h_offset)*643 + (w+w_offset);
                IBUF[c][h][w] = ifm[ifm_index];
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

void Load_WBUF1x1(DT* weight, DT WBUF1x1[32][32])
{
    for(int m=0; m<32; m++)
    {
        for(int n=0; n<32; n++)
        {
            WBUF1x1[m][n] = weight[m*32 + n];
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
            }
        }
    }
}

void Export_PWCONV1(DT* ofm, DT OBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    int c_offset = Cx*32;
    for (int c=0; c<32; c++)
    {
        for (int h=1; h<=40; h++)
        {
            for (int w=1; w<=80; w++)
            {
                int ofm_index = (c+c_offset)*323*643 + (h+h_offset)*643 + (w+w_offset);
                ofm[ofm_index] = OBUF[c][h][w];
            }
        }
    }
}

void Add_Bias(DT FM[32][42][82], DT BBUF[32], int relu)
{
    for(int h=1; h<=40; h++){
        for(int w=1; w<=80; w++){
            for(int c=0; c<32; c++){
                DT odata = FM[c][h][w];
                odata += BBUF[c];
                if(relu==1)
                {
                    if(odata<0)
                        FM[c][h][w] = 0;
                    else
                        FM[c][h][w] = odata;
                }
            }
        }
    }
}

void Load_BBUF(DT* bias, DT BBUF[32])
{
    for(int c=0; c<32; c++)
    {
        BBUF[c] = bias[c];
    }
}

void Clear_FM(DT FM[32][42][82])
{
    for(int h=0; h<42; h++){
        for(int w=0; w<82; w++){
            for(int c=0; c<32; c++){
                FM[c][h][w] = 0;
            }
        }
    }
}

void Compare(DT FM1[32][42][82], DT FM2[32][42][82])
{
    int error = 0;
    for(int h=1; h<41; h++){
        for(int w=1; w<81; w++){
            for(int c=0; c<32; c++){
                if(abs(FM1[c][h][w]-FM2[c][h][w])>0.001)
                    error++;
                    //printf("FM1[%d][%d][%d]=%f, FM2[%d][%d][%d]=%f\n", c,h,w,FM1[c][h][w],c,h,w,FM2[c][h][w]);
            }
        }
    }
    printf("error count: %d\n", error);
}

void SkyNet_(DT* ifm, DT* ofm, DT* parameter)
{
    DT FM1[32][42][82]={0};
    DT FM2[32][42][82]={0};
    DT FM3[32][42][82]={0};
    DT FM4[32][42][82]={0};
    DT FM5[32][42][82]={0};

    DT WBUF3x3[4][32][3][3]={0};
    DT WBUF1x1[4][32][32]={0};
    DT BBUF[4][32]={0};


    int weight_offset = 0;
    int bias_offset = weight_offset + config[1].oc*config[1].k*config[1].k;
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0]);
    Load_BBUF(parameter + bias_offset, BBUF[0]);
    weight_offset = bias_offset + config[1].oc;
    bias_offset = weight_offset + config[2].oc*config[2].ic*config[2].k*config[2].k;
    Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0]);
    Load_WBUF1x1(parameter + weight_offset + 32*32, WBUF1x1[1]);
    Load_BBUF(parameter + bias_offset, BBUF[1]);
    Load_BBUF(parameter + bias_offset + 32, BBUF[2]);
    
    for(int Hx=0; Hx<8; Hx++)
    {	
        Load_IFM(ifm, FM1, Hx, 0);
		for(int Wx=0; Wx<8; Wx++) 
        {
            if(Wx%2==0)
            {
                Load_IFM(ifm, FM2, Hx, Wx+1);
                {
                    DWCONV3X3(FM1, FM3, WBUF3x3[0]);
                    Add_Bias(FM3, BBUF[0], 1);
                }
            }
            else
            {
                Load_IFM(ifm, FM1, Hx, Wx+1);
                {
                    DWCONV3X3(FM2, FM3, WBUF3x3[0]);
                    Add_Bias(FM3, BBUF[0], 1);
                }
            }
            for(int Cx=0; Cx<2; Cx++)
            {
                PWCONV1X1(FM3, FM5, WBUF1x1[Cx]);
                Add_Bias(FM5, BBUF[Cx+1], 1);
                Export_PWCONV1(ofm, FM5, Hx, Wx, Cx);
                Clear_FM(FM5);
            }
            Clear_FM(FM3);
		}
	}
}

DT* data[4];
DT* data_blob;
DT* dwconv1[4];
DT* dwconv1_blob;
DT* pwconv1_blob;
DT* pwconv1[4];
DT* parameter;
DT* blob[2];
DT* reorg_blob;

void SkyNet_init()
{
    for(int p=0; p<4; p++)
    {
        data[p] = (DT*)sds_alloc(32*160*320*sizeof(DT));
        dwconv1[p] = (DT*)sds_alloc(32*160*320*sizeof(DT));
        pwconv1[p] = (DT*)sds_alloc(64*160*320*sizeof(DT));
    }
    data_blob = (DT*)sds_alloc(32*323*643*sizeof(DT));
    dwconv1_blob = (DT*)sds_alloc(32*323*643*sizeof(DT));
    pwconv1_blob = (DT*)sds_alloc(64*323*643*sizeof(DT));
    parameter = (DT*)sds_alloc(440928*sizeof(DT));
    load_weight(parameter, 440928);
}

void SkyNet()
{
    for(int p=0; p<4; p++)
        load_fm(data[p], config[0]);
    stitch(data, data_blob, config[0]);
    SkyNet_(data_blob, pwconv1_blob, parameter);
    distitch(pwconv1_blob, pwconv1, config[2]);
    for(int p=0; p<4; p++)
        check_fm(pwconv1[p], config[2]);
}