#include "CNN.h"

void stitch(DT* ifm[4], DT* ofm, layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*l.oh*l.ow + h*l.ow + w;
                    int ofm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    ofm[ofm_index] = ifm[p][ifm_index];
                }
            }
        }
    }
}

void distitch(DT* ifm, DT* ofm[4], layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                    ofm[p][ofm_index] = ifm[ifm_index];
                }
            }
        }
    }
}

void generate_fm(DT* fm, layer l)
{
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int fm_index = c*l.oh*l.ow + h*l.ow + w;
                fm[fm_index] = h + w;
            }
        }
    }
}

void check(DT* result, DT* golden, int len, layer l)
{
    int err = 0;
    for (int j = 0; j < len; j++)
    {
        if (((result[j] - golden[j]) > check_scale) || ((result[j] - golden[j]) < -check_scale))
        {
            err++;
            //printf("[%d] correct=%f,wrong=%f\n", j, tmp[j], fm[j]);
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);
}

void generate_weight(DT* weight, layer l)
{
    
}

void load_fm(DT* fm, layer l)
{
    char nstr[50];

    sprintf(nstr, "../blobs/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(fm, 1, l.ow*l.oh*l.oc * sizeof(DT), fp);
    fclose(fp);
}

void load_weight(DT *weight, int length)
{
    char nstr[50];
    sprintf(nstr, "../weights/SkyNet.wt");
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(DT), fp);
    fclose(fp);
}

void show_fm(DT* fm, layer l)
{
    for (int c=0;c<l.oc;c++)
    {
        for (int h=0;h<l.oh;h++)
        {
            for (int w=0;w<l.ow;w++)
            {
                int i = c*l.oh*l.ow + h*l.ow + w;
                std::cout << fm[i]<<", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void check_fm(DT* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    DT *tmp = (DT *)malloc(sizeof(DT)*len);

    char nstr[50];
    sprintf(nstr, "../blobs/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(DT), fp);
    fclose(fp);

    int err = 0;

    for (int j = 0; j < len; j++)
    {
        if (((fm[j] - tmp[j]) > check_scale) || ((fm[j] - tmp[j]) < -check_scale))
        {
            err++;
            //printf("[%d] correct=%f,wrong=%f\n", j, tmp[j], fm[j]);
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);

    free(tmp);
}