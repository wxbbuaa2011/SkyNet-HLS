#include "CNN.h"

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