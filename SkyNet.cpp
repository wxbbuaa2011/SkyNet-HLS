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

DT* data_blob;
DT* parameter;
DT* blob[2];
DT* reorg_blob;

void SkyNet_init()
{
    parameter = (DT*)sds_alloc(438728*sizeof(DT));
    blob[0] = (DT*)sds_alloc(2457600*sizeof(DT));
    blob[1] = (DT*)sds_alloc(2457600*sizeof(DT));
    reorg_blob = (DT*)sds_alloc(config[9].oc*config[9].oh*config[9].ow*sizeof(DT));
    load_weight(parameter, 438728);
}
void SkyNet()
{
    int input = 0;
    load_fm(blob[input], config[0]);
  
    /*******dwconv1*******/
    int weight_offset = 0;
    int bias_offset = weight_offset + config[1].oc*config[1].k*config[1].k;
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[1]);
    //check_fm(blob[1-input], config[1]);
    input = 1 - input;
    /*******pwconv1*******/
    weight_offset = bias_offset + config[1].oc;
    bias_offset = weight_offset + config[2].oc*config[2].ic*config[2].k*config[2].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[2]);
    //check_fm(blob[1-input], config[2]);
    input = 1 - input;
    /*******pool1*******/
    maxpool(blob[input], blob[1-input], config[3]);
    //check_fm(blob[1-input], config[3]);
    input = 1 - input;

    /*******dwconv2*******/
    weight_offset = bias_offset + config[2].oc;
    bias_offset = weight_offset + config[4].oc*config[4].k*config[4].k;
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[4]);
    //check_fm(blob[1-input], config[4]);
    input = 1 - input;
    /*******pwconv2*******/
    weight_offset = bias_offset + config[4].oc;
    bias_offset = weight_offset + config[5].oc*config[5].ic*config[5].k*config[5].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[5]);
    //check_fm(blob[1-input], config[5]);
    input = 1 - input;
    /*******pool2*******/
    maxpool(blob[input], blob[1-input], config[6]);
    //check_fm(blob[1-input], config[6]);
    input = 1 - input;

    /*******dwconv3*******/
    weight_offset = bias_offset + config[5].oc;
    bias_offset = weight_offset + config[7].oc*config[7].k*config[7].k;
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[7]);
    //check_fm(blob[1-input], config[7]);
    input = 1 - input;
    /*******pwconv3*******/
    weight_offset = bias_offset + config[7].oc;
    bias_offset = weight_offset + config[8].oc*config[8].ic*config[8].k*config[8].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[8]);
    check_fm(blob[1-input], config[8]);
    input = 1 - input;

    /*******reorg*******/
    reorg(blob[input],reorg_blob,config[9]);
    check_fm(reorg_blob, config[9]);

    /*******pool3*******/
    maxpool(blob[input], blob[1-input], config[10]);
    //check_fm(blob[1-input], config[10]);
    input = 1 - input;

    /*******dwconv4*******/
    weight_offset = bias_offset + config[8].oc;
    bias_offset = weight_offset + config[11].oc*config[11].k*config[11].k;
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[11]);
    //check_fm(blob[1-input], config[11]);
    input = 1 - input;
    /*******pwconv4*******/
    weight_offset = bias_offset + config[11].oc;
    bias_offset = weight_offset + config[12].oc*config[12].ic*config[12].k*config[12].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[12]);
    //check_fm(blob[1-input], config[12]);
    input = 1 - input;

    /*******dwconv5*******/
    weight_offset = bias_offset + config[12].oc;
    bias_offset = weight_offset + config[13].oc*config[13].k*config[13].k;
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[13]);
    //check_fm(blob[1-input], config[13]);
    input = 1 - input;
    /*******pwconv5*******/
    weight_offset = bias_offset + config[13].oc;
    bias_offset = weight_offset + config[14].oc*config[14].ic*config[14].k*config[14].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[14]);
    check_fm(blob[1-input], config[14]);
    input = 1 - input;

    /*******concat*******/
    concat(reorg_blob, blob[input], blob[1-input], config[9], config[14]);
    check_fm(blob[1-input], config[15]);
    input = 1 - input;

    /*******dwconv6*******/
    weight_offset = bias_offset + config[14].oc;
    bias_offset = weight_offset + config[16].oc*config[16].k*config[16].k;
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[16]);
    //check_fm(blob[1-input], config[16]);
    input = 1 - input;
    /*******pwconv6*******/
    weight_offset = bias_offset + config[16].oc;
    bias_offset = weight_offset + config[17].oc*config[17].ic*config[17].k*config[17].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[17]);
    //check_fm(blob[1-input], config[17]);
    input = 1 - input;
    /*******conv7*******/
    weight_offset = bias_offset + config[17].oc;
    bias_offset = weight_offset + config[18].oc*config[18].ic*config[18].k*config[18].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 0, config[18]);
    check_fm(blob[1-input], config[18]);
    input = 1 - input;
}