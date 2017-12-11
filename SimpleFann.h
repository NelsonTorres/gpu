#ifndef __simple_fann_h__
#define __simple_fann_h__

#include <fenv.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <stdarg.h>

#include <vector>
#include "layer.h"

using namespace std;


class SimpleFann{
  public:

    layer layers[3];
    int layerSizes[3];
    unsigned int num_layers;
    unsigned int num_connections;
    unsigned int num_neurons;
    float connections[290];
    unsigned int total_connections = 290;
    float output[1];

    float MSE_value;
    unsigned int num_MSE;

    SimpleFann( int num_layers, unsigned int* layersSize);
    SimpleFann();
    void toString();
    long long unsigned int toBytes();
};


#endif
