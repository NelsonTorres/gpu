#include <fenv.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <stdarg.h>
#include <string>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cudaProfiler.h>

#include "SimpleFann.h"
#include "SimpleFannData.h"

using namespace std;


#define SIGMOID 0
#define GAUSSIAN 1
#define TANH 2


void cudaHandler(cudaError_t error) {
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
};


__device__ __host__
float clip(float value, float lowerLimit, float upperLimit) {
  if (value < lowerLimit)
    value = lowerLimit;
  else if (value > upperLimit)
    value = upperLimit;
  return value;
}


__device__ __host__
float activation_derived(unsigned int activation_function, float value) {
  switch (activation_function) {
    case SIGMOID:
      clip(value, 0.01f, 0.99f);
      return (2.0f * value * (1.0f - value));
      /*case SIGMOID_SYMMETRIC:
        value = fann_clip(value, -0.98f, 0.98f);
        return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
      */
    case GAUSSIAN:
      return (2.0f * value * (1.0f - value));
  }
  return 0;
}

__device__
void update_slopes(float * s1, float * e1, float * output, int N1) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  unsigned i;
  for (i = 0; i < N1; i++) {
    s1[i * stride + tid] += e1[i * stride + tid ] * output[i];
  }
}


__device__
float compute_MSE(float output, float * output_desired,int  function) {
  float diff;
    diff = output_desired[0] - output;
    //update mse used for tanh (SYMMETRIC)
    //		neuron_diff = fann_update_MSE(ann, last_layer_begin, neuron_diff);
    if (diff < -.9999999)
      diff = -17.0;
    else if (diff > .9999999)
      diff = 17.0;
    else
      diff = (float) log((1.0 + diff) / (1.0 - diff));
  return activation_derived(function, output) * diff;
}


__device__
void backpropagate( float* w,float* e2, float* e3,int function,int  N2,int N3, float * input) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int i, j;
  for (i = 0; i < N2; ++i) {
    e2[(i * stride) + tid ] = 0;
    for (j = 0; j < N3; j++) {
      e2[i * stride + tid ] += e3[j * stride + tid ] * w[j + i * N2];
    }
    e2[i * stride + tid ] *= activation_derived(function, input[i]);
  }
}


__device__ __host__
float activation_switch(int activation_function, float sum) {
  float ret = -1;
  switch (activation_function) {
    case SIGMOID:
      ret = (1.0f / (1.0f + exp(-2.0f * sum)));
      break;
    case TANH:
      ret = (2.0f / (1.0f + exp(-2.0f * sum)) - 1.0f);
      break;
    case GAUSSIAN:
      ret = (exp(-sum * sum));
      break;
  }
  return ret;
};


__device__ __host__
float* fpropagate(float *w1, int function, float * input, int N1,int  N2, float* output) {
  int i, j;
  float max_sum = 150;
  for (i = 0; i < N2 ; i++) {
    float neuron_sum = 0;
    for (j = 0; j < N1; j++) {
      neuron_sum += w1[j + (i * N2)] * input[j];
    }
    neuron_sum = clip(neuron_sum, -1 * max_sum, max_sum);
    output[i] = activation_switch(function, neuron_sum);
  }
  return output;

}


__device__
void update_weights(float *w,float* e,float* s,float* prevSlopes1,float* prevSteps1,int functions,int N1, int N2) {
  int i, k;
  float next_step;
  float same_sign;
  const float increase_factor = 1.2;
  const float decrease_factor = 0.5;
  const float delta_min = 0;
  const float delta_max = 50;
  float result;
  int signal, max;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (i = tid; i < N1; i+=stride) {
    same_sign = prevSlopes1[i] * s[i];
    if (same_sign >= 0.0)
      next_step = prevSteps1[i] * increase_factor < delta_max ? prevSteps1[i] * increase_factor : delta_max;
      else {
      next_step = prevSteps1[i] * decrease_factor > delta_min ? prevSteps1[i] * decrease_factor : delta_min;
      s[i] = 0;
    }

    signal = (s[i] < 0) ? -1 : 1;
    next_step = abs(next_step) * signal;
    max = 1500 * signal;
    for (k = 0; k < N2; ++k) {
      result = w [k + (i * N1) ] + next_step;
      w [k + (i * N1) ] = result  < max ? max: result;
    }
    prevSteps1[i] = next_step;
    prevSlopes1[i] = s[i];
    s[i] = 0;
  }
}

//TODO verify final output
//TODO verify stride value
__device__
void reduce_slopes(float* s, int N1) {
  //source https://stackoverflow.com/questions/5293139/cuda-multiple-kernels-to-compute-a-single-value
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < N1; i+=stride) {
    for (int k = 0; k < stride; k++) {
      s[0] += s[k];
      s[k] = 0;
    }
  }
}

__global__
void train(float* w1,float* e1,float* s1,float* prevSlopes1, float* prevSteps1, float* w2, float* e2,float* s2,float* prevSlopes2, float* prevSteps2,  int* functions, const int num_input ,const int num_hidden_neurons, const int num_output, float* input, float* output, unsigned num_data) {
  int i;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float e3;
  float* output1 = new float[num_hidden_neurons], *output2 = new float[num_output];
  
  for (i = tid; i < num_data; i += stride) {
    output1 = fpropagate(w1, functions[0], input + (i *num_input), num_input, num_hidden_neurons, output1);
    output2 = fpropagate(w2, functions[1], output1, num_hidden_neurons, num_output, output2);
    e3 = compute_MSE(output2[0], output + (i * num_output), functions[2]);
    backpropagate(w2, e2, &e3, functions[1], num_hidden_neurons, num_output, input + (i *num_input) );
    backpropagate(w1, e1, e2, functions[0], num_input, num_hidden_neurons,  input + (i *num_input)    );
    update_slopes(s1, e1, output1, num_input);
    update_slopes(s2, e2, output2, num_hidden_neurons);
  }

  //TODO single
  __syncthreads();
  reduce_slopes(s1, num_input);
  reduce_slopes(s2, num_hidden_neurons);
  __syncthreads();

  update_weights(w1, e1, s1, prevSlopes1, prevSteps1, functions[0], num_input, num_hidden_neurons);
  update_weights(w2, e2, s2, prevSlopes2, prevSteps2, functions[1], num_hidden_neurons, num_output);
};

void executeKernel(unsigned num_threads, float* w1, float* e1, float* s1, float* prevSlopes1, float* prevSteps1 , float* w2, float* e2, float* s2, float* prevSlopes2, float* prevSteps2, int* functions, const int num_input, const int num_hidden_neurons, const int num_output, float* input, float* output, unsigned num_data ) {
  int maxBlockSize = 1024;
  int blocks = num_threads / maxBlockSize + 1;
  num_threads = num_threads / blocks * 1.0; //WARN 513/2 = 256.5 -> 256
  train<<< blocks, num_threads >>> (w1,  e1,  s1, prevSlopes1, prevSteps1, w2, e2, s2, prevSlopes2, prevSteps2, functions, num_input, num_hidden_neurons, num_output, input, output ,num_data);
  cudaError_t error = cudaGetLastError();
  cudaHandler(error);
  cudaDeviceSynchronize();
}

using namespace std;
typedef std::chrono::high_resolution_clock Clock;
//using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {

  string train = "/home/a65445/tese/DBN/datasets/train1.csv";
  const unsigned int num_input = 18;
  const unsigned int num_output = 1;
  const unsigned int num_layers = 3;
  const unsigned int num_neurons_hidden = stoi(argv[1]);
  
  int num_threads = stoi(argv[2]);
  int epochs = stoi(argv[3]);
  int i;

  srand(0);

  unsigned numNeurons = num_input + num_neurons_hidden + num_output;
  unsigned w1Size = num_input * num_neurons_hidden;
  unsigned w2Size = num_neurons_hidden * num_output;
  unsigned e1Size = num_threads * num_input;
  unsigned e2Size = num_threads * num_neurons_hidden;

  float w1[w1Size];
  float w2[w2Size];
  int functions[num_layers];

  functions[0] = SIGMOID;
  functions[1] = SIGMOID;
  functions[2] = GAUSSIAN;

  float e1[e1Size];
  float e2[e2Size];

  float s1[e1Size];
  float s2[e2Size];

  float prevSteps1[num_input];
  float prevSteps2[num_neurons_hidden];

  float prevSlopes1[num_input];
  float prevSlopes2[num_neurons_hidden];

  for (i = 0; i <w1Size; i++) {
    w1[i] = (rand()%100) / 50;
  }

  for (i = 0; i < w2Size; i++) {
    w2[i] = (rand()%100) / 50;
  }

  for (i = 0; i < e1Size; i++) {
    e1[i] = 0;
    s1[i] = 0;
  }

  for (i = 0; i < e2Size; i++) {
    e2[i] = 0;
    s2[i] = 0;
  }

  for(i = 0; i < num_input;++i){
    prevSlopes1[i] = 0;
    prevSteps1[i] = 0.0001;
  }

  for(i = 0; i <num_neurons_hidden;++i){
    prevSlopes2[i] = 0;
    prevSteps2[i] = 0.0001;
  }

  float *input = NULL, *output = NULL;
  unsigned num_data;
  num_data = readData(train.c_str(),input, output);

  float *d_w1, *d_w2, *d_input, *d_output, *d_e1, *d_e2, *d_s1, *d_s2, *d_prevSlopes1, *d_prevSlopes2, *d_prevSteps1, *d_prevSteps2;
  int* d_functions;

  cudaMalloc((void **) &d_w1, w1Size * sizeof(float));
  cudaMalloc((void **) &d_w2, w2Size * sizeof(float));
  cudaMalloc((void **) &d_input, num_data * num_input * sizeof(float));
  cudaMalloc((void **) &d_output, num_data * num_output * sizeof(float));
  cudaMalloc((void **) &d_e1, e1Size * sizeof(float));
  cudaMalloc((void **) &d_e2, e2Size * sizeof(float));
  cudaMalloc((void **) &d_s1, e1Size * sizeof(float));
  cudaMalloc((void **) &d_s2, e2Size * sizeof(float));
  cudaMalloc((void **) &d_functions, num_layers * sizeof(int));
  cudaMalloc((void **) &d_prevSlopes1, e1Size * sizeof(float));
  cudaMalloc((void **) &d_prevSlopes2, e2Size * sizeof(float));
  cudaMalloc((void **) &d_prevSteps1, e1Size * sizeof(float));
  cudaMalloc((void **) &d_prevSteps2, e2Size * sizeof(float));

  cudaMemcpy(d_w1, &w1, w1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w2, &w2, w2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_e1, &e1, e1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_e2, &e2, e2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s1, &s1, e1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s2, &s2, e2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_functions, &functions, num_layers * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prevSlopes1, &prevSlopes1, e1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prevSlopes2, &prevSlopes2, e2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prevSteps1, &prevSteps1, e1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prevSteps1, &prevSteps1, e2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, &input, num_input * num_data * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, &output,  num_output * num_data * sizeof(float), cudaMemcpyHostToDevice);

  for(i = 0; i < epochs; ++i) {
    cout << i << "\n";
    executeKernel(num_threads, d_w1, d_e1, d_s1, d_prevSlopes1, d_prevSteps1, d_w2, d_e2, d_s2, d_prevSlopes2, d_prevSteps2, d_functions, num_input, num_neurons_hidden, num_output, d_input, d_output, num_data);
  }

  cudaMemcpy(w1, d_w1, w1Size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(w2, d_w2, w2Size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(input, d_input, num_input * num_data * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(output, d_output,  num_output * num_data * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(e1, d_e1, e1Size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(e2, d_e2, e2Size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(s1, d_s1, e1Size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(s2, d_s2, e2Size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(functions, d_functions, num_layers * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(prevSlopes1, d_prevSlopes1, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(prevSlopes2, d_prevSlopes2, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(prevSteps1, d_prevSteps1, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(prevSteps2, d_prevSteps2, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);


  float output1[num_neurons_hidden], output2[num_output];
  for (i = 0; i < num_data; i++) {
    fpropagate(w2, functions[1],fpropagate(w1, functions[0], input+ (i * num_input), num_input, num_neurons_hidden, output1),num_neurons_hidden,num_output,output2);
    cout << output2[0] << " " << output[i] << endl;
  }

  cudaFree(&d_w1);
  cudaFree(&d_w2);
  cudaFree(&d_input);
  cudaFree(&d_output);
  cudaFree(&d_e1);
  cudaFree(&d_e2);
  cudaFree(&d_s1);
  cudaFree(&d_s2);
  cudaFree(&d_functions);
  cudaFree(&d_prevSlopes1);
  cudaFree(&d_prevSlopes2);
  cudaFree(&d_prevSteps1);
  cudaFree(&d_prevSteps2);

  cudaDeviceReset();

  return 0;
}



