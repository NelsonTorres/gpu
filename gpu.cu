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


#include "SimpleFann.h"
#include "SimpleFannData.h"

using namespace std;


#define SIGMOID 0
#define GAUSSIAN 1
#define TANH 2


void cudaHandler(cudaError_t error){
  if(error != cudaSuccess){
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
};


__device__ __host__
float clip (float value, float lowerLimit, float upperLimit){
	if(value < lowerLimit)
		value = lowerLimit;
	else if (value > upperLimit)
		value = upperLimit;
	return value;
}


__device__ __host__
float activation_derived(unsigned int activation_function, float steepness, float value, float sum){
	switch (activation_function)
	{
		case SIGMOID:
			clip(value,0.01f,0.99f);
			return (2.0f * value * (1.0f - value));
		/*case SIGMOID_SYMMETRIC:
			value = fann_clip(value, -0.98f, 0.98f);
			return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
		*/
		case GAUSSIAN:
			return (2.0f * steepness * value * (1.0f - value));
		}
	return 0;
}


 __device__
void fann_update_slopes_batch(SimpleFann *ann){
  unsigned int i,j,k;
  int stride =	blockDim.x;
  int tid =	threadIdx.x + blockIdx.x * blockDim.x;
  for(i = 0; i < ann->num_layers-1; i++){
    for(j = tid; j < ann->layers[i].numNeurons ; j+=stride){
      for(k = 0; k < ann->layers[i+1].numNeurons; k++){
    	  ann->layers[i].neurons[j].slope += ann->layers[i+1].neurons[k].error * ann->layers[i].neurons[j].value;
      }
    }
  }
}



__host__ __device__
void fann_compute_MSE(SimpleFann *ann, float* desired_output){
	float neuron_diff, neuron_value;
	  for(int i = 0; i< ann->layers[ann->num_layers -1].numNeurons; i++){
	    neuron_value = ann->layers[ann->num_layers -1].neurons[i].value;
	    neuron_diff = desired_output[0] - neuron_value;
	    //update mse used for tanh (SYMMETRIC)
	    //		neuron_diff = fann_update_MSE(ann, last_layer_begin, neuron_diff);
	    if(neuron_diff < -.9999999)
	      neuron_diff = -17.0;
	    else if(neuron_diff > .9999999)
	      neuron_diff = 17.0;
	    else
	      neuron_diff = (float) log((1.0 + neuron_diff) / (1.0 - neuron_diff));

	    ann->layers[ann->num_layers -1].neurons[i].error = activation_derived(ann->layers[ann->num_layers -1].neurons[i].activation_function,
	    													ann->layers[ann->num_layers -1].neurons[i].activation_steepness, ann->layers[ann->num_layers -1].neurons[i].value,
	    													ann->layers[ann->num_layers -1].neurons[i].sum) * neuron_diff;
	  }
}





 __device__
void fann_backpropagate_MSE(SimpleFann *ann){
  int tid =	threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x;

  int i,j,k;
  float act;

  for(i = ann->num_layers - 2; i > -1; --i){
	for(j = tid; j < ann->layers[i].numNeurons; j+=stride){
      ann->layers[i].neurons[j].error  = 0;
	  for(k = 0; k < ann->layers[i].numNeurons; k++){
		ann->layers[i].neurons[j].error  += ann->layers[i+1].neurons[k].error * ann->layers[i].neurons[j].connections[k];
	  }
    }

	for(j = 0; j < ann->layers[i].numNeurons; j++){
	  act = activation_derived(ann->layers[i].neurons[j].activation_function, ann->layers[i].neurons[j].activation_steepness, ann->layers[i].neurons[j].value, ann->layers[i].neurons[j].sum);
	  ann->layers[i].neurons[j].error *= act;
	}
  }
}


__device__ __host__
float activation_switch(int activation_function, float sum){
  float ret = -1;
  switch (activation_function) {
	case SIGMOID:
	  ret =  (1.0f/(1.0f + exp(-2.0f * sum)));
	  break;
	case TANH:
	  ret =  (2.0f/(1.0f + exp(-2.0f * sum)) - 1.0f);
	  break;
	case GAUSSIAN:
	  ret =  (exp(-sum * sum));
	  break;
	}
  return ret;
};

__host__
void fann_run_cpu(SimpleFann*  ann, float* input, int num_input) {

  int i,j ,k;
  float max_sum = 150;
  float neuron_sum;
  for(i = 0; i < num_input; i++){
    ann->layers[0].neurons[i].value = input[i];
  }

  for(i = 1; i < ann->num_layers; i++) {
    for(j = 0; j < ann->layers[i].numNeurons; j++) {
      neuron_sum = 0;
      for(k = 0; k < ann->layers[i-1].numNeurons; k++){
    	neuron_sum += ann->layers[i-1].neurons[k].connections[j] * ann->layers[i-1].neurons[k].value;
      }
      //neuron_sum += ann->layers[i-1].neurons[k].connections[j] * ann->layers[i-1].bias.value;
      neuron_sum = clip(neuron_sum, -1*max_sum, max_sum);

      ann->layers[i].neurons[j].sum = neuron_sum;
      ann->layers[i].neurons[j].value = activation_switch(ann->layers[i].neurons[j].activation_function , neuron_sum);
    }
  }
}

__device__
void fann_run(SimpleFann*  ann, float* input, int num_input) {

  int tid =	threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x;

  int i,j ,k;
  float max_sum = 150;
  for(i = 0; i < num_input; i++){
    ann->layers[0].neurons[i].value = input[i];
  }

  for(i = 1; i < ann->num_layers; i++) {
   for(j = tid; j < ann->layers[i].numNeurons; j+=stride) {
      float neuron_sum = 0;
      for(k = 0; k < ann->layers[i-1].numNeurons; k++){
    	neuron_sum += ann->layers[i-1].neurons[k].connections[j] * ann->layers[i-1].neurons[k].value;
      }
      neuron_sum = clip(neuron_sum, -1*max_sum, max_sum);

      ann->layers[i].neurons[j].sum = neuron_sum;
      ann->layers[i].neurons[j].value = activation_switch(ann->layers[i].neurons[j].activation_function , neuron_sum);
    }
  }
}


__device__
void run( SimpleFann*  ann, float* input, float* output,int  num_input, int  num_output){
	fann_run( ann, input, num_input);
    fann_compute_MSE(ann, output);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann);
}

__device__
void update_weights(SimpleFann* ann, SimpleFannData* data){
  int tid =	threadIdx.x + blockIdx.x * blockDim.x;
  int stride =	blockDim.x;
  int i,j, k;
  float  next_step;
  float same_sign;
  const float increase_factor = 1.2;
  const float decrease_factor = 0.5;
  const float delta_min = 0;
  const float delta_max = 50;


  for(i=0; i < ann->num_layers-1; i++){
    for(j=tid; j < ann->layers[i].numNeurons; j+=stride){
	  same_sign = ann->layers[i].neurons[j].prev_slope * ann->layers[i].neurons[j].slope;
	  if(same_sign >= 0.0)
	    next_step = ann->layers[i].neurons[j].prev_step * increase_factor < delta_max ? ann->layers[i].neurons[j].prev_step * increase_factor :  delta_max;
	  else{
	    next_step = ann->layers[i].neurons[j].prev_step * decrease_factor > delta_min ? ann->layers[i].neurons[j].prev_step * decrease_factor : delta_min;
	    ann->layers[i].neurons[j].slope = 0;
	  }

	  if(ann->layers[i].neurons[j].slope < 0){
	    for(k = 0; k < ann->layers[i].numNeurons; ++k){
	      ann->layers[i].neurons[j].connections[k] -= next_step;
	      if(ann->layers[i].neurons[j].connections[k] < -1500)
	    	ann->layers[i].neurons[j].connections[k] = -1500;
	    }
	  }else{
	    for(k = 0; k < ann->layers[i].numNeurons; ++k){
	      ann->layers[i].neurons[j].connections[k] += next_step;
	      if(ann->layers[i].neurons[j].connections[k] > 1500)
	        ann->layers[i].neurons[j].connections[k] = 1500;
	    }
	  }

	  ann->layers[i].neurons[j].prev_step = next_step;
	  ann->layers[i].neurons[j].prev_slope = ann->layers[i].neurons[j].slope;
	  ann->layers[i].neurons[j].slope = 0;
	}
  }
}

__global__
void fann_train_epoch_irpropm_parallel(SimpleFann*  ann, SimpleFannData* data){

  int i;
  for(i=0;i < data->num_data;++i){
    run(ann, data->input+(i*data->num_input) ,data->output + (i * data->num_output) , data->num_input, data->num_output);
  }
  update_weights(ann,data);

};

void execute(SimpleFann* d_ann, SimpleFannData* d_data, unsigned int num_threads){

  fann_train_epoch_irpropm_parallel<<<1,num_threads>>>(d_ann, d_data);
  cudaError_t  error = cudaGetLastError();
  cudaHandler(error);
  cudaDeviceSynchronize();

}

using namespace std;
typedef std::chrono::high_resolution_clock Clock;
//using std::chrono::high_resolution_clock;

int main(int argc, char** argv){

	//string train = "/home/nelson/Desktop/tese/DBN/datasets/train1.csv";
	string train = "/home/a65445/tese/DBN/datasets/HIGGS_2_train.csv"
	const unsigned int num_input = 597;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = stoi(argv[1]);

	unsigned int neurons[3] = {num_input, num_neurons_hidden, num_output};

    int num_threads = stoi(argv[2]);
    int epochs = stoi(argv[3]);
	srand(0);
	SimpleFann ann(num_layers, neurons);
	SimpleFannData data = SimpleFannData(train.c_str());

	ann.layers[0].setActivation(SIGMOID);
	ann.layers[1].setActivation(SIGMOID);
	ann.layers[2].setActivation(GAUSSIAN);

	SimpleFann *d_ann;
	SimpleFannData *d_data;
	cudaMalloc((void**) &d_ann, sizeof(SimpleFann));
	cudaMalloc((void**) &d_data, sizeof(SimpleFannData));
	cudaMemcpy(d_ann, &ann, sizeof(ann), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, &data, sizeof(data), cudaMemcpyHostToDevice);

	for(int i = 0 ;i<epochs;++i){
	  execute(d_ann, d_data, num_threads);
	  //fann_train_epoch_irpropm_parallel(&ann,&data);
	  //update_weights(&ann, &data);
	}

	cudaMemcpy(&ann, d_ann, sizeof(SimpleFann), cudaMemcpyDeviceToHost);
	cudaMemcpy(&data, d_data, sizeof(SimpleFannData), cudaMemcpyDeviceToHost);


	for(int i = 0; i < data.num_data ; i++){
	  fann_run_cpu(&ann, data.input+(i*num_input), data.num_input);
	  cout << ann.layers[ann.num_layers - 1].neurons[0].value << " " << data.output[i * data.num_output]<< endl;
	}

	cudaFree(d_ann);
	cudaFree(d_data);


	return 0;
}



