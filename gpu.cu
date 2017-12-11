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

#include "fann.h"
#include "fann_data.h"
#include "fann_train.h"
#include "parallel_fann.h"
#include "SimpleFann.h"
#include "SimpleFannData.h"

using namespace std;


#define SIGMOID 0
#define GAUSS 1
#define TANH 2


void cudaHandler(cudaError_t error){
  if(error != cudaSuccess){
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
};



__device__
void fann_update_slopes_batch_gpu(SimpleFann *ann){
  float tmp_error;
  unsigned int i,j,k, num_connections;

  for(i = 1; i < ann->num_layers; i++){
    for(j = 0; j < ann->layers[i].numNeurons ; j++){
      for(k = 0; k < ann->layers[i-1].numNeurons; k++){
        ann->layers[i].neurons[j].slope += ann->layers[i].neurons[j].error * ann->layers[i-1].neurons[k].value;
      }
    }
  }
}


__device__
void fann_compute_MSE_gpu(SimpleFann *ann, float* desired_output){
  float neuron_diff, neuron_value;
  for(int i = 0; i< ann->layers[ann->num_layers -1].numNeurons; i++){
    neuron_value = ann->layers[ann->num_layers -1].neurons[i].value;
    neuron_diff = *desired_output - neuron_value;
    ann->MSE_value = neuron_diff * neuron_diff;
  }
}


__device__
float fann_activation_derived(unsigned int activation_function, float steepness, float value, float sum){
	switch (activation_function)
	{
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			value = fann_clip(value, 0.01f, 0.99f);
			return (2.0f * steepness * value * (1.0f - value));
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			value = fann_clip(value, -0.98f, 0.98f);
			return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
		case FANN_GAUSSIAN:
			return (2.0f * steepness * value * (1.0f - value));
		}
	return 0;
}


__device__
void fann_backpropagate_MSE_gpu(SimpleFann *ann){
  unsigned int i,j,k;
  float error_prev_layer;
  int neuron_index = 0;

  for(i = ann->num_layers - 1; i > 0; --i){
    for(j = 0; j < ann->layers[i].numNeurons; j++){
	  for(k = 0; k < ann->layers[i-1].numNeurons; k++){
	    error_prev_layer += ann->connections[ann->total_connections -  neuron_index];
	    neuron_index++;
	  }

    }

	for(j = 0; j < ann->layers[i].numNeurons; j++){
	  ann->layers[i].neurons[j].error *= fann_activation_derived(ann->layers[i].neurons[j].activation_function, ann->layers[i].neurons[j].activation_steepness, ann->layers[i].neurons[j].value, ann->layers[i].neurons[j].sum);
	}
  }
}


__device__
float activation_switch(int activation_function, float sum){
	//sigmoid
	switch (activation_function) {
		case SIGMOID:
			return (1.0f/(1.0f + exp(-2.0f * sum)));
		case TANH:
			return (2.0f/(1.0f + exp(-2.0f * sum)) - 1.0f);
		case GAUSS:
			return (exp(-sum * sum));
	}
	return -1;
};


__device__
void fann_run_gpu(SimpleFann*  ann, float* input, int num_input) {

  int i,j ,k ;
  float neuron_sum;
  float max_sum = 1;

  for(i = 0; i < num_input; i++){
	//input[i] = 0.0;
	//ann->layers[0].neurons[i].value = 0.0 ;
    ann->layers[0].neurons[i].value = input[i];
  }

  for(i = 1; i < ann->num_layers; i++) {

    for(j=0; j < ann->layers[i].numNeurons; j++) {
      neuron_sum = 0;

      for(k=0; k < ann->num_connections; k++){
		neuron_sum += ann->connections[k] * ann->layers[i-1].neurons[j].value;
      }
    }
    //neuron_sum = fann_mult(steepness, neuron_sum);

    //  max_sum = 150/steepness;
    if(neuron_sum > max_sum)
      neuron_sum = max_sum;
    else if(neuron_sum < -1*max_sum)
      neuron_sum = -1*max_sum;
      ann->layers[i].neurons[j].sum = neuron_sum;
      ann->layers[i].neurons[j].value = activation_switch(ann->layers[i].neurons[j].activation_function , neuron_sum);

    }

  int num_output = ann->layerSizes[ann->num_layers-1];
  for(i = 0; i < num_output; i++){
  	ann->output[i] = ann->layers[ann->num_layers-1].neurons[i].value;
  }
}

__device__
void run( SimpleFann*  ann, float* input, float* output,int  num_input, int  num_output){
	fann_run_gpu( ann, input, num_input);

    fann_compute_MSE_gpu(ann, output);

    fann_backpropagate_MSE_gpu(ann);

    fann_update_slopes_batch_gpu(ann);

}



__global__
void fann_train_epoch_irpropm_parallel_gpu(SimpleFann*  ann, SimpleFannData* data){
  //TODO clean arrays
  //unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;


  int i,j,k, num_threads;
  float delta_zero = 0;
/*  for( i=0;i < size ;++i){
    train_slopes[i] = 0;
    prev_steps[i] = delta_zero;
    prev_train_slopes[i] = 0;
  }
*/
  //ann->connections[0] = 2;
  //TODO uncomment run
  #pragma omp for schedule(static)
  for(i=0;i < data->num_data;++i){
    run(ann, data->input+(i*data->num_input) ,data->output + (i * data->num_output) , data->num_input, data->num_output);
  }


  float  next_step;
  float same_sign;
  const float increase_factor = 1.2;       //1.2;
  const float decrease_factor = 0.5;       //0.5;
  const float delta_min = 0;   //0.0;
  const float delta_max = 50;   //50.0;
  const unsigned int first_weight = 0;

  #pragma omp parallel private(next_step)
  {
    #pragma omp for schedule(static)
    for(i=1; i < ann->num_layers; i++){
      for(j=0; j < ann->layers[i].numNeurons; j++){
  	    same_sign = ann->layers[i].neurons[j].prev_step * ann->layers[i].neurons[j].slope;

    	  if(same_sign >= 0.0)
    		next_step = ann->layers[i].neurons[j].prev_step * increase_factor < delta_max ? ann->layers[i].neurons[j].prev_step * increase_factor :  delta_max;
    	  else{
    		next_step = ann->layers[i].neurons[j].prev_step * decrease_factor > delta_min ? ann->layers[i].neurons[j].prev_step * decrease_factor : delta_min;
    		ann->layers[i].neurons[j].slope = 0;
    	  }


    	if(ann->layers[i].neurons[j].slope < 0){
    	  for(k =ann->layers[i-1].connection_start; k < ann->layers[i-1].connection_end; ++k){
    		ann->connections[k] -= next_step;
    	    if(ann->connections[k] < -1500)
    	      ann->connections[k] = -1500;
    	  }
    	}else{
      	  for(k =ann->layers[i-1].connection_start; k < ann->layers[i-1].connection_end; ++k){
    	    ann->connections[k] += next_step;
    	    if(ann->connections[k] > 1500)
    	      ann->connections[k] = 1500;
    	  }
    	}

    	ann->layers[i].neurons[j].prev_step = next_step;
    	ann->layers[i].neurons[j].prev_slope = ann->layers[i].neurons[j].slope;
    	ann->layers[i].neurons[j].slope = 0.0;
    	}
      }
    }

  //sync
  //merge of MSEs
  /*
  for(i=0;i<num_threads;++i){
 	//TODO += ann_vect[i]
    ann->MSE_value += ann->MSE_value;
    ann->num_MSE += ann->num_MSE;
  }*/

};


void execute(SimpleFann* d_ann, SimpleFannData* d_data){
  cout << "Executing\n";
  //cout << ann.connections[0][1][1]<<"\n";

  fann_train_epoch_irpropm_parallel_gpu<<<1,1>>>(d_ann, d_data);
  cudaError_t  error = cudaGetLastError();
  cudaHandler(error);
  cudaDeviceSynchronize();

  //cout << ann.connections[0][1][1]<<"\n";
}


using namespace std;


typedef std::chrono::high_resolution_clock Clock;
//using std::chrono::high_resolution_clock;

int main(int argc, char** argv){

	//string train = "/home/a65445/tese/DBN/datasets/HIGGS_SUSY_FINAL.csv";
	string train = "/home/nelson/Desktop/tese/DBN/datasets/train1.csv";

	const unsigned int num_input = 28;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	size_t free, total;
	const unsigned int num_neurons_hidden = stoi(argv[1]);

	unsigned int neurons[3] = {num_input, num_neurons_hidden, num_output};




    ofstream out;
	out.open(getenv("OUT"));
    int num_threads = stoi(argv[2]);
	srand(0);
	printf("Creating network.\n");
	SimpleFann ann(num_layers, neurons);
	SimpleFannData data = SimpleFannData(train.c_str());

	ann.layers[1].setActivation(SIGMOID);
	ann.layers[2].setActivation(GAUSS);

//	cout << ann.connections[0]<< "\n";
//	cout << data.input[0]<< "\n";

	/*cudaMemGetInfo 	(&free, &total);
	cout << "free: "<< free << " total: " << total << "\n";
	cout << "free: "<< free - sizeof(ann) - sizeof(data) << " total: "<< total<<"\n";
*/


	// TODO copy to device
	SimpleFann *d_ann;
	SimpleFannData *d_data;
	cudaMalloc((void**) &d_ann, sizeof(SimpleFann));
	cudaMalloc((void**) &d_data, sizeof(SimpleFannData));
	cudaMemcpy(d_ann, &ann, sizeof(ann), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, &data, sizeof(data), cudaMemcpyHostToDevice);

	execute(d_ann, d_data);
	// TODO copy to host

	cudaMemcpy(&ann, d_ann, sizeof(SimpleFann), cudaMemcpyDeviceToHost);
	cudaMemcpy(&data, d_data, sizeof(SimpleFannData), cudaMemcpyDeviceToHost);

//	cout << ann.connections[0]<< "\n";
//	cout << data.input[0]<< "\n";


/*	Clock::time_point t_train_init = Clock::now();
/*	printf("Training network.\n");
        for(int k=0;k<25000;++k) {
          //fann_train_epoch_irpropm_parallel_gpu(struct fann **ann, struct fann_train_data *data, int data_to_process)
	  //fann_train_epoch_irpropm_parallel_gpu((struct fann **)d_ann_vec,(struct fann_train_data *)d_data_train, data_to_process);
	}
	cout << error <<endl;

	Clock::time_point t_train_end = Clock::now();
	chrono::duration<double, std::milli> time_span = chrono::duration_cast<chrono::duration<double, std::milli>>(t_train_end - t_train_init);
	cout <<"TRAIN: "<< time_span.count() <<endl;
	printf("Testing network.");
	Clock::time_point t_class_init = Clock::now();
	for(i = 0; i < fann_length_train_data(data_validation); i++){
	//	calc_out = fann_run(ann, data_validation->input[i]);
	//	out << calc_out[0]<< " " << data_validation->output[i][0]<< endl;
	}

	Clock::time_point t_class_end = Clock::now();
	time_span = chrono::duration_cast<chrono::duration<double, milli>>(t_class_end - t_class_init);
	cout << "CLASSIFICATION:" << time_span.count() << endl;

	fann_destroy_train(data_train);
	fann_destroy_train(data_validation);
	fann_destroy(ann);
*/
	cout << "main : over";

//	cudaFree(d_ann);
//	cudaFree(d_data);
	return 0;
}



