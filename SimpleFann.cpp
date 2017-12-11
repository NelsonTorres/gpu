/*
 * SimpleFann.cpp
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#include "SimpleFann.h"

SimpleFann::SimpleFann(int num_layers, unsigned int* layersSize){
	this->num_layers = num_layers;

	this->layers; //= new layer*[num_layers];

	this->num_connections = 0;
	this->num_neurons=0;

	for(int i=1;i<num_layers;++i) {  num_neurons += layersSize[i];}

	this->layers[0].setNeurons(layersSize[0]);
	this->layerSizes[0] = layersSize[0];
	for(int i=1;i<num_layers;++i) {
		this->layerSizes[i] = layersSize[i];
		this->layers[i].setNeurons(layersSize[i]);
		this->layers[i].connection_start = num_connections;
		this->layers[i].connection_end = layersSize[i-1]*layersSize[i];
		num_connections += layersSize[i-1]*layersSize[i];
	}

/*
	this->connections = new float**[num_layers];

	for(int i=1;i< num_layers;++i){
		connections[i-1] = new float*[layers[i]->numNeurons];

		for(int j=0;j<layers[i]->numNeurons;++j){
			connections[i-1][j] = new float[layers[i-1]->numNeurons];

			for(int k=0;k< layers[i-1]->numNeurons;++k){
				this->connections[i-1][j][k] = rand();
			}
		}
	}*/

	for(int k=0;k< 290;++k){
		this->connections[k] = rand() / RAND_MAX;
	}

	//this->output = new float[num_output];
	this->MSE_value = 0;
	this->num_MSE = 0;
}




