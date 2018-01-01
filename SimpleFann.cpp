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
		num_connections += layersSize[i-1]*layersSize[i];
	}

	int connection = 0;
	for(int i=0; i < num_layers -1 ;++i) {
	  for(int j=0; j < layers[i].numNeurons; ++j){
	    layers[i].neurons[j].setConnections(connection, connection + layers[i+1].numNeurons);
		connection += layers[i+1].numNeurons;
		}
	}



	this->MSE_value = 0;
	this->num_MSE = 0;
}




