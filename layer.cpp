/*
 * layer.cpp
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#include "layer.h"

layer::layer() {
	numNeurons = -1;
	connection_start =-1;
	connection_end = -1;
}

layer::layer(int numNeurons) {
  this->numNeurons = numNeurons;
	connection_start =-1;
	connection_end = -1;
  //this->neurons = new Neuron[numNeurons];
}

void layer::setNeurons(int numNeurons) {
  this->numNeurons = numNeurons;
  //this->neurons = new Neuron[numNeurons];
}

layer::~layer() {
	// TODO Auto-generated destructor stub
}

void layer::setActivation(int func){
	for(int i =0;i< numNeurons;++i){
		neurons[i].setActivation(func);
	}
}

