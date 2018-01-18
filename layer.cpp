/*
 * layer.cpp
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#include "layer.h"

layer::layer() {
	numNeurons = -1;
//	bias.value = (rand()%10 )/ 10;
}

layer::layer(int numNeurons) {
  this->numNeurons = numNeurons;
}

void layer::setNeurons(int numNeurons) {
  this->numNeurons = numNeurons;
}

layer::~layer() {}

void layer::setActivation(int func){
  for(int i =0;i< numNeurons;++i){
	neurons[i].setActivation(func);
  }
}

