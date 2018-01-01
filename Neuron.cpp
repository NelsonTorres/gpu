/*
 * Neuron.cpp
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#include "Neuron.h"

Neuron::Neuron() {
	error = 0;
	slope = 0;
	sum = 0;
	value = 0;
	prev_slope = 0;
	prev_step = 0.0001;
	activation_function = 0;
	activation_steepness = 1;
	for(int i=0;i<64 ;++i){
		connections[i] = (rand()%100) / 50;
	}
}



Neuron::~Neuron() {
}

void Neuron::setActivation(int func){
	this->activation_function = func;
	error = 0;
	slope = 0;
	prev_step = 0.0001;
}

void Neuron::setConnections(int start, int end){

}
