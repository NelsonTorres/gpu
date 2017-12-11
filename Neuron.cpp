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
	prev_step = 0;
	activation_function = 0;
	activation_steepness = 0;

}



Neuron::~Neuron() {
	// TODO Auto-generated destructor stub
}

void Neuron::setActivation(int func){
	this->activation_function = func;
	error = 0;
	slope = 0;
	prev_step = 0.0001;
}
