/*
 * Neuron.h
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <stdlib.h>

class Neuron {
public:
	float sum[32];
	float value[32];
	float activation_steepness;
	int activation_function;
	float error[32];
	float slope[32];
	float prev_step;
	float prev_slope;
	float connections[597];
	Neuron();
	void setActivation(int);
	void setConnections(int start, int end);
	virtual ~Neuron();
};

#endif /* NEURON_H_ */
