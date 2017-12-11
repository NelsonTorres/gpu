/*
 * Neuron.h
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#ifndef NEURON_H_
#define NEURON_H_



class Neuron {
public:
	float sum;
	float value;
	float activation_steepness;
	int activation_function;
	float error;
	float slope;
	float prev_step;
	float prev_slope;
	Neuron();
	void setActivation(int);
	virtual ~Neuron();
};

#endif /* NEURON_H_ */
