/*
 * layer.h
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <stdlib.h>
#include "Neuron.h"

class layer {
public:
	Neuron neurons[597];
	Neuron bias;
	int numNeurons;
	layer();
	void setActivation(int);
	void setNeurons(int);
	layer(int);
	virtual ~layer();
};

#endif /* LAYER_H_ */
