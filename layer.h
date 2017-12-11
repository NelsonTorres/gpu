/*
 * layer.h
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Neuron.h"

class layer {
public:
	Neuron neurons[28];
	int numNeurons;
	int connection_start;
	int  connection_end;
	layer();
	void setActivation(int);
	void setNeurons(int);
	layer(int);
	virtual ~layer();
};

#endif /* LAYER_H_ */
