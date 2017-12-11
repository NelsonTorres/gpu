/*
 * SimpleFannData.h
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#ifndef SIMPLEFANNDATA_H_
#define SIMPLEFANNDATA_H_

#include <iostream>
#include <fstream>

class SimpleFannData{
  public:
  unsigned int num_data;
  unsigned int num_input;
  unsigned int num_output;
  float input[280], output[28];

  SimpleFannData();
  SimpleFannData(const char*);
  void toString();
  void cudaAllocate(SimpleFannData* );
};

#endif /* SIMPLEFANNDATA_H_ */
