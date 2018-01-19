/*
 * SimpleFannData.cpp
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#include "SimpleFannData.h"
#include <stdio.h>
#include <stdlib.h>


unsigned readData(const char *path, float* input, float* output) {
  unsigned num_data, num_input, num_output, i, j;

  std::ifstream file;
  file.open(path, std::ifstream::in);

  file >> num_data >> num_input >> num_output;

  input = new float[num_data * num_input];
  output = new float[num_data * num_output];

  for (i = 0; i < num_data; i++) {
    for (j = 0; j < num_input; j++) {
      file >> this->input[i * num_input + j];
    }
    for (j = 0; j < num_output; j++) {
      file >> this->output[i * num_output + j];
    }
  }
  file.close();
  return num_data;
};
