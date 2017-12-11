/*
 * SimpleFannData.cpp
 *
 *  Created on: 05/12/2017
 *      Author: nelson
 */

#include "SimpleFannData.h"
#include <stdio.h>
#include <stdlib.h>

SimpleFannData::SimpleFannData(){
	num_data = -1;
	num_input = -1;
	num_output= -1;
};

SimpleFannData::SimpleFannData(const char* path){
	num_data = -1;
	num_input = -1;
	num_output= -1;

  std::ifstream file;
  file.open (path, std::ifstream::in);

  unsigned int i, j;
  unsigned int line = 1;

  file >> num_data >> num_input >> num_output;
  line++;

  std::cout << num_data << " " <<num_input << " "<<num_output << "\n";

  for(i = 0; i < num_data; i++){
	for(j = 0; j < num_input; j++){
      file >> this->input[i* num_input +j];
	}
	for(j = 0; j < num_output; j++){
	  file >> this->output[i* num_input + j];
	}
	line++;
  }
  file.close();

};
