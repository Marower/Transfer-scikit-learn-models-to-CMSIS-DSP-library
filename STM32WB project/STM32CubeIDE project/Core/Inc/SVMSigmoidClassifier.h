#ifndef _SVM_sigmoid
#define _SVM_sigmoid

#include <stdio.h>
#include "arm_math.h"

void initARMSVMSigmoidClasificator ();
uint32_t predictSVMSigmoid (float32_t innputVector[]);

#endif
