#ifndef _SVM_linear
#define _SVM_linear

#include <stdio.h>
#include "arm_math.h"

void initARMSVMLinearClasificator ();
uint32_t predictSVMLinear (float32_t innputVector[]);

#endif
