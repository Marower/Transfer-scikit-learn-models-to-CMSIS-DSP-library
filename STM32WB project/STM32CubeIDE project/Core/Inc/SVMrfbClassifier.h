#ifndef _SVM_rbf
#define _SVM_rbf

#include <stdio.h>
#include "arm_math.h"

void initARMSVMrbfClasificator ();
uint32_t predictSVMrbf (float32_t innputVector[]);

#endif
