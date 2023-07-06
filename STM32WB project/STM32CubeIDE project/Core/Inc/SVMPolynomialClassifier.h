#ifndef _SVM_poly
#define _SVM_poly

#include <stdio.h>
#include "arm_math.h"

void initARMSVMPolynomialClasificator ();
uint32_t predictSVMPolynomial (float32_t innputVector[]);

#endif
