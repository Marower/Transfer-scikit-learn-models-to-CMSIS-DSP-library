#ifndef _gaussian_naive_bayes
#define _gaussian_naive_bayes

#include <stdio.h>
#include "arm_math.h"

void initARMGaussianNaiveBayesClasificator ();
uint32_t predictClassNaiveBayes (float32_t innputVector[]);

#endif
