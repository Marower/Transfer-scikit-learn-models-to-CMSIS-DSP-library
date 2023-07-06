#ifndef _gaussian_naive_bayes_parameters
#define _gaussian_naive_bayes_parameters
#include "arm_math.h"

//Classes: ['AF'  'Normal'  ]
#define bayesNB_OF_CLASSES 2
#define bayesVECTOR_DIMENSION 7
/*
Those parameters was generated with the scikit-learn and Marek's script.
*/
const float32_t bayesTheta[bayesNB_OF_CLASSES*bayesVECTOR_DIMENSION] = {
4.604286E-01f, 1.069502E+00f, 6.812064E-01f, 6.547679E-01f, 1.773899E-01f, 9.540820E-01f, 6.729112E-01f, 
7.577056E-01f, 8.690410E-01f, 8.090479E-01f, 8.090713E-01f, 3.220681E-02f, 1.384098E-01f, 1.114310E-01f, 
}; /**< Mean values for the Gaussians */

const float32_t bayesSigma[bayesNB_OF_CLASSES*bayesVECTOR_DIMENSION] = {
5.892907E-02f, 9.500625E-01f, 1.877134E-01f, 2.349302E-01f, 1.330251E-01f, 1.592040E+00f, 3.254051E-02f, 
2.989925E-02f, 3.906091E-02f, 2.827103E-02f, 2.865862E-02f, 1.531940E-03f, 3.271100E-02f, 2.517850E-02f, 
}; /**< Variances for the Gaussians */

const float32_t bayesClassPriors[bayesNB_OF_CLASSES] = {
5.603736E-01f, 4.396264E-01f, 
}; /**< Class prior probabilities */

const float32_t bayesEpsilon = 1.070423E-09f;

#endif
