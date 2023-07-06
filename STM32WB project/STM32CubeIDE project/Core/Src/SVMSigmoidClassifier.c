/*
 * SVMSigmoidClassifier.c
 *
 *  Created on: 25 Jan 2023
 *      Author: Marek Zylinski
 */
#include "SVMSigmoidClassifier.h"
#include "SVMParameters_sigmoid.h"

arm_svm_sigmoid_instance_f32 sigmoidSVM;

void initARMSVMSigmoidClasificator ()
{
	 /*
	    Initialization of the SVM instance parameters.
	    Additional parameters (intercept, degree, coef0 and gamma) are also coming from Python.
	   */
	  arm_svm_sigmoid_init_f32(&sigmoidSVM,
	    sigmoid_NB_SUPPORT_VECTORS,
		sigmoid_VECTOR_DIMENSION,
		sigmoidIntercept,        /* Intercept */
		sigmoidDualCoefficients,
		sigmoidSupportVectors,
		sigmoidClasses,
		sigmoidCoef0,         /* Coef0 */
		sigmoidGamma          /* Gamma */
	  );

}

uint32_t predictSVMSigmoid (float32_t innputVector[])
{
	  int32_t result;
	  arm_svm_sigmoid_predict_f32(&sigmoidSVM, innputVector, &result);
	  return result;
}
