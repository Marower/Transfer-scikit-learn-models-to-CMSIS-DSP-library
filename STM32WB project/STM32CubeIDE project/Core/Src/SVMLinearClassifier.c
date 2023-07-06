/*
 * SVMLinearClassifier.c
 *
 *  Created on: 25 Jan 2023
 *      Author: Marek Zylinski
 */

#include "SVMLinearClassifier.h"
#include "SVMParameters_linear.h"

arm_svm_linear_instance_f32 linearSVM;

void initARMSVMLinearClasificator ()
{
	  arm_svm_linear_init_f32(&linearSVM, //Parameters for the SVM function
			  linear_NB_SUPPORT_VECTORS,	//Number of support vectors
			  linear_VECTOR_DIMENSION,	//Dimension of vector space
			  linearIntercept,	//Intercept
			  linearDualCoefficients,	//Array of dual coefficients
			  linearSupportVectors,	//Array of support vectors
			  linearClasses	//Array of 2 classes ID
	  );

}

uint32_t predictSVMLinear (float32_t innputVector[])
{
	  int32_t result;
	  arm_svm_linear_predict_f32(&linearSVM, innputVector, &result);
	  return result;
}
