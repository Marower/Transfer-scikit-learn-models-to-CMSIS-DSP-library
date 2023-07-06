/*
 * SVMrfbClassifier.c
 *
 *  Created on: 25 Jan 2023
 *      Author: Marek Zylinski
 */
#include "SVMrfbClassifier.h"
#include "SVMParameters_rbf.h"


arm_svm_rbf_instance_f32 rbfSVM;

void initARMSVMrbfClasificator ()
{
	arm_svm_rbf_init_f32(&rbfSVM, //Parameters for the SVM function
				  rbf_NB_SUPPORT_VECTORS,	//Number of support vectors
				  rbf_VECTOR_DIMENSION,	//Dimension of vector space
				  rbfIntercept,	//Intercept
				  rbfDualCoefficients,	//Array of dual coefficients
				  rbfSupportVectors,	//Array of support vectors
				  rbfClasses,	//Array of 2 classes ID
				  rbfGamma	//gamma (scikit-learn terminology)
		  );

}

uint32_t predictSVMrbf (float32_t innputVector[])
{
	  int32_t result;
	  arm_svm_rbf_predict_f32(&rbfSVM, innputVector, &result);
	  return result;
}
