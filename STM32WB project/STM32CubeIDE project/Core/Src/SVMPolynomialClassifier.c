/*
 * SVMPolunomialClassifier.c
 *
 *  Created on: 25 Jan 2023
 *      Author: Marek Zylinski
 */
#include "SVMPolynomialClassifier.h"
#include "SVMParameters_poly.h"

arm_svm_polynomial_instance_f32 polySVM;

void initARMSVMPolynomialClasificator ()
{
	 /*
	    Initialization of the SVM instance parameters.
	    Additional parameters (intercept, degree, coef0 and gamma) are also coming from Python.
	   */
	  arm_svm_polynomial_init_f32(&polySVM,
		poly_NB_SUPPORT_VECTORS,
		poly_VECTOR_DIMENSION,
		polyIntercept,        /* Intercept */
		polyDualCoefficients,
		polySupportVectors,
		polyClasses,
		polyDegree,                 /* degree */
		polyCoef0,         /* Coef0 */
		polyGamma          /* Gamma */
	  );

}

uint32_t predictSVMPolynomial (float32_t innputVector[])
{
	  int32_t result;
	  arm_svm_polynomial_predict_f32(&polySVM, innputVector, &result);
	  return result;
}
