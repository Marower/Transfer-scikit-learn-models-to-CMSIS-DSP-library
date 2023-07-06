/*
 * naiveBayesClassifier.c
 *
 *  Created on: 25 Jan 2023
 *      Author: Marek Zylinski
 */
#include "naiveBayesClassifier.h"
#include "naiveBayesParameters.h"

arm_gaussian_naive_bayes_instance_f32 bayesInstance;

void initARMGaussianNaiveBayesClasificator ()
{
	bayesInstance.vectorDimension = bayesVECTOR_DIMENSION;
	bayesInstance.numberOfClasses = bayesNB_OF_CLASSES;
	bayesInstance.theta = bayesTheta;
	bayesInstance.sigma = bayesSigma;
	bayesInstance.classPriors = bayesClassPriors;
	bayesInstance.epsilon= bayesEpsilon;
}

uint32_t predictClassNaiveBayes (float32_t innputVector[])
{
	/* Result of the classifier */
	float32_t result[bayesNB_OF_CLASSES];
	float32_t temp[bayesNB_OF_CLASSES];

	uint32_t index;

	index = arm_gaussian_naive_bayes_predict_f32(&bayesInstance, innputVector, result,temp);

	return index;
}
