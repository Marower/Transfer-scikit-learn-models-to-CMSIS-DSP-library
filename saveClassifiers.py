import numpy as np

def saveSVMClassifier (clf):
    kernelsName = clf.kernel
    supportShape = clf.support_vectors_.shape
    VECDIM = clf.n_features_in_
    nbSupportVectors = supportShape[0]
    vectorDimensions = supportShape[1]

    dualCoefs = clf.dual_coef_
    dualCoefs = dualCoefs.reshape(nbSupportVectors)
    supportVectors = clf.support_vectors_
    supportVectors = supportVectors.reshape(nbSupportVectors * VECDIM)
    fileName = "SVMParameters_" + kernelsName + ".h"
    f = open(fileName, "w")
    f.write("#ifndef _SVM_%s_Parameters\n" % kernelsName)
    f.write("#define _SVM_%s_Parameters\n#include \"arm_math.h\"\n\n" % kernelsName)
    f.write("#define " + kernelsName + "_NB_SUPPORT_VECTORS  %d\n" % nbSupportVectors)

    f.write("#define " + kernelsName + "_VECTOR_DIMENSION %d\n" % vectorDimensions)

    f.write("/*\nThose parameters was generated with the scikit-learn and Marek's script.\n*/\n"
            "//Classes: [")
    classVector = "const int32_t %sClasses[2]={" % kernelsName
    for i in range(0, len(clf.classes_)):
        f.write("'%s' " % clf.classes_[i])
        classVector += " %d" % i
        if i < len(clf.classes_) - 1:
            f.write(" ")
            classVector += ", "
    f.write("]\n")
    f.write(classVector + "};\n\n")

    f.write("const float32_t " + kernelsName + "Intercept  = %ff;\n" % clf.intercept_)
    if kernelsName == 'poly' or kernelsName == 'sigmoid':
        f.write("const float32_t " + kernelsName + "Coef0 = %ff;\n" % clf.coef0)
    if kernelsName == 'poly':
        f.write("const float32_t " + kernelsName + "Degree = %ff;\n" % clf.degree)
    if kernelsName != "linear":
        f.write("const float32_t " + kernelsName + "Gamma = %ff;\n" % clf._gamma)

    f.write("const float32_t " + kernelsName + "DualCoefficients[" + kernelsName + "_NB_SUPPORT_VECTORS ] = {\n")

    for i in range(0, len(dualCoefs)):
        f.write("%E" % dualCoefs[i])
        if i < len(dualCoefs):
            f.write("f, ")
        if i > 0 and ((i + 1) % vectorDimensions) == 0:
            f.write("\n")

    f.write("}; /**< Dual coefficients */\n\nconst float32_t "
            + kernelsName + "SupportVectors[" + kernelsName + "_NB_SUPPORT_VECTORS*"
            + kernelsName + "_VECTOR_DIMENSION] = {\n")

    for i in range(0, len(supportVectors)):
        f.write("%E" % supportVectors[i])
        if i < len(supportVectors):
            f.write("f, ")
        if i > 0 and ((i + 1) % vectorDimensions) == 0:
            f.write("\n")

    f.write("}; /**<  Support vectors */\n")

    f.write("\n#endif\n")
    f.close()


def saveBayesClasificator (gnb):
    theta = list(np.reshape(gnb.theta_, np.size(gnb.theta_)))
    # Gaussian variances
    Sigma = list(np.reshape(gnb.var_, np.size(gnb.var_)))
    # Class priors
    classPriors = list(np.reshape(gnb.class_prior_, np.size(gnb.class_prior_)))
    f = open("naiveBayesParameters.h", "w")
    f.write("#ifndef _gaussian_naive_bayes_parameters\n")
    f.write("#define _gaussian_naive_bayes_parameters\n#include \"arm_math.h\"\n\n")
    f.write("//Classes: [")
    for i in range(0, len(gnb.classes_)):
        f.write("'%s' " % gnb.classes_[i])
        if i < len(gnb.classes_):
            f.write(" ")
    f.write("]\n")
    f.write("#define bayesNB_OF_CLASSES %d\n" % len(gnb.classes_))

    f.write("#define bayesVECTOR_DIMENSION %d\n" % gnb.n_features_in_)

    f.write("/*\nThose parameters was generated with the scikit-learn and Marek's script.\n*/\n"
            "const float32_t bayesTheta[bayesNB_OF_CLASSES*bayesVECTOR_DIMENSION] = {\n")

    for i in range(0, len(theta)):
        f.write("%E" % theta[i])
        if i < len(theta):
            f.write("f, ")
        if i > 0 and ((i + 1) % gnb.n_features_in_) == 0:
            f.write("\n")

    f.write("}; /**< Mean values for the Gaussians */\n\n"
            "const float32_t bayesSigma[bayesNB_OF_CLASSES*bayesVECTOR_DIMENSION] = {\n")

    for i in range(0, len(Sigma)):
        f.write("%E" % Sigma[i])
        if i < len(Sigma):
            f.write("f, ")
        if i > 0 and ((i + 1) % gnb.n_features_in_) == 0:
            f.write("\n")

    f.write("}; /**< Variances for the Gaussians */\n\nconst float32_t bayesClassPriors[bayesNB_OF_CLASSES] = {\n")

    for i in range(0, len(classPriors)):
        f.write("%E" % classPriors[i])
        if i < len(classPriors):
            f.write("f, ")

    f.write("\n}; /**< Class prior probabilities */\n\n")

    f.write("const float32_t bayesEpsilon = %Ef;\n" % gnb.epsilon_)

    f.write("\n#endif\n")
    f.close()
