from sklearn import svm
from loadDataSet import getDataset
from saveClassifiers import saveSVMClassifier
X_train, X_test, y_train, y_test = getDataset()
#I used 7 parameters

clfs = [svm.SVC(kernel='linear', C=5000), svm.SVC(kernel='poly', C=5000), svm.SVC(kernel='rbf', C=5000),
        svm.SVC(kernel='sigmoid')]

for clf in clfs:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf.kernel + " Number of mislabeled points out of a total %d points : %d", (len(X_test), (y_test != y_pred).sum()))

    saveSVMClassifier(clf)

'''
    # The parameters of the trained classifier are printed to be used
    # in CMSIS-DSP
    VECDIM = clf.n_features_in_
    supportShape = clf.support_vectors_.shape

    nbSupportVectors=supportShape[0]
    vectorDimensions=supportShape[1]

    print("nbSupportVectors = %d" % nbSupportVectors)
    print("vectorDimensions = %d" % vectorDimensions)
    print("degree = %d" % clf.degree)
    print("coef0 = %f" % clf.coef0)
    print("gamma = %f" % clf._gamma)

    print("intercept = %f" % clf.intercept_)

    dualCoefs=clf.dual_coef_
    dualCoefs=dualCoefs.reshape(nbSupportVectors)
    supportVectors=clf.support_vectors_
    supportVectors = supportVectors.reshape(nbSupportVectors*VECDIM)

    print("Dual Coefs")
    print(dualCoefs)

    print("Support Vectors")
    print(supportVectors)
    '''
