#ifndef _SVM_rbf_Parameters
#define _SVM_rbf_Parameters
#include "arm_math.h"

#define rbf_NB_SUPPORT_VECTORS  80
#define rbf_VECTOR_DIMENSION 7
/*
Those parameters was generated with the scikit-learn and Marek's script.
*/
//Classes: ['AF'  'Normal' ]
const int32_t rbfClasses[2]={ 0,  1};

const float32_t rbfIntercept  = -11.095958f;
const float32_t rbfGamma = 0.384151f;
const float32_t rbfDualCoefficients[rbf_NB_SUPPORT_VECTORS ] = {
-2.404700E+03f, -1.395370E+03f, -2.363271E+03f, -5.000000E+03f, -4.095532E-01f, -9.441651E-02f, -5.000000E+03f, 
-1.227678E+03f, -2.475305E+00f, -8.874980E+01f, -5.185038E+01f, -1.843989E+03f, -8.067159E+02f, -3.403692E+02f, 
-3.953027E+03f, -5.000000E+03f, -1.317798E+02f, -5.000000E+03f, -5.000000E+03f, -4.604758E+02f, -7.154236E+01f, 
-3.232821E+01f, -5.000000E+03f, -5.000000E+03f, -1.281997E+03f, -5.000000E+03f, -1.094777E+03f, -3.980863E+00f, 
-4.213866E+01f, -5.000000E+03f, -5.000000E+03f, -1.803084E+02f, -4.991491E+03f, -5.000000E+03f, -5.000000E+03f, 
-3.264301E+02f, -4.709985E+03f, -5.000000E+03f, -7.797796E+01f, -1.320273E+02f, -4.988375E+03f, -5.000000E+03f, 
-1.512710E+03f, -5.323166E+02f, -5.000000E+03f, -2.096228E+03f, 1.488110E+02f, 5.000000E+03f, 5.000000E+03f, 
1.841993E+03f, 3.793732E+03f, 5.000000E+03f, 4.261153E+03f, 5.000000E+03f, 3.635546E+03f, 4.620450E+03f, 
5.000000E+03f, 3.456797E+03f, 3.199087E+03f, 2.254753E+02f, 1.854529E+03f, 5.000000E+03f, 5.000000E+03f, 
4.529981E+03f, 1.864833E+03f, 2.356535E+03f, 5.000000E+03f, 5.000000E+03f, 3.527580E+03f, 5.000000E+03f, 
5.000000E+03f, 5.270984E+02f, 4.183233E+03f, 5.000000E+03f, 5.549556E+01f, 1.302696E+03f, 5.969262E+02f, 
4.427437E+01f, 1.580123E+03f, 4.539223E+03f, }; /**< Dual coefficients */

const float32_t rbfSupportVectors[rbf_NB_SUPPORT_VECTORS*rbf_VECTOR_DIMENSION] = {
7.900000E-01f, 1.042000E+00f, 9.086000E-01f, 9.150000E-01f, 8.530755E-02f, 4.424884E-01f, 9.000000E-01f, 
4.700000E-01f, 4.700000E-01f, 4.700000E-01f, 4.700000E-01f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
4.540000E-01f, 1.236000E+00f, 6.275172E-01f, 6.160000E-01f, 1.444137E-01f, 6.798588E-01f, 2.068966E-01f, 
9.240000E-01f, 9.560000E-01f, 9.392500E-01f, 9.350000E-01f, 1.084633E-02f, 2.734959E-02f, 0.000000E+00f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
6.580000E-01f, 8.940000E-01f, 7.630909E-01f, 7.860000E-01f, 8.757791E-02f, 1.701176E-01f, 3.636364E-01f, 
2.620000E-01f, 1.082000E+00f, 6.917000E-01f, 6.420000E-01f, 1.830861E-01f, 1.030157E+00f, 6.500000E-01f, 
8.040000E-01f, 1.758000E+00f, 1.097800E+00f, 9.750000E-01f, 2.763146E-01f, 1.099809E+00f, 8.000000E-01f, 
5.060000E-01f, 1.768000E+00f, 8.107500E-01f, 7.320000E-01f, 3.114884E-01f, 1.703071E+00f, 7.500000E-01f, 
6.380000E-01f, 1.524000E+00f, 8.796364E-01f, 6.700000E-01f, 3.221252E-01f, 1.581039E+00f, 7.272727E-01f, 
5.620000E-01f, 8.780000E-01f, 7.352727E-01f, 7.500000E-01f, 8.384401E-02f, 4.552318E-01f, 4.545455E-01f, 
8.040000E-01f, 1.758000E+00f, 1.097800E+00f, 9.750000E-01f, 2.763146E-01f, 1.099809E+00f, 8.000000E-01f, 
1.308000E+00f, 1.346000E+00f, 1.323667E+00f, 1.317000E+00f, 1.626858E-02f, 5.589275E-02f, 0.000000E+00f, 
5.280000E-01f, 1.382000E+00f, 8.070000E-01f, 7.560000E-01f, 2.025429E-01f, 1.001155E+00f, 4.166667E-01f, 
6.000000E-01f, 6.100000E-01f, 6.048571E-01f, 6.060000E-01f, 3.109715E-03f, 1.685230E-02f, 0.000000E+00f, 
3.140000E-01f, 1.388000E+00f, 7.065600E-01f, 7.180000E-01f, 1.894400E-01f, 1.085318E+00f, 5.200000E-01f, 
8.000000E-01f, 1.084000E+00f, 9.530000E-01f, 9.610000E-01f, 1.097165E-01f, 3.333827E-01f, 5.000000E-01f, 
6.020000E-01f, 9.500000E-01f, 8.238000E-01f, 8.280000E-01f, 9.125520E-02f, 3.914486E-01f, 4.000000E-01f, 
4.500000E-01f, 1.098000E+00f, 8.207083E-01f, 8.460000E-01f, 1.170543E-01f, 1.286805E+00f, 4.791667E-01f, 
2.500000E-01f, 1.060000E+00f, 3.672558E-01f, 2.860000E-01f, 1.715704E-01f, 1.546113E+00f, 4.883721E-01f, 
1.014000E+00f, 1.514000E+00f, 1.194571E+00f, 1.154000E+00f, 1.692521E-01f, 5.811712E-01f, 5.714286E-01f, 
6.740000E-01f, 9.300000E-01f, 7.570909E-01f, 7.200000E-01f, 8.434863E-02f, 3.293934E-01f, 2.727273E-01f, 
4.900000E-01f, 6.900000E-01f, 5.506087E-01f, 5.500000E-01f, 4.057063E-02f, 2.786683E-01f, 3.043478E-01f, 
5.760000E-01f, 1.600000E+00f, 8.004444E-01f, 7.570000E-01f, 2.273288E-01f, 1.474651E+00f, 7.222222E-01f, 
6.900000E-01f, 9.440000E-01f, 8.443077E-01f, 8.500000E-01f, 7.257110E-02f, 3.373485E-01f, 4.615385E-01f, 
9.340000E-01f, 1.504000E+00f, 1.097714E+00f, 1.026000E+00f, 1.957428E-01f, 5.943803E-01f, 5.714286E-01f, 
5.900000E-01f, 1.758000E+00f, 8.615556E-01f, 7.780000E-01f, 2.704006E-01f, 1.506058E+00f, 7.777778E-01f, 
3.122000E+00f, 3.122000E+00f, 3.122000E+00f, 3.122000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
7.500000E-01f, 1.010000E+00f, 8.713333E-01f, 8.640000E-01f, 7.078842E-02f, 3.662786E-01f, 5.555556E-01f, 
7.540000E-01f, 9.900000E-01f, 8.460000E-01f, 8.560000E-01f, 7.215262E-02f, 3.470620E-01f, 4.444444E-01f, 
8.920000E-01f, 1.396000E+00f, 1.090250E+00f, 1.070000E+00f, 1.510891E-01f, 4.578166E-01f, 6.250000E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.640000E-01f, 8.260000E-01f, 6.770000E-01f, 6.300000E-01f, 9.024613E-02f, 2.796212E-01f, 1.666667E-01f, 
6.400000E-01f, 6.820000E-01f, 6.641818E-01f, 6.660000E-01f, 1.247252E-02f, 6.033241E-02f, 0.000000E+00f, 
9.320000E-01f, 1.256000E+00f, 1.071429E+00f, 1.072000E+00f, 1.069499E-01f, 3.811771E-01f, 7.142857E-01f, 
6.200000E-01f, 8.220000E-01f, 6.826667E-01f, 6.640000E-01f, 5.734954E-02f, 2.518015E-01f, 5.000000E-01f, 
9.480000E-01f, 9.480000E-01f, 9.480000E-01f, 9.480000E-01f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
2.560000E-01f, 1.890000E+00f, 4.867692E-01f, 3.560000E-01f, 4.343137E-01f, 2.075591E+00f, 5.384615E-01f, 
3.020000E-01f, 9.320000E-01f, 4.281000E-01f, 3.740000E-01f, 1.535070E-01f, 9.511109E-01f, 3.500000E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
7.480000E-01f, 1.148000E+00f, 9.186667E-01f, 9.280000E-01f, 1.008435E-01f, 6.176763E-01f, 3.888889E-01f, 
6.600000E-01f, 9.200000E-01f, 7.892727E-01f, 7.940000E-01f, 7.900518E-02f, 3.245797E-01f, 7.272727E-01f, 
4.880000E-01f, 1.330000E+00f, 1.009250E+00f, 1.170000E+00f, 3.313797E-01f, 8.702069E-01f, 6.250000E-01f, 
6.740000E-01f, 9.300000E-01f, 7.570909E-01f, 7.200000E-01f, 8.434863E-02f, 3.293934E-01f, 2.727273E-01f, 
4.760000E-01f, 1.530000E+00f, 8.202000E-01f, 7.390000E-01f, 2.952038E-01f, 1.458162E+00f, 7.000000E-01f, 
6.520000E-01f, 8.320000E-01f, 7.341667E-01f, 7.190000E-01f, 4.971525E-02f, 2.167948E-01f, 4.166667E-01f, 
6.940000E-01f, 1.034000E+00f, 8.860000E-01f, 8.920000E-01f, 9.534299E-02f, 4.724574E-01f, 6.666667E-01f, 
7.880000E-01f, 1.012000E+00f, 8.828333E-01f, 8.810000E-01f, 6.864114E-02f, 3.505139E-01f, 6.666667E-01f, 
7.400000E-01f, 1.048000E+00f, 8.951111E-01f, 8.960000E-01f, 7.707860E-02f, 3.766590E-01f, 3.333333E-01f, 
5.760000E-01f, 5.860000E-01f, 5.804000E-01f, 5.800000E-01f, 2.164651E-03f, 8.246211E-03f, 0.000000E+00f, 
5.880000E-01f, 7.020000E-01f, 6.458182E-01f, 6.430000E-01f, 3.528339E-02f, 1.944634E-01f, 3.636364E-01f, 
5.780000E-01f, 7.780000E-01f, 6.155714E-01f, 6.040000E-01f, 4.970175E-02f, 2.375374E-01f, 1.428571E-01f, 
7.040000E-01f, 9.220000E-01f, 7.747500E-01f, 7.680000E-01f, 5.072282E-02f, 2.969781E-01f, 5.000000E-01f, 
8.300000E-01f, 1.022000E+00f, 9.337500E-01f, 9.380000E-01f, 7.192407E-02f, 2.179266E-01f, 2.500000E-01f, 
7.280000E-01f, 1.578000E+00f, 8.717778E-01f, 8.020000E-01f, 2.672460E-01f, 1.178847E+00f, 6.666667E-01f, 
6.200000E-01f, 8.540000E-01f, 7.268571E-01f, 7.280000E-01f, 7.185877E-02f, 1.914785E-01f, 2.142857E-01f, 
8.260000E-01f, 1.158000E+00f, 8.752000E-01f, 8.410000E-01f, 1.006123E-01f, 2.884718E-01f, 2.000000E-01f, 
8.680000E-01f, 9.040000E-01f, 8.912500E-01f, 8.990000E-01f, 1.492601E-02f, 3.789459E-02f, 0.000000E+00f, 
7.780000E-01f, 9.760000E-01f, 8.734667E-01f, 8.750000E-01f, 6.189632E-02f, 3.659235E-01f, 5.666667E-01f, 
6.640000E-01f, 9.220000E-01f, 8.090909E-01f, 8.460000E-01f, 8.236195E-02f, 1.685586E-01f, 4.545455E-01f, 
7.720000E-01f, 1.256000E+00f, 1.066000E+00f, 1.067000E+00f, 1.466814E-01f, 5.415718E-01f, 5.000000E-01f, 
6.600000E-01f, 8.820000E-01f, 7.729167E-01f, 7.800000E-01f, 6.570283E-02f, 4.008691E-01f, 4.583333E-01f, 
5.660000E-01f, 6.280000E-01f, 5.986154E-01f, 5.940000E-01f, 2.072655E-02f, 4.481071E-02f, 0.000000E+00f, 
3.260000E-01f, 1.566000E+00f, 7.832973E-01f, 7.680000E-01f, 2.076681E-01f, 1.667809E+00f, 6.486486E-01f, 
5.440000E-01f, 1.196000E+00f, 6.228182E-01f, 5.950000E-01f, 1.328209E-01f, 8.453165E-01f, 1.818182E-01f, 
5.980000E-01f, 1.088000E+00f, 8.550000E-01f, 8.560000E-01f, 1.120341E-01f, 5.723181E-01f, 5.000000E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
1.000000E+00f, 1.000000E+00f, 1.000000E+00f, 1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.380000E-01f, 7.500000E-01f, 6.248800E-01f, 6.180000E-01f, 5.343264E-02f, 2.637120E-01f, 1.800000E-01f, 
8.760000E-01f, 9.220000E-01f, 8.955556E-01f, 8.980000E-01f, 1.826959E-02f, 6.743886E-02f, 0.000000E+00f, 
7.280000E-01f, 8.960000E-01f, 8.026667E-01f, 7.810000E-01f, 5.530960E-02f, 2.735032E-01f, 5.833333E-01f, 
6.420000E-01f, 8.700000E-01f, 7.151111E-01f, 6.910000E-01f, 6.473193E-02f, 2.670431E-01f, 2.777778E-01f, 
2.952000E+00f, 3.000000E+00f, 2.976000E+00f, 2.976000E+00f, 3.394113E-02f, 4.800000E-02f, 0.000000E+00f, 
9.980000E-01f, 1.018000E+00f, 1.009000E+00f, 1.012000E+00f, 8.000000E-03f, 2.966479E-02f, 0.000000E+00f, 
6.960000E-01f, 1.218000E+00f, 8.388627E-01f, 8.480000E-01f, 7.510020E-02f, 7.272991E-01f, 2.156863E-01f, 
3.600000E-01f, 1.800000E+00f, 9.042222E-01f, 8.980000E-01f, 3.914964E-01f, 1.431071E+00f, 5.555556E-01f, 
8.420000E-01f, 8.880000E-01f, 8.670769E-01f, 8.640000E-01f, 1.652504E-02f, 8.192680E-02f, 0.000000E+00f, 
3.060000E-01f, 1.112000E+00f, 7.137778E-01f, 6.540000E-01f, 2.454474E-01f, 8.917017E-01f, 4.444444E-01f, 
}; /**<  Support vectors */

#endif
