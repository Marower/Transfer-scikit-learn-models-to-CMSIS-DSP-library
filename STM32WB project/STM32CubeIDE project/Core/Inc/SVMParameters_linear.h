#ifndef _SVM_linear_Parameters
#define _SVM_linear_Parameters
#include "arm_math.h"

#define linear_NB_SUPPORT_VECTORS  196
#define linear_VECTOR_DIMENSION 7
/*
Those parameters was generated with the scikit-learn and Marek's script.
*/
//Classes: ['AF'  'Normal' ]
const int32_t linearClasses[2]={ 0,  1};

const float32_t linearIntercept  = -0.467655f;
const float32_t linearDualCoefficients[linear_NB_SUPPORT_VECTORS ] = {
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -2.418621E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -3.082919E+02f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -3.163461E+02f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-1.184612E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -1.653400E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -8.787073E+02f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 4.711595E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 2.048384E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
}; /**< Dual coefficients */

const float32_t linearSupportVectors[linear_NB_SUPPORT_VECTORS*linear_VECTOR_DIMENSION] = {
3.300000E-01f, 5.700000E-01f, 4.875556E-01f, 4.990000E-01f, 5.725713E-02f, 3.247522E-01f, 3.333333E-01f, 
4.700000E-01f, 4.700000E-01f, 4.700000E-01f, 4.700000E-01f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
2.680000E-01f, 4.500000E-01f, 3.487200E-01f, 3.440000E-01f, 4.469482E-02f, 2.286744E-01f, 2.400000E-01f, 
4.540000E-01f, 1.236000E+00f, 6.275172E-01f, 6.160000E-01f, 1.444137E-01f, 6.798588E-01f, 2.068966E-01f, 
9.240000E-01f, 9.560000E-01f, 9.392500E-01f, 9.350000E-01f, 1.084633E-02f, 2.734959E-02f, 0.000000E+00f, 
1.330000E+00f, 1.474000E+00f, 1.397200E+00f, 1.386000E+00f, 5.363954E-02f, 1.694816E-01f, 6.000000E-01f, 
2.980000E-01f, 4.720000E-01f, 3.240741E-01f, 3.120000E-01f, 4.077888E-02f, 2.999133E-01f, 1.481481E-01f, 
5.920000E-01f, 8.720000E-01f, 6.868571E-01f, 6.600000E-01f, 8.237144E-02f, 3.759787E-01f, 5.714286E-01f, 
6.080000E-01f, 9.520000E-01f, 7.581818E-01f, 7.860000E-01f, 1.223583E-01f, 4.883605E-01f, 5.454545E-01f, 
4.060000E-01f, 9.600000E-01f, 6.813846E-01f, 7.980000E-01f, 2.106314E-01f, 6.173686E-01f, 3.846154E-01f, 
4.820000E-01f, 6.380000E-01f, 5.514286E-01f, 5.530000E-01f, 4.455889E-02f, 2.217927E-01f, 4.285714E-01f, 
6.200000E-01f, 9.760000E-01f, 7.045000E-01f, 6.920000E-01f, 9.119858E-02f, 4.069545E-01f, 5.000000E-01f, 
6.580000E-01f, 8.940000E-01f, 7.630909E-01f, 7.860000E-01f, 8.757791E-02f, 1.701176E-01f, 3.636364E-01f, 
6.280000E-01f, 9.400000E-01f, 7.531667E-01f, 7.480000E-01f, 1.026537E-01f, 5.021115E-01f, 5.833333E-01f, 
5.720000E-01f, 7.880000E-01f, 6.620000E-01f, 6.500000E-01f, 5.821226E-02f, 2.803854E-01f, 4.615385E-01f, 
2.960000E-01f, 4.260000E-01f, 3.934545E-01f, 4.010000E-01f, 2.644082E-02f, 1.343875E-01f, 9.090909E-02f, 
5.520000E-01f, 8.120000E-01f, 6.778333E-01f, 6.640000E-01f, 7.538728E-02f, 2.812259E-01f, 5.000000E-01f, 
4.192000E+00f, 4.192000E+00f, 4.192000E+00f, 4.192000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.120000E-01f, 7.220000E-01f, 6.088571E-01f, 6.000000E-01f, 6.317601E-02f, 2.740146E-01f, 5.000000E-01f, 
6.080000E-01f, 1.074000E+00f, 7.450769E-01f, 7.160000E-01f, 1.070121E-01f, 7.490234E-01f, 6.153846E-01f, 
9.900000E-01f, 1.886000E+00f, 1.339273E+00f, 1.404000E+00f, 2.543010E-01f, 1.315991E+00f, 7.272727E-01f, 
5.620000E-01f, 8.780000E-01f, 7.352727E-01f, 7.500000E-01f, 8.384401E-02f, 4.552318E-01f, 4.545455E-01f, 
2.560000E-01f, 5.200000E-01f, 3.217037E-01f, 3.000000E-01f, 7.867304E-02f, 5.276021E-01f, 2.592593E-01f, 
6.420000E-01f, 8.980000E-01f, 7.244615E-01f, 7.180000E-01f, 7.754205E-02f, 3.139363E-01f, 6.153846E-01f, 
5.600000E-01f, 9.360000E-01f, 6.717333E-01f, 6.600000E-01f, 7.878232E-02f, 7.005969E-01f, 5.333333E-01f, 
1.074000E+00f, 1.814000E+00f, 1.615600E+00f, 1.770000E+00f, 3.113692E-01f, 9.048271E-01f, 6.000000E-01f, 
1.308000E+00f, 1.346000E+00f, 1.323667E+00f, 1.317000E+00f, 1.626858E-02f, 5.589275E-02f, 0.000000E+00f, 
4.660000E-01f, 7.740000E-01f, 5.902667E-01f, 5.620000E-01f, 7.887012E-02f, 3.163858E-01f, 4.666667E-01f, 
5.160000E-01f, 7.860000E-01f, 6.352973E-01f, 6.220000E-01f, 6.214538E-02f, 5.334754E-01f, 5.135135E-01f, 
3.120000E-01f, 5.020000E-01f, 4.146000E-01f, 4.130000E-01f, 4.619000E-02f, 2.998266E-01f, 4.000000E-01f, 
4.780000E-01f, 6.880000E-01f, 5.692000E-01f, 5.720000E-01f, 6.646073E-02f, 2.521032E-01f, 4.666667E-01f, 
5.280000E-01f, 1.382000E+00f, 8.070000E-01f, 7.560000E-01f, 2.025429E-01f, 1.001155E+00f, 4.166667E-01f, 
4.900000E-01f, 7.240000E-01f, 6.110909E-01f, 6.320000E-01f, 7.950529E-02f, 2.910189E-01f, 5.454545E-01f, 
5.260000E-01f, 7.460000E-01f, 6.496000E-01f, 6.480000E-01f, 7.623722E-02f, 3.101935E-01f, 3.333333E-01f, 
6.000000E-01f, 6.100000E-01f, 6.048571E-01f, 6.060000E-01f, 3.109715E-03f, 1.685230E-02f, 0.000000E+00f, 
3.100000E-01f, 5.320000E-01f, 3.930909E-01f, 3.650000E-01f, 6.859329E-02f, 3.213160E-01f, 3.181818E-01f, 
8.000000E-01f, 1.084000E+00f, 9.530000E-01f, 9.610000E-01f, 1.097165E-01f, 3.333827E-01f, 5.000000E-01f, 
9.340000E-01f, 1.472000E+00f, 1.203143E+00f, 1.212000E+00f, 2.106272E-01f, 6.846634E-01f, 7.142857E-01f, 
4.180000E-01f, 4.720000E-01f, 4.450000E-01f, 4.450000E-01f, 3.818377E-02f, 5.400000E-02f, 5.000000E-01f, 
9.900000E-01f, 1.886000E+00f, 1.339273E+00f, 1.404000E+00f, 2.543010E-01f, 1.315991E+00f, 7.272727E-01f, 
5.140000E-01f, 7.600000E-01f, 6.268421E-01f, 6.280000E-01f, 6.588649E-02f, 3.620442E-01f, 5.263158E-01f, 
3.540000E-01f, 5.300000E-01f, 4.372632E-01f, 4.380000E-01f, 5.238834E-02f, 2.653224E-01f, 3.684211E-01f, 
3.720000E-01f, 7.700000E-01f, 5.245000E-01f, 5.090000E-01f, 1.055310E-01f, 4.145118E-01f, 3.750000E-01f, 
5.120000E-01f, 7.220000E-01f, 6.088571E-01f, 6.000000E-01f, 6.317601E-02f, 2.740146E-01f, 5.000000E-01f, 
6.020000E-01f, 9.500000E-01f, 8.238000E-01f, 8.280000E-01f, 9.125520E-02f, 3.914486E-01f, 4.000000E-01f, 
8.580000E-01f, 1.364000E+00f, 1.119714E+00f, 1.124000E+00f, 2.051550E-01f, 5.610597E-01f, 7.142857E-01f, 
3.940000E-01f, 6.280000E-01f, 5.462500E-01f, 5.660000E-01f, 7.429984E-02f, 4.591993E-01f, 3.750000E-01f, 
4.500000E-01f, 1.098000E+00f, 8.207083E-01f, 8.460000E-01f, 1.170543E-01f, 1.286805E+00f, 4.791667E-01f, 
3.240000E-01f, 5.080000E-01f, 3.976190E-01f, 3.900000E-01f, 4.981012E-02f, 2.656238E-01f, 1.904762E-01f, 
7.360000E-01f, 1.298000E+00f, 1.028250E+00f, 1.065000E+00f, 2.108417E-01f, 7.730175E-01f, 6.250000E-01f, 
1.014000E+00f, 1.514000E+00f, 1.194571E+00f, 1.154000E+00f, 1.692521E-01f, 5.811712E-01f, 5.714286E-01f, 
4.320000E-01f, 6.100000E-01f, 5.006250E-01f, 4.880000E-01f, 5.173635E-02f, 3.083504E-01f, 5.000000E-01f, 
3.760000E-01f, 7.160000E-01f, 4.393939E-01f, 4.100000E-01f, 8.420137E-02f, 5.483977E-01f, 3.636364E-01f, 
6.740000E-01f, 9.300000E-01f, 7.570909E-01f, 7.200000E-01f, 8.434863E-02f, 3.293934E-01f, 2.727273E-01f, 
6.540000E-01f, 1.150000E+00f, 8.148000E-01f, 7.280000E-01f, 1.413766E-01f, 6.115390E-01f, 4.400000E-01f, 
1.494000E+00f, 1.510000E+00f, 1.503750E+00f, 1.506000E+00f, 5.650959E-03f, 6.633250E-03f, 0.000000E+00f, 
4.900000E-01f, 6.900000E-01f, 5.506087E-01f, 5.500000E-01f, 4.057063E-02f, 2.786683E-01f, 3.043478E-01f, 
1.520000E+00f, 1.524000E+00f, 1.522400E+00f, 1.522000E+00f, 1.673320E-03f, 4.898979E-03f, 0.000000E+00f, 
6.900000E-01f, 9.440000E-01f, 8.443077E-01f, 8.500000E-01f, 7.257110E-02f, 3.373485E-01f, 4.615385E-01f, 
9.340000E-01f, 1.504000E+00f, 1.097714E+00f, 1.026000E+00f, 1.957428E-01f, 5.943803E-01f, 5.714286E-01f, 
2.840000E-01f, 4.880000E-01f, 3.464167E-01f, 3.350000E-01f, 4.435814E-02f, 3.277682E-01f, 3.750000E-01f, 
5.600000E-01f, 8.520000E-01f, 6.823000E-01f, 6.810000E-01f, 7.315528E-02f, 5.416752E-01f, 5.000000E-01f, 
3.940000E-01f, 5.820000E-01f, 4.711304E-01f, 4.580000E-01f, 5.644571E-02f, 3.044799E-01f, 3.043478E-01f, 
5.740000E-01f, 9.200000E-01f, 7.439091E-01f, 7.560000E-01f, 9.685036E-02f, 4.588289E-01f, 5.909091E-01f, 
3.122000E+00f, 3.122000E+00f, 3.122000E+00f, 3.122000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.560000E-01f, 8.320000E-01f, 6.953333E-01f, 6.860000E-01f, 7.502404E-02f, 3.738930E-01f, 5.833333E-01f, 
7.500000E-01f, 1.010000E+00f, 8.713333E-01f, 8.640000E-01f, 7.078842E-02f, 3.662786E-01f, 5.555556E-01f, 
7.540000E-01f, 9.900000E-01f, 8.460000E-01f, 8.560000E-01f, 7.215262E-02f, 3.470620E-01f, 4.444444E-01f, 
2.920000E-01f, 5.020000E-01f, 3.918182E-01f, 3.780000E-01f, 5.898477E-02f, 3.543050E-01f, 3.181818E-01f, 
4.100000E-01f, 7.600000E-01f, 6.158571E-01f, 6.230000E-01f, 8.046199E-02f, 4.873890E-01f, 4.285714E-01f, 
8.920000E-01f, 1.396000E+00f, 1.090250E+00f, 1.070000E+00f, 1.510891E-01f, 4.578166E-01f, 6.250000E-01f, 
5.220000E-01f, 8.860000E-01f, 6.703333E-01f, 6.660000E-01f, 9.917783E-02f, 5.329015E-01f, 5.000000E-01f, 
5.120000E-01f, 7.680000E-01f, 5.950435E-01f, 5.700000E-01f, 6.806112E-02f, 3.998300E-01f, 4.347826E-01f, 
5.640000E-01f, 8.260000E-01f, 6.770000E-01f, 6.300000E-01f, 9.024613E-02f, 2.796212E-01f, 1.666667E-01f, 
3.180000E-01f, 6.020000E-01f, 3.931429E-01f, 3.680000E-01f, 7.786031E-02f, 4.111156E-01f, 2.857143E-01f, 
6.400000E-01f, 6.820000E-01f, 6.641818E-01f, 6.660000E-01f, 1.247252E-02f, 6.033241E-02f, 0.000000E+00f, 
9.320000E-01f, 1.256000E+00f, 1.071429E+00f, 1.072000E+00f, 1.069499E-01f, 3.811771E-01f, 7.142857E-01f, 
5.560000E-01f, 8.280000E-01f, 7.087500E-01f, 7.180000E-01f, 7.342615E-02f, 3.777565E-01f, 5.625000E-01f, 
6.180000E-01f, 8.720000E-01f, 7.050000E-01f, 6.830000E-01f, 8.550811E-02f, 2.780072E-01f, 5.000000E-01f, 
7.840000E-01f, 1.090000E+00f, 9.251111E-01f, 9.220000E-01f, 9.888939E-02f, 2.844574E-01f, 7.777778E-01f, 
6.440000E-01f, 9.540000E-01f, 7.952308E-01f, 8.060000E-01f, 9.155158E-02f, 4.587548E-01f, 6.923077E-01f, 
6.200000E-01f, 8.220000E-01f, 6.826667E-01f, 6.640000E-01f, 5.734954E-02f, 2.518015E-01f, 5.000000E-01f, 
9.480000E-01f, 9.480000E-01f, 9.480000E-01f, 9.480000E-01f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
3.200000E-01f, 7.820000E-01f, 3.996471E-01f, 3.850000E-01f, 8.565395E-02f, 6.749607E-01f, 3.235294E-01f, 
3.340000E-01f, 5.320000E-01f, 3.912500E-01f, 3.720000E-01f, 5.310470E-02f, 3.327161E-01f, 3.750000E-01f, 
3.180000E-01f, 6.020000E-01f, 3.931429E-01f, 3.680000E-01f, 7.786031E-02f, 4.111156E-01f, 2.857143E-01f, 
6.920000E-01f, 1.234000E+00f, 9.062222E-01f, 9.260000E-01f, 1.607465E-01f, 5.282878E-01f, 5.555556E-01f, 
5.720000E-01f, 7.880000E-01f, 6.620000E-01f, 6.500000E-01f, 5.821226E-02f, 2.803854E-01f, 4.615385E-01f, 
7.480000E-01f, 1.148000E+00f, 9.186667E-01f, 9.280000E-01f, 1.008435E-01f, 6.176763E-01f, 3.888889E-01f, 
3.900000E-01f, 7.460000E-01f, 4.758824E-01f, 4.480000E-01f, 1.021579E-01f, 5.780069E-01f, 2.941176E-01f, 
3.240000E-01f, 5.080000E-01f, 3.976190E-01f, 3.900000E-01f, 4.981012E-02f, 2.656238E-01f, 1.904762E-01f, 
6.740000E-01f, 9.300000E-01f, 7.570909E-01f, 7.200000E-01f, 8.434863E-02f, 3.293934E-01f, 2.727273E-01f, 
4.900000E-01f, 8.560000E-01f, 6.911667E-01f, 6.880000E-01f, 9.584252E-02f, 4.213360E-01f, 4.166667E-01f, 
5.320000E-01f, 8.880000E-01f, 7.041667E-01f, 7.270000E-01f, 9.638732E-02f, 4.469273E-01f, 5.833333E-01f, 
7.480000E-01f, 1.298000E+00f, 9.195556E-01f, 9.080000E-01f, 1.795822E-01f, 4.951889E-01f, 4.444444E-01f, 
3.340000E-01f, 4.320000E-01f, 3.792174E-01f, 3.820000E-01f, 2.428422E-02f, 1.384197E-01f, 8.695652E-02f, 
2.760000E-01f, 4.980000E-01f, 3.356552E-01f, 3.320000E-01f, 4.519820E-02f, 3.533667E-01f, 3.103448E-01f, 
3.220000E-01f, 5.400000E-01f, 3.934545E-01f, 3.760000E-01f, 5.910179E-02f, 3.444358E-01f, 3.636364E-01f, 
2.960000E-01f, 4.260000E-01f, 3.934545E-01f, 4.010000E-01f, 2.644082E-02f, 1.343875E-01f, 9.090909E-02f, 
5.660000E-01f, 1.130000E+00f, 6.758182E-01f, 6.300000E-01f, 1.388962E-01f, 6.471167E-01f, 2.727273E-01f, 
6.520000E-01f, 8.320000E-01f, 7.341667E-01f, 7.190000E-01f, 4.971525E-02f, 2.167948E-01f, 4.166667E-01f, 
6.940000E-01f, 1.034000E+00f, 8.860000E-01f, 8.920000E-01f, 9.534299E-02f, 4.724574E-01f, 6.666667E-01f, 
7.580000E-01f, 8.640000E-01f, 8.085714E-01f, 8.000000E-01f, 3.246232E-02f, 2.003197E-01f, 5.000000E-01f, 
2.800000E-01f, 7.860000E-01f, 6.992135E-01f, 7.320000E-01f, 9.711035E-02f, 7.884237E-01f, 1.235955E-01f, 
7.880000E-01f, 1.012000E+00f, 8.828333E-01f, 8.810000E-01f, 6.864114E-02f, 3.505139E-01f, 6.666667E-01f, 
6.460000E-01f, 9.740000E-01f, 8.625946E-01f, 8.780000E-01f, 7.531838E-02f, 2.931211E-01f, 3.513514E-01f, 
7.400000E-01f, 1.048000E+00f, 8.951111E-01f, 8.960000E-01f, 7.707860E-02f, 3.766590E-01f, 3.333333E-01f, 
7.640000E-01f, 8.960000E-01f, 8.193636E-01f, 8.180000E-01f, 3.852216E-02f, 2.252732E-01f, 4.090909E-01f, 
7.660000E-01f, 9.040000E-01f, 8.189091E-01f, 8.020000E-01f, 4.881282E-02f, 2.402915E-01f, 5.454545E-01f, 
7.260000E-01f, 8.360000E-01f, 7.788000E-01f, 7.830000E-01f, 3.616260E-02f, 1.687009E-01f, 4.000000E-01f, 
6.540000E-01f, 1.318000E+00f, 7.298333E-01f, 6.800000E-01f, 1.854831E-01f, 9.119145E-01f, 1.666667E-01f, 
7.320000E-01f, 8.860000E-01f, 8.156667E-01f, 8.150000E-01f, 5.498319E-02f, 2.072197E-01f, 5.000000E-01f, 
7.620000E-01f, 9.660000E-01f, 8.711429E-01f, 8.740000E-01f, 6.002523E-02f, 2.651792E-01f, 5.238095E-01f, 
9.260000E-01f, 1.154000E+00f, 1.055000E+00f, 1.059000E+00f, 6.226865E-02f, 3.040789E-01f, 5.000000E-01f, 
5.880000E-01f, 7.020000E-01f, 6.458182E-01f, 6.430000E-01f, 3.528339E-02f, 1.944634E-01f, 3.636364E-01f, 
6.120000E-01f, 7.320000E-01f, 6.462105E-01f, 6.380000E-01f, 3.152350E-02f, 1.572387E-01f, 2.105263E-01f, 
5.780000E-01f, 7.780000E-01f, 6.155714E-01f, 6.040000E-01f, 4.970175E-02f, 2.375374E-01f, 1.428571E-01f, 
7.040000E-01f, 9.220000E-01f, 7.747500E-01f, 7.680000E-01f, 5.072282E-02f, 2.969781E-01f, 5.000000E-01f, 
2.520000E-01f, 9.700000E-01f, 8.296000E-01f, 9.530000E-01f, 2.375164E-01f, 9.464058E-01f, 4.000000E-01f, 
7.280000E-01f, 1.578000E+00f, 8.717778E-01f, 8.020000E-01f, 2.672460E-01f, 1.178847E+00f, 6.666667E-01f, 
7.440000E-01f, 9.240000E-01f, 8.342222E-01f, 8.610000E-01f, 5.629706E-02f, 2.875065E-01f, 4.444444E-01f, 
7.640000E-01f, 8.540000E-01f, 8.066000E-01f, 8.020000E-01f, 3.370526E-02f, 1.566652E-01f, 4.000000E-01f, 
6.200000E-01f, 8.540000E-01f, 7.268571E-01f, 7.280000E-01f, 7.185877E-02f, 1.914785E-01f, 2.142857E-01f, 
5.500000E-01f, 8.920000E-01f, 7.944167E-01f, 8.080000E-01f, 7.291682E-02f, 2.941496E-01f, 1.666667E-01f, 
7.640000E-01f, 9.380000E-01f, 8.480000E-01f, 8.440000E-01f, 5.318646E-02f, 2.352105E-01f, 6.363636E-01f, 
7.440000E-01f, 1.510000E+00f, 8.364444E-01f, 7.540000E-01f, 2.526218E-01f, 1.070652E+00f, 2.222222E-01f, 
2.560000E-01f, 1.052000E+00f, 9.147500E-01f, 9.990000E-01f, 2.671713E-01f, 7.546628E-01f, 2.500000E-01f, 
7.840000E-01f, 9.100000E-01f, 8.330769E-01f, 8.200000E-01f, 4.051844E-02f, 2.007287E-01f, 5.384615E-01f, 
6.280000E-01f, 1.278000E+00f, 6.873333E-01f, 6.460000E-01f, 1.640343E-01f, 8.773346E-01f, 1.333333E-01f, 
7.280000E-01f, 1.548000E+00f, 9.191111E-01f, 7.540000E-01f, 3.476911E-01f, 1.576765E+00f, 4.444444E-01f, 
7.620000E-01f, 9.160000E-01f, 8.458000E-01f, 8.420000E-01f, 4.766506E-02f, 2.607988E-01f, 5.000000E-01f, 
7.220000E-01f, 8.320000E-01f, 7.778182E-01f, 7.820000E-01f, 4.225120E-02f, 1.686891E-01f, 3.636364E-01f, 
3.600000E-01f, 1.022000E+00f, 6.732308E-01f, 7.200000E-01f, 1.772146E-01f, 7.770045E-01f, 3.076923E-01f, 
7.680000E-01f, 8.900000E-01f, 8.356667E-01f, 8.390000E-01f, 3.652977E-02f, 1.869010E-01f, 5.000000E-01f, 
9.280000E-01f, 1.042000E+00f, 9.841333E-01f, 9.940000E-01f, 3.602750E-02f, 2.078269E-01f, 6.000000E-01f, 
6.360000E-01f, 1.378000E+00f, 7.152000E-01f, 6.640000E-01f, 1.849151E-01f, 7.178356E-01f, 6.666667E-02f, 
7.780000E-01f, 9.760000E-01f, 8.734667E-01f, 8.750000E-01f, 6.189632E-02f, 3.659235E-01f, 5.666667E-01f, 
6.920000E-01f, 7.680000E-01f, 7.258182E-01f, 7.200000E-01f, 2.329729E-02f, 1.015874E-01f, 2.727273E-01f, 
8.760000E-01f, 1.014000E+00f, 9.490000E-01f, 9.650000E-01f, 4.673329E-02f, 1.753054E-01f, 6.250000E-01f, 
6.640000E-01f, 9.220000E-01f, 8.090909E-01f, 8.460000E-01f, 8.236195E-02f, 1.685586E-01f, 4.545455E-01f, 
6.900000E-01f, 8.340000E-01f, 7.675000E-01f, 7.650000E-01f, 4.734496E-02f, 1.500000E-01f, 2.500000E-01f, 
8.060000E-01f, 9.820000E-01f, 8.885263E-01f, 8.920000E-01f, 5.201642E-02f, 2.102570E-01f, 3.684211E-01f, 
6.020000E-01f, 7.040000E-01f, 6.578000E-01f, 6.610000E-01f, 2.862388E-02f, 1.654207E-01f, 2.500000E-01f, 
6.840000E-01f, 8.020000E-01f, 7.521667E-01f, 7.590000E-01f, 4.399552E-02f, 1.569459E-01f, 2.500000E-01f, 
8.340000E-01f, 9.420000E-01f, 8.932000E-01f, 8.850000E-01f, 3.514035E-02f, 1.426885E-01f, 4.000000E-01f, 
7.540000E-01f, 9.300000E-01f, 8.502609E-01f, 8.540000E-01f, 4.403017E-02f, 3.057777E-01f, 4.782609E-01f, 
7.720000E-01f, 1.256000E+00f, 1.066000E+00f, 1.067000E+00f, 1.466814E-01f, 5.415718E-01f, 5.000000E-01f, 
7.360000E-01f, 9.080000E-01f, 8.303529E-01f, 8.520000E-01f, 5.244156E-02f, 3.179560E-01f, 7.058824E-01f, 
7.560000E-01f, 8.780000E-01f, 8.378333E-01f, 8.420000E-01f, 3.554724E-02f, 1.481756E-01f, 3.333333E-01f, 
6.600000E-01f, 8.820000E-01f, 7.729167E-01f, 7.800000E-01f, 6.570283E-02f, 4.008691E-01f, 4.583333E-01f, 
7.480000E-01f, 1.616000E+00f, 8.833750E-01f, 8.430000E-01f, 2.007741E-01f, 1.076710E+00f, 4.375000E-01f, 
8.040000E-01f, 9.620000E-01f, 8.901333E-01f, 8.960000E-01f, 5.640398E-02f, 3.055421E-01f, 7.333333E-01f, 
3.260000E-01f, 1.566000E+00f, 7.832973E-01f, 7.680000E-01f, 2.076681E-01f, 1.667809E+00f, 6.486486E-01f, 
7.500000E-01f, 9.340000E-01f, 8.331613E-01f, 8.220000E-01f, 4.954129E-02f, 2.581550E-01f, 3.225806E-01f, 
7.740000E-01f, 9.960000E-01f, 8.458824E-01f, 8.400000E-01f, 5.646003E-02f, 3.029851E-01f, 3.529412E-01f, 
6.720000E-01f, 8.620000E-01f, 7.899512E-01f, 7.960000E-01f, 4.182700E-02f, 2.580310E-01f, 2.439024E-01f, 
7.960000E-01f, 9.100000E-01f, 8.544615E-01f, 8.520000E-01f, 4.228596E-02f, 1.730087E-01f, 3.846154E-01f, 
7.300000E-01f, 8.240000E-01f, 7.811250E-01f, 7.920000E-01f, 2.840628E-02f, 1.886690E-01f, 3.750000E-01f, 
8.240000E-01f, 1.042000E+00f, 9.134167E-01f, 9.070000E-01f, 5.584911E-02f, 3.047688E-01f, 5.000000E-01f, 
7.940000E-01f, 9.740000E-01f, 8.716923E-01f, 8.780000E-01f, 6.004913E-02f, 2.807917E-01f, 6.153846E-01f, 
7.320000E-01f, 8.500000E-01f, 7.875000E-01f, 7.770000E-01f, 3.875447E-02f, 1.663971E-01f, 3.333333E-01f, 
5.440000E-01f, 1.196000E+00f, 6.228182E-01f, 5.950000E-01f, 1.328209E-01f, 8.453165E-01f, 1.818182E-01f, 
5.980000E-01f, 1.088000E+00f, 8.550000E-01f, 8.560000E-01f, 1.120341E-01f, 5.723181E-01f, 5.000000E-01f, 
7.880000E-01f, 9.340000E-01f, 8.596667E-01f, 8.570000E-01f, 5.152110E-02f, 1.835211E-01f, 3.333333E-01f, 
7.180000E-01f, 8.800000E-01f, 7.793514E-01f, 7.720000E-01f, 3.879592E-02f, 3.214156E-01f, 3.783784E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
9.140000E-01f, 1.028000E+00f, 9.756842E-01f, 9.800000E-01f, 3.387732E-02f, 2.151279E-01f, 5.263158E-01f, 
5.840000E-01f, 6.780000E-01f, 6.357778E-01f, 6.340000E-01f, 2.749878E-02f, 1.767824E-01f, 2.222222E-01f, 
5.720000E-01f, 6.740000E-01f, 6.170588E-01f, 6.140000E-01f, 2.949676E-02f, 1.315599E-01f, 1.764706E-01f, 
9.420000E-01f, 1.054000E+00f, 9.910000E-01f, 9.790000E-01f, 4.258772E-02f, 1.365137E-01f, 5.000000E-01f, 
8.500000E-01f, 9.960000E-01f, 9.267000E-01f, 9.300000E-01f, 4.154528E-02f, 3.072328E-01f, 6.500000E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.380000E-01f, 7.500000E-01f, 6.248800E-01f, 6.180000E-01f, 5.343264E-02f, 2.637120E-01f, 1.800000E-01f, 
8.780000E-01f, 1.006000E+00f, 9.513333E-01f, 9.540000E-01f, 4.873397E-02f, 1.724645E-01f, 4.444444E-01f, 
3.900000E-01f, 1.030000E+00f, 8.624615E-01f, 9.660000E-01f, 2.367884E-01f, 9.405573E-01f, 3.846154E-01f, 
8.300000E-01f, 9.900000E-01f, 9.186667E-01f, 9.420000E-01f, 6.109010E-02f, 2.588127E-01f, 5.555556E-01f, 
8.180000E-01f, 9.540000E-01f, 8.731667E-01f, 8.840000E-01f, 4.485499E-02f, 2.049195E-01f, 6.666667E-01f, 
7.460000E-01f, 8.520000E-01f, 7.986667E-01f, 7.970000E-01f, 3.687160E-02f, 1.871363E-01f, 5.000000E-01f, 
7.140000E-01f, 8.560000E-01f, 7.849091E-01f, 7.780000E-01f, 4.383938E-02f, 1.534275E-01f, 2.727273E-01f, 
8.620000E-01f, 9.940000E-01f, 9.366000E-01f, 9.490000E-01f, 4.252603E-02f, 2.416361E-01f, 5.000000E-01f, 
7.280000E-01f, 8.960000E-01f, 8.026667E-01f, 7.810000E-01f, 5.530960E-02f, 2.735032E-01f, 5.833333E-01f, 
5.580000E-01f, 8.240000E-01f, 6.809333E-01f, 6.820000E-01f, 5.156337E-02f, 3.148460E-01f, 2.000000E-01f, 
7.240000E-01f, 8.860000E-01f, 8.026667E-01f, 8.090000E-01f, 4.589580E-02f, 1.915620E-01f, 3.333333E-01f, 
7.640000E-01f, 8.920000E-01f, 8.221333E-01f, 8.100000E-01f, 4.324988E-02f, 1.950795E-01f, 3.333333E-01f, 
6.420000E-01f, 8.700000E-01f, 7.151111E-01f, 6.910000E-01f, 6.473193E-02f, 2.670431E-01f, 2.777778E-01f, 
6.720000E-01f, 8.520000E-01f, 7.850588E-01f, 8.060000E-01f, 5.498235E-02f, 1.733551E-01f, 2.941176E-01f, 
7.560000E-01f, 1.606000E+00f, 8.786667E-01f, 7.900000E-01f, 2.736074E-01f, 8.529877E-01f, 1.111111E-01f, 
7.740000E-01f, 1.608000E+00f, 8.947500E-01f, 7.960000E-01f, 2.883251E-01f, 1.168378E+00f, 2.500000E-01f, 
6.860000E-01f, 7.940000E-01f, 7.385946E-01f, 7.360000E-01f, 3.211737E-02f, 2.501360E-01f, 2.702703E-01f, 
6.760000E-01f, 7.720000E-01f, 7.301667E-01f, 7.260000E-01f, 3.062332E-02f, 1.415627E-01f, 2.500000E-01f, 
3.600000E-01f, 1.800000E+00f, 9.042222E-01f, 8.980000E-01f, 3.914964E-01f, 1.431071E+00f, 5.555556E-01f, 
6.540000E-01f, 7.740000E-01f, 7.162105E-01f, 7.160000E-01f, 3.463393E-02f, 2.253797E-01f, 4.210526E-01f, 
7.160000E-01f, 8.040000E-01f, 7.540000E-01f, 7.560000E-01f, 2.657408E-02f, 1.638780E-01f, 3.333333E-01f, 
6.820000E-01f, 8.420000E-01f, 7.350909E-01f, 7.340000E-01f, 4.071721E-02f, 2.104281E-01f, 3.636364E-01f, 
6.700000E-01f, 8.500000E-01f, 7.436667E-01f, 7.470000E-01f, 4.200420E-02f, 2.420826E-01f, 4.444444E-01f, 
3.060000E-01f, 1.112000E+00f, 7.137778E-01f, 6.540000E-01f, 2.454474E-01f, 8.917017E-01f, 4.444444E-01f, 
}; /**<  Support vectors */

#endif