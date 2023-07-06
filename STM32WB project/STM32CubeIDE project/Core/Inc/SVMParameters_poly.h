#ifndef _SVM_poly_Parameters
#define _SVM_poly_Parameters
#include "arm_math.h"

#define poly_NB_SUPPORT_VECTORS  151
#define poly_VECTOR_DIMENSION 7
/*
Those parameters was generated with the scikit-learn and Marek's script.
*/
//Classes: ['AF'  'Normal' ]
const int32_t polyClasses[2]={ 0,  1};

const float32_t polyIntercept  = 0.534915f;
const float32_t polyCoef0 = 0.000000f;
const float32_t polyDegree = 3.000000f;
const float32_t polyGamma = 0.384151f;
const float32_t polyDualCoefficients[poly_NB_SUPPORT_VECTORS ] = {
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -1.573452E+03f, -5.000000E+03f, -5.000000E+03f, -6.821400E+01f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -4.397371E+03f, -5.000000E+03f, 
-3.728256E+02f, -5.000000E+03f, -1.035676E+02f, -5.000000E+03f, -5.000000E+03f, -5.047725E+02f, -5.000000E+03f, 
-1.981294E+02f, -5.000000E+03f, -3.490608E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -7.759160E+02f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.151993E+01f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-7.830398E+02f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -7.282299E+01f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
-5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, -5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 3.409747E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 2.541441E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 9.621695E+02f, 5.000000E+03f, 
7.004262E+02f, 5.000000E+03f, 5.000000E+03f, 2.883495E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 4.352611E+03f, 4.416636E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
4.902500E+03f, 5.000000E+03f, 5.000000E+03f, 3.452316E+03f, 1.396177E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 1.009649E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 2.205471E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 2.905070E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 2.667539E+03f, 5.000000E+03f, 4.586992E+03f, 
5.000000E+03f, 5.000000E+03f, 5.000000E+03f, 5.000000E+03f, }; /**< Dual coefficients */

const float32_t polySupportVectors[poly_NB_SUPPORT_VECTORS*poly_VECTOR_DIMENSION] = {
3.300000E-01f, 5.700000E-01f, 4.875556E-01f, 4.990000E-01f, 5.725713E-02f, 3.247522E-01f, 3.333333E-01f, 
4.700000E-01f, 4.700000E-01f, 4.700000E-01f, 4.700000E-01f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
2.680000E-01f, 4.500000E-01f, 3.487200E-01f, 3.440000E-01f, 4.469482E-02f, 2.286744E-01f, 2.400000E-01f, 
4.540000E-01f, 1.236000E+00f, 6.275172E-01f, 6.160000E-01f, 1.444137E-01f, 6.798588E-01f, 2.068966E-01f, 
9.240000E-01f, 9.560000E-01f, 9.392500E-01f, 9.350000E-01f, 1.084633E-02f, 2.734959E-02f, 0.000000E+00f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
1.330000E+00f, 1.474000E+00f, 1.397200E+00f, 1.386000E+00f, 5.363954E-02f, 1.694816E-01f, 6.000000E-01f, 
2.980000E-01f, 4.720000E-01f, 3.240741E-01f, 3.120000E-01f, 4.077888E-02f, 2.999133E-01f, 1.481481E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
3.040000E-01f, 5.020000E-01f, 4.116000E-01f, 4.080000E-01f, 5.134240E-02f, 2.347424E-01f, 4.000000E-01f, 
4.060000E-01f, 9.600000E-01f, 6.813846E-01f, 7.980000E-01f, 2.106314E-01f, 6.173686E-01f, 3.846154E-01f, 
4.820000E-01f, 6.380000E-01f, 5.514286E-01f, 5.530000E-01f, 4.455889E-02f, 2.217927E-01f, 4.285714E-01f, 
6.580000E-01f, 8.940000E-01f, 7.630909E-01f, 7.860000E-01f, 8.757791E-02f, 1.701176E-01f, 3.636364E-01f, 
4.640000E-01f, 3.016000E+00f, 2.145000E+00f, 2.550000E+00f, 1.142416E+00f, 2.993275E+00f, 7.500000E-01f, 
3.040000E-01f, 5.020000E-01f, 4.116000E-01f, 4.080000E-01f, 5.134240E-02f, 2.347424E-01f, 4.000000E-01f, 
5.720000E-01f, 7.880000E-01f, 6.620000E-01f, 6.500000E-01f, 5.821226E-02f, 2.803854E-01f, 4.615385E-01f, 
2.960000E-01f, 4.260000E-01f, 3.934545E-01f, 4.010000E-01f, 2.644082E-02f, 1.343875E-01f, 9.090909E-02f, 
2.860000E-01f, 4.560000E-01f, 3.505833E-01f, 3.270000E-01f, 5.233497E-02f, 3.291443E-01f, 3.750000E-01f, 
5.120000E-01f, 7.220000E-01f, 6.088571E-01f, 6.000000E-01f, 6.317601E-02f, 2.740146E-01f, 5.000000E-01f, 
5.620000E-01f, 8.780000E-01f, 7.352727E-01f, 7.500000E-01f, 8.384401E-02f, 4.552318E-01f, 4.545455E-01f, 
2.560000E-01f, 5.200000E-01f, 3.217037E-01f, 3.000000E-01f, 7.867304E-02f, 5.276021E-01f, 2.592593E-01f, 
1.074000E+00f, 1.814000E+00f, 1.615600E+00f, 1.770000E+00f, 3.113692E-01f, 9.048271E-01f, 6.000000E-01f, 
1.308000E+00f, 1.346000E+00f, 1.323667E+00f, 1.317000E+00f, 1.626858E-02f, 5.589275E-02f, 0.000000E+00f, 
3.460000E-01f, 1.686000E+00f, 7.802857E-01f, 6.870000E-01f, 3.682348E-01f, 1.771804E+00f, 7.857143E-01f, 
3.120000E-01f, 5.020000E-01f, 4.146000E-01f, 4.130000E-01f, 4.619000E-02f, 2.998266E-01f, 4.000000E-01f, 
4.780000E-01f, 6.880000E-01f, 5.692000E-01f, 5.720000E-01f, 6.646073E-02f, 2.521032E-01f, 4.666667E-01f, 
4.880000E-01f, 2.256000E+00f, 1.318889E+00f, 1.218000E+00f, 5.829572E-01f, 2.127811E+00f, 8.888889E-01f, 
5.260000E-01f, 7.460000E-01f, 6.496000E-01f, 6.480000E-01f, 7.623722E-02f, 3.101935E-01f, 3.333333E-01f, 
4.860000E-01f, 2.704000E+00f, 1.171000E+00f, 1.080000E+00f, 8.015902E-01f, 2.814984E+00f, 6.666667E-01f, 
6.000000E-01f, 6.100000E-01f, 6.048571E-01f, 6.060000E-01f, 3.109715E-03f, 1.685230E-02f, 0.000000E+00f, 
3.140000E-01f, 1.388000E+00f, 7.065600E-01f, 7.180000E-01f, 1.894400E-01f, 1.085318E+00f, 5.200000E-01f, 
3.100000E-01f, 5.320000E-01f, 3.930909E-01f, 3.650000E-01f, 6.859329E-02f, 3.213160E-01f, 3.181818E-01f, 
8.000000E-01f, 1.084000E+00f, 9.530000E-01f, 9.610000E-01f, 1.097165E-01f, 3.333827E-01f, 5.000000E-01f, 
4.180000E-01f, 4.720000E-01f, 4.450000E-01f, 4.450000E-01f, 3.818377E-02f, 5.400000E-02f, 5.000000E-01f, 
3.060000E-01f, 5.160000E-01f, 3.753913E-01f, 3.580000E-01f, 6.615803E-02f, 4.535416E-01f, 3.478261E-01f, 
3.540000E-01f, 5.300000E-01f, 4.372632E-01f, 4.380000E-01f, 5.238834E-02f, 2.653224E-01f, 3.684211E-01f, 
5.120000E-01f, 7.220000E-01f, 6.088571E-01f, 6.000000E-01f, 6.317601E-02f, 2.740146E-01f, 5.000000E-01f, 
6.020000E-01f, 9.500000E-01f, 8.238000E-01f, 8.280000E-01f, 9.125520E-02f, 3.914486E-01f, 4.000000E-01f, 
3.940000E-01f, 6.280000E-01f, 5.462500E-01f, 5.660000E-01f, 7.429984E-02f, 4.591993E-01f, 3.750000E-01f, 
3.240000E-01f, 5.080000E-01f, 3.976190E-01f, 3.900000E-01f, 4.981012E-02f, 2.656238E-01f, 1.904762E-01f, 
3.760000E-01f, 7.160000E-01f, 4.393939E-01f, 4.100000E-01f, 8.420137E-02f, 5.483977E-01f, 3.636364E-01f, 
6.740000E-01f, 9.300000E-01f, 7.570909E-01f, 7.200000E-01f, 8.434863E-02f, 3.293934E-01f, 2.727273E-01f, 
1.494000E+00f, 1.510000E+00f, 1.503750E+00f, 1.506000E+00f, 5.650959E-03f, 6.633250E-03f, 0.000000E+00f, 
2.500000E-01f, 1.640000E+00f, 6.280000E-01f, 4.900000E-01f, 3.931429E-01f, 2.429807E+00f, 8.947368E-01f, 
4.900000E-01f, 6.900000E-01f, 5.506087E-01f, 5.500000E-01f, 4.057063E-02f, 2.786683E-01f, 3.043478E-01f, 
1.520000E+00f, 1.524000E+00f, 1.522400E+00f, 1.522000E+00f, 1.673320E-03f, 4.898979E-03f, 0.000000E+00f, 
6.900000E-01f, 9.440000E-01f, 8.443077E-01f, 8.500000E-01f, 7.257110E-02f, 3.373485E-01f, 4.615385E-01f, 
2.840000E-01f, 4.880000E-01f, 3.464167E-01f, 3.350000E-01f, 4.435814E-02f, 3.277682E-01f, 3.750000E-01f, 
3.940000E-01f, 5.820000E-01f, 4.711304E-01f, 4.580000E-01f, 5.644571E-02f, 3.044799E-01f, 3.043478E-01f, 
3.122000E+00f, 3.122000E+00f, 3.122000E+00f, 3.122000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
2.500000E-01f, 6.980000E-01f, 3.343077E-01f, 2.780000E-01f, 1.168817E-01f, 7.081666E-01f, 2.692308E-01f, 
7.500000E-01f, 1.010000E+00f, 8.713333E-01f, 8.640000E-01f, 7.078842E-02f, 3.662786E-01f, 5.555556E-01f, 
7.540000E-01f, 9.900000E-01f, 8.460000E-01f, 8.560000E-01f, 7.215262E-02f, 3.470620E-01f, 4.444444E-01f, 
2.920000E-01f, 5.020000E-01f, 3.918182E-01f, 3.780000E-01f, 5.898477E-02f, 3.543050E-01f, 3.181818E-01f, 
2.800000E-01f, 2.334000E+00f, 7.786000E-01f, 4.480000E-01f, 6.667687E-01f, 2.133843E+00f, 7.000000E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.640000E-01f, 8.260000E-01f, 6.770000E-01f, 6.300000E-01f, 9.024613E-02f, 2.796212E-01f, 1.666667E-01f, 
3.180000E-01f, 6.020000E-01f, 3.931429E-01f, 3.680000E-01f, 7.786031E-02f, 4.111156E-01f, 2.857143E-01f, 
6.400000E-01f, 6.820000E-01f, 6.641818E-01f, 6.660000E-01f, 1.247252E-02f, 6.033241E-02f, 0.000000E+00f, 
6.200000E-01f, 8.220000E-01f, 6.826667E-01f, 6.640000E-01f, 5.734954E-02f, 2.518015E-01f, 5.000000E-01f, 
9.480000E-01f, 9.480000E-01f, 9.480000E-01f, 9.480000E-01f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
3.200000E-01f, 7.820000E-01f, 3.996471E-01f, 3.850000E-01f, 8.565395E-02f, 6.749607E-01f, 3.235294E-01f, 
4.640000E-01f, 1.104000E+00f, 6.650000E-01f, 5.880000E-01f, 2.252786E-01f, 7.057903E-01f, 3.000000E-01f, 
3.020000E-01f, 9.320000E-01f, 4.281000E-01f, 3.740000E-01f, 1.535070E-01f, 9.511109E-01f, 3.500000E-01f, 
3.340000E-01f, 5.320000E-01f, 3.912500E-01f, 3.720000E-01f, 5.310470E-02f, 3.327161E-01f, 3.750000E-01f, 
3.180000E-01f, 6.020000E-01f, 3.931429E-01f, 3.680000E-01f, 7.786031E-02f, 4.111156E-01f, 2.857143E-01f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
5.720000E-01f, 7.880000E-01f, 6.620000E-01f, 6.500000E-01f, 5.821226E-02f, 2.803854E-01f, 4.615385E-01f, 
7.480000E-01f, 1.148000E+00f, 9.186667E-01f, 9.280000E-01f, 1.008435E-01f, 6.176763E-01f, 3.888889E-01f, 
3.900000E-01f, 7.460000E-01f, 4.758824E-01f, 4.480000E-01f, 1.021579E-01f, 5.780069E-01f, 2.941176E-01f, 
3.240000E-01f, 5.080000E-01f, 3.976190E-01f, 3.900000E-01f, 4.981012E-02f, 2.656238E-01f, 1.904762E-01f, 
6.740000E-01f, 9.300000E-01f, 7.570909E-01f, 7.200000E-01f, 8.434863E-02f, 3.293934E-01f, 2.727273E-01f, 
3.340000E-01f, 4.320000E-01f, 3.792174E-01f, 3.820000E-01f, 2.428422E-02f, 1.384197E-01f, 8.695652E-02f, 
2.760000E-01f, 4.980000E-01f, 3.356552E-01f, 3.320000E-01f, 4.519820E-02f, 3.533667E-01f, 3.103448E-01f, 
3.220000E-01f, 5.400000E-01f, 3.934545E-01f, 3.760000E-01f, 5.910179E-02f, 3.444358E-01f, 3.636364E-01f, 
2.960000E-01f, 4.260000E-01f, 3.934545E-01f, 4.010000E-01f, 2.644082E-02f, 1.343875E-01f, 9.090909E-02f, 
5.660000E-01f, 1.130000E+00f, 6.758182E-01f, 6.300000E-01f, 1.388962E-01f, 6.471167E-01f, 2.727273E-01f, 
6.520000E-01f, 8.320000E-01f, 7.341667E-01f, 7.190000E-01f, 4.971525E-02f, 2.167948E-01f, 4.166667E-01f, 
6.940000E-01f, 1.034000E+00f, 8.860000E-01f, 8.920000E-01f, 9.534299E-02f, 4.724574E-01f, 6.666667E-01f, 
7.880000E-01f, 1.012000E+00f, 8.828333E-01f, 8.810000E-01f, 6.864114E-02f, 3.505139E-01f, 6.666667E-01f, 
7.660000E-01f, 9.040000E-01f, 8.189091E-01f, 8.020000E-01f, 4.881282E-02f, 2.402915E-01f, 5.454545E-01f, 
5.760000E-01f, 5.860000E-01f, 5.804000E-01f, 5.800000E-01f, 2.164651E-03f, 8.246211E-03f, 0.000000E+00f, 
6.680000E-01f, 6.840000E-01f, 6.760000E-01f, 6.750000E-01f, 5.393599E-03f, 2.607681E-02f, 0.000000E+00f, 
7.620000E-01f, 9.660000E-01f, 8.711429E-01f, 8.740000E-01f, 6.002523E-02f, 2.651792E-01f, 5.238095E-01f, 
7.540000E-01f, 7.680000E-01f, 7.583636E-01f, 7.580000E-01f, 4.365151E-03f, 1.685230E-02f, 0.000000E+00f, 
5.880000E-01f, 7.020000E-01f, 6.458182E-01f, 6.430000E-01f, 3.528339E-02f, 1.944634E-01f, 3.636364E-01f, 
6.120000E-01f, 7.320000E-01f, 6.462105E-01f, 6.380000E-01f, 3.152350E-02f, 1.572387E-01f, 2.105263E-01f, 
7.040000E-01f, 9.220000E-01f, 7.747500E-01f, 7.680000E-01f, 5.072282E-02f, 2.969781E-01f, 5.000000E-01f, 
7.040000E-01f, 7.180000E-01f, 7.109565E-01f, 7.100000E-01f, 4.215537E-03f, 1.907878E-02f, 0.000000E+00f, 
6.840000E-01f, 7.000000E-01f, 6.915000E-01f, 6.920000E-01f, 4.833595E-03f, 2.097618E-02f, 0.000000E+00f, 
6.060000E-01f, 6.300000E-01f, 6.175385E-01f, 6.180000E-01f, 6.226494E-03f, 3.605551E-02f, 0.000000E+00f, 
7.280000E-01f, 1.578000E+00f, 8.717778E-01f, 8.020000E-01f, 2.672460E-01f, 1.178847E+00f, 6.666667E-01f, 
8.620000E-01f, 8.760000E-01f, 8.693333E-01f, 8.660000E-01f, 5.196152E-03f, 1.612452E-02f, 0.000000E+00f, 
5.980000E-01f, 6.080000E-01f, 6.047692E-01f, 6.060000E-01f, 2.650593E-03f, 1.249000E-02f, 0.000000E+00f, 
6.680000E-01f, 6.840000E-01f, 6.748333E-01f, 6.720000E-01f, 5.936533E-03f, 1.939072E-02f, 0.000000E+00f, 
6.200000E-01f, 8.540000E-01f, 7.268571E-01f, 7.280000E-01f, 7.185877E-02f, 1.914785E-01f, 2.142857E-01f, 
7.280000E-01f, 1.548000E+00f, 9.191111E-01f, 7.540000E-01f, 3.476911E-01f, 1.576765E+00f, 4.444444E-01f, 
6.540000E-01f, 6.780000E-01f, 6.680000E-01f, 6.700000E-01f, 6.831301E-03f, 2.569047E-02f, 0.000000E+00f, 
3.600000E-01f, 1.022000E+00f, 6.732308E-01f, 7.200000E-01f, 1.772146E-01f, 7.770045E-01f, 3.076923E-01f, 
6.800000E-01f, 7.020000E-01f, 6.906667E-01f, 6.900000E-01f, 6.786796E-03f, 2.000000E-02f, 0.000000E+00f, 
6.560000E-01f, 6.660000E-01f, 6.613333E-01f, 6.610000E-01f, 3.446562E-03f, 1.600000E-02f, 0.000000E+00f, 
6.540000E-01f, 6.800000E-01f, 6.695000E-01f, 6.700000E-01f, 7.821881E-03f, 3.059412E-02f, 0.000000E+00f, 
7.780000E-01f, 9.760000E-01f, 8.734667E-01f, 8.750000E-01f, 6.189632E-02f, 3.659235E-01f, 5.666667E-01f, 
6.280000E-01f, 6.520000E-01f, 6.398462E-01f, 6.420000E-01f, 6.348834E-03f, 2.416609E-02f, 0.000000E+00f, 
6.520000E-01f, 6.680000E-01f, 6.600000E-01f, 6.600000E-01f, 4.830459E-03f, 1.854724E-02f, 0.000000E+00f, 
7.200000E-01f, 8.100000E-01f, 7.541818E-01f, 7.440000E-01f, 3.256323E-02f, 1.403567E-01f, 2.727273E-01f, 
6.640000E-01f, 9.220000E-01f, 8.090909E-01f, 8.460000E-01f, 8.236195E-02f, 1.685586E-01f, 4.545455E-01f, 
6.900000E-01f, 8.340000E-01f, 7.675000E-01f, 7.650000E-01f, 4.734496E-02f, 1.500000E-01f, 2.500000E-01f, 
5.880000E-01f, 5.980000E-01f, 5.938571E-01f, 5.940000E-01f, 2.983471E-03f, 1.039230E-02f, 0.000000E+00f, 
6.160000E-01f, 6.280000E-01f, 6.210000E-01f, 6.200000E-01f, 3.486237E-03f, 1.000000E-02f, 0.000000E+00f, 
6.700000E-01f, 6.900000E-01f, 6.778333E-01f, 6.780000E-01f, 6.235286E-03f, 1.637071E-02f, 0.000000E+00f, 
7.720000E-01f, 1.256000E+00f, 1.066000E+00f, 1.067000E+00f, 1.466814E-01f, 5.415718E-01f, 5.000000E-01f, 
7.360000E-01f, 9.080000E-01f, 8.303529E-01f, 8.520000E-01f, 5.244156E-02f, 3.179560E-01f, 7.058824E-01f, 
6.720000E-01f, 7.000000E-01f, 6.878333E-01f, 6.900000E-01f, 9.475647E-03f, 1.800000E-02f, 0.000000E+00f, 
6.600000E-01f, 8.820000E-01f, 7.729167E-01f, 7.800000E-01f, 6.570283E-02f, 4.008691E-01f, 4.583333E-01f, 
7.480000E-01f, 1.616000E+00f, 8.833750E-01f, 8.430000E-01f, 2.007741E-01f, 1.076710E+00f, 4.375000E-01f, 
3.260000E-01f, 1.566000E+00f, 7.832973E-01f, 7.680000E-01f, 2.076681E-01f, 1.667809E+00f, 6.486486E-01f, 
5.980000E-01f, 6.120000E-01f, 6.040000E-01f, 6.020000E-01f, 4.472136E-03f, 2.306513E-02f, 0.000000E+00f, 
7.940000E-01f, 9.960000E-01f, 8.574000E-01f, 8.360000E-01f, 6.830032E-02f, 1.437498E-01f, 2.000000E-01f, 
8.000000E-01f, 8.180000E-01f, 8.080000E-01f, 8.060000E-01f, 5.734884E-03f, 2.374868E-02f, 0.000000E+00f, 
8.240000E-01f, 1.042000E+00f, 9.134167E-01f, 9.070000E-01f, 5.584911E-02f, 3.047688E-01f, 5.000000E-01f, 
5.980000E-01f, 1.088000E+00f, 8.550000E-01f, 8.560000E-01f, 1.120341E-01f, 5.723181E-01f, 5.000000E-01f, 
7.200000E-01f, 7.400000E-01f, 7.305455E-01f, 7.300000E-01f, 6.875517E-03f, 2.190890E-02f, 0.000000E+00f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
7.920000E-01f, 8.160000E-01f, 7.996471E-01f, 7.960000E-01f, 7.976067E-03f, 1.833030E-02f, 0.000000E+00f, 
5.720000E-01f, 6.740000E-01f, 6.170588E-01f, 6.140000E-01f, 2.949676E-02f, 1.315599E-01f, 1.764706E-01f, 
1.000000E+00f, 1.000000E+00f, 1.000000E+00f, 1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
7.120000E-01f, 7.500000E-01f, 7.316364E-01f, 7.320000E-01f, 1.447254E-02f, 2.097618E-02f, 0.000000E+00f, 
7.520000E-01f, 7.780000E-01f, 7.663636E-01f, 7.660000E-01f, 9.330303E-03f, 1.536229E-02f, 0.000000E+00f, 
6.220000E-01f, 6.320000E-01f, 6.260800E-01f, 6.260000E-01f, 2.481935E-03f, 1.918333E-02f, 0.000000E+00f, 
-1.000000E+00f, -1.000000E+00f, -1.000000E+00f, -1.000000E+00f, 0.000000E+00f, 0.000000E+00f, 0.000000E+00f, 
7.460000E-01f, 7.600000E-01f, 7.532000E-01f, 7.540000E-01f, 5.432413E-03f, 2.200000E-02f, 0.000000E+00f, 
3.900000E-01f, 1.030000E+00f, 8.624615E-01f, 9.660000E-01f, 2.367884E-01f, 9.405573E-01f, 3.846154E-01f, 
6.040000E-01f, 6.300000E-01f, 6.200000E-01f, 6.190000E-01f, 7.358930E-03f, 3.218695E-02f, 0.000000E+00f, 
5.680000E-01f, 6.040000E-01f, 5.850476E-01f, 5.820000E-01f, 1.089255E-02f, 4.019950E-02f, 0.000000E+00f, 
7.280000E-01f, 8.960000E-01f, 8.026667E-01f, 7.810000E-01f, 5.530960E-02f, 2.735032E-01f, 5.833333E-01f, 
7.300000E-01f, 7.440000E-01f, 7.372727E-01f, 7.380000E-01f, 3.926599E-03f, 1.766352E-02f, 0.000000E+00f, 
7.080000E-01f, 7.300000E-01f, 7.160000E-01f, 7.150000E-01f, 7.385489E-03f, 2.630589E-02f, 0.000000E+00f, 
6.020000E-01f, 6.200000E-01f, 6.086154E-01f, 6.080000E-01f, 5.188745E-03f, 1.280625E-02f, 0.000000E+00f, 
6.880000E-01f, 7.800000E-01f, 7.215385E-01f, 7.140000E-01f, 2.839195E-02f, 1.217867E-01f, 2.307692E-01f, 
6.420000E-01f, 8.700000E-01f, 7.151111E-01f, 6.910000E-01f, 6.473193E-02f, 2.670431E-01f, 2.777778E-01f, 
9.920000E-01f, 1.084000E+00f, 1.028286E+00f, 1.016000E+00f, 3.343508E-02f, 7.954873E-02f, 1.428571E-01f, 
6.880000E-01f, 7.060000E-01f, 6.996667E-01f, 7.020000E-01f, 5.959459E-03f, 1.989975E-02f, 0.000000E+00f, 
8.000000E-01f, 8.220000E-01f, 8.106000E-01f, 8.110000E-01f, 6.669999E-03f, 1.414214E-02f, 0.000000E+00f, 
3.600000E-01f, 1.800000E+00f, 9.042222E-01f, 8.980000E-01f, 3.914964E-01f, 1.431071E+00f, 5.555556E-01f, 
9.220000E-01f, 9.340000E-01f, 9.251111E-01f, 9.240000E-01f, 3.887301E-03f, 1.469694E-02f, 0.000000E+00f, 
8.660000E-01f, 8.860000E-01f, 8.748889E-01f, 8.740000E-01f, 8.069146E-03f, 2.262742E-02f, 0.000000E+00f, 
5.960000E-01f, 6.220000E-01f, 6.108571E-01f, 6.120000E-01f, 7.553472E-03f, 2.727636E-02f, 0.000000E+00f, 
6.780000E-01f, 6.960000E-01f, 6.852000E-01f, 6.840000E-01f, 5.977736E-03f, 1.600000E-02f, 0.000000E+00f, 
6.700000E-01f, 8.500000E-01f, 7.436667E-01f, 7.470000E-01f, 4.200420E-02f, 2.420826E-01f, 4.444444E-01f, 
3.060000E-01f, 1.112000E+00f, 7.137778E-01f, 6.540000E-01f, 2.454474E-01f, 8.917017E-01f, 4.444444E-01f, 
}; /**<  Support vectors */

#endif
