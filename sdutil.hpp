#pragma once

#include <cassert>

#include <vector>

//
// for matrix and vector
//

// MKL support
#define EIGEN_USE_MKL_ALL
#include "Eigen/eigen"

//typedef Eigen::MatrixXd MatrixXd;
//typedef Eigen::VectorXd VectorXd;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Vector3d;


#define ASSERT_MATRIX_SIZE(mat,nrow,ncol) assert(mat.rows()==(nrow) && mat.cols()==(ncol))


//
// Special constants
//
//namespace {

// identity tensor
const double Delta[3][3] = { 
	1, 0, 0, 
	0, 1, 0, 
	0, 0, 1, 
};

// permutation tensor
const double Epsil[3][3][3] = { 
	0,  0,  0, // (0,0,*)
	0,  0,  1, // (0,1,*)
	0,  -1, 0, // (0,2,*)
	0,  0,  -1, // (1,0,*)
	0,  0,  0, // (1,1,*)
	1,  0,  0, // (1,2,*)
	0,  1,  0, // (2,0,*)
	-1, 0,  0, // (2,1,*)
	0,  0,  0, // (2,2,*)
};

// Math
const double PI = 3.14159265358979323846;
const double TwoPI = PI * 2;
const double FourPI = PI * 4;
const double SixPI = PI * 6;
const double EightPI = PI * 8;
const double SqrtPI = sqrt(PI);


//}

//
// Math functions
//
double math_erf(double x);
double math_erfc(double x);
