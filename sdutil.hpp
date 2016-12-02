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



// identity
const double Delta[3][3] = { 
	1, 0, 0, 
	0, 1, 0, 
	0, 0, 1, 
};

// permutation
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




