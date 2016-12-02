#pragma once



#include "sdutil.hpp"






//
//
//
struct StokesianDynamics
{

	double dens;
	double visc;

	double aref;

	// cutoff surface distance 
	double dr_min;

	// dr<dr_lub, add lubrication
	double dr_lub;
	// cutoff distance for lub.
	//double dr_lubmin;

	int npart;

	//
	VectorXd x;
	VectorXd u;

	VectorXd uinf, einf;
	VectorXd fext;


	//
	MatrixXd muf, mus, mes;
	MatrixXd rfu, rfe, rse;
	MatrixXd lubfu, lubfe, lubse;


	//
	int npair;
	int *pairs[2];

	// (xj-xi)/r
	VectorXd ex, ey, ez;
	// r, 1/r
	VectorXd dr, dr_inv;
	// lubrication, dst=r-ai-aj, 1/dst, log(1/dst)
	VectorXd dst, dst_inv, log_dst_inv;

};


//
void sd_check_dist(StokesianDynamics &sd);

//
void sd_form_mob(StokesianDynamics &sd);

//
void sd_inv_mob(StokesianDynamics &sd);

//
void sd_corr_lub(StokesianDynamics &sd);

void sd_set_flow(StokesianDynamics &sd, 
	const double U0[3], const double O0[3], const double E0[5]);

void sd_save_csv(StokesianDynamics &sd, const char outfilename[]);



