#pragma once



#include "sdutil.hpp"



struct SDEwald
{
	double eps_tol;
	
	double xi;

	double len[3];

	double rmax;
	int rnum[3];

	double kmax;
	int knum[3];

};



//
//
//
struct StokesianDynamics
{

	double dens;
	double visc;

	// reference length
	double aref;

	// cutoff surface distance 
	double dr_min;

	// dr<dr_lub, add lubrication
	double dr_lub;
	// cutoff distance for lub.
	//double dr_lubmin;

	// 0=unbounded, 1=periodic
	int bc_mode;

	//double problo[3];
	//double probhi[3];
	//double problen[3];

	double prob_lx, prob_ly, prob_lz;

	//
	// Ewald code
	//

	double ewald_eps;

	double ewald_xi;

	double ewald_rmax;
	int ewald_rmax_x, ewald_rmax_y, ewald_rmax_z;

	double ewald_kmax;
	int ewald_kmax_x, ewald_kmax_y, ewald_kmax_z;

	//
	// data
	//

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



