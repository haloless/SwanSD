

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>

#include "sd.hpp"

#include <mkl.h>

void test_eigen() {
	MatrixXd m = MatrixXd::Random(3,3);
	std::cout << "m = " << std::endl << m << std::endl;

	VectorXd v(3);
	std::cout << "v = " << v << std::endl;
	v << 1, 2, 3;
	std::cout << "v = " << v << std::endl;

	VectorXd mv = m * v;
	std::cout << "m*v = " << std::endl << mv << std::endl;

	MatrixXd minv = m.inverse();
	std::cout << "m^-1 = " << minv << std::endl;

	MatrixXd msym = m + m.transpose();


}

void test_eigen_svd() {
	Matrix3d a;
	//a << 0.792207329559554,	0.0357116785741896,	0.678735154857774,
	//	0.959492426392903,	0.849129305868777,	0.757740130578333,
	//	0.655740699156587,	0.933993247757551,	0.743132468124916;
	a << 0, 0, 0, 0, 2, 0, 0, 0, 10;

	Eigen::JacobiSVD<Matrix3d> svd(a, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Matrix3d u = svd.matrixU();
	Matrix3d v = svd.matrixV();
	Vector3d svec = svd.singularValues();
	Matrix3d s = svec.asDiagonal();
	

	std::cout << "A = " << std::endl << a << std::endl;
	std::cout << "U = " << std::endl << u << std::endl;
	std::cout << "S = " << std::endl << s << std::endl;
	std::cout << "V = " << std::endl << v << std::endl;

	std::cout << "U*S*V' = " << std::endl << u * s * v.transpose() << std::endl;

	Matrix3d sinv = s.transpose();
	for (int i=0; i<3; i++) {
		if (abs(svec(i)) > 1.0e-8) {
			sinv(i,i) = 1.0 / svec(i);
		} else {
			sinv(i,i) = 0.0;
		}
	}

	Matrix3d ainv = v * sinv * u.transpose();
	std::cout << "Ainv = " << std::endl << ainv << std::endl;

}

void test_mkl_svd() {
	double a[3][3] = {
		//0.792207329559554,	0.0357116785741896,	0.678735154857774,
		//0.959492426392903,	0.849129305868777,	0.757740130578333,
		//0.655740699156587,	0.933993247757551,	0.743132468124916,
		0, 0, 0,
		0, 2, 0,
		0, 0, 10,
	};

	MKL_INT m = 3, n = 3;
	MKL_INT lda = 3, ldu = 3, ldvt = 3;
	double superb[3];

	double svec[3], u[3][3], vt[3][3];

	MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
		m, n, &a[0][0], lda, svec, &u[0][0], ldu, &vt[0][0], ldvt, superb);
	
	if (info > 0) {
		std::cerr << "DGESVD failed" << std::endl;
	}

	std::cout << "U = " << std::endl;
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			std::cout << u[i][j] << ",";
		}
		std::cout << std::endl;
	}

	std::cout << "S = " << std::endl;
	for (int i=0; i<3; i++) {
		std::cout << svec[i] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "V = " << std::endl;
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			std::cout << vt[j][i] << ",";
		}
		std::cout << std::endl;
	}
}

void test_mkl_gels() {
	// input matrix
	double a[3][3] = {
		//0.792207329559554,	0.0357116785741896,	0.678735154857774,
		//0.959492426392903,	0.849129305868777,	0.757740130578333,
		//0.655740699156587,	0.933993247757551,	0.743132468124916,
		0, 0, 0,
		0, 2, 0,
		0, 0, 10,
	};

	// use unit matrix as RHS to get inverse
	double b[3][3] = {
		1, 0, 0, 
		0, 1, 0,
		0, 0, 1,
	};

	MKL_INT m = 3, n = 3;
	MKL_INT lda = 3, ldb = 3;

	// GELS requires full-rank
	//MKL_INT info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', 
	//	m, n, 3, &a[0][0], lda, &b[0][0], ldb);

	double rcond = 1.0e-6;
	double svec[3];
	MKL_INT rank = 0;
	// GELSS use SVD
	MKL_INT info = LAPACKE_dgelss(LAPACK_ROW_MAJOR, 
		m, n, 3, &a[0][0], lda, &b[0][0], ldb, svec, rcond, &rank);
	
	if (info != 0) {
		std::cerr << "DGELS failed = " << info << std::endl;
	}

	std::cout << "PAinv = " << std::endl;
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			std::cout << b[i][j] << ",";
		}
		std::cout << std::endl;
	}
}



void test_sd(int argc, char *argv[]) {

	StokesianDynamics sd;

	sd.dens = 1.0;
	sd.visc = 1.0;

	sd.aref = 1.0;

	sd.dr_min = 2.0 + 1.0e-3;

	sd.dr_lub = 4.0;
	//sd.dr_lubmin = 2.0 + 1.0e-3;

	const int npart = 15;
	const int n11 = npart * 11;
	const int n6 = npart * 6;
	const int n5 = npart * 5;
	const int npairmax = npart * (npart-1) / 2;

	sd.npart = npart;

	sd.x.resize(n6); sd.x.setZero();
	sd.u.resize(n6); sd.u.setZero();



	// allocate matrix
	sd.muf.resize(n6,n6); sd.muf.setZero();
	sd.mus.resize(n6,n5); sd.mus.setZero();
	sd.mes.resize(n5,n5); sd.mes.setZero();
	sd.rfu.resize(n6,n6); sd.rfu.setZero();
	sd.rfe.resize(n6,n5); sd.rfe.setZero();
	sd.rse.resize(n5,n5); sd.rse.setZero();
	sd.lubfu.resize(n6,n6); sd.lubfu.setZero();
	sd.lubfe.resize(n6,n5); sd.lubfe.setZero();
	sd.lubse.resize(n5,n5); sd.lubse.setZero();


	//
	sd.npair = 0;
	sd.pairs[0] = new int[npairmax];
	sd.pairs[1] = new int[npairmax];
	sd.ex.resize(npairmax);
	sd.ey.resize(npairmax);
	sd.ez.resize(npairmax);
	sd.dr.resize(npairmax);
	sd.dr_inv.resize(npairmax);
	sd.dst.resize(npairmax);
	sd.dst_inv.resize(npairmax);
	sd.log_dst_inv.resize(npairmax);

	// count all pairs
	for (int i=0; i<npart; i++) {
		for (int j=i+1; j<npart; j++) {
			sd.pairs[0][sd.npair] = i;
			sd.pairs[1][sd.npair] = j;
			sd.npair += 1;
		}
	}


	//
	VectorXd fext(n6); fext.setZero();
	VectorXd eext(n5); eext.setZero();
	VectorXd uext(n6); uext.setZero();

	VectorXd upart(n6); upart.setZero();

	if (0) {
		const double pos[npart][3] = {
			//-5.0, 0.2, 0.0,
			//5.0, -0.2, 0.0,
			//0.9, 0.5, 0.1,
			//1.0, 0.5, 0.1,
			1.2, 0.5, 0.1,
			-0.8, -0.6, -0.3,
		};
		for (int i=0; i<npart; i++) {
			int ioff = i * 6;
			sd.x(ioff+0) = pos[i][0];
			sd.x(ioff+1) = pos[i][1];
			sd.x(ioff+2) = pos[i][2];
		}
	}
	if (1) {
		const double drcen = 4.0;
		const double x0 = -drcen*(npart-1)/2.0;
		for (int i=0; i<npart; i++) {
			int ioff6 = i * 6;

			double xx = x0 + drcen*i;
			double yy = 0;
			double zz = 0;
			sd.x(ioff6+0) = xx;
			sd.x(ioff6+1) = yy;
			sd.x(ioff6+2) = zz;

			fext(ioff6+0) = 0.0;
			fext(ioff6+1) = 1.0;
			fext(ioff6+2) = 0.0;

		}
	}

	//
	// Begin test
	//

	//
	sd_check_dist(sd);

	//
	sd_form_mob(sd);

	//std::cout << "Muf=" << std::endl;
	//std::cout << sd.muf << std::endl;
	//std::cout << "Mus=" << std::endl;
	//std::cout << sd.mus << std::endl;
	//std::cout << "Mes=" << std::endl;
	//std::cout << sd.mes << std::endl;

	//
	sd_inv_mob(sd);

	//
	sd_corr_lub(sd);

	//
	MatrixXd Rfu = sd.rfu + sd.lubfu;
	MatrixXd Rfe = sd.rfe + sd.lubfe;
	MatrixXd Rse = sd.rse + sd.lubse;

	//std::cout << "Rfu=" << std::endl;
	//std::cout << Rfu << std::endl;
	//std::cout << "Rfe=" << std::endl;
	//std::cout << Rfe << std::endl;
	//std::cout << "Rse=" << std::endl;
	//std::cout << Rse << std::endl;

	{
		VectorXd rhs = fext + Rfe*eext;
		
		upart = Rfu.ldlt().solve(rhs);

		for (int i=0; i<npart; i++) {
			std::cout << upart(i*6+1) << std::endl;
		}
	}

}



void init_sd(StokesianDynamics &sd, int npart) {

	sd.dens = 1.0;
	sd.visc = 1.0;

	sd.aref = 1.0;

	sd.dr_min = 2.0 + 1.0e-3;

	sd.dr_lub = 4.0;
	//sd.dr_lubmin = 2.0 + 1.0e-4;

	const int n11 = npart * 11;
	const int n6 = npart * 6;
	const int n5 = npart * 5;
	const int npairmax = npart * (npart-1) / 2;

	sd.npart = npart;

	// allocate vector
	sd.x.resize(n6); sd.x.setZero();
	sd.u.resize(n6); sd.u.setZero();

	sd.uinf.resize(n6); sd.uinf.setZero();
	sd.einf.resize(n5); sd.einf.setZero();
	sd.fext.resize(n6); sd.fext.setZero();


	// allocate matrix
	sd.muf.resize(n6,n6); sd.muf.setZero();
	sd.mus.resize(n6,n5); sd.mus.setZero();
	sd.mes.resize(n5,n5); sd.mes.setZero();
	sd.rfu.resize(n6,n6); sd.rfu.setZero();
	sd.rfe.resize(n6,n5); sd.rfe.setZero();
	sd.rse.resize(n5,n5); sd.rse.setZero();
	sd.lubfu.resize(n6,n6); sd.lubfu.setZero();
	sd.lubfe.resize(n6,n5); sd.lubfe.setZero();
	sd.lubse.resize(n5,n5); sd.lubse.setZero();


	//
	sd.npair = 0;
	sd.pairs[0] = new int[npairmax];
	sd.pairs[1] = new int[npairmax];
	sd.ex.resize(npairmax);
	sd.ey.resize(npairmax);
	sd.ez.resize(npairmax);
	sd.dr.resize(npairmax);
	sd.dr_inv.resize(npairmax);
	sd.dst.resize(npairmax);
	sd.dst_inv.resize(npairmax);
	sd.log_dst_inv.resize(npairmax);

	// count all pairs
	for (int i=0; i<npart; i++) {
		for (int j=i+1; j<npart; j++) {
			sd.pairs[0][sd.npair] = i;
			sd.pairs[1][sd.npair] = j;
			sd.npair += 1;
		}
	}
}

void save_sd(StokesianDynamics &sd, int step) {
	char outfilename[256];
	sprintf(outfilename, "output%06d.csv", step);

	sd_save_csv(sd, outfilename);
}

void ext_force(StokesianDynamics &sd)
{
	const int np = sd.npart;

	const double kref = 1.0e2;
	const double aref = sd.aref;
	const double drmin = sd.dr_min * aref;

	const VectorXd &x = sd.x;
	VectorXd &fext = sd.fext;

	fext.setZero();

	// add pairwise repulsive force
	for (int i=0; i<np; i++) {
		for (int j=i+1; j<np; j++) {
			int ioff = i * 6;
			int joff = j * 6;

			double dx = x(joff+0) - x(ioff+0);
			double dy = x(joff+1) - x(ioff+1);
			double dz = x(joff+2) - x(ioff+2);
			double dr = sqrt(dx*dx + dy*dy + dz*dz);

			if (dr < drmin) {
				double disp = drmin - dr;
				double ff = kref * (disp / aref);

				double ex = dx / dr;
				double ey = dy / dr;
				double ez = dz / dr;

				fext(ioff+0) -= ff * ex;
				fext(ioff+1) -= ff * ey;
				fext(ioff+2) -= ff * ez;

				fext(joff+0) += ff * ex;
				fext(joff+1) += ff * ey;
				fext(joff+2) += ff * ez;
			}
		}
	}
}


void sim_sd(int argc, char *argv[]) {
	
	StokesianDynamics sd;

	const int npart = atoi(argv[1]);
	const int n6 = npart * 6;
	const int n5 = npart * 5;

	init_sd(sd, npart);

	// imposed flow
	const double shear_rate = 1.0;
	const double U0[3] = { 0.0, 0.0, 0.0, };
	const double O0[3] = { 0.0, 0.0, -shear_rate/2 };
	const double E0[5] = { 0.0, shear_rate/2, 0.0, 0.0, 0.0 }; // Exx,Exy,Exz,Eyz,Eyy

	// load particle
	{
		const char posfile[] = "input_pos.csv";
		FILE *fp = fopen(posfile, "r");
		if (!fp) {
			std::cerr << "Failed to open " << posfile << std::endl;
			exit(1);
		}

		const int buflen = 2048;
		char buf[buflen];
		// header
		fgets(buf, buflen, fp);

		for (int i=0; i<npart; i++) {
			// read line
			fgets(buf, buflen, fp);

			const double scale = 1.05;
			int idx;
			double dd, xx, yy, zz;
			sscanf(buf, "%d,%lf,%lf,%lf,%lf", &idx, &dd, &xx, &yy, &zz);
			if (idx != i) {
				std::cerr << "Error at i=" << i << ": " << buf << std::endl;
			}

			sd.x(i*6+0) = xx * scale;
			sd.x(i*6+1) = yy * scale;
			sd.x(i*6+2) = zz * scale;

		}

		fclose(fp);
	}

	//
	VectorXd uold(n6); uold.setZero();


	const double dt = 1.0e-3;
	const int maxstep = 10000;
	double t = 0.0;
	int step = 0;
	int save = 0;

	save_sd(sd, save);
	save += 1;

	for (step=1; step<=maxstep; step++) {
		
		//
		ext_force(sd);

		// set Uinf, Einf
		sd_set_flow(sd, U0, O0, E0);

		//
		sd_check_dist(sd);

		//
		sd_form_mob(sd);

		//
		sd_inv_mob(sd);

		//
		sd_corr_lub(sd);

		//
		MatrixXd Rfu = sd.rfu + sd.lubfu;
		MatrixXd Rfe = sd.rfe + sd.lubfe;
		MatrixXd Rse = sd.rse + sd.lubse;

		VectorXd rhs = sd.fext + Rfe * sd.einf;

		VectorXd sol = Rfu.ldlt().solve(rhs);

		// recover velocity
		sd.u = sol + sd.uinf;

		//
		sd.x += 1.5*dt*sd.u - 0.5*dt*uold;

		//
		uold = sd.u;

		t += dt;

		std::cout << "step=" << step 
			<< "; dt=" << dt
			<< "; time=" << t 
			<< std::endl;

		if (step % 100 == 0) {
			save_sd(sd, save);
			save += 1;
		}
	}
}



int main(int argc, char *argv[])
{

	//test_eigen();
	//test_eigen_svd();
	//test_mkl_svd();
	test_mkl_gels();

	//test_sd(argc, argv);

	//sim_sd(argc, argv);


	return 0;
}





