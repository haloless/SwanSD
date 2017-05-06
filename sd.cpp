
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <iostream>
#include <utility>
#include <algorithm>


#include "sd.hpp"

#include "sddata.hpp"

typedef Eigen::Matrix<double,3,5> MatrixR3C5d;
typedef Eigen::Matrix<double,5,3> MatrixR5C3d;
typedef Eigen::Matrix<double,5,5> Matrix5d;
typedef Eigen::Matrix<double,12,12> Matrix12d;
typedef Eigen::Matrix<double,12,10> MatrixR12C10d;
typedef Eigen::Matrix<double,10,10> Matrix10d;


static const int mesid[2][5] = {
	0, 0, 0, 1, 1, 
	2, 1, 2, 2, 2,
};

// Complementary index 
// Useful with the Signature tensor
//	int k = 3 - i - j;
//	if (k == -1) k = 2;
//	if (k == 3)  k = 0;
static const int compid[3][3] = {
	0, 2, 1, 
	2, 1, 0,
	1, 0, 2,
};


void sd_check_dist(StokesianDynamics &sd)
{
	const double aref = sd.aref;

	// position
	const VectorXd &x = sd.x;

	const int npair = sd.npair;
	for (int ipair=0; ipair<sd.npair; ipair++) {
		//
		int i = sd.pairs[0][ipair];
		int j = sd.pairs[1][ipair];

		// offset
		int ioff = 6 * i;
		int joff = 6 * j;

		double dx = x(joff) - x(ioff);
		double dy = x(joff+1) - x(ioff+1);
		double dz = x(joff+2) - x(ioff+2);
		double dr = sqrt(dx*dx + dy*dy + dz*dz);

		double ex = dx / dr;
		double ey = dy / dr;
		double ez = dz / dr;

		// truncate at min distance
		if (dr < sd.dr_min) {
			dr = sd.dr_min;
		}

		double rdr = 1.0 / dr;

		// save pair configuration
		sd.ex(ipair) = ex;
		sd.ey(ipair) = ey;
		sd.ez(ipair) = ez;
		sd.dr(ipair) = dr;
		sd.dr_inv(ipair) = rdr;

		if (dr < sd.dr_lub) {
			// surface distance
			double dst = dr - 2.0*aref;
			double rdst = 1.0 / dst;

			sd.dst(ipair) = dst;
			sd.dst_inv(ipair) = rdst;
			sd.log_dst_inv(ipair) = log(rdst);
		}
	}
}


//
// self contribution
// this is simple, so given in this special routine.
// NOTE 
// e = [exx-ezz,2*exy,2*exz,2*eyz,eyy-ezz], ezz = -exx-eyy
// reduce to e= [2exx+eyy,2exy,2exz,2eyz,2eyy+exx]
// s = [sxx,sxy,sxz,syz,syy]
// SE part is 
// [ 2 0 0 0 1 ]
// [ 0 2 0 0 0 ]
// [ 0 0 2 0 0 ]
// [ 0 0 0 2 0 ]
// [ 1 0 0 0 2 ]
//
static void mob_self(
	Matrix3d &ma, Matrix3d &mb, Matrix3d &mbt, Matrix3d &mc,
	MatrixR3C5d &mgt, MatrixR3C5d &mht, Matrix5d &mm)
{
	ma.setZero();
	mb.setZero();
	mbt.setZero();
	mc.setZero();
	mgt.setZero();
	mht.setZero();
	mm.setZero();

	for (int i=0; i<3; i++) {
		ma(i,i) = 1.0;
		mc(i,i) = 0.75;
	}

	mm(0,4) = 0.9;
	mm(4,0) = 0.9;

	for (int i=0; i<5; i++) {
		mm(i,i) = 1.8;
	}
}

//
// scalars for pair, so actually X12A,Y12A,etc.
//
static void mob_scalars(double dr_inv,
	double &xa, double &ya,
	double &yb,
	double &xc, double &yc,
	double &xg, double &yg,
	double &yh,
	double &xm, double &ym, double &zm)
{
	double dr_inv2 = dr_inv * dr_inv;
	double dr_inv3 = dr_inv2 * dr_inv;
	double dr_inv4 = dr_inv3 * dr_inv;
	double dr_inv5 = dr_inv4 * dr_inv;

	// 
	xa = 1.5 * dr_inv - dr_inv3;
	ya = 0.75 * dr_inv + 0.5 * dr_inv3;

	//
	yb = -0.75 * dr_inv2;

	//
	xc = 0.75 * dr_inv3;
	yc = -0.375 * dr_inv3;

	//
	xg = 2.25 * dr_inv2 - 3.6 * dr_inv4;
	yg = 1.2 * dr_inv4;

	//
	yh = -1.125 * dr_inv3;

	//
	xm = -4.5 * dr_inv3 + 10.8 * dr_inv5;
	ym = 2.25 * dr_inv3 - 7.2 * dr_inv5;
	zm = 1.8 * dr_inv5;

	return;
}





static void mob_ewald_self(double xi)
{
	double xi2 = xi * xi;
	double xia2 = xi2;
	double xiaspi = xi / SqrtPI;

	double selfa = 1.0 - xiaspi * (6.0 - 40.0/3.0*xia2);
	double selfc = 0.75 - xiaspi * xia2 * 10.0;
	double selfm = 0.9 - xiaspi * xia2 * (12.0-30.24*xia2);


}

//
// convert E,S for symmetric matrix
// E1=Exx-Ezz=2Exx+Eyy, E2=2Exy, E3 = 2Exz, E4=2Eyz, E5=Eyy-Ezz=2Eyy+Exx
// S1=Sxx, S2=Sxy=Syx, S3=Sxz=Szx, S4=Syz=Szy, S5=Syy
//
static void mob_reduce_m53t(double gt[3][3][3], MatrixR3C5d &mgt)
{
	for (int i=0; i<3; i++) {
		mgt(i,0) = gt[i][0][0] - gt[i][2][2];
		mgt(i,1) = 2.0 * gt[i][0][1];
		mgt(i,2) = 2.0 * gt[i][0][2];
		mgt(i,3) = 2.0 * gt[i][1][2];
		mgt(i,4) = gt[i][1][1] - gt[i][2][2];
	}
}

//
// convert E,S for symmetric matrix
// E1=Exx-Ezz=2Exx+Eyy, E2=2Exy, E3 = 2Exz, E4=2Eyz, E5=Eyy-Ezz=2Eyy+Exx
// S1=Sxx, S2=Sxy=Syx, S3=Sxz=Szx, S4=Syz=Szy, S5=Syy
//
static void mob_reduce_m55(double m[3][3][3][3], Matrix5d &mm) 
{
	for (int i=0; i<5; i++) {
		int i0 = mesid[0][i];
		int i1 = mesid[1][i];

		if (i==0 || i==4) {
			mm(i,0) = m[i0][i0][0][0] - m[i0][i0][2][2] - m[i1][i1][0][0] + m[i1][i1][2][2];
			mm(i,1) = 2.0 * (m[i0][i0][0][1] - m[i1][i1][0][1]);
			mm(i,2) = 2.0 * (m[i0][i0][0][2] - m[i1][i1][0][2]);
			mm(i,3) = 2.0 * (m[i0][i0][1][2] - m[i1][i1][1][2]);
			mm(i,4) = m[i0][i0][1][1] - m[i0][i0][2][2] - m[i1][i1][1][1] + m[i1][i1][2][2];
		} else {
			mm(i,0) = 2.0 * (m[i0][i1][0][0] - m[i0][i1][2][2]);
			mm(i,1) = 4.0 * m[i0][i1][0][1];
			mm(i,2) = 4.0 * m[i0][i1][0][2];
			mm(i,3) = 4.0 * m[i0][i1][1][2];
			mm(i,4) = 2.0 * (m[i0][i1][1][1] - m[i0][i1][2][2]);
		}
	}
}

static void mob_pair_mat(
	double dr_inv, double ex, double ey, double ez,
	double xa, double ya,
	double yb,
	double xc, double yc,
	double xg, double yg,
	double yh,
	double xm, double ym, double zm,
	Matrix3d &ma, Matrix3d &mb, Matrix3d &mbt, Matrix3d &mc,
	MatrixR3C5d &mgt, MatrixR3C5d &mht, Matrix5d &mm)
{
	// ei*ej
	Vector3d e(ex,ey,ez);
	Matrix3d ee = e * e.transpose();

	// moba, mobc
	for (int i=0; i<3; i++) {

		ma(i,i) = xa*ee(i,i) + ya*(1.0-ee(i,i));
		
		mc(i,i) = xc*ee(i,i) + yc*(1.0-ee(i,i));

		for (int j=i+1; j<3; j++) {
			ma(i,j) = xa*ee(i,j) - ya*ee(i,j);
			ma(j,i) = ma(i,j);

			mc(i,j) = xc*ee(i,j) - yc*ee(i,j);
			mc(j,i) = mc(i,j);
		}
	}

	// mobb, mobbt
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			int k = compid[i][j];

			mb(i,j) = yb * Epsil[i][j][k] * e(k);
			mbt(j,i) = -mb(i,j);
		}
	}

	// mobgt, mobht
	double gt[3][3][3] = { 0 };
	double ht[3][3][3] = { 0 };
	for (int k=0; k<3; k++) {
		for (int i=0; i<3; i++) {
			for (int j=0; j<3; j++) {
				int l = compid[j][k];
				int m = compid[i][k];

				gt[k][i][j] = -xg * (ee(i,j) - 1.0/3.0*Delta[i][j]) * e(k)
					- yg * (e(i)*Delta[j][k] + e(j)*Delta[i][k] - 2.0*ee(i,j)*e(k));

				ht[k][i][j] = yh * (ee(i,l)*Epsil[j][k][l] + ee(j,m)*Epsil[i][k][m]);
			}
		}
	}

	//
	double m[3][3][3][3] = {0};
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			for (int k=0; k<3; k++) {
				for (int l=0; l<3; l++) {
					m[i][j][k][l] = 1.5 * xm * (ee(i,j) - 1.0/3.0*Delta[i][j]) * (ee(k,l) - 1.0/3.0*Delta[k][l])
						+ 0.5 * ym * (ee(i,k)*Delta[j][l] + ee(j,k)*Delta[i][l] + 
						ee(i,l)*Delta[j][k] + ee(j,l)*Delta[i][k] - 
						4.0*ee(i,j)*ee(k,l))
						+ 0.5 * zm * (Delta[i][k]*Delta[j][l] + Delta[j][k]*Delta[i][l] - Delta[i][j]*Delta[k][l] +
						ee(i,j)*Delta[k][l] + ee(k,l)*Delta[i][j] - ee(i,k)*Delta[j][l] - 
						ee(j,k)*Delta[i][l] - ee(i,l)*Delta[j][k] - ee(j,l)*Delta[i][k] + 
						ee(i,j)*ee(k,l));
				}
			}
		}
	}

	// convert E,S for symmetric matrix
	// E1=Exx-Ezz, E2=2Exy, E3 = 2Exz, E4=2Eyz, E5=Eyy-Ezz
	// S1=Sxx, S2=Sxy=Syx, S3=Sxz=Szx, S4=Syz=Szy, S5=Syy
	mob_reduce_m53t(gt, mgt);
	mob_reduce_m53t(ht, mht);
	//for (int i=0; i<3; i++) {
	//	mgt(i,0) = gt[i][0][0] - gt[i][2][2];
	//	mgt(i,1) = 2.0 * gt[i][0][1];
	//	mgt(i,2) = 2.0 * gt[i][0][2];
	//	mgt(i,3) = 2.0 * gt[i][1][2];
	//	mgt(i,4) = gt[i][1][1] - gt[i][2][2];

	//	mht(i,0) = ht[i][0][0] - ht[i][2][2];
	//	mht(i,1) = 2.0 * ht[i][0][1];
	//	mht(i,2) = 2.0 * ht[i][0][2];
	//	mht(i,3) = 2.0 * ht[i][1][2];
	//	mht(i,4) = ht[i][1][1] - ht[i][2][2];
	//}

	mob_reduce_m55(m, mm);
	//for (int i=0; i<5; i++) {
	//	int i0 = mesid[0][i];
	//	int i1 = mesid[1][i];

	//	if (i==0 || i==4) {
	//		mm(i,0) = m[i0][i0][0][0] - m[i0][i0][2][2] - m[i1][i1][0][0] + m[i1][i1][2][2];
	//		mm(i,1) = 2.0 * (m[i0][i0][0][1] - m[i1][i1][0][1]);
	//		mm(i,2) = 2.0 * (m[i0][i0][0][2] - m[i1][i1][0][2]);
	//		mm(i,3) = 2.0 * (m[i0][i0][1][2] - m[i1][i1][1][2]);
	//		mm(i,4) = m[i0][i0][1][1] - m[i0][i0][2][2] - m[i1][i1][1][1] + m[i1][i1][2][2];
	//	} else {
	//		mm(i,0) = 2.0 * (m[i0][i1][0][0] - m[i0][i1][2][2]);
	//		mm(i,1) = 4.0 * m[i0][i1][0][1];
	//		mm(i,2) = 4.0 * m[i0][i1][0][2];
	//		mm(i,3) = 4.0 * m[i0][i1][1][2];
	//		mm(i,4) = 2.0 * (m[i0][i1][1][1] - m[i0][i1][2][2]);
	//	}
	//}
}

static void mob_pair(
	double dr_inv, double dx, double dy, double dz,
	Matrix3d &ma, Matrix3d &mb, Matrix3d &mbt, Matrix3d &mc,
	MatrixR3C5d &mgt, MatrixR3C5d &mht, Matrix5d &mm)
{

	// scalars for pair, so actually X12A,Y12A,etc.
	double xa, ya;
	double yb;
	double xc, yc;
	double xg, yg;
	double yh;
	double xm, ym, zm;

	mob_scalars(dr_inv, 
		xa, ya, yb, xc, yc, 
		xg, yg, yh, 
		xm, ym, zm);

	// fill matrix
	mob_pair_mat(dr_inv, dx, dy, dz,
		xa, ya, yb, xc, yc, xg, yg, yh, xm, ym, zm,
		ma, mb, mbt, mc, mgt, mht, mm);

}

inline void add_matrix_self(int i, 
	MatrixXd &muf, MatrixXd &mus, MatrixXd &mes,
	const Matrix3d &ma, const Matrix3d &mc, const Matrix5d &mm) 
{	
	{ // MUF
		int ioff = i * 6;
		// u
		muf.block<3,3>(ioff,ioff) += ma;

		// omega
		muf.block<3,3>(ioff+3,ioff+3) += mc;
	}

	{ // MUS
		// nothing here
	}

	{ // MES
		int ioff = i * 5;
		// E
		mes.block<5,5>(ioff,ioff) += mm;
	}
}

inline void add_matrix_pair(int i, int j, 
	MatrixXd &muf, MatrixXd &mus, MatrixXd &mes,
	const Matrix3d &ma, 
	const Matrix3d &mb, const Matrix3d &mbt, 
	const Matrix3d &mc,
	const MatrixR3C5d &mgt, 
	const MatrixR3C5d &mht,
	const Matrix5d &mm,
	bool add_to_j)
{
	//
	const int i6 = i * 6;
	const int j6 = j * 6;

	const int i5 = i * 5;
	const int j5 = j * 5;

	// UF
	muf.block<3,3>(i6  ,j6  ) = ma;
	muf.block<3,3>(i6+3,j6  ) = mb;
	muf.block<3,3>(i6  ,j6+3) = mbt;
	muf.block<3,3>(i6+3,j6+3) = mc;

	if (add_to_j) {
		muf.block<3,3>(j6  ,i6  ) = ma.transpose();
		muf.block<3,3>(j6+3,i6  ) = mb.transpose();
		muf.block<3,3>(j6  ,i6+3) = mbt.transpose();
		muf.block<3,3>(j6+3,i6+3) = mc.transpose();
	}

	// US
	mus.block<3,5>(i6,j5) = mgt;
	mus.block<3,5>(i6+3,j5) = mht;

	if (add_to_j) {
		mus.block<3,5>(j6,i5) = -mgt;
		mus.block<3,5>(j6+3,i5) = mht;
	}

	// ES
	mes.block<5,5>(i5,j5) = mm;

	if (add_to_j) {
		mes.block<5,5>(j5,i5) = mm.transpose();
	}
}

void sd_form_mob(StokesianDynamics &sd)
{
	const int np = sd.npart;
	const int n6 = np * 6;
	const int n5 = np * 5;
	const int n3 = np * 3;

	MatrixXd &muf = sd.muf;
	MatrixXd &mus = sd.mus;
	MatrixXd &mes = sd.mes;

	muf.setZero();
	mus.setZero();
	mes.setZero();

	// local matrix
	Matrix3d ma, mb, mbt, mc;
	MatrixR3C5d mgt, mht;
	Matrix5d mm;

	// self part, same for all
	mob_self(ma, mb, mbt, mc, mgt, mht, mm);

	// set self part
	for (int i=0; i<np; i++) {
		add_matrix_self(i, muf, mus, mes, ma, mc, mm);
	}

	// set pair part
	for (int ipair=0; ipair<sd.npair; ipair++) {
		int i = sd.pairs[0][ipair];
		int j = sd.pairs[1][ipair];

		//
		mob_pair(sd.dr_inv(ipair), sd.ex(ipair), sd.ey(ipair), sd.ez(ipair),
			ma, mb, mbt, mc, mgt, mht, mm);

		//
		add_matrix_pair(i, j, muf, mus, mes, 
			ma, mb, mbt, mc, mgt, mht, mm,
			true); // also add to j 

	}


}

static void lub_scalars(
	double dr, double ex, double ey, double ez,
	double dst, double dst_inv, double dst_log,
	double &x11a, double &x12a, double &y11a, double &y12a,
	double &y11b, double &y12b,
	double &x11c, double &x12c, double &y11c, double &y12c,
	double &x11g, double &x12g, double &y11g, double &y12g,
	double &y11h, double &y12h,
	double &xm, double &ym, double &zm)
{
	if (dr <= 2.1) { // asympototic 
		const double xi = dst;
		const double xi1 = dst_inv; // 1/xi
		const double dlx = dst_log; // log(1/xi) = -log(xi)

		const double xdlx = xi * dlx; // xi * log(1/xi)
		const double dlx1 = dlx + xdlx; // log(1/xi) + xi*log(1/xi)

		const double ug1 = 94.0 / 375.0;
		const double ug2 = 31.0 / 375.0;
		const double ug4 = 39.0 / 280.0;
		const double ug5 = 113.0 / 1500.0;
		const double ug6 = 47.0 / 840.0;
		const double ug7 = 137.0 / 1500.0;

		// Rfu: a, b, c 
		double csa1 = dlx / 6.0;
		double csa2 = xdlx / 6.0;
		double csa3 = dlx1 / 6.0;
		double csa4 = 0.25*xi1 + 0.225*dlx;
		double csa5 = dlx / 15.0;

		x11a = csa4 - 1.23041 + (3.0/112.0)*xdlx + 1.8918*xi;
		x12a = -x11a + 0.00312 - 0.0011*xi;
		y11a = csa1 - 0.39394 + 0.95665*xi;
		y12a = -y11a + 0.00463606 - 0.007049*xi;

		y11b = -csa1 + 0.408286 - xdlx/12.0 - 0.84055*xi;
		y12b = -y11b + 0.00230818 - 0.007508*xi;

		x11c = 0.0479 - csa2 + 0.12494*xi;
		x12c = -0.031031 + csa2 - 0.174476 * xi;
		y11c = 4.0*csa5 - 0.605434 + ug1*xdlx + 0.939139*xi;
		y12c = csa5 - 0.212032 + ug2*xdlx + 0.452843*xi;

		// Rsu: g, h
		double csg1 = csa4 + ug4 * xdlx;
		double csg2 = dlx/12.0 + xdlx/24.0;

		x11g = csg1 - 1.16897 + 1.47882 * xi;
		x12g = -csg1 + 1.178967 - 1.480493 * xi;
		y11g = csg2 - 0.2041 + 0.442226 * xi;
		y12g = -csg2 + 0.216365 - 0.469830 * xi;

		y11h = 0.5 * csa5 - 0.143777 + ug7 * xdlx + 0.264207 * xi;
		y12h = 2.0 * csa5 - 0.298166 + ug5 * xdlx + 0.534123 * xi;

		// Rse: m
		xm   = 1.0/3.0 * xi1 + 0.3 * dlx - 1.48163 + 0.335714 * xdlx + 1.413604 * xi;
		ym   = csa3 - 0.423489 + 0.827286 * xi;
		zm   = 0.0129151 - 0.042284 * xi;

	} else { // interpolation from table
		{
			int ia = int((dr-2.1)/0.05);
			int ib = ia + 1;
			double ra = r2babc[ia][RSABC];
			double rb = r2babc[ib][RSABC];
			assert(dr>=ra && dr<rb);
			double cb = (dr-ra) / (rb-ra);
			double ca = 1.0 - cb;

			x11a = ca * r2babc[ia][X11AS] + cb * r2babc[ib][X11AS];
			x12a = ca * r2babc[ia][X12AS] + cb * r2babc[ib][X12AS];
			y11a = ca * r2babc[ia][Y11AS] + cb * r2babc[ib][Y11AS];
			y12a = ca * r2babc[ia][Y12AS] + cb * r2babc[ib][Y12AS];

			y11b = ca * r2babc[ia][Y11BS] + cb * r2babc[ib][Y11BS];
			y12b = ca * r2babc[ia][Y12BS] + cb * r2babc[ib][Y12BS];

			y11c = ca * r2babc[ia][Y11CS] + cb * r2babc[ib][Y11CS];
			y12c = ca * r2babc[ia][Y12CS] + cb * r2babc[ib][Y12CS];
			x11c = ca * r2babc[ia][X11CS] + cb * r2babc[ib][X11CS];
			x12c = ca * r2babc[ia][X12CS] + cb * r2babc[ib][X12CS];
		}

		{
			int ia, ib;
			if (dr < 2.2) {
				ia = int((dr-2.1)/0.01);
				ib = ia + 1;
			} else {
				ia = int((dr-2.2)/0.05) + 10;
				ib = ia + 1;
			}
			double ra = r2bgh[ia][RSGH];
			double rb = r2bgh[ib][RSGH];
			assert(dr>=ra && dr<rb);
			double cb = (dr-ra) / (rb-ra);
			double ca = 1.0 - cb;

			x11g = ca * r2bgh[ia][X11GS] + cb * r2bgh[ib][X11GS];
			x12g = ca * r2bgh[ia][X12GS] + cb * r2bgh[ib][X12GS];
			y11g = ca * r2bgh[ia][Y11GS] + cb * r2bgh[ib][Y11GS];
			y12g = ca * r2bgh[ia][Y12GS] + cb * r2bgh[ib][Y12GS];

			y11h = ca * r2bgh[ia][Y11HS] + cb * r2bgh[ib][Y11HS];
			y12h = ca * r2bgh[ia][Y12HS] + cb * r2bgh[ib][Y12HS];

			xm = ca * r2bm[ia][XMS] + cb * r2bm[ib][XMS];
			ym = ca * r2bm[ia][YMS] + cb * r2bm[ib][YMS];
			zm = ca * r2bm[ia][ZMS] + cb * r2bm[ib][ZMS];
		}
	}

}



//
// 
static void lub_pair(
	double dr, double dx, double dy, double dz,
	double dst, double dst_inv, double dst_log,
	Matrix12d &tabc, MatrixR12C10d &tght, Matrix10d &tzm
	)
{

	// scalars for lubrication
	// lubrication is two-body
	// so scalars here contain both (11) and (12) terms

	double x11a, x12a, y11a, y12a;
	double y11b, y12b;
	double x11c, x12c, y11c, y12c;

	double x11g, x12g, y11g, y12g;
	double y11h, y12h;

	double xm, ym, zm;

	lub_scalars(dr, dx, dy, dz, 
		dst, dst_inv, dst_log, 
		x11a, x12a, 
		y11a, y12a, y11b, y12b, 
		x11c, x12c, y11c, y12c,
		x11g, x12g, y11g, y12g, 
		y11h, y12h,
		xm, ym, zm);

	if (0) {
		std::cout << "dr=" << dr << std::endl;
		std::cout << "x11g=" << x11g << std::endl;
		std::cout << "x12g=" << x12g << std::endl;
		std::cout << "y11g=" << y11g << std::endl;
		std::cout << "y12g=" << y12g << std::endl;
		std::cout << "y11h=" << y11h << std::endl;
		std::cout << "y12h=" << y12h << std::endl;
	}

	//
	tabc.setZero();
	tght.setZero();
	tzm.setZero();

	// ei*ej
	Vector3d e(dx,dy,dz);
	Matrix3d ee = e * e.transpose();

	//
	//
	//   | 3    3    3    3    | 5    5    |
	// --|---------------------|-----------|
	// 3 | a11  bt11 a12  bt12 | gt11 gt12 |
	// 3 |      c11  b12  c12  | ht11 ht12 |
	// 3 |           a22  bt22 | gt21 gt22 |
	// 3 |                c22  | ht21 ht22 |
	// --|---------------------|-----------|
	// 5 |                     | m11  m12  |
	// 5 |                     |      m22  |
	// --|---------------------|-----------|
	//

	//
	// Rfu, use a,b,c
	//
	// | a11  bt11 a12  bt12 |
	// |      c11  b12  c12  |
	// |           a22  bt22 |
	// |                c22  |
	//

	
	{ // a11 & a22, half
		Matrix3d mat; mat.setZero();
		for (int i=0; i<3; i++) {
			mat(i,i) = (x11a-y11a)*ee(i,i) + y11a;
			for (int j=i+1; j<3; j++) {
				mat(i,j) = (x11a-y11a) * ee(i,j);
			}
		}

		tabc.block<3,3>(0,0) = mat;
		tabc.block<3,3>(6,6) = mat;
	}

	{ // a12, full
		Matrix3d mat; mat.setZero();
		for (int i=0; i<3; i++) {
			mat(i,i) = (x12a-y12a)*ee(i,i) + y12a;
			for (int j=i+1; j<3; j++) {
				double mij = (x12a-y12a) * ee(i,j);
				mat(i,j) = mij;
				mat(j,i) = mij;
			}
		}

		tabc.block<3,3>(0,6) = mat;
	}


	{ // c11 & c22, half
		Matrix3d mat; mat.setZero();
		for (int i=0; i<3; i++) {
			mat(i,i) = (x11c-y11c) * ee(i,i) + y11c;
			for (int j=i+1; j<3; j++) {
				mat(i,j) = (x11c-y11c) * ee(i,j);
			}
		}

		tabc.block<3,3>(3,3) = mat;
		tabc.block<3,3>(9,9) = mat;
	}

	{ // c12, full
		Matrix3d mat; mat.setZero();
		for (int i=0; i<3; i++) {
			mat(i,i) = (x12c-y12c) * ee(i,i) + y12c;
			for (int j=i+1; j<3; j++) {
				double mij = (x12c-y12c) * ee(i,j);
				mat(i,j) = mij;
				mat(j,i) = mij;
			}
		}

		tabc.block<3,3>(3,9) = mat;
	}

	{ // bt11 & bt22=-bt11, full
		Matrix3d mat; mat.setZero();
		mat(0,0) = 0.0;
		mat(0,1) = -y11b * e(2);
		mat(0,2) = y11b * e(1);
		mat(1,0) = -mat(0,1);
		mat(1,1) = 0.0;
		mat(1,2) = -y11b * e(0);
		mat(2,0) = -mat(0,2);
		mat(2,1) = -mat(1,2);
		mat(2,2) = 0.0;

		tabc.block<3,3>(0,3) = mat;
		tabc.block<3,3>(6,9) = -mat;
	}

	{ // bt12 & b12=bt12, full
		Matrix3d mat; mat.setZero();
		mat(0,0) = 0.0;
		mat(0,1) = y12b * e(2);
		mat(0,2) = -y12b * e(1);
		mat(1,0) = -mat(0,1);
		mat(1,1) = 0.0;
		mat(1,2) = y12b * e(0);
		mat(2,0) = -mat(0,2);
		mat(2,1) = -mat(1,2);
		mat(2,2) = 0.0;

		tabc.block<3,3>(0,9) = mat;
		tabc.block<3,3>(3,6) = mat;
	}

	//
	// Rfe, gt,ht
	//

	{ // gt11, gt22=-gt11
		double c13x11g   = 1.0/3.0 * x11g;
		double c2y11g    = 2.0*y11g;
		double xm2y11g   = x11g - c2y11g;
		double comd11    = ee(0,0) * xm2y11g;
		double comd22    = ee(1,1) * xm2y11g;
		double comd33    = ee(2,2) * xm2y11g;
		double c2ymx11   = c2y11g - c13x11g;
		double con34     = comd11 - c13x11g;
		double con56     = comd11 + y11g;
		double con712    = comd22 + y11g;
		double con89     = comd33 + y11g;
		double con1011   = comd22 - c13x11g;

		MatrixR3C5d mat;
		mat(0,0) = e(0) * (comd11+c2ymx11);
		mat(0,1) = e(1) * con56;
		mat(0,2) = e(2) * con56;
		mat(0,3) = e(0) * ee(1,2) * xm2y11g;
		mat(0,4) = e(0) * con1011;
		mat(1,0) = e(1) * con34;
		mat(1,1) = e(0) * con712;
		mat(1,2) = mat(0,3);
		mat(1,3) = e(2) * con712;
		mat(1,4) = e(1) * (comd22+c2ymx11);
		mat(2,0) = e(2) * con34;
		mat(2,1) = mat(0,3);
		mat(2,2) = e(0) * con89;
		mat(2,3) = e(1) * con89;
		mat(2,4) = e(2) * con1011;

		tght.block<3,5>(0,0) = mat;
		tght.block<3,5>(6,5) = -mat;
	}

	{ // gt21, gt12=-gt21
		double c13x12g = 1.0/3.0 * x12g;
		double c2y12g = 2.0 * y12g;
		double xm2y12g = x12g - c2y12g;
		double cumd11 = ee(0,0) * xm2y12g;
		double cumd22 = ee(1,1) * xm2y12g;
		double cumd33 = ee(2,2) * xm2y12g;
		double c2ymx12 = c2y12g - c13x12g;
		double cun34 = cumd11 - c13x12g;
		double cun56 = cumd11 + y12g;
		double cun712 = cumd22 + y12g;
		double cun89 = cumd33 + y12g;
		double cun1011 = cumd22 - c13x12g;

		MatrixR3C5d mat;
		mat(0,0) = e(0) * (cumd11+c2ymx12);
		mat(0,1) = e(1) * cun56;
		mat(0,2) = e(2) * cun56;
		mat(0,3) = e(0) * ee(1,2) * xm2y12g;
		mat(0,4) = e(0) * cun1011;
		mat(1,0) = e(1) * cun34;
		mat(1,1) = e(0) * cun712;
		mat(1,2) = mat(0,3);
		mat(1,3) = e(2) * cun712;
		mat(1,4) = e(1) * (cumd22+c2ymx12);
		mat(2,0) = e(2) * cun34;
		mat(2,1) = mat(0,3);
		mat(2,2) = e(0) * cun89;
		mat(2,3) = e(1) * cun89;
		mat(2,4) = e(2) * cun1011;

		tght.block<3,5>(6,0) = mat;
		tght.block<3,5>(0,5) = -mat;
	}

	{ // ht11, ht22=ht11
		double d11md22 = ee(0,0) - ee(1,1);
		double d22md33 = ee(1,1) - ee(2,2);
		double d33md11 = ee(2,2) - ee(0,0);
		double y11hd12 = y11h * ee(0,1);
		double y11hd13 = y11h * ee(0,2);
		double y11hd23 = y11h * ee(1,2);
		double cyhd12a = 2.0 * y11hd12;

		MatrixR3C5d mat;
		mat(0,0) = 0.0;
		mat(0,1) = -y11hd13;
		mat(0,2) = y11hd12;
		mat(0,3) = y11h * d22md33;
		mat(0,4) = -2.0 * y11hd23;
		mat(1,0) = 2.0 * y11hd13;
		mat(1,1) = y11hd23;
		mat(1,2) = y11h * d33md11;
		mat(1,3) = -y11hd12;
		mat(1,4) = 0.0;
		mat(2,0) = -cyhd12a;
		mat(2,1) = y11h * d11md22;
		mat(2,2) = -y11hd23;
		mat(2,3) = y11hd13;
		mat(2,4) = cyhd12a;

		tght.block<3,5>(3,0) = mat;
		tght.block<3,5>(9,5) = mat;
	}

	{ // ht12, ht21=ht12
		double d11md22 = ee(0,0) - ee(1,1);
		double d22md33 = ee(1,1) - ee(2,2);
		double d33md11 = ee(2,2) - ee(0,0);
		double y12hd12 = y12h * ee(0,1);
		double y12hd13 = y12h * ee(0,2);
		double y12hd23 = y12h * ee(1,2);
		double cyhd12b = 2.0 * y12hd12;

		MatrixR3C5d mat;
		mat(0,0) = 0.0;
		mat(0,1) = -y12h * ee(0,2);
		mat(0,2) = y12h * ee(0,1);
		mat(0,3) = y12h * d22md33;
		mat(0,4) = -2.0 * y12hd23;
		mat(1,0) = 2.0 * y12hd13;
		mat(1,1) = y12hd23;
		mat(1,2) = y12h * d33md11;
		mat(1,3) = -y12hd12;
		mat(1,4) = 0.0;
		mat(2,0) = -cyhd12b;
		mat(2,1) = y12h * d11md22;
		mat(2,2) = -y12hd23;
		mat(2,3) = y12hd13;
		mat(2,4) = cyhd12b;

		tght.block<3,5>(3,5) = mat;
		tght.block<3,5>(9,0) = mat;
	}


	//
	// Rse, m
	//
	{
		double m[3][3][3][3] = { 0 };
		for (int i=0; i<3; i++) {
			for (int j=0; j<3; j++) {
				for (int k=0; k<3; k++) {
					for (int l=0; l<3; l++) {
						m[i][j][k][l] = 
							1.5 * xm * (ee(i,j) - 1.0/3.0*Delta[i][j]) * (ee(k,l) - 1.0/3.0*Delta[k][l])
							+ 0.5 * ym * (ee(i,k)*Delta[j][l] + ee(j,k)*Delta[i][l] + 
							ee(i,l)*Delta[j][k] + ee(j,l)*Delta[i][k] - 
							4.0*ee(i,j)*ee(k,l))
							+ 0.5 * zm * (Delta[i][k]*Delta[j][l] + Delta[j][k]*Delta[i][l] - Delta[i][j]*Delta[k][l] +
							ee(i,j)*Delta[k][l] + ee(k,l)*Delta[i][j] - ee(i,k)*Delta[j][l] - 
							ee(j,k)*Delta[i][l] - ee(i,l)*Delta[j][k] - ee(j,l)*Delta[i][k] + 
							ee(i,j)*ee(k,l));
					}
				}
			}
		}

		Matrix5d mat;
		mob_reduce_m55(m, mat);
		//for (int i=0; i<5; i++) {
		//	int i0 = mesid[0][i];
		//	int i1 = mesid[1][i];

		//	if (i==0 || i==4) {
		//		mat(i,0) = m[i0][i0][0][0] - m[i0][i0][2][2] - m[i1][i1][0][0] + m[i1][i1][2][2];
		//		mat(i,1) = 2.0 * (m[i0][i0][0][1] - m[i1][i1][0][1]);
		//		mat(i,2) = 2.0 * (m[i0][i0][0][2] - m[i1][i1][0][2]);
		//		mat(i,3) = 2.0 * (m[i0][i0][1][2] - m[i1][i1][1][2]);
		//		mat(i,4) = m[i0][i0][1][1] - m[i0][i0][2][2] - m[i1][i1][1][1] + m[i1][i1][2][2];
		//	} else {
		//		mat(i,0) = 2.0 * (m[i0][i1][0][0] - m[i0][i1][2][2]);
		//		mat(i,1) = 4.0 * m[i0][i1][0][1];
		//		mat(i,2) = 4.0 * m[i0][i1][0][2];
		//		mat(i,3) = 4.0 * m[i0][i1][1][2];
		//		mat(i,4) = 2.0 * (m[i0][i1][1][1] - m[i0][i1][2][2]);
		//	}
		//}

		// m11, m12=m11, m22=m11
		tzm.block<5,5>(0,0) = mat;
		tzm.block<5,5>(0,5) = mat;
		tzm.block<5,5>(5,5) = mat;
	}

	if (1) {
		// fill symmetric ABCM
		for (int i=0; i<12; i++) {
			for (int j=0; j<i; j++) {
				tabc(i,j) = tabc(j,i);
			}
		}
		for (int i=0; i<10; i++) {
			for (int j=0; j<i; j++) {
				tzm(i,j) = tzm(j,i);
			}
		}
	}

	if (0) {
		std::cout << "ABC=" << std::endl;
		std::cout << tabc << std::endl;
		std::cout << "GHT=" << std::endl;
		std::cout << tght << std::endl;
		std::cout << "ZM=" << std::endl;
		std::cout << tzm << std::endl;
	}
}


//
// Invert M
//
void sd_inv_mob(StokesianDynamics &sd)
{
	const int np = sd.npart;
	const int n6 = np * 6;
	const int n5 = np * 5;

	const MatrixXd &muf = sd.muf;
	const MatrixXd &mus = sd.mus;
	const MatrixXd &mes = sd.mes;
	ASSERT_MATRIX_SIZE(muf, n6, n6);
	ASSERT_MATRIX_SIZE(mus, n6, n5);
	ASSERT_MATRIX_SIZE(mes, n5, n5);

	MatrixXd &rfu = sd.rfu;
	MatrixXd &rfe = sd.rfe;
	MatrixXd &rse = sd.rse;
	ASSERT_MATRIX_SIZE(rfu, n6, n6);
	ASSERT_MATRIX_SIZE(rfe, n6, n5);
	ASSERT_MATRIX_SIZE(rse, n5, n5);

	//
	MatrixXd R1(n6,n6);
	R1 = muf.inverse();

	MatrixXd R2(n5,n6);
	R2 = mus.transpose() * R1;

	MatrixXd R3(n5,n5);
	R3 = R2 * mus - mes;

	//
	rse = -R3.inverse();

	rfe = -R2.transpose() * rse;

	rfu = -rfe * R2 + R1;
}

void sd_corr_lub(StokesianDynamics &sd) 
{
	const int np = sd.npart;
	const int n6 = np * 6;
	const int n5 = np * 5;

	MatrixXd &lubfu = sd.lubfu;
	MatrixXd &lubfe = sd.lubfe;
	MatrixXd &lubse = sd.lubse;
	ASSERT_MATRIX_SIZE(lubfu, n6, n6);
	ASSERT_MATRIX_SIZE(lubfe, n6, n5);
	ASSERT_MATRIX_SIZE(lubse, n5, n5);

	lubfu.setZero();
	lubfe.setZero();
	lubse.setZero();

	// set pair lubrication
	for (int ipair=0; ipair<sd.npair; ipair++) {
		// check pair distance
		double dr = sd.dr(ipair);
		if (dr >= sd.dr_lub) continue;

		//
		Matrix12d tabc;
		MatrixR12C10d tght;
		Matrix10d tzm;

		lub_pair(dr, sd.ex(ipair), sd.ey(ipair), sd.ez(ipair),
			sd.dst(ipair), sd.dst_inv(ipair), sd.log_dst_inv(ipair), 
			tabc, tght, tzm);

		int i = sd.pairs[0][ipair];
		int j = sd.pairs[1][ipair];
		int ioff6 = i * 6;
		int joff6 = j * 6;
		int ioff5 = i * 5;
		int joff5 = j * 5;

		// put submatrix to correct position
		lubfu.block<6,6>(ioff6,ioff6) += tabc.block<6,6>(0,0);
		lubfu.block<6,6>(ioff6,joff6) += tabc.block<6,6>(0,6);
		lubfu.block<6,6>(joff6,ioff6) += tabc.block<6,6>(6,0);
		lubfu.block<6,6>(joff6,joff6) += tabc.block<6,6>(6,6);

		lubfe.block<6,5>(ioff6,ioff5) += tght.block<6,5>(0,0);
		lubfe.block<6,5>(ioff6,joff5) += tght.block<6,5>(0,5);
		lubfe.block<6,5>(joff6,ioff5) += tght.block<6,5>(6,0);
		lubfe.block<6,5>(joff6,joff5) += tght.block<6,5>(6,5);

		lubse.block<5,5>(ioff5,ioff5) += tzm.block<5,5>(0,0);
		lubse.block<5,5>(ioff5,joff5) += tzm.block<5,5>(0,5);
		lubse.block<5,5>(joff5,ioff5) += tzm.block<5,5>(5,0);
		lubse.block<5,5>(joff5,joff5) += tzm.block<5,5>(5,5);

	}
}



//
void sd_set_flow(StokesianDynamics &sd, 
	const double U0[3], const double O0[3], const double E0[5])
{
	const VectorXd &x = sd.x;
	VectorXd &uinf = sd.uinf;
	VectorXd &einf = sd.einf;

	const double Efull[3][3] = {
		E0[0], E0[1], E0[2],
		E0[1], E0[4], E0[3],
		E0[2], E0[3], -E0[0]-E0[4],
	};

	const int np = sd.npart;
	
	for (int i=0; i<np; i++) {
		int iu = i * 6;
		int ie = i * 5;

		double xx = x(iu+0);
		double yy = x(iu+1);
		double zz = x(iu+2);

		uinf(iu+0) = U0[0] + O0[1]*zz - O0[2]*yy
			+ Efull[0][0]*xx + Efull[0][1]*yy + Efull[0][2]*zz;
		uinf(iu+1) = U0[1] + O0[2]*xx - O0[0]*zz
			+ Efull[1][0]*xx + Efull[1][1]*yy + Efull[1][2]*zz;
		uinf(iu+2) = U0[2] + O0[0]*yy - O0[1]*xx
			+ Efull[2][0]*xx + Efull[2][1]*yy + Efull[2][2]*zz;

		uinf(iu+3) = O0[0];
		uinf(iu+4) = O0[1];
		uinf(iu+5) = O0[2];

		einf(ie+0) = Efull[0][0] - Efull[2][2];
		einf(ie+1) = 2.0 * Efull[0][1];
		einf(ie+2) = 2.0 * Efull[1][2];
		einf(ie+3) = 2.0 * Efull[0][2];
		einf(ie+4) = Efull[1][1] - Efull[2][2];
	}

}



void sd_save_csv(StokesianDynamics &sd, const char outfilename[]) {

	FILE *fp = fopen(outfilename, "w");
	if (!fp) {
		std::cerr << "Failed to save " << outfilename << std::endl;
		exit(1);
	}

	fprintf(fp, "x,y,z,u,v,w\n");

	for (int i=0; i<sd.npart; i++) {
		double xx = sd.x(i*6+0);
		double yy = sd.x(i*6+1);
		double zz = sd.x(i*6+2);
		
		double uu = sd.u(i*6+0);
		double vv = sd.u(i*6+1);
		double ww = sd.u(i*6+2);

		fprintf(fp, "%lf,%lf,%lf,%lf,%lf,%lf\n", 
			xx, yy, zz, 
			uu, vv, ww);
	}


	fclose(fp);

	std::cout << "Saved " << outfilename << std::endl;
}
