
#include <iostream>


#include "sd.hpp"


const double ewald_safe_factor = 1.15;



static double ewald_real(double r, double xi) {

	double xir = xi * r / ewald_safe_factor;

	double xir2 = xir * xir;

	double r2 = r * r;

	double val = math_erfc(xir) * (0.75 + 1.5/r2) / r + 
		exp(-xir2) * xi / SqrtPI * (4.5 + 3.0*xir2 + (3.0+xir2*(14.0+xir2*(4.0+xir2)))/r2);

	return val;
}

static double ewald_recip(double k, double xi) {
	double kx = k / xi / ewald_safe_factor;
	double kx2 = kx * kx;
	double k2 = k * k;

	double val = SixPI * exp(-kx2/4.0) * (1.0+k2/3.0)/k2 * (1.0+kx2*(0.25+kx2/8.0));

	return val;
}



void sd_ewald_setup(StokesianDynamics &sd)
{
	assert(sd.bc_mode == 1);

	const double xi = sd.ewald_xi;
	const double ewald_eps = sd.ewald_eps;

	const double lx = sd.prob_lx;
	const double ly = sd.prob_ly;
	const double lz = sd.prob_lz;

	double lmin = std::min(lx, std::min(ly,lz));
	double lmax = std::max(lx, std::max(ly,lz));

	std::cout << "xi=" << xi << std::endl;
	std::cout << "eps=" << ewald_eps << std::endl;

	double dr = lmin;
	double dk = TwoPI / lmax;

	const double red_factor = 0.99;

	// real space cutoff
	double rmax = 10.0 * sqrt(-log(ewald_eps)) / xi;
	while (ewald_real(rmax*red_factor,xi) + 6.0*ewald_real(rmax*red_factor+dr,xi) < ewald_eps) {
		rmax *= red_factor;
	}

	// set real
	sd.ewald_rmax = rmax;
	sd.ewald_rmax_x = (int) (rmax / lx) + 1;
	sd.ewald_rmax_y = (int) (rmax / ly) + 1;
	sd.ewald_rmax_z = (int) (rmax / lz) + 1;

	std::cout << "In real space: " << std::endl
		<< "rmax = " << rmax << std::endl
		<< "rmax_x = " << sd.ewald_rmax_x << std::endl
		<< "rmax_y = " << sd.ewald_rmax_y << std::endl
		<< "rmax_z = " << sd.ewald_rmax_z << std::endl;

	// wave space cutoff
	double kmax = 10.0 * sqrt(-log(ewald_eps)) * 2.0 * xi;
	while (ewald_recip(kmax*red_factor,xi) + 6.0*ewald_recip(kmax*red_factor+dk,xi) < ewald_eps) {
		kmax *= red_factor;
	}

	// set wave
	sd.ewald_kmax = kmax;
	sd.ewald_kmax_x = (int) (kmax * lx / TwoPI);
	sd.ewald_kmax_y = (int) (kmax * ly / TwoPI);
	sd.ewald_kmax_z = (int) (kmax * lz / TwoPI);

	std::cout << "In wave space:" << std::endl
		<< "kmax = " << kmax << std::endl
		<< "kmax_x = " << sd.ewald_kmax_x << std::endl
		<< "kmax_y = " << sd.ewald_kmax_y << std::endl
		<< "kmax_z = " << sd.ewald_kmax_z << std::endl;

}

