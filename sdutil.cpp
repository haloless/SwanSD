
//#include "sdutil.hpp"

// Intel's header, for ERF, ERFC
#include <mathimf.h>

#define MATH_WRAP_DECL(func) \
	double math_##func (double x) {\
	return func (x); \
	}


double math_erf(double x) {
	return erf(x);
}

double math_erfc(double x) {
	return erfc(x);
}

double math_gamma(double x) {
	return gamma(x);
}

//MATH_WRAP_DECL(erf);
//
//MATH_WRAP_DECL(erfc);
//
//MATH_WRAP_DECL(gamma);

