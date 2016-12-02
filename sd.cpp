
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <iostream>
#include <utility>
#include <algorithm>


#include "sd.hpp"


typedef Eigen::Matrix<double,3,5> Matrix_3_5d;
typedef Eigen::Matrix<double,5,3> Matrix_5_3d;
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


//	int k = 3 - i - j;
//	if (k == -1) k = 2;
//	if (k == 3)  k = 0;
static const int compid[3][3] = {
	0, 2, 1, 
	2, 1, 0,
	1, 0, 2,
};

enum R2BABC {
	RSABC = 0,
	X11AS = 1,
	X12AS,
	Y11AS,
	Y12AS,
	Y11BS,
	Y12BS,
	X11CS,
	X12CS,
	Y11CS,
	Y12CS,
	R2BABC_NUM,
};

enum R2BGH {
	RSGH = 0,
	X11GS = 1,
	X12GS,
	Y11GS,
	Y12GS,
	Y11HS,
	Y12HS,
	R2BGH_NUM,
};

enum R2BM {
	RSM = 0,
	XMS = 1,
	YMS,
	ZMS,
	R2BM_NUM,
};

static const double r2babc[][R2BABC_NUM] = {
2.10000000e+00,	1.98069000e+00,	-1.97823200e+00,	8.52200000e-02,	-8.14540000e-02,	-7.87158000e-02,	8.02421000e-02,	2.19500000e-02,	-1.00995000e-02,	1.59990000e-01,	5.80310000e-03 ,
2.15000000e+00,	1.14261000e+00,	-1.14019200e+00,	5.37500000e-02,	-5.03280000e-02,	-4.82308000e-02,	4.95261000e-02,	1.67400000e-02,	-6.79550000e-03,	1.04880000e-01,	-1.96900000e-04,
2.20000000e+00,	7.40120000e-01,	-7.37752000e-01,	3.67000000e-02,	-3.35920000e-02,	-3.18458000e-02,	3.29531000e-02,	1.30800000e-02,	-4.73150000e-03,	7.36000000e-02,	-2.24620000e-03,
2.25000000e+00,	5.11730000e-01,	-5.09422000e-01,	2.63700000e-02,	-2.35600000e-02,	-2.20328000e-02,	2.29871000e-02,	1.04000000e-02,	-3.37250000e-03,	5.38700000e-02,	-2.82460000e-03,
2.30000000e+00,	3.69290000e-01,	-3.67042000e-01,	1.96700000e-02,	-1.71210000e-02,	-1.57498000e-02,	1.65771000e-02,	8.37000000e-03,	-2.45050000e-03,	4.06100000e-02,	-2.81500000e-03,
2.35000000e+00,	2.74840000e-01,	-2.72662000e-01,	1.50800000e-02,	-1.27850000e-02,	-1.15348000e-02,	1.22561000e-02,	6.79000000e-03,	-1.80550000e-03,	3.12900000e-02,	-2.57600000e-03,
2.40000000e+00,	2.09440000e-01,	-2.07332000e-01,	1.18300000e-02,	-9.75800000e-03,	-8.60950000e-03,	9.24210000e-03,	5.56000000e-03,	-1.34810000e-03,	2.45200000e-02,	-2.26330000e-03,
2.45000000e+00,	1.62630000e-01,	-1.60603000e-01,	9.46000000e-03,	-7.58300000e-03,	-6.52500000e-03,	7.08110000e-03,	4.58000000e-03,	-1.01760000e-03,	1.94900000e-02,	-1.94570000e-03,
2.50000000e+00,	1.28260000e-01,	-1.26311000e-01,	7.67000000e-03,	-5.98200000e-03,	-5.00680000e-03,	5.49710000e-03,	3.80000000e-03,	-7.75600000e-04,	1.56600000e-02,	-1.65200000e-03,
2.55000000e+00,	1.02480000e-01,	-1.00617000e-01,	6.30000000e-03,	-4.77900000e-03,	-3.88220000e-03,	4.31510000e-03,	3.17000000e-03,	-5.96300000e-04,	1.27100000e-02,	-1.39250000e-03,
2.60000000e+00,	8.28000000e-02,	-8.10230000e-02,	5.24000000e-03,	-3.86200000e-03,	-3.03670000e-03,	3.42010000e-03,	2.65000000e-03,	-4.62100000e-04,	1.04000000e-02,	-1.16880000e-03,
2.65000000e+00,	6.75400000e-02,	-6.58530000e-02,	4.39000000e-03,	-3.15100000e-03,	-2.39320000e-03,	2.73410000e-03,	2.23000000e-03,	-3.60600000e-04,	8.57000000e-03,	-9.78800000e-04,
2.70000000e+00,	5.55600000e-02,	-5.39560000e-02,	3.71000000e-03,	-2.59300000e-03,	-1.89830000e-03,	2.20110000e-03,	1.88000000e-03,	-2.83200000e-04,	7.11000000e-03,	-8.18900000e-04,
2.75000000e+00,	4.60300000e-02,	-4.45190000e-02,	3.15000000e-03,	-2.14900000e-03,	-1.51410000e-03,	1.78310000e-03,	1.59000000e-03,	-2.23800000e-04,	5.92000000e-03,	-6.84900000e-04,
2.80000000e+00,	3.83800000e-02,	-3.69560000e-02,	2.70000000e-03,	-1.79200000e-03,	-1.21350000e-03,	1.45210000e-03,	1.34000000e-03,	-1.77800000e-04,	4.95000000e-03,	-5.73000000e-04,
2.85000000e+00,	3.21700000e-02,	-3.08400000e-02,	2.32000000e-03,	-1.50300000e-03,	-9.76500000e-04,	1.18910000e-03,	1.14000000e-03,	-1.41900000e-04,	4.17000000e-03,	-4.79600000e-04,
2.90000000e+00,	2.71000000e-02,	-2.58530000e-02,	2.00000000e-03,	-1.26700000e-03,	-7.88600000e-04,	9.78100000e-04,	9.70000000e-04,	-1.13900000e-04,	3.52000000e-03,	-4.01600000e-04,
2.95000000e+00,	2.29200000e-02,	-2.17570000e-02,	1.73000000e-03,	-1.07100000e-03,	-6.38700000e-04,	8.07100000e-04,	8.30000000e-04,	-9.17000000e-05,	2.97000000e-03,	-3.36500000e-04,
3.00000000e+00,	1.94500000e-02,	-1.83680000e-02,	1.50000000e-03,	-9.10000000e-04,	-5.18600000e-04,	6.68100000e-04,	7.20000000e-04,	-7.41000000e-05,	2.51000000e-03,	-2.82300000e-04,
3.05000000e+00,	1.65500000e-02,	-1.55470000e-02,	1.30000000e-03,	-7.74000000e-04,	-4.21900000e-04,	5.55100000e-04,	6.10000000e-04,	-6.01000000e-05,	2.14000000e-03,	-2.36900000e-04,
3.10000000e+00,	1.40900000e-02,	-1.31570000e-02,	1.13000000e-03,	-6.60000000e-04,	-3.43600000e-04,	4.62100000e-04,	5.20000000e-04,	-4.88000000e-05,	1.82000000e-03,	-1.98700000e-04,
3.15000000e+00,	1.20200000e-02,	-1.11750000e-02,	9.90000000e-04,	-5.63000000e-04,	-2.80100000e-04,	3.85100000e-04,	4.40000000e-04,	-3.97000000e-05,	1.55000000e-03,	-1.66800000e-04,
3.20000000e+00,	1.02800000e-02,	-9.49800000e-03,	8.60000000e-04,	-4.82000000e-04,	-2.28500000e-04,	3.21100000e-04,	3.80000000e-04,	-3.25000000e-05,	1.32000000e-03,	-1.40000000e-04,
3.25000000e+00,	8.78000000e-03,	-8.07200000e-03,	7.40000000e-04,	-4.12000000e-04,	-1.86200000e-04,	2.68100000e-04,	3.20000000e-04,	-2.65000000e-05,	1.12000000e-03,	-1.17300000e-04,
3.30000000e+00,	7.50000000e-03,	-6.85500000e-03,	6.50000000e-04,	-3.51000000e-04,	-1.51700000e-04,	2.23000000e-04,	2.70000000e-04,	-2.17000000e-05,	9.50000000e-04,	-9.82000000e-05,
3.35000000e+00,	6.39000000e-03,	-5.81300000e-03,	5.60000000e-04,	-3.00000000e-04,	-1.23300000e-04,	1.85700000e-04,	2.30000000e-04,	-1.76000000e-05,	8.10000000e-04,	-8.20000000e-05,
3.40000000e+00,	5.43000000e-03,	-4.91500000e-03,	4.90000000e-04,	-2.54000000e-04,	-9.99000000e-05,	1.54300000e-04,	2.00000000e-04,	-1.44000000e-05,	6.80000000e-04,	-6.84000000e-05,
3.45000000e+00,	4.61000000e-03,	-4.14100000e-03,	4.10000000e-04,	-2.15000000e-04,	-8.07000000e-05,	1.27800000e-04,	1.70000000e-04,	-1.17000000e-05,	5.70000000e-04,	-5.67000000e-05,
3.50000000e+00,	3.88000000e-03,	-3.47000000e-03,	3.50000000e-04,	-1.82000000e-04,	-6.48000000e-05,	1.05300000e-04,	1.40000000e-04,	-9.50000000e-06,	4.80000000e-04,	-4.68000000e-05,
3.55000000e+00,	3.25000000e-03,	-2.89300000e-03,	3.00000000e-04,	-1.52000000e-04,	-5.16000000e-05,	8.61000000e-05,	1.20000000e-04,	-7.60000000e-06,	4.00000000e-04,	-3.84000000e-05,
3.60000000e+00,	2.69000000e-03,	-2.38300000e-03,	2.50000000e-04,	-1.26000000e-04,	-4.08000000e-05,	6.99000000e-05,	9.00000000e-05,	-6.10000000e-06,	3.20000000e-04,	-3.12000000e-05,
3.65000000e+00,	2.20000000e-03,	-1.93800000e-03,	2.10000000e-04,	-1.03000000e-04,	-3.18000000e-05,	5.60000000e-05,	8.00000000e-05,	-4.80000000e-06,	2.70000000e-04,	-2.50000000e-05,
3.70000000e+00,	1.77000000e-03,	-1.54700000e-03,	1.60000000e-04,	-8.20000000e-05,	-2.43000000e-05,	4.40000000e-05,	7.00000000e-05,	-3.70000000e-06,	2.10000000e-04,	-1.97000000e-05,
3.75000000e+00,	1.38000000e-03,	-1.20400000e-03,	1.30000000e-04,	-6.40000000e-05,	-1.82000000e-05,	3.37000000e-05,	5.00000000e-05,	-2.90000000e-06,	1.60000000e-04,	-1.51000000e-05,
3.80000000e+00,	1.04000000e-03,	-9.01000000e-04,	1.00000000e-04,	-4.80000000e-05,	-1.30000000e-05,	2.49000000e-05,	3.00000000e-05,	-2.10000000e-06,	1.20000000e-04,	-1.12000000e-05,
3.85000000e+00,	7.40000000e-04,	-6.34000000e-04,	7.00000000e-05,	-3.40000000e-05,	-8.80000000e-06,	1.73000000e-05,	3.00000000e-05,	-1.40000000e-06,	8.00000000e-05,	-7.80000000e-06,
3.90000000e+00,	4.60000000e-04,	-3.97000000e-04,	4.00000000e-05,	-2.10000000e-05,	-5.30000000e-06,	1.07000000e-05,	2.00000000e-05,	-9.00000000e-07,	5.00000000e-05,	-4.90000000e-06,
3.95000000e+00,	2.20000000e-04,	-1.87000000e-04,	2.00000000e-05,	-1.00000000e-05,	-2.40000000e-06,	4.90000000e-06,	1.00000000e-05,	-4.00000000e-07,	2.00000000e-05,	-2.30000000e-06,
4.00000000e+00,	0.00000000e+00,	0.00000000e+00	,   0.00000000e+00,	0.00000000e+00	,	0.00000000e+00	,0.00000000e+00	,   0.00000000e+00,	0.00000000e+00,		0.00000000e+00,	0.00000000e+00 ,
};

static const double r2bgh[][R2BGH_NUM] = {
2.10000000e+00,	2.02675700e+00,	-2.01782500e+00,	4.13886800e-02,	-3.20663998e-02,	-1.93897170e-02,	7.95613997e-02,
2.11000000e+00,	1.79415700e+00,	-1.78525500e+00,	3.75377800e-02,	-2.84543998e-02,	-1.93518170e-02,	7.23663997e-02,
2.12000000e+00,	1.60105700e+00,	-1.59219500e+00,	3.42245800e-02,	-2.53723998e-02,	-1.91942170e-02,	6.60993997e-02,
2.13000000e+00,	1.43840700e+00,	-1.42959500e+00,	3.13491800e-02,	-2.27203998e-02,	-1.89486170e-02,	6.05963997e-02,
2.14000000e+00,	1.29974700e+00,	-1.29095500e+00,	2.88322800e-02,	-2.04252998e-02,	-1.86386170e-02,	5.57303997e-02,
2.15000000e+00,	1.18030700e+00,	-1.17156500e+00,	2.66164800e-02,	-1.84231998e-02,	-1.82817170e-02,	5.14003997e-02,
2.16000000e+00,	1.07649700e+00,	-1.06780500e+00,	2.46530800e-02,	-1.66682998e-02,	-1.78912170e-02,	4.75263997e-02,
2.17000000e+00,	9.85595000e-01,	-9.76940000e-01,	2.29039800e-02,	-1.51228998e-02,	-1.74775170e-02,	4.40443997e-02,
2.18000000e+00,	9.05431000e-01,	-8.96828000e-01,	2.13386800e-02,	-1.37559998e-02,	-1.70483170e-02,	4.09023997e-02,
2.19000000e+00,	8.34337000e-01,	-8.25778000e-01,	1.99318800e-02,	-1.25411998e-02,	-1.66097170e-02,	3.80543997e-02,
2.20000000e+00,	7.70931000e-01,	-7.62428000e-01,	1.86613800e-02,	-1.14591998e-02,	-1.61666170e-02,	3.54663997e-02,
2.25000000e+00,	5.36914000e-01,	-5.28699000e-01,	1.38389800e-02,	-7.50679980e-03,	-1.39867170e-02,	2.54763997e-02,
2.30000000e+00,	3.89924000e-01,	-3.82021000e-01,	1.06807800e-02,	-5.11039980e-03,	-1.20005170e-02,	1.88113997e-02,
2.35000000e+00,	2.91795000e-01,	-2.84233000e-01,	8.49198000e-03,	-3.58899980e-03,	-1.02675170e-02,	1.41783997e-02,
2.40000000e+00,	2.23413000e-01,	-2.16214000e-01,	6.90678000e-03,	-2.58849980e-03,	-8.78215700e-03,	1.08573997e-02,
2.45000000e+00,	1.74196000e-01,	-1.67376000e-01,	5.71738000e-03,	-1.91129980e-03,	-7.51826700e-03,	8.42039970e-03,
2.50000000e+00,	1.37866000e-01,	-1.31428000e-01,	4.79848000e-03,	-1.44179980e-03,	-6.44545700e-03,	6.60139970e-03,
2.55000000e+00,	1.10486000e-01,	-1.04435000e-01,	4.07128000e-03,	-1.10879980e-03,	-5.53487700e-03,	5.22039970e-03,
2.60000000e+00,	8.94920000e-02,	-8.38270000e-02,	3.48418000e-03,	-8.67899800e-04,	-4.76101700e-03,	4.16079970e-03,
2.65000000e+00,	7.31530000e-02,	-6.78690000e-02,	3.00218000e-03,	-6.90400800e-04,	-4.10210700e-03,	3.33870970e-03,
2.70000000e+00,	6.02730000e-02,	-5.53630000e-02,	2.60068000e-03,	-5.57201800e-04,	-3.53977700e-03,	2.69489970e-03,
2.75000000e+00,	5.00040000e-02,	-4.54540000e-02,	2.26268000e-03,	-4.55599800e-04,	-3.05875700e-03,	2.18679970e-03,
2.80000000e+00,	4.17330000e-02,	-3.75310000e-02,	1.97518000e-03,	-3.76798800e-04,	-2.64624700e-03,	1.78249970e-03,
2.85000000e+00,	3.50110000e-02,	-3.11420000e-02,	1.72848000e-03,	-3.14699800e-04,	-2.29165700e-03,	1.45918970e-03,
2.90000000e+00,	2.95020000e-02,	-2.59500000e-02,	1.51548000e-03,	-2.65099800e-04,	-1.98608700e-03,	1.19879970e-03,
2.95000000e+00,	2.49530000e-02,	-2.17020000e-02,	1.33048000e-03,	-2.24899800e-04,	-1.72215700e-03,	9.87899700e-04,
3.00000000e+00,	2.11710000e-02,	-1.82060000e-02,	1.16878000e-03,	-1.91899800e-04,	-1.49368700e-03,	8.16400700e-04,
3.05000000e+00,	1.80080000e-02,	-1.53100000e-02,	1.02707000e-03,	-1.64399800e-04,	-1.29544700e-03,	6.76300700e-04,
3.10000000e+00,	1.53460000e-02,	-1.28990000e-02,	9.02260000e-04,	-1.41497800e-04,	-1.12308700e-03,	5.61199700e-04,
3.15000000e+00,	1.30940000e-02,	-1.08820000e-02,	7.92050000e-04,	-1.21899800e-04,	-9.72917000e-04,	4.66298700e-04,
3.20000000e+00,	1.11780000e-02,	-9.18700000e-03,	6.94320000e-04,	-1.05299800e-04,	-8.41807000e-04,	3.87701700e-04,
3.25000000e+00,	9.54200000e-03,	-7.75598000e-03,	6.07540000e-04,	-9.09988000e-05,	-7.27120000e-04,	3.22401700e-04,
3.30000000e+00,	8.13800000e-03,	-6.54299000e-03,	5.30230000e-04,	-7.84998000e-05,	-6.26604000e-04,	2.68101700e-04,
3.35000000e+00,	6.92901000e-03,	-5.51198000e-03,	4.61250000e-04,	-6.77002000e-05,	-5.38357000e-04,	2.22399700e-04,
3.40000000e+00,	5.88300000e-03,	-4.63101000e-03,	3.99530000e-04,	-5.81001000e-05,	-4.60730000e-04,	1.84301700e-04,
3.45000000e+00,	4.97500000e-03,	-3.87700000e-03,	3.44250000e-04,	-4.96996000e-05,	-3.92344000e-04,	1.52200700e-04,
3.50000000e+00,	4.18500000e-03,	-3.22699000e-03,	2.94620000e-04,	-4.22695000e-05,	-3.31976000e-04,	1.24897700e-04,
3.55000000e+00,	3.49400000e-03,	-2.66901000e-03,	2.50000000e-04,	-3.56794000e-05,	-2.78612000e-04,	1.01998700e-04,
3.60000000e+00,	2.88800000e-03,	-2.18600000e-03,	2.09830000e-04,	-2.98401000e-05,	-2.31361000e-04,	8.25007000e-05,
3.65000000e+00,	2.35590000e-03,	-1.76600000e-03,	1.73600000e-04,	-2.45399000e-05,	-1.89457000e-04,	6.57997000e-05,
3.70000000e+00,	1.88630000e-03,	-1.40199000e-03,	1.40880000e-04,	-1.98698000e-05,	-1.52240000e-04,	5.15990000e-05,
3.75000000e+00,	1.47180000e-03,	-1.08500000e-03,	1.11300000e-04,	-1.56597000e-05,	-1.19136000e-04,	3.94993000e-05,
3.80000000e+00,	1.10500000e-03,	-8.08000000e-04,	8.45200000e-05,	-1.18399000e-05,	-8.96460000e-05,	2.89995000e-05,
3.85000000e+00,	7.79300000e-04,	-5.66010000e-04,	6.02500000e-05,	-8.42010000e-06,	-6.33400000e-05,	2.00998000e-05,
3.90000000e+00,	4.89100000e-04,	-3.52990000e-04,	3.82200000e-05,	-5.33000000e-06,	-3.98430000e-05,	1.23996000e-05,
3.95000000e+00,	2.30600000e-04,	-1.65000000e-04,	1.82000000e-05,	-2.52950000e-06,	-1.88240000e-05,	5.69970000e-06,
4.00000000e+00,	0.00000000e+00,	0.00000000e+00,		0.00000000e+00,	0.00000000e+00,		0.00000000e+00,		0.00000000e+00,
};

static const double r2bm[][R2BM_NUM] = {
2.10000000e+00,	2.75887000e+00,	8.07955110e-02,	8.64000330e-03,
2.11000000e+00,	2.44633000e+00,	7.36955110e-02,	8.31444330e-03,
2.12000000e+00,	2.18653000e+00,	6.76211110e-02,	8.00222330e-03,
2.13000000e+00,	1.96742000e+00,	6.23600110e-02,	7.69444330e-03,
2.14000000e+00,	1.78033000e+00,	5.77778110e-02,	7.41000330e-03,
2.15000000e+00,	1.61896000e+00,	5.37511110e-02,	7.13111330e-03,
2.16000000e+00,	1.47851000e+00,	5.01989110e-02,	6.86555330e-03,
2.17000000e+00,	1.35532000e+00,	4.70433110e-02,	6.61666330e-03,
2.18000000e+00,	1.24655000e+00,	4.42178110e-02,	6.37222330e-03,
2.19000000e+00,	1.14993000e+00,	4.16755110e-02,	6.14222330e-03,
2.20000000e+00,	1.06364000e+00,	3.93933110e-02,	5.91889330e-03,
2.25000000e+00,	7.43872000e-01,	3.06900110e-02,	4.92555330e-03,
2.30000000e+00,	5.41620000e-01,	2.49100110e-02,	4.11889330e-03,
2.35000000e+00,	4.05796000e-01,	2.07933110e-02,	3.46000330e-03,
2.40000000e+00,	3.10632000e-01,	1.76922110e-02,	2.91666330e-03,
2.45000000e+00,	2.41868000e-01,	1.52489110e-02,	2.47444330e-03,
2.50000000e+00,	1.90947000e-01,	1.32633110e-02,	2.09777330e-03,
2.55000000e+00,	1.52487000e-01,	1.16033110e-02,	1.78000330e-03,
2.60000000e+00,	1.22971000e-01,	1.01933110e-02,	1.52000330e-03,
2.65000000e+00,	9.99960000e-02,	8.98000100e-03,	1.30555330e-03,
2.70000000e+00,	8.19111000e-02,	7.91889100e-03,	1.11333330e-03,
2.75000000e+00,	6.75067000e-02,	6.99667100e-03,	9.58893300e-04,
2.80000000e+00,	5.59467000e-02,	6.18000100e-03,	8.23333300e-04,
2.85000000e+00,	4.65911000e-02,	5.46222100e-03,	7.08889300e-04,
2.90000000e+00,	3.89544000e-02,	4.83111100e-03,	6.15555300e-04,
2.95000000e+00,	3.26756000e-02,	4.26222100e-03,	5.33333300e-04,
3.00000000e+00,	2.74956000e-02,	3.75889100e-03,	4.58889300e-04,
3.05000000e+00,	2.31911000e-02,	3.31222100e-03,	4.01111300e-04,
3.10000000e+00,	1.95933000e-02,	2.91778100e-03,	3.45555300e-04,
3.15000000e+00,	1.65811000e-02,	2.56444100e-03,	2.98889300e-04,
3.20000000e+00,	1.40344000e-02,	2.24444100e-03,	2.54444300e-04,
3.25000000e+00,	1.18778000e-02,	1.96555100e-03,	2.17777300e-04,
3.30000000e+00,	1.00367000e-02,	1.71333100e-03,	1.92222300e-04,
3.35000000e+00,	8.48110000e-03,	1.48000100e-03,	1.63333300e-04,
3.40000000e+00,	7.14000000e-03,	1.28555100e-03,	1.42222300e-04,
3.45000000e+00,	5.98444000e-03,	1.10444100e-03,	1.23333300e-04,
3.50000000e+00,	4.99778000e-03,	9.41111000e-04,	9.88893000e-05,
3.55000000e+00,	4.14222000e-03,	7.90001000e-04,	9.00003000e-05,
3.60000000e+00,	3.39333000e-03,	6.67781000e-04,	7.11113000e-05,
3.65000000e+00,	2.74444000e-03,	5.51111000e-04,	6.22223000e-05,
3.70000000e+00,	2.18333000e-03,	4.37781000e-04,	4.44444000e-05,
3.75000000e+00,	1.68333000e-03,	3.45555000e-04,	4.22222000e-05,
3.80000000e+00,	1.26333000e-03,	2.63333000e-04,	3.44444000e-05,
3.85000000e+00,	8.88890000e-04,	1.88889000e-04,	2.33333000e-05,
3.90000000e+00,	5.56670000e-04,	1.12222000e-04,	2.00000000e-05,
3.95000000e+00,	2.64440000e-04,	5.33330000e-05,	1.55555000e-05,
4.00000000e+00,	0.00000000e+00,	0.00000000e+00,	0.00000000e+00,
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


// self contribution
static void mob_self(
	Matrix3d &ma, Matrix3d &mb, Matrix3d &mbt, Matrix3d &mc,
	Matrix_3_5d &mgt, Matrix_3_5d &mht, Matrix5d &mm)
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

static void mob_pair(
	double dr_inv, double dx, double dy, double dz,
	Matrix3d &ma, Matrix3d &mb, Matrix3d &mbt, Matrix3d &mc,
	Matrix_3_5d &mgt, Matrix_3_5d &mht, Matrix5d &mm)
{

	double dr_inv2 = dr_inv * dr_inv;
	double dr_inv3 = dr_inv2 * dr_inv;
	double dr_inv4 = dr_inv3 * dr_inv;
	double dr_inv5 = dr_inv4 * dr_inv;

	// ei*ej
	Vector3d e(dx,dy,dz);
	Matrix3d ee = e * e.transpose();

	// scalars for pair, so actually X12A,Y12A,etc.
	// 
	double xa = 1.5 * dr_inv - dr_inv3;
	double ya = 0.75 * dr_inv + 0.5 * dr_inv3;

	//
	double yb = -0.75 * dr_inv2;

	//
	double xc = 0.75 * dr_inv3;
	double yc = -0.375 * dr_inv3;

	//
	double xg = 2.25 * dr_inv2 - 3.6 * dr_inv4;
	double yg = 1.2 * dr_inv4;

	//
	double yh = -1.125 * dr_inv3;

	//
	double xm = -4.5 * dr_inv3 + 10.8 * dr_inv5;
	double ym = 2.25 * dr_inv3 - 7.2 * dr_inv5;
	double zm = 1.8 * dr_inv5;

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

	//
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
	for (int i=0; i<3; i++) {
		mgt(i,0) = gt[i][0][0] - gt[i][2][2];
		mgt(i,1) = 2.0 * gt[i][0][1];
		mgt(i,2) = 2.0 * gt[i][0][2];
		mgt(i,3) = 2.0 * gt[i][1][2];
		mgt(i,4) = gt[i][1][1] - gt[i][2][2];

		mht(i,0) = ht[i][0][0] - ht[i][2][2];
		mht(i,1) = 2.0 * ht[i][0][1];
		mht(i,2) = 2.0 * ht[i][0][2];
		mht(i,3) = 2.0 * ht[i][1][2];
		mht(i,4) = ht[i][1][1] - ht[i][2][2];
	}

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
	Matrix_3_5d mgt, mht;
	Matrix5d mm;

	// self part, same for all
	mob_self(ma, mb, mbt, mc, mgt, mht, mm);

	// set self part
	for (int i=0; i<np; i++) {
		int ioff = i * 6;
		// u
		muf.block<3,3>(ioff,ioff) = ma;
		// omega
		muf.block<3,3>(ioff+3,ioff+3) = mc;

		ioff = i * 5;
		// E
		mes.block<5,5>(ioff,ioff) = mm;

	} // end self

	// set pair part
	for (int ipair=0; ipair<sd.npair; ipair++) {
		int i = sd.pairs[0][ipair];
		int j = sd.pairs[1][ipair];

		//
		mob_pair(sd.dr_inv(ipair), sd.ex(ipair), sd.ey(ipair), sd.ez(ipair),
			ma, mb, mbt, mc, mgt, mht, mm);

		//
		int ioff = i * 6;
		int joff = j * 6;

		muf.block<3,3>(ioff  ,joff  ) = ma;
		muf.block<3,3>(ioff+3,joff  ) = mb;
		muf.block<3,3>(ioff  ,joff+3) = mbt;
		muf.block<3,3>(ioff+3,joff+3) = mc;

		muf.block<3,3>(joff  ,ioff  ) = ma.transpose();
		muf.block<3,3>(joff+3,ioff  ) = mb.transpose();
		muf.block<3,3>(joff  ,ioff+3) = mbt.transpose();
		muf.block<3,3>(joff+3,ioff+3) = mc.transpose();

		//
		int i5 = i * 5;
		int j5 = j * 5;

		mus.block<3,5>(ioff,j5) = mgt;
		mus.block<3,5>(joff,i5) = -mgt;

		mus.block<3,5>(ioff+3,j5) = mht;
		mus.block<3,5>(joff+3,i5) = mht;

		//
		mes.block<5,5>(i5,j5) = mm;
		mes.block<5,5>(j5,i5) = mm.transpose();

	} // end pair


}


//
// 
static void lub_pair(
	double dr, double dx, double dy, double dz,
	double dst, double dst_inv, double dst_log,
	//Matrix3d &ma, Matrix3d &mb, Matrix3d &mbt, Matrix3d &mc,
	//Matrix_3_5d &mgt, Matrix_3_5d &mht, Matrix5d &mm
	Matrix12d &tabc, MatrixR12C10d &tght, Matrix10d &tzm
	)
{
	//ma.setZero();
	//mb.setZero();
	//mbt.setZero();
	//mc.setZero();
	//mgt.setZero();
	//mht.setZero();
	//mm.setZero();

	// scalars for lubrication
	double x11a, x12a, y11a, y12a;
	double y11b, y12b;
	double x11c, x12c, y11c, y12c;

	double x11g, x12g, y11g, y12g;
	double y11h, y12h;

	double xm, ym, zm;

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

		x11a = csa4 - 1.23041 + 0.026785714*xdlx + 1.8918*xi;
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
		// TODO
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

		Matrix5d mat;
		for (int i=0; i<5; i++) {
			int i0 = mesid[0][i];
			int i1 = mesid[1][i];

			if (i==0 || i==4) {
				mat(i,0) = m[i0][i0][0][0] - m[i0][i0][2][2] - m[i1][i1][0][0] + m[i1][i1][2][2];
				mat(i,1) = 2.0 * (m[i0][i0][0][1] - m[i1][i1][0][1]);
				mat(i,2) = 2.0 * (m[i0][i0][0][2] - m[i1][i1][0][2]);
				mat(i,3) = 2.0 * (m[i0][i0][1][2] - m[i1][i1][1][2]);
				mat(i,4) = m[i0][i0][1][1] - m[i0][i0][2][2] - m[i1][i1][1][1] + m[i1][i1][2][2];
			} else {
				mat(i,0) = 2.0 * (m[i0][i1][0][0] - m[i0][i1][2][2]);
				mat(i,1) = 4.0 * m[i0][i1][0][1];
				mat(i,2) = 4.0 * m[i0][i1][0][2];
				mat(i,3) = 4.0 * m[i0][i1][1][2];
				mat(i,4) = 2.0 * (m[i0][i1][1][1] - m[i0][i1][2][2]);
			}
		}

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
