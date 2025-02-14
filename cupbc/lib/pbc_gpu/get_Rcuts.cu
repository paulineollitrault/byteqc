/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf,
https://github.com/hongzhouye/pyscf/tree/rsdf_direct),
which is licensed under the Apache License 2.0. The original copyright:
    Copyright 2014-2020 The PySCF Developers. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Author: Qiming Sun <osirpt.sun@gmail.com>
    Author: Hong-Zhou Ye <hzyechem@gmail.com>

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "config.h"
#include "get_Rcuts.cuh"

#define MACHEP 1.11022302462515654042E-16
#define big 4.503599627370496e15
#define biginv 2.22044604925031308085e-16

__device__ double gammaincc(double a, double x) {
    double ans, ax, c, yc, r, t, y, z;
    double pk, pkm1, pkm2, qk, qkm1, qkm2;
    if ((x <= 0) || (a <= 0))
        return (1.0);

    ax = a * log(x) - x - lgamma(a);
    ax = exp(ax);
    if ((x < 1.0) || (x < a)) {
        r = a;
        c = 1.0;
        ans = 1.0;

        do {
            r += 1.0;
            c *= x / r;
            ans += c;
        } while (c / ans > MACHEP);

        return (1 - ans * ax / a);
    }

    y = 1.0 - a;
    z = x + y + 1.0;
    c = 0.0;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = x + 1.0;
    qkm1 = z * x;
    ans = pkm1 / qkm1;

    do {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        yc = y * c;
        pk = pkm1 * z - pkm2 * yc;
        qk = qkm1 * z - qkm2 * yc;
        if (qk != 0) {
            r = pk / qk;
            t = fabs((ans - r) / r);
            ans = r;
        } else
            t = 1.0;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if (fabs(pk) > big) {
            pkm2 *= biginv;
            pkm1 *= biginv;
            qkm2 *= biginv;
            qkm1 *= biginv;
        }
    } while (t > MACHEP);

    return (ans * ax);
}
__device__ inline double _gamma(double s) { return tgamma(s); }
__device__ inline double _Gamma(double s, double x) {
    return gammaincc(s, x) * tgamma(s);
}
__device__ inline double _get_multipole(double l, double alp) {
    return 0.5 * M_PI * sqrt(2 * l + 1) / pow(alp, l + 1.5);
}
__device__ inline int fact(int n) {
    if (n == 0)
        return 1;
    for (int i = n - 1; i > 0; i--)
        n *= i;
    return n;
}
__device__ inline int comb(int n, int k) {
    return fact(n) / fact(k) / fact(n - k);
}

__device__ bool feval(double R, double *l_facs, int *ls, int num, double eta2,
    int l3, double comp) {
    double I = 0;
    for (int i = 0; i < num; i++)
        I += l_facs[i] * _Gamma(ls[i] + l3 + 0.5, eta2 * R * R) /
             pow(R, ls[i] + l3 + 1);
    // printf("Feval(%le)=%le\n", R, I);
    return I < comp * min(1.0 / R, 1.0);
}

template <int EST, typename... ARG>
__device__ double _binary_search(
    double xlo, double xhi, double xtol, ARG... args) {
    int count = 0;
    bool y = feval(xlo, args...);
    if (y)
        return xlo;
    bool xhi_rescaled;
    while (1) {
        bool y = feval(xhi, args...);
        if (y) {
            xhi_rescaled = count > 0;
            break;
        }
        xhi *= 1.5;
        assert(count <= 5);
        count++;
    }
    if (xhi_rescaled && xhi / 1.5 > xlo)
        xlo = xhi / 1.5;

    count = 0;
    double xmi;
    while (xhi - xlo > xtol) {
        assert(count <= 20);
        count++;
        xmi = 0.5 * (xhi + xlo);
        y = feval(xmi, args...);
        if (y)
            xhi = xmi;
        else
            xlo = xmi;
    }
    return xhi;
}

__device__ void _get_bincoeff(
    double d, double e1, double e2, int l1, int l2, double *cbins) {
    double d1 = -e2 / (e1 + e2) * d;
    double d2 = e1 / (e1 + e2) * d;
    int lmax = l1 + l2;
    double cl;
    int lpmin, lpmax, l1p, l2p;
    for (int l = 0; l < lmax + 1; l++) {
        cl = 0;
        lpmin = max(-l, l - 2 * l2);
        lpmax = min(l, 2 * l1 - l);
        for (int lp = lpmin; lp < lpmax + 1; lp += 2) {
            l1p = (l + lp) / 2;
            l2p = (l - lp) / 2;
            cl += pow(d1, l1 - l1p) * pow(d2, l2 - l2p) * comb(l1, l1p) *
                  comb(l2, l2p);
        }
        cbins[l] = cl;
    }
}
template <int EST>
__global__ void GINT_get_3c2e_Rcuts_ker(int nijd, int nbasaux, double *Rcuts,
    double *dijs, double *Qijs, int *dijs_ij, int *ls, double *es, double *cs,
    int *lks, double *eks, double *cks, double omega, double precision) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nijd * nbasaux)
        return;
    int ksh = idx % nbasaux;
    int ijsh = idx / nbasaux;
    double d = dijs[ijsh];

    ijsh = dijs_ij[ijsh];
    int ish = floor(sqrt(2 * ijsh + 0.25) - 0.5);
    int jsh = ijsh - ish * (ish + 1) / 2;
    int l1 = ls[ish];
    double e1 = es[ish];
    double c1 = cs[ish];
    int l2 = ls[jsh];
    double e2 = es[jsh];
    double c2 = cs[jsh];

    int l3 = lks[ksh];
    double e3 = eks[ksh];
    double c3 = cks[ksh];

    double eta2 = 1.0 / (1.0 / (e1 + e2) + 1.0 / e2 + 1.0 / pow(omega, 2));
    double prec0 = precision * (eta2 < 1.0 ? eta2 : 1.0);

    double e12 = e1 + e2;
    double l12 = l1 + l2;
    double eta1 = 1.0 / (1.0 / e12 + 1.0 / e3);
    eta2 = 1.0 / (1.0 / eta1 + 1.0 / pow(omega, 2));
    double eta12 = 1.0 / (1.0 / e1 + 1.0 / e2);
    double fac = c1 * c2 * c3 * 0.5 / M_PI;

    // printf("ish:%d jsh:%d ksh:%d l1:%d l2:%d l3:%d e1:%le e2:%le e3:%le
    // c1:%le c2:%le c3:%le prec0:%le eta1:%le eta2:%le eta12:%le fac:%le
    // d:%le\n", ish, jsh, ksh,
    // l1,l2,l3,e1,e2,e3,c1,c2,c3,prec0,eta1,eta2,eta12,fac,d);
    if (EST == 0) {
        double O3 = _get_multipole(l3, e3);
        int ls[10];
        double O12s;
        double l_facs[10];
        int num;
        if (d < 1e-3) {
            num = l12 + 1 - abs(l1 - l2);
            for (int i = 0; i < num; i++) {
                ls[i] = abs(l1 - l2) + i;
                O12s = _get_multipole(ls[i], e12);
                l_facs[i] = O12s * O3 * pow(e12, (0.5 * (ls[i] - l12))) *
                            sqrt(_gamma(l12 > 1.0 ? l12 : 1.0) /
                                 _gamma(ls[i] > 1.0 ? ls[i] : 1.0));
                // printf("l_facs:%le ls:%d O12s:%le O3:%le\n", l_facs[i],
                // ls[i], O12s, O3);
            }
        } else {
            fac *= exp(-eta12 * d * d);
            num = l12 + 1;
            double tmp[10];
            _get_bincoeff(d, e1, e2, l1, l2, tmp);
            for (int i = 0; i < num; i++) {
                ls[i] = i;
                O12s = _get_multipole(ls[i], e12);
                l_facs[i] = O12s * O3 * abs(tmp[i]);
                // printf("l_facs:%le ls:%d O12s:%le O3:%le tmp:%le\n",
                // l_facs[i], ls[i], O12s, O3, tmp[i]);
            }
        }
        Rcuts[idx] = _binary_search<EST>(
            5, 20, 1, l_facs, ls, num, eta2, l3, prec0 / fac);
    } else {
    }
    return;
}

void GINT_get_3c2e_Rcuts(int nijd, int nbasaux, double *Rcuts, double *dijs,
    double *Qijs, int *dijs_ij, int *ls, double *es, double *cs, int *lks,
    double *eks, double *cks, double omega, double precision, int estimator) {
    switch (estimator) {
    case 0: // ME
        GINT_get_3c2e_Rcuts_ker<0><<<(nijd * nbasaux + 255) / 256, 256>>>(nijd,
            nbasaux, Rcuts, dijs, Qijs, dijs_ij, ls, es, cs, lks, eks, cks,
            omega, precision);
        break;
    case 1: // ISF0
        GINT_get_3c2e_Rcuts_ker<1><<<(nijd * nbasaux + 255) / 256, 256>>>(nijd,
            nbasaux, Rcuts, dijs, Qijs, dijs_ij, ls, es, cs, lks, eks, cks,
            omega, precision);
        break;
    case 2: // ISF
        GINT_get_3c2e_Rcuts_ker<2><<<(nijd * nbasaux + 255) / 256, 256>>>(nijd,
            nbasaux, Rcuts, dijs, Qijs, dijs_ij, ls, es, cs, lks, eks, cks,
            omega, precision);
        break;
    case 3: // ISFQ0
        GINT_get_3c2e_Rcuts_ker<3><<<(nijd * nbasaux + 255) / 256, 256>>>(nijd,
            nbasaux, Rcuts, dijs, Qijs, dijs_ij, ls, es, cs, lks, eks, cks,
            omega, precision);
        break;
    case 4: // ISFQL
        GINT_get_3c2e_Rcuts_ker<4><<<(nijd * nbasaux + 255) / 256, 256>>>(nijd,
            nbasaux, Rcuts, dijs, Qijs, dijs_ij, ls, es, cs, lks, eks, cks,
            omega, precision);
        break;
    }
}
