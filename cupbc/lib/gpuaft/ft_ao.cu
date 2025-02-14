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
#include <cuComplex.h>
#include "cuda_alloc.cuh"
#include "config.h"
#include "ft_ao.h"

int PBCsizeof_env(const int *shls_slice, const int *atm, const int natm,
    const int *bas, const int nbas, const double *env) {
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    int ish, ia, np, nc;
    int nenv = 0;
    for (ish = ish0; ish < ish1; ish++) {
        ia = bas[ATOM_OF + ish * BAS_SLOTS];
        nenv = MAX(atm[PTR_COORD + ia * ATM_SLOTS] + 3, nenv);
        np = bas[NPRIM_OF + ish * BAS_SLOTS];
        nc = bas[NCTR_OF + ish * BAS_SLOTS];
        nenv = MAX(bas[PTR_EXP + ish * BAS_SLOTS] + np, nenv);
        nenv = MAX(bas[PTR_COEFF + ish * BAS_SLOTS] + np * nc, nenv);
    }
    return nenv;
}

template <int Fz>
__host__ __device__ void eval_gz(cuDoubleComplex *out, const double aij,
    const double *rij, const cuDoubleComplex fac, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv);

template <>
__host__ __device__ void eval_gz<0>(cuDoubleComplex *out, const double aij,
    const double *rij, const cuDoubleComplex fac, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv) {
    const double *kx = Gv;
    const double *ky = kx + NGv;
    const double *kz = ky + NGv;
    const double cutoff = EXP_CUTOFF * aij * 4;
    double kR, kk;
    kk = kx[iGv] * kx[iGv] + ky[iGv] * ky[iGv] + kz[iGv] * kz[iGv];
    if (kk < cutoff) {
        kR = kx[iGv] * rij[0] + ky[iGv] * rij[1] + kz[iGv] * rij[2];
        out[0] =
            cuCmul(cuCmul(make_cuDoubleComplex(exp(-.25 * kk / aij), 0.0), fac),
                make_cuDoubleComplex(cos(kR), -sin(kR)));
    } else {
        out[0] = make_cuDoubleComplex(0.0, 0.0);
    }
}

/*
 * Gv = dot(b.T,gxyz) + kpt
 * kk = dot(Gv, Gv)
 * kr = dot(rij, Gv) = dot(rij,b.T, gxyz) + dot(rij,kpt) = dot(br, gxyz) +
 * dot(rij,kpt) out = fac * exp(-.25 * kk / aij) * (cos(kr) - sin(kr) *
 * _Complex_I);
 *
 * b: the first 9 elements are 2\pi*inv(a^T), then 3 elements for k_{ij},
 * followed by 3*NGv floats for Gbase
 */
template <>
__host__ __device__ void eval_gz<1>(cuDoubleComplex *out, const double aij,
    const double *rij, const cuDoubleComplex fac, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv) {
    const int nx = gs.x;
    const int ny = gs.y;
    double br[3]; // dot(rij, b)
    br[0] = rij[0] * b[0];
    br[1] = rij[1] * b[4];
    br[2] = rij[2] * b[8];
    const double *kpt = b + 9;
    double kr[3];
    kr[0] = rij[0] * kpt[0];
    kr[1] = rij[1] * kpt[1];
    kr[2] = rij[2] * kpt[2];
    const double *Gxbase = b + 12;
    const double *Gybase = Gxbase + nx;
    const double *Gzbase = Gybase + ny;

    const double *kx = Gv;
    const double *ky = kx + NGv;
    const double *kz = ky + NGv;
    double kkx, kky, kkz;
    cuDoubleComplex csx, csy, csz;
    const int *gx = gxyz;
    const int *gy = gx + NGv;
    const int *gz = gy + NGv;

    const double cutoff = EXP_CUTOFF * aij * 4;
    int ix, iy, iz;
    double Gr;
    kkx = .25 * kx[iGv] * kx[iGv] / aij;
    kky = .25 * ky[iGv] * ky[iGv] / aij;
    kkz = .25 * kz[iGv] * kz[iGv] / aij;
    if (kkx + kky + kkz < cutoff) {
        ix = gx[iGv];
        Gr = Gxbase[ix] * br[0] + kr[0];
        csx = cuCmul(make_cuDoubleComplex(exp(-kkx), 0.0),
            make_cuDoubleComplex(cos(Gr), -sin(Gr)));
        iy = gy[iGv];
        Gr = Gybase[iy] * br[1] + kr[1];
        csy = cuCmul(make_cuDoubleComplex(exp(-kky), 0.0),
            make_cuDoubleComplex(cos(Gr), -sin(Gr)));
        iz = gz[iGv];
        Gr = Gzbase[iz] * br[2] + kr[2];
        csz = cuCmul(fac, cuCmul(make_cuDoubleComplex(exp(-kkz), 0.0),
                              make_cuDoubleComplex(cos(Gr), -sin(Gr))));
        out[0] = cuCmul(cuCmul(csx, csy), csz);
    } else {
        out[0] = make_cuDoubleComplex(0.0, 0.0);
    }
}

template <>
__host__ __device__ void eval_gz<2>(cuDoubleComplex *out, const double aij,
    const double *rij, const cuDoubleComplex fac, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv) {
    const int nx = gs.x;
    const int ny = gs.y;
    double br[3]; // dot(rij, b)
    br[0] = rij[0] * b[0];
    br[0] += rij[1] * b[1];
    br[0] += rij[2] * b[2];
    br[1] = rij[0] * b[3];
    br[1] += rij[1] * b[4];
    br[1] += rij[2] * b[5];
    br[2] = rij[0] * b[6];
    br[2] += rij[1] * b[7];
    br[2] += rij[2] * b[8];
    const double *kpt = b + 9;
    double kr[3];
    kr[0] = rij[0] * kpt[0];
    kr[1] = rij[1] * kpt[1];
    kr[2] = rij[2] * kpt[2];
    const double *Gxbase = b + 12;
    const double *Gybase = Gxbase + nx;
    const double *Gzbase = Gybase + ny;

    const double *kx = Gv;
    const double *ky = kx + NGv;
    const double *kz = ky + NGv;
    cuDoubleComplex csx, csy, csz;
    const int *gx = gxyz;
    const int *gy = gx + NGv;
    const int *gz = gy + NGv;

    const double cutoff = EXP_CUTOFF * aij * 4;
    int ix, iy, iz;
    double Gr, kk;
    kk = kx[iGv] * kx[iGv] + ky[iGv] * ky[iGv] + kz[iGv] * kz[iGv];
    if (kk < cutoff) {
        ix = gx[iGv];
        iy = gy[iGv];
        iz = gz[iGv];
        Gr = Gxbase[ix] * br[0] + kr[0];
        csx = make_cuDoubleComplex(cos(Gr), -sin(Gr));
        Gr = Gybase[iy] * br[1] + kr[1];
        csy = make_cuDoubleComplex(cos(Gr), -sin(Gr));
        Gr = Gzbase[iz] * br[2] + kr[2];
        csz = cuCmul(fac, make_cuDoubleComplex(cos(Gr), -sin(Gr)));
        out[0] = cuCmul(
            cuCmul(cuCmul(make_cuDoubleComplex(exp(-.25 * kk / aij), 0.0), csx),
                csy),
            csz);
    } else {
        out[0] = make_cuDoubleComplex(0.0, 0.0);
    }
}
