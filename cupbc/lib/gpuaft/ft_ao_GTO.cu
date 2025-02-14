/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ByteQC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include <assert.h>
#include "ft_ao.h"
#include "cuda_alloc.cuh"
#include "ft_ao_template.cu"

#undef I

template <bool prim, int I, int J, int Fz>
__device__ int GTO_ft_aopair_drv(cuDoubleComplex *out, const int i_sh,
    const int j_sh, const int2 &dims, cuDoubleComplex fac, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t nGv,
    const size_t iGv, const CINTEnvVars &envs) {
    constexpr int nfi = (I + 1) * (I + 2) / 2;
    constexpr int nfj = (J + 1) * (J + 2) / 2;
    constexpr int nf = nfi * nfj;
    cuDoubleComplex gctr[nf];

    int *bas = envs.bas;
    double *env = envs.env;
    const int i_prim = bas(NPRIM_OF, i_sh);
    const int j_prim = bas(NPRIM_OF, j_sh);
    const double *ai = env + bas(PTR_EXP, i_sh);
    const double *aj = env + bas(PTR_EXP, j_sh);
    const double *ci = env + bas(PTR_COEFF, i_sh);
    const double *cj = env + bas(PTR_COEFF, j_sh);

    int has_value;

    if (prim)
        has_value = GTO_ft_aopair_lazy_ker<I, J, Fz>(gctr, envs, envs.rj, Gv, b,
            gxyz, gs, nGv, iGv, dims, i_prim, j_prim, i_sh, j_sh, ai, aj, ci,
            cj, fac);
    else
        has_value = GTO_ft_aopair_early_ker<I, J, Fz>(gctr, envs, envs.rj, Gv,
            b, gxyz, gs, nGv, iGv, dims, i_prim, j_prim, i_sh, j_sh, ai, aj, ci,
            cj, fac);

    cuDoubleComplex *pout;
    if (has_value) {
#pragma unroll
        for (int j = 0; j < nfj; j++) {
            pout = out + j * dims.x * nGv;
#pragma unroll
            for (int i = 0; i < nfi; i++) {
                out[i * nGv + iGv] = gctr[j * nfi + i];
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < nfj; j++) {
            pout = out + j * dims.x * nGv;
#pragma unroll
            for (int i = 0; i < nfi; i++) {
                pout[i * nGv + iGv] = Z0;
            }
        }
    }
    return has_value;
}

template <bool prim, int I, int J, int Fz>
__global__ void GTO_ft_ovlp_cart(cuDoubleComplex *out, const int2 shlsi,
    const int2 shlsj, const double *Gv, const double *b, const int *gxyz,
    const int3 gs, const int nGv, int *atm, const int natm, int *bas,
    const int nbas, double *env, const int *ao_loc, const double phase) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ish0 = shlsi.x;
    size_t ish1 = shlsi.y;
    size_t jsh0 = shlsj.x;
    size_t jsh1 = shlsj.y;
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;

    if (idx >= nGv * nish * njsh)
        return;
    size_t iGv = idx % nGv;
    idx /= nGv;
    int ish = idx % nish + ish0;
    idx /= nish;
    int jsh = idx + jsh0;

    const cuDoubleComplex fac = make_cuDoubleComplex(cos(phase), sin(phase));
    const int nrow = ao_loc[ish1] - ao_loc[ish0];
    const int ncol = ao_loc[jsh1] - ao_loc[jsh0];
    const size_t off =
        ao_loc[ish] - ao_loc[ish0] + (ao_loc[jsh] - ao_loc[jsh0]) * nrow;
    int2 dims = {nrow, ncol};

    CINTEnvVars envs;
    GTO_ft_init1e_envs<I, J>(&envs, ish, jsh, atm, natm, bas, nbas, env);
    GTO_ft_aopair_drv<prim, I, J, Fz>(
        out + off * nGv, ish, jsh, dims, fac, Gv, b, gxyz, gs, nGv, iGv, envs);
}

template <bool prim, int I, int J, int Fz>
void GTO_ft_fill_s1(cuDoubleComplex *mat, const int comp, const int *shls_slice,
    const int *d_ao_loc, const double phase, const double *d_Gv,
    const double *d_b, const int *d_gxyz, const int *gs, const int nGv,
    int *d_atm, const int natm, int *d_bas, const int nbas, double *d_env) {
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];

    size_t nth = (jsh1 - jsh0);
    nth *= (ish1 - ish0);
    nth *= nGv;
    int2 shlsi = {ish0, ish1};
    int2 shlsj = {jsh0, jsh1};
    int3 gs_ = {gs[0], gs[1], gs[2]};

    GTO_ft_ovlp_cart<prim, I, J, Fz><<<(nth + 255) / 256, 256>>>(mat, shlsi,
        shlsj, d_Gv, d_b, d_gxyz, gs_, nGv, d_atm, natm, d_bas, nbas, d_env,
        d_ao_loc, phase);
}

template <bool prim, int I, int J>
void GTO_ft_fill(const int eval_gz, cuDoubleComplex *mat, const int comp,
    const int *shls_slice, const int *d_ao_loc, const double phase,
    const double *d_Gv, const double *d_b, const int *d_gxyz, const int *gs,
    const int nGv, int *d_atm, const int natm, int *d_bas, const int nbas,
    double *d_env) {
    if (eval_gz != 0) {
        assert(gxyz != NULL);
    }
    switch (eval_gz) {
    case 0:
        GTO_ft_fill_s1<prim, I, J, 0>(mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 1:
        GTO_ft_fill_s1<prim, I, J, 1>(mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 2:
        GTO_ft_fill_s1<prim, I, J, 2>(mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    default:
        printf("Unexpected eval_gz:%d!\n", eval_gz);
        assert(false);
    }
}
template <bool prim>
void GTO_ft_fill(const int eval_gz, cuDoubleComplex *mat, void *d_buf,
    size_t bufsize, const int comp, const int *shls_slice, const int *d_ao_loc,
    const double phase, const double *d_Gv, const double *d_b,
    const int *d_gxyz, const int *gs, const int nGv, int *atm, const int natm,
    int *bas, const int nbas, double *env, const int nenv) {
    DEVICE_INIT_MEMPOOL(int, d_bas, bas, BAS_SLOTS * nbas, d_buf, bufsize);
    DEVICE_INIT_MEMPOOL(int, d_atm, atm, ATM_SLOTS * natm, d_buf, bufsize);
    DEVICE_INIT_MEMPOOL(double, d_env, env, nenv, d_buf, bufsize);

    const int li = bas(ANG_OF, shls_slice[0]);
    const int lj = bas(ANG_OF, shls_slice[2]);
    const int l = (li << 3) + lj;
    switch (l) {
    case 0:
        GTO_ft_fill<prim, 0, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 8:
        GTO_ft_fill<prim, 1, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 16:
        GTO_ft_fill<prim, 2, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 24:
        GTO_ft_fill<prim, 3, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 32:
        GTO_ft_fill<prim, 4, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 40:
        GTO_ft_fill<prim, 5, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 48:
        GTO_ft_fill<prim, 6, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    case 56:
        GTO_ft_fill<prim, 7, 0>(eval_gz, mat, comp, shls_slice, d_ao_loc, phase,
            d_Gv, d_b, d_gxyz, gs, nGv, d_atm, natm, d_bas, nbas, d_env);
        break;
    default:
        printf("Unexpected i:%d and j:%d!\n", li, lj);
        assert(false);
    }
}

/*
 * Fourier transform AO pairs and add to mat (inplace)
 */
extern "C" {
void GTO_ft_fill_drv(const int eval_gz, cuDoubleComplex *mat, void *d_buf,
    size_t bufsize, const int comp, const int *shls_slice, const int *d_ao_loc,
    const double phase, const double *d_Gv, const double *d_b,
    const int *d_gxyz, const int *gs, const int nGv, int *atm, const int natm,
    int *bas, const int nbas, double *env, const int nenv) {
    const int i_prim = bas(NPRIM_OF, shls_slice[0]);
    const int j_prim = bas(NPRIM_OF, shls_slice[2]);
    if (i_prim * j_prim < 3)
        GTO_ft_fill<true>(eval_gz, mat, d_buf, bufsize, comp, shls_slice,
            d_ao_loc, phase, d_Gv, d_b, d_gxyz, gs, nGv, atm, natm, bas, nbas,
            env, nenv);
    else
        GTO_ft_fill<false>(eval_gz, mat, d_buf, bufsize, comp, shls_slice,
            d_ao_loc, phase, d_Gv, d_b, d_gxyz, gs, nGv, atm, natm, bas, nbas,
            env, nenv);
}
}
