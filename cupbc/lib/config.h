/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf)
and libcint (https://github.com/sunqm/libcint),
which are licensed under the Apache License 2.0. The original copyright:
    Copyright 2014-2020 The PySCF/libcint Developers. All Rights Reserved.

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

*/

// ng[*] from cint.h of libcint
#define IINC 0
#define JINC 1
#define KINC 2
#define LINC 3
#define GSHIFT 4
#define POS_E1 5
#define POS_E2 6
#define SLOT_RYS_ROOTS 6
#define TENSOR 7

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028
#endif
#define SQRTPI 1.7724538509055160272981674833411451

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// global parameters in env
// Overall cutoff for integral prescreening, value needs to be ~ln(threshold)
#define PTR_EXPCUTOFF 0
// R_C of (r-R_C) in dipole, GIAO operators
#define PTR_COMMON_ORIG 1
// R_O in 1/|r-R_O|
#define PTR_RINV_ORIG 4
// ZETA parameter for Gaussian charge distribution (Gaussian nuclear model)
#define PTR_RINV_ZETA 7
// omega parameter in range-separated coulomb operator
// LR interaction: erf(omega*r12)/r12 if omega > 0
// SR interaction: erfc(omega*r12)/r12 if omega < 0
#define PTR_RANGE_OMEGA 8
// Yukawa potential and Slater-type geminal e^{-zeta r}
#define PTR_F12_ZETA 9
// Gaussian type geminal e^{-zeta r^2}
#define PTR_GTG_ZETA 10
#define NGRIDS 11
#define PTR_GRIDS 12
#define PTR_ENV_START 20

// slots of atm
#define CHARGE_OF 0
#define PTR_COORD 1
#define NUC_MOD_OF 2
#define PTR_ZETA 3
#define PTR_FRAC_CHARGE 3
#define RESERVE_ATMLOT1 4
#define RESERVE_ATMLOT2 5
#define ATM_SLOTS 6

// slots of bas
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define KAPPA_OF 4
#define PTR_EXP 5
#define PTR_COEFF 6
#define RESERVE_BASLOT 7
#define BAS_SLOTS 8

#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]
#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]

#define CINTcommon_fac_sp(l)                                                   \
    (l == 0 ? 0.282094791773878143 : (l == 1 ? 0.488602511902919921 : 1))
#define Z1 make_cuDoubleComplex(1.0, 0.0)
#define Z0 make_cuDoubleComplex(0.0, 0.0)
#define ISNZ0(z1) ((z1).x != 0.0 || (z1).y != 0.0)
#define square_dist(r1, r2)                                                    \
    ((r1[0] - r2[0]) * (r1[0] - r2[0]) + (r1[1] - r2[1]) * (r1[1] - r2[1]) +   \
        (r1[2] - r2[2]) * (r1[2] - r2[2]));

#ifdef __CUDACC__
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status = call;                                             \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(status));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#define CUDA_LAST_CHECK()                                                      \
    do {                                                                       \
        cudaError_t status = cudaGetLastError();                               \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(status));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif

// #define NVTX_TRACE_ON

#ifdef NVTX_TRACE_ON
#ifdef __CUDACC__
#include "cuda_profiler_api.h"
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(r) nvtxRangePush(r);
#define NVTX_POP nvtxRangePOP();
#endif
#else
#define NVTX_PUSH(r)
#define NVTX_POP()
#endif
