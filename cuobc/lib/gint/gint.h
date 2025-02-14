/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf)
and GPU4PySCF (https://github.com/bytedance/gpu4pyscf),
which are licensed under the Apache License 2.0.
The original copyright:
    Copyright 2014-2020 The GPU4PySCF/PySCF Developers. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

// slots of atom
#define CHARGE_OF 0
#define PTR_COORD 1
#define NUC_MOD_OF 2
#define PTR_ZETA 3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
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

// boundaries for gint
// up to g functions
#define GPU_LMAX 5
#define GPU_CART_MAX 21
#define THREADS 256

// 1 roots upto (ps|ss)   6
// 2 roots upto (pp|ps)   24
// 3 roots upto (dp|pp)   72
// 4 roots upto (dd|dp)   162
// 5 roots upto (fd|dd)   324
// 6 roots upto (ff|fd)   576
// 7 roots upto (gf|ff)   960
// 8 roots upto (gg|gf)   1500
// 9 roots upto (hg|gg)   2250
// 10 roots upto (hh|hg)  3240
// 11 roots upto (hh|hh)  3888
// UGSIZE = 3*(i+1)*(j+1)*(k+1)*(l+1)
#define UGSIZE1 6
#define UGSIZE2 24
#define UGSIZE3 72
#define UGSIZE4 162
#define UGSIZE5 324
#define UGSIZE6 576
#define UGSIZE7 960
#define UGSIZE8 1500
#define UGSIZE9 2250
#define UGSIZE10 3240
#define UGSIZE11 3888

#ifndef HAVE_DEFINED_GINTENVVAS_H
#define HAVE_DEFINED_GINTENVVAS_H
typedef struct {
    int16_t i_l;
    int16_t j_l;
    int16_t k_l;
    int16_t l_l;
    int16_t nfi;
    int16_t nfj;
    int16_t nfk;
    int16_t nfl;
    int nf;
    int nrys_roots;
    int g_size;
    int g_size_ij;
    int16_t ibase;
    int16_t kbase;
    int16_t ijmin;
    int16_t ijmax;
    int16_t klmin;
    int16_t klmax;
    int nao;
    int stride_ijmax;
    int stride_ijmin;
    int stride_klmax;
    int stride_klmin;
    double fac;

    int nprim_ij;
    int nprim_kl;
} GINTEnvVars;

// ProdContractionType <-> (li, lj, nprimi, nprimj)
typedef struct {
    int l_bra;    // angular of bra
    int l_ket;    // angular of ket
    int nprim_12; // nprimi * nprimj
    int npairs;   // nbas_bra * nbas_ket in this contraction type
} ContractionProdType;

typedef struct {
    int ntasks_ij;
    int ntasks_kl;
    int bas_ij;
    int bas_kl;
    int primitive_ij;
    int primitive_kl;
} BasisProdOffsets;

typedef struct {
    int nbas;    // len(bas_coords)
    int ncptype; // len(cptype)
    ContractionProdType *cptype;
    int *bas_pairs_locs;       // len(bas_pair2bra) = sum(cptype[:].nparis)
    int *primitive_pairs_locs; // len(a12) =
                               // sum(cptype[:].nparis*cptype[:].nprim_12)
    int *bas_pair2shls;
    double *aexyz;

    // Data below held on GPU global memory
    double *bas_coords; // basis coordinates
    int *bas_pair2bra;
    int *bas_pair2ket;
    int *ao_loc;
    double *a12;
    double *e12;
    double *x12;
    double *y12;
    double *z12;
} BasisProdCache;
#endif
