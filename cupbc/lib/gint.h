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
#include "config.h"
#include <stddef.h>

// boundaries for gint
// up to g functions
#define GPU_LMAX        10
#define GPU_CART_MAX    55
#define THREADSX        8
#define THREADSY        8
#define THREADS         (THREADSX * THREADSY)
#define SHARED_MEM_NFIJ_MAX     18
#define BLOCKS_SCREEN   20
#define WARPSIZE 32
#define MIN_BLOCKS 16

// 4 roots upto (fs|dd)   108
// 5 roots upto (fs|ff)   192
// 6 roots upto (gs|gf)   300
// 7 roots upto (hs|gg)   450
// 8 roots upto (hs|hh)   648
// UGSIZE = 3*(i+1)*(j+1)*(k+1)*(l+1)
#define UGSIZE4       108
#define UGSIZE5       192
#define UGSIZE6       300
#define UGSIZE7       450
#define UGSIZE8       648
#define UGSIZE9       3*(6+1)*(6+1)*(5+1)
#define UGSIZE10      3*(7+1)*(6+1)*(6+1)
#define UGSIZE11      3*(7+1)*(7+1)*(7+1)

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
        int nao_ij;
        int nao_kl;
        int nao;
        int stride_ijmax;
        int stride_ijmin;
        int stride_klmax;
        int stride_klmin;
        double fac;
        double omega;
        double expcutoff;
        int nprim_ij;
        int nprim_kl;
        int16_t *idx;
        double *uw;
        double *uw_lr;
} GINTEnvVars;

// ProdContractionType <-> (li, lj, nprimi, nprimj)
typedef struct {
    int l_bra;  // angular of bra
    int l_ket;  // angular of ket
    int nprim_12;  // nprimi * nprimj
    int npairs;  // nbas_bra * nbas_ket in this contraction type
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
    int nbas;  // len(bas_coords)
    int ncptype;  // len(cptype)
    ContractionProdType *cptype;
    int *bas_pairs_locs;  // len(bas_pair2bra) = sum(cptype[:].nparis)
    int *primitive_pairs_locs;  // len(a12) = sum(cptype[:].nparis*cptype[:].nprim_12)
    //int *primitive_offset;
    int *bas_pair2shls;
    double *aexyz;

    // Data below held on GPU global memory
    double *bas_coords;  // basis coordinates
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
