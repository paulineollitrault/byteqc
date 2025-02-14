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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"

typedef struct {
    size_t stride_i;
    size_t stride_j;
    size_t stride_k;
    size_t stride_l;
    int ao_offsets_i;
    int ao_offsets_j;
    int ao_offsets_k;
    int ao_offsets_l;
    int nao;
    double *data;
} ERITensor;

#include "fill_ints.cu"
#include "g2e.cu"
#include "rys_roots.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"

__host__ static int GINTfill_int2e_tasks(ERITensor &eri,
    BasisProdOffsets &offsets, GINTEnvVars &envs, BasisProdCache &bpcache,
    cudaStream_t stream) {
    int nrys_roots = envs.nrys_roots;
    size_t ntasks = offsets.ntasks_ij;
    ntasks *= offsets.ntasks_kl;
    int nprim = envs.nprim_ij * envs.nprim_kl;
    int type_ijkl;

    dim3 threads(THREADS);
    ntasks *= nprim;
    dim3 blocks((ntasks + THREADS - 1) / THREADS);
    ntasks *= nrys_roots;
    dim3 blocks2((ntasks + THREADS - 1) / THREADS);
    switch (nrys_roots) {
    case 1:
        type_ijkl =
            (envs.i_l << 3) | (envs.j_l << 2) | (envs.k_l << 1) | envs.l_l;
        switch (type_ijkl) {
        case 0b0000:
            nprim <= 1 ? GINTfill_int2e_kernel0000<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0000<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case 0b0010:
            nprim <= 1 ? GINTfill_int2e_kernel0010<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0010<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case 0b1000:
            nprim <= 1 ? GINTfill_int2e_kernel1000<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1000<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        default:
            fprintf(stderr, "nroots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2:
        type_ijkl =
            (envs.i_l << 6) | (envs.j_l << 4) | (envs.k_l << 2) | envs.l_l;
        switch (type_ijkl) {
        case (0 << 6) | (0 << 4) | (1 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel0011<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0011<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (0 << 6) | (0 << 4) | (2 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel0020<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0020<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (0 << 6) | (0 << 4) | (2 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel0021<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0021<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (0 << 6) | (0 << 4) | (3 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel0030<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0030<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1010<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1010<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (1 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel1011<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1011<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (2 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1020<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1020<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1100<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1100<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1110<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1110<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2000<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2000<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2010<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2010<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2100<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2100<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel3000<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3000<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        default:
            fprintf(stderr, "roots=2 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 3:
        type_ijkl =
            (envs.i_l << 6) | (envs.j_l << 4) | (envs.k_l << 2) | envs.l_l;
        switch (type_ijkl) {
        case (0 << 6) | (0 << 4) | (2 << 2) | 2:
            nprim <= 1 ? GINTfill_int2e_kernel0022<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0022<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (0 << 6) | (0 << 4) | (3 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel0031<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0031<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (0 << 6) | (0 << 4) | (3 << 2) | 2:
            nprim <= 1 ? GINTfill_int2e_kernel0032<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel0032<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (2 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel1021<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1021<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (2 << 2) | 2:
            nprim <= 1 ? GINTfill_int2e_kernel1022<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1022<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (3 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1030<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1030<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (3 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel1031<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1031<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (1 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel1111<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1111<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (2 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1120<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1120<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (2 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel1121<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1121<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (3 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel1130<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel1130<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (1 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel2011<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2011<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (2 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2020<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2020<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (2 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel2021<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2021<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (3 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2030<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2030<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2110<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2110<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (1 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel2111<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2111<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (2 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2120<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2120<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (2 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2200<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2200<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (2 << 6) | (2 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel2210<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel2210<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel3010<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3010<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (1 << 2) | 1:
            nprim <= 1 ? GINTfill_int2e_kernel3011<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3011<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (2 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel3020<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3020<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (1 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel3100<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3100<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (1 << 4) | (1 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel3110<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3110<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        case (3 << 6) | (2 << 4) | (0 << 2) | 0:
            nprim <= 1 ? GINTfill_int2e_kernel3200<false>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache)
                       : GINTfill_int2e_kernel3200<true>
                <<<blocks, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        default:
            GINTfill_int2e_kernel<3, UGSIZE3>
                <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
            break;
        }
        break;
    case 4:
        GINTfill_int2e_kernel<4, UGSIZE4>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 5:
        GINTfill_int2e_kernel<5, UGSIZE5>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 6:
        GINTfill_int2e_kernel<6, UGSIZE6>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 7:
        GINTfill_int2e_kernel<7, UGSIZE7>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 8:
        GINTfill_int2e_kernel<8, UGSIZE8>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 9:
        GINTfill_int2e_kernel<9, UGSIZE9>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 10:
        GINTfill_int2e_kernel<10, UGSIZE10>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    case 11:
        GINTfill_int2e_kernel<11, UGSIZE11>
            <<<blocks2, threads, 0, stream>>>(eri, offsets, envs, bpcache);
        break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int2e_kernel: %s\n",
            cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
__host__ void GINTdel_basis_prod(BasisProdCache **pbp) {
    BasisProdCache *bpcache = *pbp;
    if (bpcache == NULL) {
        return;
    }

    if (bpcache->cptype != NULL) {
        free(bpcache->cptype);
        free(bpcache->primitive_pairs_locs);
    }
    if (bpcache->aexyz != NULL) {
        free(bpcache->aexyz);
    }

    free(bpcache);
    *pbp = NULL;
}

void GINTinit_basis_prod(BasisProdCache **pbp, double diag_fac, int *ao_loc,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype, int *atm, int natm,
    int *bas, int nbas, double *env) {
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp = bpcache;

    GINTinit_contraction_types(bpcache, bas_pair2shls, bas_pairs_locs, ncptype,
        atm, natm, bas, nbas, env);
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 5);
    GINTinit_aexyz(aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env);
    bpcache->aexyz = aexyz;
    bpcache->bas_pair2shls = bas_pair2shls;

    // initialize ao_loc on GPU
    DEVICE_INIT(int, d_ao_loc, ao_loc, nbas + 1);
    bpcache->ao_loc = d_ao_loc;

    // initialize basis coordinates on GPU memory
    bpcache->nbas = nbas;
    double *bas_coords = (double *)malloc(sizeof(double) * nbas * 3);
    GINTsort_bas_coordinates(bas_coords, atm, natm, bas, nbas, env);
    DEVICE_INIT(double, d_bas_coords, bas_coords, nbas * 3);
    bpcache->bas_coords = d_bas_coords;
    free(bas_coords);

    // initialize pair data on GPU memory
    DEVICE_INIT(double, d_aexyz, aexyz, n_primitive_pairs * 5);
    DEVICE_INIT(int, d_bas_pair2shls, bas_pair2shls, n_bas_pairs * 2);
    bpcache->a12 = d_aexyz;
    bpcache->e12 = d_aexyz + n_primitive_pairs * 1;
    bpcache->x12 = d_aexyz + n_primitive_pairs * 2;
    bpcache->y12 = d_aexyz + n_primitive_pairs * 3;
    bpcache->z12 = d_aexyz + n_primitive_pairs * 4;
    bpcache->bas_pair2bra = d_bas_pair2shls;
    bpcache->bas_pair2ket = d_bas_pair2shls + n_bas_pairs;
}

void GINTinit_basis_prod_cpu(BasisProdCache **pbp, double diag_fac,
    double *bas_coords, int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
    int *atm, int natm, int *bas, int nbas, double *env) {
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp = bpcache;

    GINTinit_contraction_types(bpcache, bas_pair2shls, bas_pairs_locs, ncptype,
        atm, natm, bas, nbas, env);
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 5);
    GINTinit_aexyz(aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env);
    bpcache->aexyz = aexyz;
    bpcache->bas_pair2shls = bas_pair2shls;

    // initialize basis coordinates on GPU memory
    bpcache->nbas = nbas;
    GINTsort_bas_coordinates(bas_coords, atm, natm, bas, nbas, env);
}

void GINTinit_basis_prod_gpu(BasisProdCache **pbpin, BasisProdCache **pbp,
    int *ao_loc, double *bas_coords, double *aexyz, int *bas_pair2shls) {
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memcpy(bpcache, *pbpin, sizeof(BasisProdCache));
    *pbp = bpcache;

    bpcache->ao_loc = ao_loc;
    bpcache->bas_coords = bas_coords;

    int ncptype = bpcache->ncptype;
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    bpcache->a12 = aexyz;
    bpcache->e12 = aexyz + n_primitive_pairs * 1;
    bpcache->x12 = aexyz + n_primitive_pairs * 2;
    bpcache->y12 = aexyz + n_primitive_pairs * 3;
    bpcache->z12 = aexyz + n_primitive_pairs * 4;
    bpcache->bas_pair2bra = bas_pair2shls;
    bpcache->bas_pair2ket = bas_pair2shls + n_bas_pairs;
}

int GINTfill_int2e(BasisProdCache *bpcache, double *eri, int nao, int *strides,
    int *ao_offsets, int *bins_locs_ij, int *bins_locs_kl, int nbins,
    int cp_ij_id, int cp_kl_id, cudaStream_t stream) {
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    GINTinit_EnvVars(&envs, cp_ij, cp_kl);

    // Data and buffers to be allocated on-device. Allocate them here to
    // reduce the calls to malloc
    int kl_bin, ij_bin1;
    ERITensor eritensor;
    eritensor.stride_i = strides[0];
    eritensor.stride_j = strides[1];
    eritensor.stride_k = strides[2];
    eritensor.stride_l = strides[3];
    eritensor.ao_offsets_i = ao_offsets[0];
    eritensor.ao_offsets_j = ao_offsets[1];
    eritensor.ao_offsets_k = ao_offsets[2];
    eritensor.ao_offsets_l = ao_offsets[3];
    eritensor.nao = nao;
    eritensor.data = eri;

    BasisProdOffsets offsets;
    int *bas_pairs_locs = bpcache->bas_pairs_locs;
    int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (kl_bin = 0; kl_bin < nbins; kl_bin++) {
        int bas_kl0 = bins_locs_kl[kl_bin];
        int bas_kl1 = bins_locs_kl[kl_bin + 1];
        int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }
        // ij_bin + kl_bin < nbins <~> e_ij*e_kl < cutoff
        ij_bin1 = nbins - kl_bin;
        int bas_ij0 = bins_locs_ij[0];
        int bas_ij1 = bins_locs_ij[ij_bin1];
        int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ntasks_kl;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = bas_pairs_locs[cp_kl_id] + bas_kl0;
        offsets.primitive_ij =
            primitive_pairs_locs[cp_ij_id] + bas_ij0 * envs.nprim_ij;
        offsets.primitive_kl =
            primitive_pairs_locs[cp_kl_id] + bas_kl0 * envs.nprim_kl;

        int err =
            GINTfill_int2e_tasks(eritensor, offsets, envs, *bpcache, stream);
        if (err != 0) {
            return err;
        }
    }
    return 0;
}
}
