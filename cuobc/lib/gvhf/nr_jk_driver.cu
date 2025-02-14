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
#include <stdlib.h>
#include <cuda_runtime.h>

#include "gint/gint.h"
#include "gint/config.h"
#include "gint/g2e.h"

typedef struct {
    int nao;
    int n_dm;
    double *dm;
    double *vj;
    double *vk;
} JKMatrix;

#include "g2e.cu"
#include "contract_jk.cu"
#include "gint/rys_roots.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"

__host__ static int GINTrun_tasks_jk(JKMatrix &jk, BasisProdOffsets &offsets,
    GINTEnvVars &envs, BasisProdCache &bpcache, cudaStream_t stream) {
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
        if (envs.nf == 1) {
            GINTint2e_jk_kernel0000<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
        } else {
            GINTint2e_jk_kernel1000<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
        }
        break;
    case 2:
        type_ijkl =
            (envs.i_l << 6) | (envs.j_l << 4) | (envs.k_l << 2) | envs.l_l;
        switch (type_ijkl) {
        case (1 << 6) | (0 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel1010<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (1 << 6) | (0 << 4) | (1 << 2) | 1:
            GINTint2e_jk_kernel1011<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel1100<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (1 << 6) | (1 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel1110<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel2000<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel2010<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel2100<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel3000<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        default:
            GINTint2e_jk_kernel<2, UGSIZE2>
                <<<blocks, threads, 0, stream>>>(jk, offsets, envs, bpcache);
            break;
        }
        break;
    case 3:
        type_ijkl =
            (envs.i_l << 6) | (envs.j_l << 4) | (envs.k_l << 2) | envs.l_l;
        switch (type_ijkl) {
        case (1 << 6) | (1 << 4) | (1 << 2) | 1:
            GINTint2e_jk_kernel1111<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (1 << 2) | 1:
            GINTint2e_jk_kernel2011<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (2 << 2) | 0:
            GINTint2e_jk_kernel2020<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (0 << 4) | (2 << 2) | 1:
            GINTint2e_jk_kernel2021<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel2110<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (1 << 2) | 1:
            GINTint2e_jk_kernel2111<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (1 << 4) | (2 << 2) | 0:
            GINTint2e_jk_kernel2120<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (2 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel2200<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (2 << 6) | (2 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel2210<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel3010<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (1 << 2) | 1:
            GINTint2e_jk_kernel3011<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (0 << 4) | (2 << 2) | 0:
            GINTint2e_jk_kernel3020<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (1 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel3100<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (1 << 4) | (1 << 2) | 0:
            GINTint2e_jk_kernel3110<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        case (3 << 6) | (2 << 4) | (0 << 2) | 0:
            GINTint2e_jk_kernel3200<<<blocks, threads, 0, stream>>>(
                jk, offsets, envs, bpcache);
            break;
        default:
            GINTint2e_jk_kernel<3, UGSIZE3>
                <<<blocks, threads, 0, stream>>>(jk, offsets, envs, bpcache);
            break;
        }
        break;
    case 4:
        GINTint2e_jk_kernel<4, UGSIZE4>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 5:
        GINTint2e_jk_kernel<5, UGSIZE5>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 6:
        GINTint2e_jk_kernel<6, UGSIZE6>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 7:
        GINTint2e_jk_kernel<7, UGSIZE7>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 8:
        GINTint2e_jk_kernel<8, UGSIZE8>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 9:
        GINTint2e_jk_kernel<9, UGSIZE9>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 10:
        GINTint2e_jk_kernel<10, UGSIZE10>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    case 11:
        GINTint2e_jk_kernel<11, UGSIZE11>
            <<<blocks2, threads, 0, stream>>>(jk, offsets, envs, bpcache);
        break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTint2e_jk_kernel: %s\n",
            cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
__host__ int GINTbuild_jk(BasisProdCache *bpcache, double *vj, double *vk,
    double *dm, int nao, int n_dm, int *bins_locs_ij, int *bins_locs_kl,
    int nbins, int cp_ij_id, int cp_kl_id, cudaStream_t stream) {
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    GINTinit_EnvVars(&envs, cp_ij, cp_kl);
    if (envs.nrys_roots >= 12) {
        return 2;
    }

    envs.nao = nao;

    JKMatrix jk;
    jk.n_dm = n_dm;
    jk.nao = nao;
    jk.dm = dm;
    jk.vj = vj;
    jk.vk = vk;

    BasisProdOffsets offsets;
    int *bas_pairs_locs = bpcache->bas_pairs_locs;
    int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
    int kl_bin, ij_bin1;
    for (kl_bin = 0; kl_bin < nbins; kl_bin++) {
        int bas_kl0 = bins_locs_kl[kl_bin];
        int bas_kl1 = bins_locs_kl[kl_bin + 1];
        int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }
        // ij_bin + kl_bin < nbins <~> e_ij*e_kl > cutoff
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

        int err = GINTrun_tasks_jk(jk, offsets, envs, *bpcache, stream);
        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
