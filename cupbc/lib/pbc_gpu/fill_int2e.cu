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
*/

#include <stdio.h>
#include "gint.h"
#include "rys_roots.cu"
#include "fill_int2e.cuh"
#include "cuda_alloc.cuh"
#include "reduce.cu"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define POLYFIT_ORDER 5
#define SQRTPIE4 .8862269254527580136
#define PIE4 .7853981633974483096

#include "write_int3c.cu"

#include "g2e.cu"

template <typename... Args>
unsigned long long GINTfill_prescreen_ij_num(void *gpu_buf, Args... args) {
    unsigned long long *indij = (unsigned long long *)gpu_buf;
    cudaMemset(indij, 0, sizeof(unsigned long long));
    GINTfill_prescreen_ij_num_kernel<<<216, 1024>>>(*indij, args...);
    unsigned long long h_indij;
    cudaMemcpy(
        &h_indij, indij, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return h_indij;
}

template unsigned long long GINTfill_prescreen_ij_num<>(void *gpu_buf,
    int pair_id0, int pair_id1, int *d_bas_pair2bra, int *d_bas_pair2ket,
    int iL0, int iL1, int jL0, int jL1, int *d_refuniqshl_map,
    double *d_uniq_dcut2s, int *d_bas, int nbas, int *d_atm, double *d_env,
    double *d_Ls, int prescreen_mask);

template <typename... Args>
void GINTfill_prescreen_ij(void *gpu_buf, Args... args) {
    unsigned long long *indij = (unsigned long long *)gpu_buf;
    cudaMemset(indij, 0, sizeof(unsigned long long));
    GINTfill_prescreen_ij_kernel<<<BLOCKNUM, 32>>>(*indij, args...);
}

template void GINTfill_prescreen_ij<>(void *gpu_buf, size_t *idxij,
    int pair_id0, int pair_id1, int *d_bas_pair2bra, int *d_bas_pair2ket,
    int iL0, int iL1, int jL0, int jL1, int *d_refuniqshl_map,
    double *d_uniq_dcut2s, int *d_bas, int nbas, int *d_atm, double *d_env,
    double *d_Ls, int prescreen_mask);

template <typename... Args>
void GINTfill_prescreen_k_num(void *gpu_buf, size_t buf_size, size_t numij,
    size_t *h_numk, Args... args) {
    MALLOC_ALIGN_MEMPOOL(size_t, numk, numij, gpu_buf, buf_size);
    GINTfill_prescreen_k_num_kernel<<<BLOCKNUM, 32>>>(numij, numk, args...);
    cudaMemcpy(h_numk, numk, sizeof(size_t) * numij, cudaMemcpyDeviceToHost);
}

template void GINTfill_prescreen_k_num<>(void *gpu_buf, size_t buf_size,
    size_t numij, size_t *h_numk, size_t *idxij, int pair_id0,
    int *d_bas_pair2bra, int *d_bas_pair2ket, int iL0, int iL1, int jL0,
    int jL1, int ksh0, int ksh1, double *d_Ls, int *d_refuniqshl_map,
    int nbasauxuniq, double *d_uniqexp, double dcut_binsize,
    double *d_uniq_Rcut2s, int *d_uniqshlpr_dij_loc, int *d_auxuniqshl_map,
    int *d_bas, int nbas, int *d_atm, double *d_env, int prescreen_mask);

template <typename... Args>
void GINTfill_prescreen_ijk(void *gpu_buf, Args... args) {
    unsigned long long *indijk = (unsigned long long *)gpu_buf;
    cudaMemset(indijk, 0, sizeof(unsigned long long));
    GINTfill_prescreen_ijk_kernel<<<BLOCKNUM, 32>>>(*indijk, args...);
}

template void GINTfill_prescreen_ijk<>(void *gpu_buf, size_t numij, size_t nijk,
    size_t *idxij, int *shlijk, double *rij, int pair_id0, int *d_bas_pair2bra,
    int *d_bas_pair2ket, int iL0, int iL1, int jL0, int jL1, int ksh0, int ksh1,
    double *d_Ls, int *d_refuniqshl_map, int nbasauxuniq, double *d_uniqexp,
    double dcut_binsize, double *d_uniq_Rcut2s, int *d_uniqshlpr_dij_loc,
    int *d_auxuniqshl_map, int *d_bas, int nbas, int *d_atm, double *d_env,
    int prescreen_mask);

template <int I, int K, int L, typename GINTEnvVars, typename size_t,
    typename... Args>
void GINTfill_prescreen_int2e(GINTEnvVars envs, size_t nijk, Args... args) {
    constexpr int NROOTS = (I + K + L) / 2 + 1;
    constexpr int ISSPECIAL = NROOTS <= 4 || I * K * L == 0;
    if constexpr (ISSPECIAL) {
        GINTfill_prescreen_int2e_kernel<I, 0, K, L>
            <<<BLOCKNUM, 32>>>(envs, nijk, args...);
    } else {
        constexpr int TOTAL = 2 * NROOTS + 2;
        constexpr int A = TOTAL / 3;
        constexpr int B = A + ((TOTAL - 3 * A) > 1);
        constexpr int C = A + ((TOTAL - 3 * A) > 0);
        constexpr int UGSIZE = 3 * A * B * C;
        size_t blocks = nijk;
        blocks *= NROOTS * 2;
        blocks *= envs.nprim_ij * envs.nprim_kl;
        blocks = (blocks + 31) / 32;
        GINTfill_prescreen_int2e_kernel<NROOTS, UGSIZE>
            <<<blocks, 32>>>(envs, nijk, args...);
    }
}
template <int I, typename... Args>
void GINTfill_prescreen_int2e(int k, int l, Args... args) {
    int kl = k << 3 | l;
    switch (kl) {
    case (0 << 3 | 0):
        GINTfill_prescreen_int2e<I, 0, 0>(args...);
        break;
    case (1 << 3 | 0):
        GINTfill_prescreen_int2e<I, 1, 0>(args...);
        break;
    case (1 << 3 | 1):
        GINTfill_prescreen_int2e<I, 1, 1>(args...);
        break;
    case (2 << 3 | 0):
        GINTfill_prescreen_int2e<I, 2, 0>(args...);
        break;
    case (2 << 3 | 1):
        GINTfill_prescreen_int2e<I, 2, 1>(args...);
        break;
    case (2 << 3 | 2):
        GINTfill_prescreen_int2e<I, 2, 2>(args...);
        break;
    case (3 << 3 | 0):
        GINTfill_prescreen_int2e<I, 3, 0>(args...);
        break;
    case (3 << 3 | 1):
        GINTfill_prescreen_int2e<I, 3, 1>(args...);
        break;
    case (3 << 3 | 2):
        GINTfill_prescreen_int2e<I, 3, 2>(args...);
        break;
    case (3 << 3 | 3):
        GINTfill_prescreen_int2e<I, 3, 3>(args...);
        break;
    case (4 << 3 | 0):
        GINTfill_prescreen_int2e<I, 4, 0>(args...);
        break;
    case (4 << 3 | 1):
        GINTfill_prescreen_int2e<I, 4, 1>(args...);
        break;
    case (4 << 3 | 2):
        GINTfill_prescreen_int2e<I, 4, 2>(args...);
        break;
    case (4 << 3 | 3):
        GINTfill_prescreen_int2e<I, 4, 3>(args...);
        break;
    case (4 << 3 | 4):
        GINTfill_prescreen_int2e<I, 4, 4>(args...);
        break;
    case (5 << 3 | 0):
        GINTfill_prescreen_int2e<I, 5, 0>(args...);
        break;
    case (5 << 3 | 1):
        GINTfill_prescreen_int2e<I, 5, 1>(args...);
        break;
    case (5 << 3 | 2):
        GINTfill_prescreen_int2e<I, 5, 2>(args...);
        break;
    case (5 << 3 | 3):
        GINTfill_prescreen_int2e<I, 5, 3>(args...);
        break;
    case (5 << 3 | 4):
        GINTfill_prescreen_int2e<I, 5, 4>(args...);
        break;
    case (5 << 3 | 5):
        GINTfill_prescreen_int2e<I, 5, 5>(args...);
        break;
    case (6 << 3 | 0):
        GINTfill_prescreen_int2e<I, 6, 0>(args...);
        break;
    case (6 << 3 | 1):
        GINTfill_prescreen_int2e<I, 6, 1>(args...);
        break;
    case (6 << 3 | 2):
        GINTfill_prescreen_int2e<I, 6, 2>(args...);
        break;
    case (6 << 3 | 3):
        GINTfill_prescreen_int2e<I, 6, 3>(args...);
        break;
    case (6 << 3 | 4):
        GINTfill_prescreen_int2e<I, 6, 4>(args...);
        break;
    case (6 << 3 | 5):
        GINTfill_prescreen_int2e<I, 6, 5>(args...);
        break;
    case (6 << 3 | 6):
        GINTfill_prescreen_int2e<I, 6, 6>(args...);
        break;
    default:
        fprintf(stderr,
            "ERROR: need support higher than i orbital, add more 'case' in "
            "%s:%ld!!! (K = %d, L = %d)\n",
            __FILE__, __LINE__, k, l);
    }
}
template <typename... Args>
void GINTfill_prescreen_int2e(int i, int k, int l, Args... args) {
    switch (i) {
    case 0:
        GINTfill_prescreen_int2e<0>(k, l, args...);
        break;
    case 1:
        GINTfill_prescreen_int2e<1>(k, l, args...);
        break;
    case 2:
        GINTfill_prescreen_int2e<2>(k, l, args...);
        break;
    case 3:
        GINTfill_prescreen_int2e<3>(k, l, args...);
        break;
    case 4:
        GINTfill_prescreen_int2e<4>(k, l, args...);
        break;
    case 5:
        GINTfill_prescreen_int2e<5>(k, l, args...);
        break;
    case 6:
        GINTfill_prescreen_int2e<6>(k, l, args...);
        break;
    default:
        fprintf(stderr,
            "ERROR: need support higher than i orbital, add more "
            "'case' in "
            "%s:%ld!!!(I = %d)\n",
            __FILE__, __LINE__, i);
    }
}

template void GINTfill_prescreen_int2e<>(int i, int k, int l, GINTEnvVars envs,
    size_t nijk, int *shlijk, double *rij, ERITensor eritensor,
    BasisProdCache auxbpcache, int *ao_loc, int primitive_ij, int nksh,
    double diag_fac, int *d_bas, int nbas, int *d_atm, double *d_env,
    int prescreen_mask, int16_t *d_idx4c);
