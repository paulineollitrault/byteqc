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
#include <cublas_v2.h>
#include <cuComplex.h>
#include "omp.h"
#include "fill_ints_sr.h"
#include "cuda_alloc.cuh"
#include "linalg.cuh"
#include "util.cuh"
#include "fill_int2e.cuh"
#include <cuda_runtime.h>

#define IMGBLK 80

void prescreen_int2e_k(size_t nij, size_t nijk, size_t *idxij,
    GINTEnvVars &envs, ERITensor &eritensor, BasisProdCache *auxbpcache,
    BasisProdCache *bpcache, int primitive_ij, int *d_bas_pair2bra,
    int *d_bas_pair2ket, int pair_id0, int iL0, int iL1, int jL0, int jL1,
    int ksh0, int ksh1, double diag_fac, double *d_Ls, int *d_refuniqshl_map,
    int nbasauxuniq, double *d_uniqexp, double dcut_binsize,
    double *d_uniq_Rcut2s, int *d_uniqshlpr_dij_loc, int *d_auxuniqshl_map,
    int *d_bas, int nbas, int *d_atm, double *d_env, int prescreen_mask,
    int16_t *idx4c, void *gpu_buf, unsigned long buf_size) {
    MALLOC_ALIGN_MEMPOOL(int, shlijk, 3 * nijk, gpu_buf, buf_size);
    MALLOC_ALIGN_MEMPOOL(double, rij, 6 * nijk, gpu_buf, buf_size);
    GINTfill_prescreen_ijk(gpu_buf, nij, nijk, idxij, shlijk, rij, pair_id0,
        d_bas_pair2bra, d_bas_pair2ket, iL0, iL1, jL0, jL1, ksh0, ksh1, d_Ls,
        d_refuniqshl_map, nbasauxuniq, d_uniqexp, dcut_binsize, d_uniq_Rcut2s,
        d_uniqshlpr_dij_loc, d_auxuniqshl_map, d_bas, nbas, d_atm, d_env,
        prescreen_mask);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    GINTfill_prescreen_int2e(envs.i_l, envs.k_l, envs.l_l, envs, nijk, shlijk,
        rij, eritensor, *auxbpcache, bpcache->ao_loc, primitive_ij, ksh1 - ksh0,
        diag_fac, d_bas, nbas, d_atm, d_env, prescreen_mask, idx4c);
    cudaStreamDestroy(stream);
}
void prescreen_int2e(GINTEnvVars *c_envs, int cp_ij_id, int aux_id_range,
    ERITensor eritensor, BasisProdCache *auxbpcache, BasisProdCache *bpcache,
    int *d_bas_pair2bra, int *d_bas_pair2ket, int iL0, int iL1, int jL0,
    int jL1, double diag_fac, double *d_Ls, int *d_refuniqshl_map,
    int nbasauxuniq, double *d_uniqexp, double *d_uniq_dcut2s,
    double dcut_binsize, double *d_uniq_Rcut2s, int *d_uniqshlpr_dij_loc,
    int *d_auxuniqshl_map, int *d_bas, int nbas, int *d_atm, double *d_env,
    int prescreen_mask, int16_t **c_idx4c, void *gpu_buf,
    unsigned long buf_size) {
    int pair_id0 = bpcache->bas_pairs_locs[cp_ij_id];
    int pair_id1 = bpcache->bas_pairs_locs[cp_ij_id + 1];
    size_t maxij = pair_id1 - pair_id0;
    maxij *= jL1 - jL0;
    maxij *= iL1 - iL0;
    size_t numij = GINTfill_prescreen_ij_num(gpu_buf, pair_id0, pair_id1,
        d_bas_pair2bra, d_bas_pair2ket, iL0, iL1, jL0, jL1, d_refuniqshl_map,
        d_uniq_dcut2s, d_bas, nbas, d_atm, d_env, d_Ls, prescreen_mask);
    MALLOC_ALIGN_MEMPOOL(size_t, idxij, numij, gpu_buf, buf_size);
    GINTfill_prescreen_ij(gpu_buf, idxij, pair_id0, pair_id1, d_bas_pair2bra,
        d_bas_pair2ket, iL0, iL1, jL0, jL1, d_refuniqshl_map, d_uniq_dcut2s,
        d_bas, nbas, d_atm, d_env, d_Ls, prescreen_mask);
    if (numij == 0)
        return;
    size_t *h_numk = (size_t *)malloc(sizeof(size_t) * (numij + 1));
    const size_t maxijk = (buf_size - 4096) / (3 * 4 + 6 * 8);
    size_t nijk, ij0, ij1, nij;
    for (int aux_id = 0; aux_id < aux_id_range; aux_id++) {
        // printf("%d%d%d%d\n", c_envs[aux_id].i_l, c_envs[aux_id].j_l,
        //     c_envs[aux_id].k_l, c_envs[aux_id].l_l);
        int ksh0 = auxbpcache->bas_pairs_locs[aux_id];
        int ksh1 = auxbpcache->bas_pairs_locs[aux_id + 1];
        GINTfill_prescreen_k_num(gpu_buf, buf_size, numij, h_numk, idxij,
            pair_id0, d_bas_pair2bra, d_bas_pair2ket, iL0, iL1, jL0, jL1, ksh0,
            ksh1, d_Ls, d_refuniqshl_map, nbasauxuniq, d_uniqexp, dcut_binsize,
            d_uniq_Rcut2s, d_uniqshlpr_dij_loc, d_auxuniqshl_map, d_bas, nbas,
            d_atm, d_env, prescreen_mask);
        int primitive_ij = auxbpcache->primitive_pairs_locs[aux_id] -
                           ksh0 * c_envs[aux_id].nprim_ij;

        ij0 = 0;
        nijk = 0;
        for (ij1 = 0; ij1 < numij + 1; ij1++) {
            if (ij1 == numij || nijk + h_numk[ij1] > maxijk) {
                nij = ij1 - ij0;
                prescreen_int2e_k(nij, nijk, idxij + ij0, c_envs[aux_id],
                    eritensor, auxbpcache, bpcache, primitive_ij,
                    d_bas_pair2bra, d_bas_pair2ket, pair_id0, iL0, iL1, jL0,
                    jL1, ksh0, ksh1, diag_fac, d_Ls, d_refuniqshl_map,
                    nbasauxuniq, d_uniqexp, dcut_binsize, d_uniq_Rcut2s,
                    d_uniqshlpr_dij_loc, d_auxuniqshl_map, d_bas, nbas, d_atm,
                    d_env, prescreen_mask, c_idx4c[aux_id], gpu_buf, buf_size);
                nijk = h_numk[ij1];
                ij0 = ij1;
            } else {
                nijk += h_numk[ij1];
            }
        }
    }
    free(h_numk);
}

void GPBC_nr_fill_ao_ints_bvk_kk(cuDoubleComplex *out, int bvk_nimgs,
    size_t dij, size_t dm, int nkpts, cuDoubleComplex *d_expkL,
    int *cell_loc_bvk, int cp_ij_id, int aux_id_range, void *gpu_buf,
    unsigned long buf_size, int nkpts_ij, int comp, int *kptij_idx,
    int *shls_slice, int *ao_loc, int *auxao_loc, int *atm, int natm, int *bas,
    int nbas, double *env, int nenv, int *bas_pair2shls, int *bas_pairs_locs,
    int ncptype, BasisProdCache *auxbpcache, double diag_fac, double *Ls,
    int nLs, size_t *strides, int *ao_offsets, double omega,
    int *refuniqshl_map, int *auxuniqshl_map, int uniqmaxshell, int nbasauxuniq,
    double *uniqexp, double *uniq_dcut2s, double dcut_binsize,
    double *uniq_Rcut2s, int *uniqshlpr_dij_loc, int prescreen_mask) {
    assert(comp == 1);
    NVTX_PUSH("SR ker kk");
    cublasHandle_t handle = _handle.get_cublas_handle();

    size_t dijmc = dij * dm;
    size_t dijmk = dij * dm * nkpts;
    cuDoubleComplex *d_bufkk = out;

    MEMSET(d_bufkk, 0.0, 2 * ALIGN_256((size_t)nkpts * dijmk * sizeof(double)));

    MALLOC_ALIGN_MEMPOOL(cuDoubleComplex, d_bufkL,
        MIN(bvk_nimgs, IMGBLK) * dijmk, gpu_buf, buf_size);
    MALLOC_ALIGN_MEMPOOL(double, d_bufL, bvk_nimgs *dijmc, gpu_buf, buf_size);

    int all_n_bas_pairs = bas_pairs_locs[ncptype];
    int *bas_pair2bra = bas_pair2shls;
    DEVICE_INIT_MEMPOOL(int, d_bas_pair2bra, bas_pair2bra, 2L * all_n_bas_pairs,
        gpu_buf, buf_size);
    int *d_bas_pair2ket = d_bas_pair2bra + all_n_bas_pairs;

    int auxnbas = auxbpcache->nbas;
    DEVICE_INIT_MEMPOOL(int, d_bas, bas, (size_t)(nbas + auxnbas) * BAS_SLOTS,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        int, d_atm, atm, 2L * natm * ATM_SLOTS, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_env, env, nenv, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_Ls, Ls, nLs, gpu_buf, buf_size);

    size_t uniq_dcut2s_length = uniqmaxshell * (uniqmaxshell + 3L) / 2 + 1;
    size_t uniq_Rcut2s_length =
        (size_t)(nbasauxuniq)*uniqshlpr_dij_loc[uniq_dcut2s_length];
    DEVICE_INIT_MEMPOOL(
        int, d_refuniqshl_map, refuniqshl_map, nbas, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        int, d_auxuniqshl_map, auxuniqshl_map, auxnbas, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        double, d_uniqexp, uniqexp, (uniqmaxshell + 1L), gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_uniq_dcut2s, uniq_dcut2s, uniq_dcut2s_length,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_uniq_Rcut2s, uniq_Rcut2s, uniq_Rcut2s_length,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(int, d_uniqshlpr_dij_loc, uniqshlpr_dij_loc,
        (uniq_dcut2s_length + 1L), gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(int, d_ao_loc, ao_loc, nbas + 1L, gpu_buf, buf_size);

    // initize bpcache
    BasisProdCache *bpcache_kl =
        (BasisProdCache *)malloc(sizeof(BasisProdCache));
    GINTinit_contraction_types_rsdf(bpcache_kl, bas_pair2shls, bas_pairs_locs,
        ncptype, cp_ij_id, bas, nbas);
    bpcache_kl->ao_loc = d_ao_loc;

    int16_t *c_idx4c[aux_id_range];
    GINTEnvVars c_envs[aux_id_range];
    for (int cp_aux_id = 0; cp_aux_id < aux_id_range; cp_aux_id++) {
        GINTinit_idx4c_envs(auxbpcache, bpcache_kl, cp_aux_id, omega,
            &c_idx4c[cp_aux_id], &c_envs[cp_aux_id], &gpu_buf, &buf_size, NULL);
    }

    ERITensor eritensor;
    eritensor.stride_k = strides[1];
    eritensor.stride_l = strides[2];
    eritensor.ao_offsets_k = ao_offsets[1];
    eritensor.ao_offsets_l = ao_offsets[2];

    NVTX_PUSH("SR ERI kk");
    cuDoubleComplex alpha = Z1;
    int64_t extentA[] = {static_cast<int64_t>(dijmc), bvk_nimgs};
    int64_t extentB[] = {2, bvk_nimgs, nkpts};
    for (int iL0_bvk = 0; iL0_bvk < bvk_nimgs; iL0_bvk += IMGBLK) {
        int iLcount_bvk = MIN(IMGBLK, bvk_nimgs - iL0_bvk);
        for (int iL_bvk = iL0_bvk; iL_bvk < iL0_bvk + iLcount_bvk; iL_bvk++) {
            MEMSET(d_bufL, 0.0, sizeof(double) * dijmc * bvk_nimgs);
            int iL0 = cell_loc_bvk[iL_bvk];
            int iL1 = cell_loc_bvk[iL_bvk + 1];
            for (int jL0_bvk = 0; jL0_bvk < bvk_nimgs; jL0_bvk += IMGBLK) {
                int jLcount_bvk = MIN(IMGBLK, bvk_nimgs - jL0_bvk);
                for (int jL_bvk = jL0_bvk; jL_bvk < jL0_bvk + jLcount_bvk;
                     jL_bvk++) {
                    int jL0 = cell_loc_bvk[jL_bvk];
                    int jL1 = cell_loc_bvk[jL_bvk + 1];
                    eritensor.data = d_bufL + dijmc * jL_bvk;
                    prescreen_int2e(c_envs, cp_ij_id, aux_id_range, eritensor,
                        auxbpcache, bpcache_kl, d_bas_pair2bra, d_bas_pair2ket,
                        iL0, iL1, jL0, jL1, diag_fac, d_Ls, d_refuniqshl_map,
                        nbasauxuniq, d_uniqexp, d_uniq_dcut2s, dcut_binsize,
                        d_uniq_Rcut2s, d_uniqshlpr_dij_loc, d_auxuniqshl_map,
                        d_bas, nbas, d_atm, d_env, prescreen_mask, c_idx4c,
                        gpu_buf, buf_size);
                }
            }
            contraction<double, double>("ij", d_bufL, NULL, extentA, "bjk",
                (double *)d_expkL, NULL, extentB, "bik",
                (double *)(d_bufkL + (iL_bvk - iL0_bvk) * (size_t)dijmk), NULL,
                NULL, 0);
        }
        checkcuBlasError(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_C, dijmk,
            nkpts, iLcount_bvk, &alpha, d_bufkL, CUDA_C_64F, dijmk,
            d_expkL + (size_t)iL0_bvk, CUDA_C_64F, bvk_nimgs, &alpha, d_bufkk,
            CUDA_C_64F, dijmk, CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    NVTX_POP();

    GINTdel_basis_prod_rsdf(&bpcache_kl);

    cudaDeviceSynchronize();
    NVTX_POP();
}

__global__ void Phase_factor_contraction(int nkpts, int bvk_nimgs, int iL_bvk,
    cuDoubleComplex *d_bufexp, cuDoubleComplex *d_expkL) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nkpts * bvk_nimgs)
        return;

    int jL_bvk = idx % bvk_nimgs;
    int k = idx / bvk_nimgs;

    int kiLj = k * bvk_nimgs + jL_bvk;
    int kiLi = k * bvk_nimgs + iL_bvk;

    d_bufexp[kiLj] = cuCmul(d_expkL[kiLj], cuConj(d_expkL[kiLi]));
}
void GPBC_nr_fill_ao_ints_bvk_k(cuDoubleComplex *out, int bvk_nimgs, size_t dij,
    size_t dm, int nkpts, cuDoubleComplex *d_expkL, int *cell_loc_bvk,
    int cp_ij_id, int aux_id_range, void *gpu_buf, unsigned long buf_size,
    int nkpts_ij, int comp, int *kptij_idx, int *shls_slice, int *ao_loc,
    int *auxao_loc, int *atm, int natm, int *bas, int nbas, double *env,
    int nenv, int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
    BasisProdCache *auxbpcache, double diag_fac, double *Ls, int nLs,
    size_t *strides, int *ao_offsets, double omega, int *refuniqshl_map,
    int *auxuniqshl_map, int uniqmaxshell, int nbasauxuniq, double *uniqexp,
    double *uniq_dcut2s, double dcut_binsize, double *uniq_Rcut2s,
    int *uniqshlpr_dij_loc, int prescreen_mask) {
    assert(comp == 1);
    NVTX_PUSH("SR ker k");
    cublasHandle_t handle = _handle.get_cublas_handle();

    size_t dijmc = dij * dm;
    size_t dijmk = dijmc * nkpts;
    cuDoubleComplex *d_bufk = out;

    MEMSET(d_bufk, 0.0, 2 * ALIGN_256(dijmk * sizeof(double)));
    MALLOC_ALIGN_MEMPOOL(
        cuDoubleComplex, d_bufexp, bvk_nimgs * nkpts, gpu_buf, buf_size);
    MALLOC_ALIGN_MEMPOOL(double, d_bufL, bvk_nimgs *dijmc, gpu_buf, buf_size);

    int all_n_bas_pairs = bas_pairs_locs[ncptype];
    int *bas_pair2bra = bas_pair2shls;
    DEVICE_INIT_MEMPOOL(int, d_bas_pair2bra, bas_pair2bra, 2L * all_n_bas_pairs,
        gpu_buf, buf_size);
    int *d_bas_pair2ket = d_bas_pair2bra + all_n_bas_pairs;

    int auxnbas = auxbpcache->nbas;
    DEVICE_INIT_MEMPOOL(int, d_bas, bas, (size_t)(nbas + auxnbas) * BAS_SLOTS,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        int, d_atm, atm, 2L * natm * ATM_SLOTS, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_env, env, nenv, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_Ls, Ls, nLs, gpu_buf, buf_size);

    size_t uniq_dcut2s_length = uniqmaxshell * (uniqmaxshell + 3L) / 2 + 1;
    size_t uniq_Rcut2s_length =
        nbasauxuniq * uniqshlpr_dij_loc[uniq_dcut2s_length];
    DEVICE_INIT_MEMPOOL(
        int, d_refuniqshl_map, refuniqshl_map, nbas, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        int, d_auxuniqshl_map, auxuniqshl_map, auxnbas, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        double, d_uniqexp, uniqexp, (uniqmaxshell + 1L), gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_uniq_dcut2s, uniq_dcut2s, uniq_dcut2s_length,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_uniq_Rcut2s, uniq_Rcut2s, uniq_Rcut2s_length,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(int, d_uniqshlpr_dij_loc, uniqshlpr_dij_loc,
        (uniq_dcut2s_length + 1L), gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(int, d_ao_loc, ao_loc, nbas + 1, gpu_buf, buf_size);

    // initize bpcache
    BasisProdCache *bpcache_kl =
        (BasisProdCache *)malloc(sizeof(BasisProdCache));
    GINTinit_contraction_types_rsdf(bpcache_kl, bas_pair2shls, bas_pairs_locs,
        ncptype, cp_ij_id, bas, nbas);
    bpcache_kl->ao_loc = d_ao_loc;

    int16_t *c_idx4c[aux_id_range];
    GINTEnvVars c_envs[aux_id_range];
    for (int cp_aux_id = 0; cp_aux_id < aux_id_range; cp_aux_id++) {
        GINTinit_idx4c_envs(auxbpcache, bpcache_kl, cp_aux_id, omega,
            &c_idx4c[cp_aux_id], &c_envs[cp_aux_id], &gpu_buf, &buf_size, NULL);
    }

    ERITensor eritensor;
    eritensor.stride_k = strides[1];
    eritensor.stride_l = strides[2];
    eritensor.ao_offsets_k = ao_offsets[1];
    eritensor.ao_offsets_l = ao_offsets[2];

    NVTX_PUSH("SR ERI k");
    int64_t extentA[] = {static_cast<int64_t>(dijmc), bvk_nimgs};
    int64_t extentB[] = {2, bvk_nimgs, nkpts};
    for (int iL_bvk = 0; iL_bvk < bvk_nimgs; iL_bvk++) {
        MEMSET(d_bufL, 0.0, sizeof(double) * dijmc * bvk_nimgs);
        int iL0 = cell_loc_bvk[iL_bvk];
        int iL1 = cell_loc_bvk[iL_bvk + 1];
        for (int jL_bvk = 0; jL_bvk < bvk_nimgs; jL_bvk++) {
            int jL0 = cell_loc_bvk[jL_bvk];
            int jL1 = cell_loc_bvk[jL_bvk + 1];
            eritensor.data = d_bufL + dijmc * jL_bvk;
            prescreen_int2e(c_envs, cp_ij_id, aux_id_range, eritensor,
                auxbpcache, bpcache_kl, d_bas_pair2bra, d_bas_pair2ket, iL0,
                iL1, jL0, jL1, diag_fac, d_Ls, d_refuniqshl_map, nbasauxuniq,
                d_uniqexp, d_uniq_dcut2s, dcut_binsize, d_uniq_Rcut2s,
                d_uniqshlpr_dij_loc, d_auxuniqshl_map, d_bas, nbas, d_atm,
                d_env, prescreen_mask, c_idx4c, gpu_buf, buf_size);
        }
        Phase_factor_contraction<<<(nkpts * bvk_nimgs + THREADS - 1) / THREADS,
            THREADS>>>(nkpts, bvk_nimgs, iL_bvk, d_bufexp, d_expkL);
        contraction<double, double>("ij", d_bufL, NULL, extentA, "bjk",
            (double *)d_bufexp, NULL, extentB, "bik", (double *)d_bufk, NULL,
            NULL, 0, 1.0, 1.0);
    }

    NVTX_POP();

    GINTdel_basis_prod_rsdf(&bpcache_kl);

    cudaDeviceSynchronize();
    NVTX_POP();
}

void GINT_nr3c_g(void *out, void *gpu_buf, unsigned long buf_size, int comp,
    int nimgs, double *Ls, int nLs, int *shls_slice, int *ao_loc,
    int *auxao_loc, int *refuniqshl_map, int *auxuniqshl_map, int uniqmaxshell,
    int nbasauxuniq, double *uniqexp, double *uniq_dcut2s, double dcut_binsize,
    double *uniq_Rcut2s, int *uniqshlpr_dij_loc, int *atm, int natm, int *bas,
    int nbas, double *env, int nenv, int cp_ij_id, int aux_id_range,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
    BasisProdCache *auxbpcache, double diag_fac, int prescreen_mask) {
    assert(comp == 1);
    NVTX_PUSH("SR ker g");
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];
    const int di = ao_loc[shls_slice[1]] - ao_loc[ish0];
    const int dj = ao_loc[shls_slice[3]] - ao_loc[jsh0];
    const size_t dij = di * dj;
    const size_t dm = auxao_loc[shls_slice[5]] - auxao_loc[shls_slice[4]];
    size_t dijmc = dij * dm;
    const double omega = fabs(env[PTR_RANGE_OMEGA]);
    cublasHandle_t handle = _handle.get_cublas_handle();

    double *d_bufL = (double *)out;
    MEMSET(d_bufL, 0.0, ALIGN_256(dijmc * sizeof(double)));

    int all_n_bas_pairs = bas_pairs_locs[ncptype];
    int *bas_pair2bra = bas_pair2shls;
    DEVICE_INIT_MEMPOOL(int, d_bas_pair2bra, bas_pair2bra, 2L * all_n_bas_pairs,
        gpu_buf, buf_size);
    int *d_bas_pair2ket = d_bas_pair2bra + all_n_bas_pairs;

    int auxnbas = auxbpcache->nbas;
    DEVICE_INIT_MEMPOOL(int, d_bas, bas, (size_t)(nbas + auxnbas) * BAS_SLOTS,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        int, d_atm, atm, 2L * natm * ATM_SLOTS, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_env, env, nenv, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_Ls, Ls, nLs, gpu_buf, buf_size);

    size_t uniq_dcut2s_length = uniqmaxshell * (uniqmaxshell + 3L) / 2 + 1;
    size_t uniq_Rcut2s_length =
        (size_t)(nbasauxuniq)*uniqshlpr_dij_loc[uniq_dcut2s_length];
    DEVICE_INIT_MEMPOOL(
        int, d_refuniqshl_map, refuniqshl_map, nbas, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        int, d_auxuniqshl_map, auxuniqshl_map, auxnbas, gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(
        double, d_uniqexp, uniqexp, (uniqmaxshell + 1L), gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_uniq_dcut2s, uniq_dcut2s, uniq_dcut2s_length,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(double, d_uniq_Rcut2s, uniq_Rcut2s, uniq_Rcut2s_length,
        gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(int, d_uniqshlpr_dij_loc, uniqshlpr_dij_loc,
        (uniq_dcut2s_length + 1L), gpu_buf, buf_size);
    DEVICE_INIT_MEMPOOL(int, d_ao_loc, ao_loc, nbas + 1, gpu_buf, buf_size);

    BasisProdCache *bpcache_kl =
        (BasisProdCache *)malloc(sizeof(BasisProdCache));
    GINTinit_contraction_types_rsdf(bpcache_kl, bas_pair2shls, bas_pairs_locs,
        ncptype, cp_ij_id, bas, nbas);
    bpcache_kl->ao_loc = d_ao_loc;

    int16_t *c_idx4c[aux_id_range];
    GINTEnvVars c_envs[aux_id_range];
    for (int cp_aux_id = 0; cp_aux_id < aux_id_range; cp_aux_id++) {
        GINTinit_idx4c_envs(auxbpcache, bpcache_kl, cp_aux_id, omega,
            &c_idx4c[cp_aux_id], &c_envs[cp_aux_id], &gpu_buf, &buf_size, NULL);
    }

    ERITensor eritensor;
    eritensor.stride_k = dj * dm;
    eritensor.stride_l = dm;
    eritensor.ao_offsets_k = ao_loc[ish0];
    eritensor.ao_offsets_l = ao_loc[jsh0];
    eritensor.data = d_bufL;
    NVTX_PUSH("SR ERI g")
    prescreen_int2e(c_envs, cp_ij_id, aux_id_range, eritensor, auxbpcache,
        bpcache_kl, d_bas_pair2bra, d_bas_pair2ket, 0, nimgs, 0, nimgs,
        diag_fac, d_Ls, d_refuniqshl_map, nbasauxuniq, d_uniqexp, d_uniq_dcut2s,
        dcut_binsize, d_uniq_Rcut2s, d_uniqshlpr_dij_loc, d_auxuniqshl_map,
        d_bas, nbas, d_atm, d_env, prescreen_mask, c_idx4c, gpu_buf, buf_size);
    NVTX_POP();

    GINTdel_basis_prod_rsdf(&bpcache_kl);

    cudaDeviceSynchronize();
    NVTX_POP();
}
