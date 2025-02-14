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

#include "gint.h"

#ifdef __CUDACC__
#include <cuComplex.h>
#define Complex cuDoubleComplex
#else
#include <complex.h>
#define Complex double complex
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_FUNC(NAME)                                                         \
    void NAME(Complex *out, int bvk_nimgs, size_t dij, size_t dm, int nkpts,   \
        Complex *expkL, int *cell_loc_bvk, int cp_ij_id, int aux_id_range,     \
        void *gpu_buf, unsigned long buf_size, int nkpts_ij, int comp,         \
        int *kptij_idx, int *shls_slice, int *ao_loc, int *auxao_loc,          \
        int *atm, int natm, int *bas, int nbas, double *env, int nenv,         \
        int *bas_pair2shls, int *bas_pairs_locs, int ncptype,                  \
        BasisProdCache *auxbpcache, double diag_fac, double *Ls, int nLs,      \
        size_t *strides, int *ao_offsets, double omega, int *refuniqshl_map,   \
        int *auxuniqshl_map, int uniqmaxshell, int nbasauxuniq,                \
        double *uniqexp, double *uniq_dcut2s, double dcut_binsize,             \
        double *uniq_Rcut2s, int *uniqshlpr_dij_loc, int prescreen_mask);

DEF_FUNC(GPBC_nr_fill_ao_ints_bvk_kk)
DEF_FUNC(GPBC_nr_fill_ao_ints_bvk_k)

void GINT_nr3c_g(void *out, void *gpu_buf, unsigned long buf_size, int comp,
    int nimgs, double *Ls, int nLs, int *shls_slice, int *ao_loc,
    int *auxao_loc, int *refuniqshl_map, int *auxuniqshl_map, int uniqmaxshell,
    int nbasauxuniq, double *uniqexp, double *uniq_dcut2s, double dcut_binsize,
    double *uniq_Rcut2s, int *uniqshlpr_dij_loc, int *atm, int natm, int *bas,
    int nbas, double *env, int nenv, int cp_ij_id, int aux_id_range,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
    BasisProdCache *auxbpcache, double diag_fac, int prescreen_mask);
#ifdef __cplusplus
}
#endif
