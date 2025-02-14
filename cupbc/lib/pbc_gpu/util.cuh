/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf),
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

*/

void GINTinit_contraction_types_rsdf(BasisProdCache *bpcache,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype, int cp_ij_id,
    int *bas, int nbas);
void GINTinit_bpcache_kl(BasisProdCache *bpcache, double *d_aexyz,
    double *d_bas_coords, int *d_prescreen_bas_pair2bra, const int cp_ij_id);
void GINTinit_basis_prod_rsdf(BasisProdCache *bpcache, double diag_fac,
    double *d_bas_coords, int *bas_pair2shls, int *bas_pairs_locs,
    int *d_auxbas_pairs_locs, int aux_id_range, int ncptype, int cp_ij_id,
    int *prescreen_n_bas_pairs, int auxnbas, double *d_auxbas_coords,
    int *d_prescreen_bas_pair2bra, int *d_prescreen_bas_aux,
    int *d_prescreen_n_bas_pairs, int *d_refuniqshl_map, int *d_auxuniqshl_map,
    int nbasauxuniq, double *d_uniqexp, double *d_uniq_dcut2s,
    double dcut_binsize, double *d_uniq_Rcut2s, int *d_uniqshlpr_dij_loc,
    double *d_aexyz, int *d_bas, int *d_atm, double *d_env, double *d_Ls,
    int *d_bas_pair2bra, int *d_bas_pair2ket, int nbas, int jL, int &iL,
    int iL1, int *h_iL, int stream_id, int prescreen_mask, cudaStream_t stream = NULL);
void GINTinit_idx4c_envs(const BasisProdCache *bpcache_ij,
    const BasisProdCache *bpcache_kl, const int cp_aux_id, const double omega,
    int16_t **c_idx4c, GINTEnvVars *c_envs, void **ptr_buf,
    size_t *avail_buf_size, cudaStream_t stream);
__host__ void GINTdel_basis_prod_rsdf(BasisProdCache **pbp);
