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

#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "config.h"
#include "cuda_alloc.cuh"
#include "gint.h"
#include "util.cuh"
void GINTinit_contraction_types_rsdf(BasisProdCache *bpcache,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype, int cp_ij_id,
    int *bas, int nbas) {
    bpcache->ncptype = 1;
    bpcache->nbas = nbas;
    bpcache->bas_pair2shls = bas_pair2shls;
    bpcache->bas_pairs_locs = bas_pairs_locs;

    ContractionProdType *cptype =
        (ContractionProdType *)malloc(sizeof(ContractionProdType));
    bpcache->cptype = cptype;
    int *primitive_pairs_locs = (int *)malloc(sizeof(int) * (2));
    bpcache->primitive_pairs_locs = primitive_pairs_locs;

    int n_bas_pairs = bas_pairs_locs[ncptype];
    int *bas_pair2bra = bas_pair2shls;
    int *bas_pair2ket = bas_pair2shls + n_bas_pairs;
    int n_primitive_pairs = 0;

    int pair_id = bas_pairs_locs[cp_ij_id];
    int npairs = bas_pairs_locs[cp_ij_id + 1] - bas_pairs_locs[cp_ij_id];
    int ish = bas_pair2bra[pair_id];
    int jsh = bas_pair2ket[pair_id];

    int li = bas[ANG_OF + ish * BAS_SLOTS];
    int lj = bas[ANG_OF + jsh * BAS_SLOTS];
    int npi = bas[NPRIM_OF + ish * BAS_SLOTS];
    int npj = bas[NPRIM_OF + jsh * BAS_SLOTS];
    cptype->l_bra = li;
    cptype->l_ket = lj;
    cptype->nprim_12 = npi * npj;
    cptype->npairs = npairs;
    n_primitive_pairs += npairs * npi * npj;

    primitive_pairs_locs[1] = n_primitive_pairs;
    primitive_pairs_locs[0] = 0;
}

void GINTinit_bpcache_kl(BasisProdCache *bpcache, double *d_aexyz,
    double *d_bas_coords, int *d_prescreen_bas_pair2bra, const int cp_ij_id) {
    int pair_id0 = bpcache->bas_pairs_locs[cp_ij_id];
    int pair_id1 = bpcache->bas_pairs_locs[cp_ij_id + 1];
    int n_bas_pairs = pair_id1 - pair_id0;

    int ncptype = bpcache->ncptype;
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    bpcache->bas_coords = d_bas_coords;

    bpcache->a12 = d_aexyz;
    bpcache->e12 = d_aexyz + n_primitive_pairs * 1;
    bpcache->x12 = d_aexyz + n_primitive_pairs * 2;
    bpcache->y12 = d_aexyz + n_primitive_pairs * 3;
    bpcache->z12 = d_aexyz + n_primitive_pairs * 4;

    bpcache->bas_pair2bra = d_prescreen_bas_pair2bra;
    bpcache->bas_pair2ket = d_prescreen_bas_pair2bra + n_bas_pairs;
}

__device__ static void get_rc(
    double *rc, double *ri, double *rj, double ei, double ej) {
    double eij = ei + ej;
    rc[0] = (ri[0] * ei + rj[0] * ej) / eij;
    rc[1] = (ri[1] * ei + rj[1] * ej) / eij;
    rc[2] = (ri[2] * ei + rj[2] * ej) / eij;
}

// no-constant
// d_aexyz, d_bas_coords,
// d_prescreen_bas_pair2bra, d_prescreen_n_bas_pairs
__global__ void GINTprescreen_aexyz_kernel(double *d_aexyz,
    const int n_primitive_pairs, const int pair_id0, const int pair_id1,
    const int nbas, const int *d_auxbas_pairs_locs, const int aux_id_range,
    double *d_bas_coords, int const *d_refuniqshl_map,
    int const *d_auxuniqshl_map, int nbasauxuniq, double *d_uniqexp,
    double const *d_uniq_dcut2s, double dcut_binsize,
    double const *d_uniq_Rcut2s, int const *d_uniqshlpr_dij_loc,
    int const *bas_pair2bra, int const *bas_pair2ket, const double diag_fac,
    int const *bas, int const *atm, double const *env, const int jL,
    const int iL1, int *h_iL, double const *Ls, int *d_prescreen_bas_pair2bra,
    int *d_prescreen_bas_aux, int *d_prescreen_n_bas_pairs,
    int *h_prescreen_n_bas_pairs, int stream_id, int prescreen_mask) {
    cg::grid_group grid = cg::this_grid();
    size_t idx = grid.thread_rank();
    size_t stride = grid.size();

    int iL = *h_iL;
    for (; iL < iL1; iL++) {
        int *d_prescreen_bas_pair2ket =
            d_prescreen_bas_pair2bra + (pair_id1 - pair_id0);
        double *a12 = d_aexyz;
        double *e12 = a12 + n_primitive_pairs;
        double *x12 = e12 + n_primitive_pairs;
        double *y12 = x12 + n_primitive_pairs;
        double *z12 = y12 + n_primitive_pairs;

        // reuse the calculation for the initial of d_bas_coords for the
        // corresponding part.
        double *ibas_x = d_bas_coords;
        double *ibas_y = ibas_x + nbas;
        double *ibas_z = ibas_y + nbas;
        double *jbas_x = ibas_z + nbas;
        double *jbas_y = jbas_x + nbas;
        double *jbas_z = jbas_y + nbas;
        size_t pair_id;

        // int off = 0;
        size_t prescreen_bas_idx;
        size_t auxprescreen_bas_idx;
        for (pair_id = idx + pair_id0; pair_id < pair_id1; pair_id += stride) {
            int ish, jsh, ia, ja, iptrxyz, jptrxyz;
            int ip, jp, npi, npj, li, lj;
            double const *ai, *aj, *ci, *cj;
            double *ri, *rj;
            double aij, norm;
            double ixyz[3];
            double jxyz[3];
            double dij2, dij2_cut;
            int Ish, Jsh, IJsh;
            int count;
            ish = bas_pair2bra[pair_id];
            jsh = bas_pair2ket[pair_id];

            npi = bas[NPRIM_OF + ish * BAS_SLOTS];
            npj = bas[NPRIM_OF + jsh * BAS_SLOTS];
            // off = offset[pair_id-pair_id0]; d_bas[ATOM_OF + ib * BAS_SLOTS];
            // ib is similar to ish and jsh
            ia = bas[ATOM_OF + ish * BAS_SLOTS];
            ja = bas[ATOM_OF + jsh * BAS_SLOTS];
            // printf("ia %d and ja %d : \n", ia, ja);
            li = bas[ANG_OF + ish * BAS_SLOTS];
            lj = bas[ANG_OF + jsh * BAS_SLOTS];
            // ptr shift
            ai = env + bas[PTR_EXP + ish * BAS_SLOTS]; // offset
            aj = env + bas[PTR_EXP + jsh * BAS_SLOTS];
            ci = env + bas[PTR_COEFF + ish * BAS_SLOTS];
            cj = env + bas[PTR_COEFF + jsh * BAS_SLOTS];
            iptrxyz = atm[PTR_COORD + ia * ATM_SLOTS];
            jptrxyz = atm[PTR_COORD + ja * ATM_SLOTS];

            ixyz[0] = env[iptrxyz + 0] + Ls[iL * 3 + 0];
            ixyz[1] = env[iptrxyz + 1] + Ls[iL * 3 + 1];
            ixyz[2] = env[iptrxyz + 2] + Ls[iL * 3 + 2];

            jxyz[0] = env[jptrxyz + 0] + Ls[jL * 3 + 0];
            jxyz[1] = env[jptrxyz + 1] + Ls[jL * 3 + 1];
            jxyz[2] = env[jptrxyz + 2] + Ls[jL * 3 + 2];

            ri = ixyz;
            rj = jxyz;

            dij2 = square_dist(ri, rj);
            Ish = d_refuniqshl_map[ish];
            Jsh = d_refuniqshl_map[jsh];
            IJsh = (Ish >= Jsh) ? (Ish * (Ish + 1) / 2 + Jsh)
                                : (Jsh * (Jsh + 1) / 2 + Ish);
            dij2_cut = d_uniq_dcut2s[IJsh];
            if (dij2 <= dij2_cut || prescreen_mask < 1) {
                size_t idij = (size_t)(sqrt(dij2) / dcut_binsize);
                const double *uniq_Rcut2s_K =
                    d_uniq_Rcut2s +
                    (d_uniqshlpr_dij_loc[IJsh] + idij) * nbasauxuniq;
                double rc[3];
                double ei = d_uniqexp[Ish];
                double ej = d_uniqexp[Jsh];
                get_rc(rc, ri, rj, ei, ej);
                int flag = 0;
                for (int aux_id = 0; aux_id < aux_id_range; aux_id++) {
                    size_t ksh0 = d_auxbas_pairs_locs[aux_id];
                    int ksh1 = d_auxbas_pairs_locs[aux_id + 1];
                    for (int ksh = ksh0; ksh < ksh1; ksh++) {
                        int Ksh = d_auxuniqshl_map[ksh];
                        double Rcut2 = uniq_Rcut2s_K[Ksh];
                        int kptrxyz =
                            atm[PTR_COORD +
                                bas[ATOM_OF + (ksh + nbas) * BAS_SLOTS] *
                                    ATM_SLOTS];
                        double *rk = (double *)(env + kptrxyz);
                        double Rijk2 = square_dist(rc, rk);
                        if (Rijk2 < Rcut2 || prescreen_mask < 2) {
                            if (flag == 0)
                                prescreen_bas_idx =
                                    atomicAdd(d_prescreen_n_bas_pairs, 1);
                            auxprescreen_bas_idx = atomicAdd(
                                d_prescreen_n_bas_pairs + aux_id + 1, 1);
                            size_t offset = auxprescreen_bas_idx +
                                         ksh0 * (pair_id1 - pair_id0);
                            d_prescreen_bas_aux[2 * offset] = ksh;
                            d_prescreen_bas_aux[2 * offset + 1] =
                                prescreen_bas_idx;
                            flag++;
                        }
                    }
                }
                if (flag == 0)
                    continue;
                d_prescreen_bas_pair2bra[prescreen_bas_idx] = ish;
                d_prescreen_bas_pair2ket[prescreen_bas_idx] = jsh;

                ibas_x[ish] = ri[0];
                ibas_y[ish] = ri[1];
                ibas_z[ish] = ri[2];

                jbas_x[jsh] = rj[0];
                jbas_y[jsh] = rj[1];
                jbas_z[jsh] = rj[2];

                norm = CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
                // dij2 = square_dist(ri,rj);
                int prescreen_primitive_idx = npi * npj * prescreen_bas_idx;
                for (count = prescreen_primitive_idx, ip = 0; ip < npi; ip++) {
                    for (jp = 0; jp < npj; jp++, count++) {
                        aij = ai[ip] + aj[jp];
                        a12[count] = aij;
                        e12[count] = norm * ci[ip] * cj[jp] *
                                     exp(-dij2 * ai[ip] * aj[jp] / aij);
                        x12[count] = (ai[ip] * ri[0] + aj[jp] * rj[0]) / aij;
                        y12[count] = (ai[ip] * ri[1] + aj[jp] * rj[1]) / aij;
                        z12[count] = (ai[ip] * ri[2] + aj[jp] * rj[2]) / aij;
                    }
                }

                if (ish == jsh) {
                    for (count = 0; count < npi * npj; count++) {
                        e12[prescreen_primitive_idx + count] *= diag_fac;
                    }
                }
            }
        }
        grid.sync();
        if (idx == 0) {
            *h_iL = iL;
            for (int aux_id = 0; aux_id < aux_id_range + 1; aux_id++) {
                h_prescreen_n_bas_pairs[aux_id] =
                    d_prescreen_n_bas_pairs[aux_id];
                d_prescreen_n_bas_pairs[aux_id] = 0;
            }
        }
        grid.sync();
        if (h_prescreen_n_bas_pairs[0] > 0) {
            return;
        }
    }
}

void GINTinit_aexyz_rsdf(BasisProdCache *bpcache, double diag_fac, int cp_ij_id,
    int *prescreen_n_bas_pairs, int auxnbas, int nbas, int *d_auxbas_pairs_locs,
    int aux_id_range, double *d_bas_coords, double *d_auxbas_coords,
    int *d_prescreen_bas_pair2bra, int *d_prescreen_bas_aux,
    int *d_prescreen_n_bas_pairs, int *d_refuniqshl_map, int *d_auxuniqshl_map,
    int nbasauxuniq, double *d_uniqexp, double *d_uniq_dcut2s,
    double dcut_binsize, double *d_uniq_Rcut2s, int *d_uniqshlpr_dij_loc,
    double *d_aexyz, int *d_bas, int *d_atm, double *d_env, double *d_Ls,
    int *d_bas_pair2bra, int *d_bas_pair2ket, int jL, int &iL, int iL1,
    int *h_iL, int stream_id, int prescreen_mask,
    cudaStream_t stream = NULL) {
    int pair_id0 = bpcache->bas_pairs_locs[cp_ij_id]; // const
    int pair_id1 = bpcache->bas_pairs_locs[cp_ij_id + 1];
    int n_bas_pairs = pair_id1 - pair_id0;

    int ncptype = bpcache->ncptype;
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];

    *h_iL = iL;
    int blocks_num = (n_bas_pairs + 639) / 640 > BLOCKS_SCREEN
                         ? BLOCKS_SCREEN
                         : (n_bas_pairs + 639) / 640;
    dim3 threads(640);
    dim3 blocks(blocks_num);
    void *kernelArgs[] = {
        (void *)&d_aexyz,
        (void *)&n_primitive_pairs,
        (void *)&pair_id0,
        (void *)&pair_id1,
        (void *)&nbas,
        (void *)&d_auxbas_pairs_locs,
        (void *)&aux_id_range,
        (void *)&d_bas_coords,
        (void *)&d_refuniqshl_map,
        (void *)&d_auxuniqshl_map,
        (void *)&nbasauxuniq,
        (void *)&d_uniqexp,
        (void *)&d_uniq_dcut2s,
        (void *)&dcut_binsize,
        (void *)&d_uniq_Rcut2s,
        (void *)&d_uniqshlpr_dij_loc,
        (void *)&d_bas_pair2bra,
        (void *)&d_bas_pair2ket,
        (void *)&diag_fac,
        (void *)&d_bas,
        (void *)&d_atm,
        (void *)&d_env,
        (void *)&jL,
        (void *)&iL1,
        (void *)&h_iL,
        (void *)&d_Ls,
        (void *)&d_prescreen_bas_pair2bra,
        (void *)&d_prescreen_bas_aux,
        (void *)&d_prescreen_n_bas_pairs,
        (void *)&prescreen_n_bas_pairs,
        (void *)&stream_id,
        (void *)&prescreen_mask,
    };
    cudaLaunchCooperativeKernel((void *)GINTprescreen_aexyz_kernel, blocks,
        threads, kernelArgs, 0, stream);
    cudaStreamSynchronize(stream);

    iL = *h_iL;
    CUDA_LAST_CHECK();
    if (*prescreen_n_bas_pairs == 0) {
        return;
    }
}

void GINTinit_basis_prod_rsdf(BasisProdCache *bpcache, double diag_fac,
    double *d_bas_coords, int *bas_pair2shls, int *bas_pairs_locs,
    int *d_auxbas_pairs_locs, int aux_id_range, int ncptype, int cp_ij_id,
    int *prescreen_n_bas_pairs, int auxnbas, double *d_auxbas_coords,
    int *d_prescreen_bas_pair2bra, int *d_prescreen_bas_aux,
    int *d_prescreen_n_bas_pairs, int *d_refuniqshl_map,
    int *d_auxuniqshl_map, int nbasauxuniq, double *d_uniqexp,
    double *d_uniq_dcut2s, double dcut_binsize, double *d_uniq_Rcut2s,
    int *d_uniqshlpr_dij_loc, double *d_aexyz, int *d_bas, int *d_atm,
    double *d_env, double *d_Ls, int *d_bas_pair2bra, int *d_bas_pair2ket,
    int nbas, int jL, int &iL, int iL1, int *h_iL, int stream_id,
    int prescreen_mask, cudaStream_t stream) {
    GINTinit_aexyz_rsdf(bpcache, diag_fac, cp_ij_id, prescreen_n_bas_pairs,
        auxnbas, nbas, d_auxbas_pairs_locs, aux_id_range, d_bas_coords,
        d_auxbas_coords, d_prescreen_bas_pair2bra, d_prescreen_bas_aux,
        d_prescreen_n_bas_pairs, d_refuniqshl_map, d_auxuniqshl_map,
        nbasauxuniq, d_uniqexp, d_uniq_dcut2s, dcut_binsize, d_uniq_Rcut2s,
        d_uniqshlpr_dij_loc, d_aexyz, d_bas, d_atm, d_env, d_Ls, d_bas_pair2bra,
        d_bas_pair2ket, jL, iL, iL1, h_iL, stream_id, prescreen_mask, stream);
}

void GINTinit_EnvVars(GINTEnvVars *envs, ContractionProdType *cp_ij,
    ContractionProdType *cp_kl, double omega) {
    int i_l = cp_ij->l_bra;
    int j_l = cp_ij->l_ket;
    int k_l = cp_kl->l_bra;
    int l_l = cp_kl->l_ket;
    int nfi = (i_l + 1) * (i_l + 2) / 2;
    int nfj = (j_l + 1) * (j_l + 2) / 2;
    int nfk = (k_l + 1) * (k_l + 2) / 2;
    int nfl = (l_l + 1) * (l_l + 2) / 2;
    int nroots = (i_l + j_l + k_l + l_l) / 2 + 1;
    double fac = (M_PI * M_PI * M_PI) * 2 / SQRTPI;

    envs->i_l = i_l;
    envs->j_l = j_l;
    envs->k_l = k_l;
    envs->l_l = l_l;

    envs->nfi = nfi;
    envs->nfj = nfj;
    envs->nfk = nfk;
    envs->nfl = nfl;
    envs->nf = nfi * nfj * nfk * nfl;
    envs->nrys_roots = nroots;
    envs->fac = fac;

    // int ibase = i_l >= j_l;
    int kbase = k_l >= l_l;
    envs->ibase = 1;
    envs->kbase = kbase;

    int li1 = i_l + 1;
    int lj1 = j_l + 1;
    int lk1 = k_l + 1;
    int ll1 = l_l + 1;
    int di = nroots;
    int dj = di * li1;
    // int dk = dj * lj1;
    int dl = dj * lk1;
    envs->g_size_ij = dj * lj1;
    envs->g_size = dl * ll1;
    envs->omega = omega;

    envs->ijmin = j_l;
    envs->ijmax = i_l;
    envs->stride_ijmax = nroots;
    envs->stride_ijmin = nroots * li1;

    if (kbase) {
        envs->klmin = l_l;
        envs->klmax = k_l;
        envs->stride_klmax = dj * lj1;
        envs->stride_klmin = dj * lj1 * lk1;
    } else {
        envs->klmin = k_l;
        envs->klmax = l_l;
        envs->stride_klmax = dj * lj1;
        envs->stride_klmin = dj * lj1 * ll1;
    }

    envs->nprim_ij = cp_ij->nprim_12;
    envs->nprim_kl = cp_kl->nprim_12;
}
/*
 * GTO = x^{nx}y^{ny}z^{nz}e^{-ar^2}
 */
void CINTcart_comp(int *nx, int *ny, int *nz, const int lmax) {
    int inc = 0;
    int lx, ly, lz;

    for (lx = lmax; lx >= 0; lx--) {
        for (ly = lmax - lx; ly >= 0; ly--) {
            lz = lmax - lx - ly;
            nx[inc] = lx;
            ny[inc] = ly;
            nz[inc] = lz;
            inc++;
        }
    }
}
void GINTinit_2c_gidx(int *idx, int li, int lj) {
    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    int i, j, n;
    int stride_i, stride_j;
    int *idy = idx + nfij;
    int *idz = idx + nfij * 2;

    int i_nx[nfi], i_ny[nfi], i_nz[nfi];
    int j_nx[nfj], j_ny[nfj], j_nz[nfj];
    CINTcart_comp(i_nx, i_ny, i_nz, li);
    CINTcart_comp(j_nx, j_ny, j_nz, lj);

    if (li >= lj) {
        stride_i = 1;
        stride_j = li + 1;
    } else {
        stride_i = li + 1;
        stride_j = 1;
    }
    for (n = 0, j = 0; j < nfj; j++) {
        for (i = 0; i < nfi; i++, n++) {
            idx[n] = stride_j * j_nx[j] + stride_i * i_nx[i];
            idy[n] = stride_j * j_ny[j] + stride_i * i_ny[i];
            idz[n] = stride_j * j_nz[j] + stride_i * i_nz[i];
        }
    }
}

void GINTinit_4c_idx(
    int16_t *idx, int *ij_idx, int *kl_idx, GINTEnvVars *envs) {
    int nfi = envs->nfi;
    int nfj = envs->nfj;
    int nfk = envs->nfk;
    int nfl = envs->nfl;
    int nfij = nfi * nfj;
    int nfkl = nfk * nfl;
    int nroots = envs->nrys_roots;
    int g_size = envs->g_size;
    int g_size_ij = envs->g_size_ij / nroots;
    int *ij_idy = ij_idx + nfij;
    int *ij_idz = ij_idy + nfij;
    int *kl_idy = kl_idx + nfkl;
    int *kl_idz = kl_idy + nfkl;
    // int16_t *idy = idx + nf;
    // int16_t *idz = idx + nf * 2;
    int16_t *idy = idx + 1;
    int16_t *idz = idx + 2;
    int ofx = 0;
    int ofy = g_size / nroots;
    int ofz = g_size / nroots * 2;
    int n, ij, kl;
    for (n = 0, kl = 0; kl < nfkl; kl++) {
        for (ij = 0; ij < nfij; ij++, n++) {
            idx[n * 3] = ofx + ij_idx[ij] + kl_idx[kl] * g_size_ij;
            idy[n * 3] = ofy + ij_idy[ij] + kl_idy[kl] * g_size_ij;
            idz[n * 3] = ofz + ij_idz[ij] + kl_idz[kl] * g_size_ij;
        }
    }
    // TODO: copy to constant memory or global memory depends on the size of nf
}
void GINTinit_idx4c_envs(const BasisProdCache *bpcache_ij,
    const BasisProdCache *bpcache_kl, const int cp_aux_id, const double omega,
    int16_t **c_idx4c, GINTEnvVars *c_envs, void **ptr_buf,
    size_t *avail_buf_size, cudaStream_t stream) {
    ContractionProdType *cp_ij = bpcache_ij->cptype + cp_aux_id;
    ContractionProdType *cp_kl = bpcache_kl->cptype;
    GINTinit_EnvVars(c_envs, cp_ij, cp_kl, omega);
    // int *bas_pairs_locs_ij = bpcache_ij->bas_pairs_locs;
    // int ntasks_ij = bas_pairs_locs_ij[cp_aux_id + 1] -
    // bas_pairs_locs_ij[cp_aux_id];
    //         printf("nprim:%d, ij: %d, kl: %d, roots:%d, i:%d, j:%d, k:%d,
    //         l:%d, ntask_kl: %d\n", c_envs->nprim_ij * c_envs->nprim_kl,
    //         c_envs->nprim_ij, c_envs->nprim_kl, c_envs->nrys_roots ,
    //         c_envs->i_l, c_envs->j_l, c_envs->k_l, c_envs->l_l, ntasks_ij);
    if (c_envs->nrys_roots > 3) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * c_envs->nf * 3);
        int *idx_ij =
            (int *)malloc(sizeof(int) * c_envs->nfi * c_envs->nfj * 3);
        int *idx_kl =
            (int *)malloc(sizeof(int) * c_envs->nfk * c_envs->nfl * 3);

        GINTinit_2c_gidx(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
        GINTinit_2c_gidx(idx_kl, cp_kl->l_bra, cp_kl->l_ket);
        GINTinit_4c_idx(idx4c, idx_ij, idx_kl, c_envs);
        MALLOC_ALIGN_MEMPOOL(
            int16_t, tmp_idx4c, c_envs->nf * 3, *ptr_buf, *avail_buf_size);
        CUDA_CHECK(cudaMemcpyAsync(tmp_idx4c, idx4c,
            sizeof(int16_t) * c_envs->nf * 3, cudaMemcpyHostToDevice, stream));
        *c_idx4c = tmp_idx4c;
        free(idx4c);
        free(idx_ij);
        free(idx_kl);
    }
}

__host__ void GINTdel_basis_prod_rsdf(BasisProdCache **pbp) {
    BasisProdCache *bpcache = *pbp;
    if (bpcache == NULL) {
        return;
    }
    if (bpcache->cptype != NULL) {
        free(bpcache->cptype);
    }
    if (bpcache->primitive_pairs_locs != NULL) {
        free(bpcache->primitive_pairs_locs);
    }
    free(bpcache);
    *pbp = NULL;
}
