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
#include "config.h"
#include "cuda_alloc.cuh"
#include "gint.h"

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

void GINTinit_contraction_types(BasisProdCache *bpcache, int *bas_pair2shls,
    int *bas_pairs_locs, int ncptype, int *atm, int natm, int *bas, int nbas,
    double *env) {
    bpcache->ncptype = ncptype;
    bpcache->bas_pair2shls = bas_pair2shls;
    bpcache->bas_pairs_locs = bas_pairs_locs;

    ContractionProdType *cptype =
        (ContractionProdType *)malloc(sizeof(ContractionProdType) * ncptype);
    bpcache->cptype = cptype;
    int *primitive_pairs_locs = (int *)malloc(sizeof(int) * (ncptype + 1));
    bpcache->primitive_pairs_locs = primitive_pairs_locs;

    int n;
    int n_bas_pairs = bas_pairs_locs[ncptype];
    int *bas_pair2bra = bas_pair2shls;
    int *bas_pair2ket = bas_pair2shls + n_bas_pairs;
    int n_primitive_pairs = 0;
    primitive_pairs_locs[0] = 0;

    for (n = 0; n < ncptype; n++, cptype++) {
        int pair_id = bas_pairs_locs[n];
        int npairs = bas_pairs_locs[n + 1] - bas_pairs_locs[n];
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
        primitive_pairs_locs[n + 1] = n_primitive_pairs;
    }
}

void GINTinit_aexyz(double *aexyz, BasisProdCache *bpcache, double diag_fac,
    int *atm, int natm, int *bas, int nbas, double *env, int iL, int jL,
    double *Ls) {
    int ncptype = bpcache->ncptype;
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    int *bas_pair2bra = bpcache->bas_pair2shls;
    int *bas_pair2ket = bpcache->bas_pair2shls + n_bas_pairs;
    double *a12 = aexyz;
    double *e12 = a12 + n_primitive_pairs;
    double *x12 = e12 + n_primitive_pairs;
    double *y12 = x12 + n_primitive_pairs;
    double *z12 = y12 + n_primitive_pairs;
    int pair_id, count;
    int ish, jsh, ia, ja;
    int ip, jp, npi, npj, li, lj;
    double *ai, *aj, *ci, *cj, ri[3], rj[3], *iptrxyz, *jptrxyz;
    double rx, ry, rz, dist_ij, aij, norm;
    int off = 0;
    for (pair_id = 0; pair_id < n_bas_pairs; pair_id++) {
        ish = bas_pair2bra[pair_id];
        jsh = bas_pair2ket[pair_id];
        npi = bas[NPRIM_OF + ish * BAS_SLOTS];
        npj = bas[NPRIM_OF + jsh * BAS_SLOTS];
        ia = bas[ATOM_OF + ish * BAS_SLOTS];
        ja = bas[ATOM_OF + jsh * BAS_SLOTS];
        li = bas[ANG_OF + ish * BAS_SLOTS];
        lj = bas[ANG_OF + jsh * BAS_SLOTS];
        // ptr shift
        ai = env + bas[PTR_EXP + ish * BAS_SLOTS];
        aj = env + bas[PTR_EXP + jsh * BAS_SLOTS];
        ci = env + bas[PTR_COEFF + ish * BAS_SLOTS];
        cj = env + bas[PTR_COEFF + jsh * BAS_SLOTS];
        iptrxyz = env + atm[PTR_COORD + ia * ATM_SLOTS];
        jptrxyz = env + atm[PTR_COORD + ja * ATM_SLOTS];
        ri[0] = iptrxyz[0] + Ls[iL * 3 + 0];
        ri[1] = iptrxyz[1] + Ls[iL * 3 + 1];
        ri[2] = iptrxyz[2] + Ls[iL * 3 + 2];
        rj[0] = jptrxyz[0] + Ls[jL * 3 + 0];
        rj[1] = jptrxyz[1] + Ls[jL * 3 + 1];
        rj[2] = jptrxyz[2] + Ls[jL * 3 + 2];
        // Run shift_bas here
        // env_loc[ri+0] = env[ri+0] + Ls[iL*3+0]
        // env_loc[ri+1] = env[ri+1] + Ls[iL*3+1]
        // env_loc[ri+2] = env[ri+2] + Ls[iL*3+2]
        norm = CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);

        rx = ri[0] - rj[0];
        ry = ri[1] - rj[1];
        rz = ri[2] - rj[2];
        dist_ij = rx * rx + ry * ry + rz * rz;

        for (count = off, ip = 0; ip < npi; ip++) {
            for (jp = 0; jp < npj; jp++, count++) {
                aij = ai[ip] + aj[jp];
                a12[count] = aij;
                e12[count] = norm * ci[ip] * cj[jp] *
                             exp(-dist_ij * ai[ip] * aj[jp] / aij);
                x12[count] = (ai[ip] * ri[0] + aj[jp] * rj[0]) / aij;
                y12[count] = (ai[ip] * ri[1] + aj[jp] * rj[1]) / aij;
                z12[count] = (ai[ip] * ri[2] + aj[jp] * rj[2]) / aij;
            }
        }

        if (ish == jsh) {
            for (count = 0; count < npi * npj; count++) {
                e12[off + count] *= diag_fac;
            }
        }
        off += npi * npj;
    }
}

void GINTinit_aexyz_aux(double *aexyz, BasisProdCache *bpcache, double diag_fac,
    int *atm, int natm, int *bas, int nbas, double *env) {
    int ncptype = bpcache->ncptype;
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    int *bas_pair2bra = bpcache->bas_pair2shls;
    // int *bas_pair2ket = bpcache->bas_pair2shls + n_bas_pairs;
    double *a12 = aexyz;
    double *e12 = a12 + n_primitive_pairs;
    double *x12 = e12 + n_primitive_pairs;
    double *y12 = x12 + n_primitive_pairs;
    double *z12 = y12 + n_primitive_pairs;
    int pair_id, count;
    int ish, ia;
    int ip, npi, npj, li;
    double *ai, *ci, *ri;
    double aij, norm;
    int off = 0;

    for (pair_id = 0; pair_id < n_bas_pairs; pair_id++) {
        ish = bas_pair2bra[pair_id];
        npi = bas[NPRIM_OF + ish * BAS_SLOTS];
        npj = 1;
        ia = bas[ATOM_OF + ish * BAS_SLOTS];
        li = bas[ANG_OF + ish * BAS_SLOTS];
        ai = env + bas[PTR_EXP + ish * BAS_SLOTS];
        ci = env + bas[PTR_COEFF + ish * BAS_SLOTS];
        ri = env + atm[PTR_COORD + ia * ATM_SLOTS];
        norm = CINTcommon_fac_sp(li);
        for (count = off, ip = 0; ip < npi; ip++, count++) {
            aij = ai[ip];
            a12[count] = aij;
            e12[count] = norm * ci[ip];
            x12[count] = (ai[ip] * ri[0]) / aij;
            y12[count] = (ai[ip] * ri[1]) / aij;
            z12[count] = (ai[ip] * ri[2]) / aij;
        }

        off += npi * npj;
    }
}

void GINTsort_bas_coordinates(
    double *bas_coords, int *atm, int natm, int *bas, int nbas, double *env) {
    int ib, atm_id, ptr_coord;
    double *bas_x = bas_coords;
    double *bas_y = bas_x + nbas;
    double *bas_z = bas_y + nbas;
    for (ib = 0; ib < nbas; ib++) {
        atm_id = bas[ATOM_OF + ib * BAS_SLOTS];
        ptr_coord = atm[PTR_COORD + atm_id * ATM_SLOTS];
        bas_x[ib] = env[ptr_coord];
        bas_y[ib] = env[ptr_coord + 1];
        bas_z[ib] = env[ptr_coord + 2];
    }
}

void GINTinit_basis_prod(BasisProdCache **pbp, double diag_fac, int *ao_loc,
    int *bas_pair2shls, int *bas_pairs_locs, int ncptype, int *atm, int natm,
    int *bas, int nbas, double *env, int iL, int jL, double *Ls) {
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp = bpcache;

    GINTinit_contraction_types(bpcache, bas_pair2shls, bas_pairs_locs, ncptype,
        atm, natm, bas, nbas, env);
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 5);
    GINTinit_aexyz(
        aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env, iL, jL, Ls);
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

void GINTinit_contraction_types_aux(BasisProdCache *bpcache, int *bas_pair2shls,
    int *bas_pairs_locs, int ncptype, int *atm, int natm, int *bas, int nbas,
    double *env) {
    bpcache->ncptype = ncptype;
    bpcache->bas_pair2shls = bas_pair2shls;
    bpcache->bas_pairs_locs = bas_pairs_locs;

    ContractionProdType *cptype =
        (ContractionProdType *)malloc(sizeof(ContractionProdType) * ncptype);
    bpcache->cptype = cptype;
    int *primitive_pairs_locs = (int *)malloc(sizeof(int) * (ncptype + 1));
    bpcache->primitive_pairs_locs = primitive_pairs_locs;

    int n;
    int *bas_pair2bra = bas_pair2shls;
    int n_primitive_pairs = 0;
    primitive_pairs_locs[0] = 0;

    for (n = 0; n < ncptype; n++, cptype++) {
        int pair_id = bas_pairs_locs[n];
        int npairs = bas_pairs_locs[n + 1] - bas_pairs_locs[n];
        int ish = bas_pair2bra[pair_id];
        int li = bas[ANG_OF + ish * BAS_SLOTS];
        int lj = 0;
        int npi = bas[NPRIM_OF + ish * BAS_SLOTS];
        int npj = 1;
        cptype->l_bra = li;
        cptype->l_ket = lj;
        cptype->nprim_12 = npi * npj;
        cptype->npairs = npairs;
        n_primitive_pairs += npairs * npi * npj;
        primitive_pairs_locs[n + 1] = n_primitive_pairs;
    }
}

void GINTinit_basis_prod_aux_cpu(BasisProdCache **pbp, double diag_fac,
    double *bas_coords, int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
    int *atm, int natm, int *bas, int nbas, double *env) {
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp = bpcache;

    GINTinit_contraction_types_aux(bpcache, bas_pair2shls, bas_pairs_locs,
        ncptype, atm, natm, bas, nbas, env);
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 5);
    GINTinit_aexyz_aux(aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env);
    bpcache->aexyz = aexyz;
    bpcache->bas_pair2shls = bas_pair2shls;

    // initialize basis coordinates on GPU memory
    bpcache->nbas = nbas;
    GINTsort_bas_coordinates(bas_coords, atm, natm, bas, nbas, env);
}

void GINTinit_basis_prod_aux_gpu(BasisProdCache **pbpin, BasisProdCache **pbp,
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
}
