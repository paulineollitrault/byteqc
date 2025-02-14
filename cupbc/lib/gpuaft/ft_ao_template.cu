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

template <int I, int J, int Fz>
__host__ __device__ static void aopair_rr_igtj_lazy(cuDoubleComplex *g,
    const double ai, const double aj, const CINTEnvVars *envs, const double *rj,
    const cuDoubleComplex fac, const double *Gv, const double *b,
    const int *gxyz, const int3 &gs, const size_t NGv, const size_t iGv) {
    constexpr int nmax = I + J;
    constexpr int dj = I < J ? (I + 1) : (I + J + 1);
    constexpr int g_size =
        I < J ? ((I + 1) * (I + J + 1)) : ((I + J + 1) * (J + 1));
    const double aij = ai + aj;
    const double a2 = .5 / aij;
    const double *ri = envs->ri;
    double rij[3], rirj[3], rijri[3];
    cuDoubleComplex *gx = g;
    cuDoubleComplex *gy = gx + g_size;
    cuDoubleComplex *gz = gy + g_size;
    const double *kx = Gv;
    const double *ky = kx + NGv;
    const double *kz = ky + NGv;
    size_t off0, off1, off2;
    int i, j;
    double ia2;

    rirj[0] = ri[0] - rj[0];
    rirj[1] = ri[1] - rj[1];
    rirj[2] = ri[2] - rj[2];
    rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
    rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
    rij[2] = (ai * ri[2] + aj * rj[2]) / aij;
    rijri[0] = rij[0] - ri[0];
    rijri[1] = rij[1] - ri[1];
    rijri[2] = rij[2] - ri[2];

    gx[0] = Z1;
    gy[0] = Z1;
    eval_gz<Fz>(gz, aij, rij, fac, Gv, b, gxyz, gs, NGv, iGv);
    if (nmax > 0) {
        if (ISNZ0(gz[0])) {
            gx[1] =
                cuCmul(make_cuDoubleComplex(rijri[0], -kx[iGv] * a2), gx[0]);
            gy[1] =
                cuCmul(make_cuDoubleComplex(rijri[1], -ky[iGv] * a2), gy[0]);
            gz[1] =
                cuCmul(make_cuDoubleComplex(rijri[2], -kz[iGv] * a2), gz[0]);
        }
    }

#pragma unroll
    for (i = 1; i < nmax; i++) {
        off0 = (i - 1);
        off1 = i;
        off2 = (i + 1);
        ia2 = i * a2;
        if (ISNZ0(gz[0])) {
            gx[off2] = cuCadd(cuCmul(make_cuDoubleComplex(ia2, 0.0), gx[off0]),
                cuCmul(
                    make_cuDoubleComplex(rijri[0], -kx[iGv] * a2), gx[off1]));
            gy[off2] = cuCadd(cuCmul(make_cuDoubleComplex(ia2, 0.0), gy[off0]),
                cuCmul(
                    make_cuDoubleComplex(rijri[1], -ky[iGv] * a2), gy[off1]));
            gz[off2] = cuCadd(cuCmul(make_cuDoubleComplex(ia2, 0.0), gz[off0]),
                cuCmul(
                    make_cuDoubleComplex(rijri[2], -kz[iGv] * a2), gz[off1]));
        }
    }

#pragma unroll
    for (j = 1; j <= J; j++) {
        int ptr = dj * j;
#pragma unroll
        for (i = ptr; i <= ptr + nmax - j; i++) {
            off0 = i - dj;       // [i,  j-1]
            off1 = (i + 1) - dj; // [i+1,j-1]
            off2 = i;            // [i,  j  ]
            if (ISNZ0(gz[0])) {
                gx[off2] = cuCadd(gx[off1],
                    cuCmul(make_cuDoubleComplex(rirj[0], 0.0), gx[off0]));
                gy[off2] = cuCadd(gy[off1],
                    cuCmul(make_cuDoubleComplex(rirj[1], 0.0), gy[off0]));
                gz[off2] = cuCadd(gz[off1],
                    cuCmul(make_cuDoubleComplex(rirj[2], 0.0), gz[off0]));
            }
        }
    }
}

template <int I, int J>
__host__ __device__ static void inner_prod_direct(const cuDoubleComplex *g,
    cuDoubleComplex *buf, const size_t ni, const double *Gv, const int empty) {
    int n = 0;
    constexpr int g_size =
        I < J ? (I + 1) * (I + J + 1) : (J + 1) * (I + J + 1);
    constexpr int di = 1;
    constexpr int dj = I < J ? I + 1 : I + J + 1;
    constexpr int nfi = (I + 1) * (I + 2) / 2;
    constexpr int nf = nfi * (J + 1) * (J + 2) / 2;
    const cuDoubleComplex *gz = g + g_size * 2;

    int ix = dj * J + di * I;
    int iy = g_size;
    int iz = 2 * g_size;

    if (empty) {
        if (ISNZ0(gz[0])) {
#pragma unroll
            for (int ljx = J; ljx >= 0; ljx--) {
#pragma unroll
                for (int ljy = J - ljx; ljy >= 0; ljy--) {
#pragma unroll
                    for (int lix = I; lix >= 0; lix--) {
#pragma unroll
                        for (int liy = I - lix; liy >= 0; liy--) {
                            buf[n] = cuCmul(cuCmul(g[ix], g[iy]), g[iz]);
                            iy -= di;
                            iz += di;
                            n += 1;
                        }
                        iy += di * (I - lix + 2);
                        iz -= di * (I - lix + 1);
                        ix -= di;
                    }
                    ix += di * (I + 1);
                    iy -= di * (I + 1);

                    iy -= dj;
                    iz += dj;

                    n += (ni - nfi) * 1;
                }
                iy += dj * (J - ljx + 2);
                iz -= dj * (J - ljx + 1);
                ix -= dj;
            }
        } else {
#pragma unroll
            for (int i = 0; i < nf; i++)
                buf[i] = Z0;
        }
    } else {
        if (ISNZ0(gz[0]))
#pragma unroll
            for (int ljx = J; ljx >= 0; ljx--) {
#pragma unroll
                for (int ljy = J - ljx; ljy >= 0; ljy--) {
#pragma unroll
                    for (int lix = I; lix >= 0; lix--) {
#pragma unroll
                        for (int liy = I - lix; liy >= 0; liy--) {
                            buf[n] = cuCadd(
                                buf[n], cuCmul(cuCmul(g[ix], g[iy]), g[iz]));
                            iy -= di;
                            iz += di;
                            n += 1;
                        }
                        iy += di * (I - lix + 2);
                        iz -= di * (I - lix + 1);
                        ix -= di;
                    }
                    ix += di * (I + 1);
                    iy -= di * (I + 1);

                    iy -= dj;
                    iz += dj;

                    n += (ni - nfi) * 1;
                }
                iy += dj * (J - ljx + 2);
                iz -= dj * (J - ljx + 1);
                ix -= dj;
            }
    }
}

template <int I, int J, int Fz>
__host__ __device__ static int GTO_ft_aopair_lazy_ker(cuDoubleComplex *buf,
    const CINTEnvVars &envs, const double *rj, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv, const int2 &dims, const int i_prim, const int j_prim,
    const int i_sh, const int j_sh, const double *ai, const double *aj,
    const double *ci, const double *cj, const cuDoubleComplex fac) {
    constexpr size_t leng =
        3 * (I < J ? ((I + 1) * (I + J + 1)) : ((J + 1) * (I + J + 1)));
    cuDoubleComplex g[leng];

    double fac1i, fac1j;
    double aij, dij, eij;

    const double *ri = envs.ri;
    double rrij = square_dist(ri, rj);
    constexpr double fac1 =
        SQRTPI * M_PI * CINTcommon_fac_sp(I) * CINTcommon_fac_sp(J);

    int empty = 1;
    for (int jp = 0; jp < j_prim; jp++) {
        fac1j = fac1 * cj[jp];
        for (int ip = 0; ip < i_prim; ip++) {
            aij = ai[ip] + aj[jp];
            eij = (ai[ip] * aj[jp] / aij) * rrij;
            if (eij <= EXP_CUTOFF) {
                dij = exp(-eij) / (aij * sqrt(aij));
                fac1i = fac1j * dij * ci[ip];
                aopair_rr_igtj_lazy<I, J, Fz>(g, ai[ip], aj[jp], &envs, rj,
                    cuCmul(make_cuDoubleComplex(fac1i, 0.0), fac), Gv, b, gxyz,
                    gs, NGv, iGv);
                inner_prod_direct<I, J>(g, buf, dims.x, Gv, empty);
                empty = 0;
            }
        }
    }
    if (empty) {
        constexpr int nf = (I + 1) * (I + 2) * (J + 1) * (J + 2) / 4;
#pragma unroll
        for (int i = 0; i < nf; i++)
            buf[i] = Z0;
    }
    return !empty;
}

template <int I, int J>
__device__ static void vrr1d_withGv(cuDoubleComplex *g, const double *rijri,
    const double aij, const double *Gv, const size_t NGv, const size_t iGv) {
    if (I + J == 0)
        return;
    int cumxyz = 1;
    const double *kx = Gv;
    const double *ky = kx + NGv;
    const double *kz = ky + NGv;
    int i, m, l;
    double a2;
    cuDoubleComplex *p0, *p1, *p2, *dec1, *dec2;
    double ka2[3];
    double &kxa2 = ka2[0];
    double &kya2 = ka2[1];
    double &kza2 = ka2[2];
    a2 = .5 / aij;
    kxa2 = kx[iGv] * a2;
    kya2 = ky[iGv] * a2;
    kza2 = kz[iGv] * a2;

    p0 = g + 1;
    p0[0] = cuCmul(make_cuDoubleComplex(rijri[0], -kxa2), g[0]);
    p0[1] = cuCmul(make_cuDoubleComplex(rijri[1], -kya2), g[0]);
    p0[2] = cuCmul(make_cuDoubleComplex(rijri[2], -kza2), g[0]);
    cumxyz += 3;

#pragma unroll
    for (l = 1; l < I + J; l++) {
        p0 = g + cumxyz;
        dec1 = p0 - _LEN_CART[l];
        dec2 = dec1 - _LEN_CART[l - 1];
#pragma unroll
        for (i = 0; i < _LEN_CART[l + 1]; i++) {
            m = DEC1_XYZ(l + 1, i);
            double kxa2_ = ka2[m];
            p1 = dec1 + ADDR_IF_L_DEC1(l + 1, i);
            p2 = dec2 + ADDR_IF_L_DEC2(l + 1, i);
            if (ADDR_IF_L_DEC2(l + 1, i) < 0) {
                p0[0] = cuCmul(make_cuDoubleComplex(rijri[m], -kxa2_), p1[0]);
            } else {
                a2 = .5 / aij * DEC1_XYZ_ORDER(l + 1, i);
                p0[0] = cuCadd(cuCmul(make_cuDoubleComplex(a2, 0.0), p2[0]),
                    cuCmul(make_cuDoubleComplex(rijri[m], -kxa2_), p1[0]));
            }
            p0 += 1;
        }
        cumxyz += _LEN_CART[l + 1];
    }
}

template <int I, int J, int Fz>
__host__ __device__ static void aopair_rr_igtj_early(cuDoubleComplex *g,
    const double ai, const double aj, const CINTEnvVars *envs, const double *rj,
    const cuDoubleComplex fac, const double *Gv, const double *b,
    const int *gxyz, const int3 &gs, const size_t NGv, const size_t iGv) {
    const double aij = ai + aj;
    const double *ri = envs->ri;
    double rij[3], rijri[3];

    rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
    rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
    rij[2] = (ai * ri[2] + aj * rj[2]) / aij;
    rijri[0] = rij[0] - ri[0];
    rijri[1] = rij[1] - ri[1];
    rijri[2] = rij[2] - ri[2];

    eval_gz<Fz>(g, aij, rij, fac, Gv, b, gxyz, gs, NGv, iGv);
    vrr1d_withGv<I, J>(g, rijri, aij, Gv, NGv, iGv);
}
template <int I, int J>
__device__ static void prim_to_ctr(cuDoubleComplex *gc,
    const cuDoubleComplex *gp, const double coeff, const int empty) {
    double c;
    constexpr int offset_g1d = _CUM_LEN_CART[I] - _LEN_CART[I];
    constexpr int nf = _CUM_LEN_CART[I + J] - offset_g1d;
    if (empty) {
        c = coeff;
#pragma unroll
        for (int i = 0; i < nf; i++) {
            gc[i] = cuCmul(gp[i], make_cuDoubleComplex(c, 0.0));
        }
    } else {
        c = coeff;
        if (c != 0) {
#pragma unroll
            for (int i = 0; i < nf; i++) {
                gc[i] =
                    cuCadd(gc[i], cuCmul(gp[i], make_cuDoubleComplex(c, 0.0)));
            }
        }
    }
}
__device__ static void vrr2d_ket_inc1_withGv(cuDoubleComplex *out,
    const cuDoubleComplex *g, double *rirj, const int li, const int lj) {
    if (lj == 0) {
        for (int i = 0; i < _LEN_CART[li]; i++) {
            out[i] = g[i];
        }
        return;
    }
    const int row_10 = _LEN_CART[li + 1];
    const int row_00 = _LEN_CART[li];
    const int col_00 = _LEN_CART[lj - 1];
    const cuDoubleComplex *g00 = g;
    const cuDoubleComplex *g10 = g + row_00 * col_00;
    int i, j;
    const cuDoubleComplex *p00, *p10;
    cuDoubleComplex *p01 = out;

    p01 = out;
    for (j = STARTX_IF_L_DEC1(lj); j < _LEN_CART[lj - 1]; j++) {
        for (i = 0; i < row_00; i++) {
            p00 = g00 + (j * row_00 + i);
            p10 = g10 + (j * row_10 + WHEREX_IF_L_INC1(i));
            p01[0] = cuCadd(
                p10[0], cuCmul(make_cuDoubleComplex(rirj[0], 0.0), p00[0]));
            p01++;
        }
    }
    for (j = STARTY_IF_L_DEC1(lj); j < _LEN_CART[lj - 1]; j++) {
        for (i = 0; i < row_00; i++) {
            p00 = g00 + (j * row_00 + i);
            p10 = g10 + (j * row_10 + WHEREY_IF_L_INC1(i));
            p01[0] = cuCadd(
                p10[0], cuCmul(make_cuDoubleComplex(rirj[1], 0.0), p00[0]));
            p01++;
        }
    }
    j = STARTZ_IF_L_DEC1(lj);
    if (j < _LEN_CART[lj - 1]) {
        for (i = 0; i < row_00; i++) {
            p00 = g00 + (j * row_00 + i);
            p10 = g10 + (j * row_10 + WHEREZ_IF_L_INC1(i));
            p01[0] = cuCadd(
                p10[0], cuCmul(make_cuDoubleComplex(rirj[2], 0.0), p00[0]));
            p01++;
        }
    }
}
template <int I, int J>
__device__ static void vrr2d_withGv(cuDoubleComplex *out, cuDoubleComplex *g,
    cuDoubleComplex *gbuf2, const double *ri, const double *rj) {
    constexpr int nmax = I + J;
    cuDoubleComplex *g00, *g01, *gswap, *pg00, *pg01;
    int row_01, col_01, row_00, col_00;
    double rirj[3];
    rirj[0] = ri[0] - rj[0];
    rirj[1] = ri[1] - rj[1];
    rirj[2] = ri[2] - rj[2];

    g00 = gbuf2;
    g01 = g;

#pragma unroll
    for (int j = 1; j < J; j++) {
        gswap = g00;
        g00 = g01;
        g01 = gswap;
        pg00 = g00;
        pg01 = g01;
#pragma unroll
        for (int i = I; i <= nmax - j; i++) {
            vrr2d_ket_inc1_withGv(pg01, pg00, rirj, i, j);
            row_01 = _LEN_CART[i];
            col_01 = _LEN_CART[j];
            row_00 = _LEN_CART[i];
            col_00 = _LEN_CART[j - 1];
            pg00 += row_00 * col_00;
            pg01 += row_01 * col_01;
        }
    }
    vrr2d_ket_inc1_withGv(out, g01, rirj, I, J);
}

template <int I, int J, int Fz>
__device__ static int GTO_ft_aopair_early_ker(cuDoubleComplex *buf,
    const CINTEnvVars &envs, const double *rj, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv, const int2 &dims, const int i_prim, const int j_prim,
    const int i_sh, const int j_sh, const double *ai, const double *aj,
    const double *ci, const double *cj, const cuDoubleComplex fac) {
    const double *ri = envs.ri;
    double fac1i, fac1j;
    double aij, dij, eij;
    int ip, jp;
    int empty;
    constexpr int nfi = (I + 1) * (I + 2) / 2;
    constexpr int nfj = (J + 1) * (J + 2) / 2;
    constexpr int nf = nfi * nfj;
    constexpr size_t len1 = bufsize(I, J);
    constexpr size_t lenj = len1;
    cuDoubleComplex gctr[len1 + lenj];
    cuDoubleComplex *g = gctr + lenj;
    cuDoubleComplex *g1d = g;

    constexpr int offset_g1d = _CUM_LEN_CART[I] - _LEN_CART[I];
    double rrij = square_dist(ri, rj);
    double fac1 = SQRTPI * M_PI * CINTcommon_fac_sp(I) * CINTcommon_fac_sp(J);

    empty = 1;
    for (jp = 0; jp < j_prim; jp++) {
        fac1j = fac1 * cj[jp];
        for (ip = 0; ip < i_prim; ip++) {
            aij = ai[ip] + aj[jp];
            eij = (ai[ip] * aj[jp] / aij) * rrij;
            if (eij <= EXP_CUTOFF) {
                dij = exp(-eij) / (aij * sqrt(aij));
                fac1i = fac1j * dij;
                aopair_rr_igtj_early<I, J, Fz>(g, ai[ip], aj[jp], &envs, rj,
                    cuCmul(make_cuDoubleComplex(fac1i, 0.0), fac), Gv, b, gxyz,
                    gs, NGv, iGv);
                prim_to_ctr<I, J>(gctr, g1d + offset_g1d, ci[ip], empty);
                empty = 0;
            }
        }
    }
    if (!empty) {
        vrr2d_withGv<I, J>(buf, gctr, gctr + lenj, ri, rj);
    } else {
#pragma unroll
        for (int i = 0; i < nf; i++)
            buf[i] = Z0;
    }
    return !empty;
}

template <int I, int J>
__host__ __device__ static void GTO_ft_init1e_envs(CINTEnvVars *envs,
    const int i_sh, const int j_sh, int *atm, const int natm, int *bas,
    const int nbas, double *env) {
    // envs->natm = natm;
    // envs->nbas = nbas;
    envs->atm = atm;
    envs->bas = bas;
    envs->env = env;
    // envs->shls = shls;

    envs->i_l = bas(ANG_OF, i_sh);
    envs->j_l = bas(ANG_OF, j_sh);
    assert(envs->i_l == I && envs->j_l == J);
    // envs->x_ctr[0] = bas(NCTR_OF, i_sh);
    // envs->x_ctr[1] = bas(NCTR_OF, j_sh);
    // envs->nfi = (envs->i_l + 1) * (envs->i_l + 2) / 2;
    // envs->nfj = (envs->j_l + 1) * (envs->j_l + 2) / 2;
    // envs->nf = envs->nfi * envs->nfj;
    // envs->common_factor = 1;

    // envs->gbits = ng[GSHIFT];
    // envs->ncomp_e1 = ng[POS_E1];
    // envs->ncomp_tensor = ng[TENSOR];

    // envs->li_ceil = envs->i_l + ng[IINC];
    // envs->lj_ceil = envs->j_l + ng[JINC];
    // if (ng[RYS_ROOTS] > 0)
    // {
    //     envs->nrys_roots = ng[RYS_ROOTS];
    // }
    // else
    // {
    //     envs->nrys_roots = (envs->li_ceil + envs->lj_ceil) / 2 + 1;
    // }

    envs->ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
    envs->rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));

    // int dli, dlj;
    // if (envs->li_ceil < envs->lj_ceil)
    // {
    //     dli = envs->li_ceil + 1;
    //     dlj = envs->li_ceil + envs->lj_ceil + 1;
    // }
    // else
    // {
    //     dli = envs->li_ceil + envs->lj_ceil + 1;
    //     dlj = envs->lj_ceil + 1;
    // }
    // envs->g_stride_i = 1;
    // envs->g_stride_j = dli;
    // envs->g_size = dli * dlj;

    // envs->lk_ceil = 1;
    // envs->ll_ceil = 1;
    // envs->g_stride_k = 0;
    // envs->g_stride_l = 0;
}

// Lastsum (no bvk)
#pragma region
template <bool prim, int I, int J, int Fz>
__host__ __device__ void GTO_ft_aopair_drv_lastsum(cuDoubleComplex *out,
    const int i_sh, const int j_sh, const int2 dims,
    const cuDoubleComplex *expkL, const double *Gv, const double *b,
    const int *gxyz, const int3 &gs, const size_t NGv, const size_t iGv,
    const CINTEnvVars &envs, const double *Ls, const int nimgs, const int nkpts,
    const size_t nijg, const size_t naoj) {
    const int *atm = envs.atm;
    const int *bas = envs.bas;
    const double *env = envs.env;
    constexpr int nfi = (I + 1) * (I + 2) / 2;
    constexpr int nfj = (J + 1) * (J + 2) / 2;
    constexpr int nf = nfi * nfj;
    cuDoubleComplex *pbuf;
    int empty = 1;
    for (int jL = 0; jL < nimgs; jL++) {
        double rj[3];
        int ptr = atm[PTR_COORD + bas[ATOM_OF + j_sh * BAS_SLOTS] * ATM_SLOTS];
        rj[0] = env[ptr + 0] + Ls[jL * 3 + 0];
        rj[1] = env[ptr + 1] + Ls[jL * 3 + 1];
        rj[2] = env[ptr + 2] + Ls[jL * 3 + 2];

        const int i_prim = bas(NPRIM_OF, i_sh);
        const int j_prim = bas(NPRIM_OF, j_sh);
        const double *ai = env + bas(PTR_EXP, i_sh);
        const double *aj = env + bas(PTR_EXP, j_sh);
        const double *ci = env + bas(PTR_COEFF, i_sh);
        const double *cj = env + bas(PTR_COEFF, j_sh);

        cuDoubleComplex buf_ij[nf];
        cuDoubleComplex expk;
        if (prim) {
            GTO_ft_aopair_lazy_ker<I, J, Fz>(buf_ij, envs, rj, Gv, b, gxyz, gs,
                NGv, iGv, dims, i_prim, j_prim, i_sh, j_sh, ai, aj, ci, cj, Z1);
        } else {
            GTO_ft_aopair_early_ker<I, J, Fz>(buf_ij, envs, rj, Gv, b, gxyz, gs,
                NGv, iGv, dims, i_prim, j_prim, i_sh, j_sh, ai, aj, ci, cj, Z1);
        }
        int i_bvk = jL;
        if (!empty) {
            for (int k = 0; k < nkpts; k++) {
                expk = expkL[k * nimgs + i_bvk];
#pragma unroll
                for (int j = 0; j < nfj; j++) {
#pragma unroll
                    for (int i = 0; i < nfi; i++) {
                        pbuf = out + k * nijg + (i * naoj + j) * NGv + iGv;
                        pbuf[0] = cuCadd(
                            pbuf[0], cuCmul(buf_ij[(i + nfi * j)], expk));
                    }
                }
            }
        } else {
            for (int k = 0; k < nkpts; k++) {
                expk = expkL[k * nimgs + i_bvk];
#pragma unroll
                for (int j = 0; j < nfj; j++) {
#pragma unroll
                    for (int i = 0; i < nfi; i++) {
                        pbuf = out + k * nijg + (i * naoj + j) * NGv + iGv;
                        pbuf[0] = cuCmul(buf_ij[(i + nfi * j)], expk);
                    }
                }
            }
        }
        empty = 0;
    }
}

template <bool prim, int I, int J, int Fz>
__global__ void GTO_ft_ovlp_cart_lastsum(cuDoubleComplex *out, const int2 shlsi,
    const int2 shlsj, const cuDoubleComplex *expkL, const double *Gv,
    const double *b, const int *gxyz, const int3 gs, const int nGv, int *atm,
    const int natm, int *bas, const int nbas, double *env, const double *Ls,
    const int *ao_loc, const int nimgs, const int nkpts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ish0 = shlsi.x;
    size_t ish1 = shlsi.y;
    size_t jsh0 = shlsj.x;
    size_t jsh1 = shlsj.y;
    size_t nshi = ish1 - ish0;
    size_t nshj = jsh1 - jsh0;

    // extern __shared__ cuDoubleComplex s_expkL[];
    // __syncthreads();
    // for (int i = threadIdx.x; i < nkpts * bvk_nimgs; i += blockDim.x)
    //     s_expkL[i] = expkL[i];
    // __syncthreads();

    if (idx >= nGv * nshi * nshj)
        return;
    size_t iGv = idx % nGv;
    idx /= nGv;
    int ish = idx % nshi + ish0;
    idx /= nshi;
    int jsh = idx + jsh0;

    const size_t ao_loc_i0 = ao_loc[ish0];
    const size_t ao_loc_i = ao_loc[ish];
    const size_t ao_loc_j0 = ao_loc[jsh0];
    const size_t ao_loc_j = ao_loc[jsh];

    const size_t naoi = ao_loc[ish1] - ao_loc_i0;
    const size_t naoj = ao_loc[jsh1] - ao_loc_j0;
    const size_t nijg = naoi * naoj * nGv;

    const int di = ao_loc[ish + 1] - ao_loc_i;
    const int dj = ao_loc[jsh + 1] - ao_loc_j;
    int2 dims = {di, dj};
    const int ip = ao_loc_i - ao_loc_i0;
    const int jp = ao_loc_j - ao_loc_j0;
    const size_t out_off = (ip * naoj + jp) * nGv;

    CINTEnvVars envs;
    // int ng[] = { 0, 0, 0, 0, 0, 1, 0, 1 };
    // int shls[2] = { ish, jsh };
    NVTX_PUSH("GTO_ft_init1e_envs");
    GTO_ft_init1e_envs<I, J>(&envs, ish, jsh, atm, natm, bas, nbas, env);
    NVTX_POP();
    // assert(envs->ncomp_e1 * envs->ncomp_tensor == 1 && envs->x_ctr[0] == 1 &&
    // envs->x_ctr[1] == 1 && envs->gbits == 0);
    NVTX_PUSH("GTO_ft_aopair_drv_lastsum");
    GTO_ft_aopair_drv_lastsum<prim, I, J, Fz>(out + out_off, ish, jsh, dims,
        expkL, Gv, b, gxyz, gs, nGv, iGv, envs, Ls, nimgs, nkpts, nijg, naoj);
    NVTX_POP();
}

template <bool prim, int I, int J, int Fz>
static void GINT_ft_latsum(cuDoubleComplex *out, const int nkpts,
    const int comp, const int nimgs, double *env_loc, const double *Ls,
    const cuDoubleComplex *expkL, const int *shls_slice, const int *ao_loc,
    const double *sGv, const double *b, const int *sgxyz, const int3 &gs,
    const int nGv, int *atm, const int natm, int *bas, const int nbas) {
    NVTX_PUSH("GINT_ft_latsum");
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];

    size_t nth = (jsh1 - jsh0);
    nth *= (ish1 - ish0);
    nth *= nGv;
    int2 shlsi = {ish0, ish1};
    int2 shlsj = {jsh0, jsh1};
    GTO_ft_ovlp_cart_lastsum<prim, I, J, Fz><<<(nth + 255) / 256, 256>>>(out,
        shlsi, shlsj, expkL, sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas,
        env_loc, Ls, ao_loc, nimgs, nkpts);
    NVTX_POP();
}

template <bool prim, int I, int J>
void _PBC_ft_latsum(const int eval_gz, cuDoubleComplex *out, const int nkpts,
    const int comp, const int nimgs, double *env_loc, const double *Ls,
    const cuDoubleComplex *expkL, const int *shls_slice, const int *ao_loc,
    const double *sGv, const double *b, const int *sgxyz, const int *gs,
    const int nGv, int *atm, const int natm, int *bas, const int nbas) {
    int3 gs_ = {gs[0], gs[1], gs[2]};
    if (eval_gz != 0) {
        assert(sgxyz != NULL);
    }
    if (eval_gz == 0)
        GINT_ft_latsum<prim, I, J, 0>(out, nkpts, comp, nimgs, env_loc, Ls,
            expkL, shls_slice, ao_loc, sGv, b, sgxyz, gs_, nGv, atm, natm, bas,
            nbas);
    else if (eval_gz == 1)
        GINT_ft_latsum<prim, I, J, 1>(out, nkpts, comp, nimgs, env_loc, Ls,
            expkL, shls_slice, ao_loc, sGv, b, sgxyz, gs_, nGv, atm, natm, bas,
            nbas);
    else if (eval_gz == 2)
        GINT_ft_latsum<prim, I, J, 2>(out, nkpts, comp, nimgs, env_loc, Ls,
            expkL, shls_slice, ao_loc, sGv, b, sgxyz, gs_, nGv, atm, natm, bas,
            nbas);
    else
        assert(false);
}
#pragma endregion

// Bvk
#pragma region
template <bool prim, int I, int J, int Fz>
__host__ __device__ void GTO_ft_aopair_drv_bvk(cuDoubleComplex *out,
    const int i_sh, const int j_sh, const int2 dims,
    const cuDoubleComplex *expkL, const int *cell_loc_bvk, const double *Gv,
    const double *b, const int *gxyz, const int3 &gs, const size_t NGv,
    const size_t iGv, const CINTEnvVars &envs, const double *Ls,
    const int8_t *ovlp_mask, const int nimgs, const int nkpts,
    const int bvk_nimgs, const size_t nijg, const size_t naoj) {
    const int *atm = envs.atm;
    const int *bas = envs.bas;
    const double *env = envs.env;
    constexpr int nfi = (I + 1) * (I + 2) / 2;
    constexpr int nfj = (J + 1) * (J + 2) / 2;
    constexpr int nf = nfi * nfj;
    cuDoubleComplex *pbuf;
    int empty = 1;
    for (int jL = 0; jL < nimgs; jL++) {
        if (!ovlp_mask[jL]) {
            if (empty)
                for (int k = 0; k < nkpts; k++)
#pragma unroll
                    for (int j = 0; j < nfj; j++)
#pragma unroll
                        for (int i = 0; i < nfi; i++) {
                            pbuf = out + k * nijg + (i * naoj + j) * NGv + iGv;
                            pbuf[0] = Z0;
                        }
            empty = 0;
            continue;
        }
        double rj[3];
        int ptr = atm[PTR_COORD + bas[ATOM_OF + j_sh * BAS_SLOTS] * ATM_SLOTS];
        rj[0] = env[ptr + 0] + Ls[jL * 3 + 0];
        rj[1] = env[ptr + 1] + Ls[jL * 3 + 1];
        rj[2] = env[ptr + 2] + Ls[jL * 3 + 2];

        const int i_prim = bas(NPRIM_OF, i_sh);
        const int j_prim = bas(NPRIM_OF, j_sh);
        const double *ai = env + bas(PTR_EXP, i_sh);
        const double *aj = env + bas(PTR_EXP, j_sh);
        const double *ci = env + bas(PTR_COEFF, i_sh);
        const double *cj = env + bas(PTR_COEFF, j_sh);

        cuDoubleComplex buf_ij[nf];
        cuDoubleComplex expk;
        if (prim) {
            GTO_ft_aopair_lazy_ker<I, J, Fz>(buf_ij, envs, rj, Gv, b, gxyz, gs,
                NGv, iGv, dims, i_prim, j_prim, i_sh, j_sh, ai, aj, ci, cj, Z1);
        } else {
            GTO_ft_aopair_early_ker<I, J, Fz>(buf_ij, envs, rj, Gv, b, gxyz, gs,
                NGv, iGv, dims, i_prim, j_prim, i_sh, j_sh, ai, aj, ci, cj, Z1);
        }
        int i_bvk = cell_loc_bvk[jL];
        if (!empty) {
            for (int k = 0; k < nkpts; k++) {
                expk = expkL[k * bvk_nimgs + i_bvk];
#pragma unroll
                for (int j = 0; j < nfj; j++) {
#pragma unroll
                    for (int i = 0; i < nfi; i++) {
                        pbuf = out + k * nijg + (i * naoj + j) * NGv + iGv;
                        pbuf[0] = cuCadd(
                            pbuf[0], cuCmul(buf_ij[(i + nfi * j)], expk));
                    }
                }
            }
        } else {
            for (int k = 0; k < nkpts; k++) {
                expk = expkL[k * bvk_nimgs + i_bvk];
#pragma unroll
                for (int j = 0; j < nfj; j++) {
#pragma unroll
                    for (int i = 0; i < nfi; i++) {
                        pbuf = out + k * nijg + (i * naoj + j) * NGv + iGv;
                        pbuf[0] = cuCmul(buf_ij[(i + nfi * j)], expk);
                    }
                }
            }
        }
        empty = 0;
    }
}

template <bool prim, int I, int J, int Fz>
__global__ void GTO_ft_ovlp_cart_bvk(cuDoubleComplex *out, const int2 shlsi,
    const int2 shlsj, const cuDoubleComplex *expkL, const int *cell_loc_bvk,
    const double *Gv, const double *b, const int *gxyz, const int3 gs,
    const int nGv, int *atm, const int natm, int *bas, const int nbas,
    double *env, const double *Ls, const int8_t *ovlp_mask, const int *ao_loc,
    const int nimgs, const int nkpts, const int bvk_nimgs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ish0 = shlsi.x;
    size_t ish1 = shlsi.y;
    size_t jsh0 = shlsj.x;
    size_t jsh1 = shlsj.y;
    size_t nshi = ish1 - ish0;
    size_t nshj = jsh1 - jsh0;

    // extern __shared__ cuDoubleComplex s_expkL[];
    // __syncthreads();
    // for (int i = threadIdx.x; i < nkpts * bvk_nimgs; i += blockDim.x)
    //     s_expkL[i] = expkL[i];
    // __syncthreads();

    if (idx >= nGv * nshi * nshj)
        return;
    size_t iGv = idx % nGv;
    idx /= nGv;
    int ish = idx % nshi + ish0;
    idx /= nshi;
    int jsh = idx + jsh0;

    const size_t ao_loc_i0 = ao_loc[ish0];
    const size_t ao_loc_i = ao_loc[ish];
    const size_t ao_loc_j0 = ao_loc[jsh0];
    const size_t ao_loc_j = ao_loc[jsh];

    const size_t naoi = ao_loc[ish1] - ao_loc_i0;
    const size_t naoj = ao_loc[jsh1] - ao_loc_j0;
    const size_t nijg = naoi * naoj * nGv;
    const size_t ovlp_mask_off = (ish * nbas + jsh - nbas) * nimgs;

    const int di = ao_loc[ish + 1] - ao_loc_i;
    const int dj = ao_loc[jsh + 1] - ao_loc_j;
    int2 dims = {di, dj};
    const int ip = ao_loc_i - ao_loc_i0;
    const int jp = ao_loc_j - ao_loc_j0;
    const size_t out_off = (ip * naoj + jp) * nGv;

    CINTEnvVars envs;
    // int ng[] = { 0, 0, 0, 0, 0, 1, 0, 1 };
    // int shls[2] = { ish, jsh };
    NVTX_PUSH("GTO_ft_init1e_envs");
    GTO_ft_init1e_envs<I, J>(&envs, ish, jsh, atm, natm, bas, nbas, env);
    NVTX_POP();
    // assert(envs->ncomp_e1 * envs->ncomp_tensor == 1 && envs->x_ctr[0] == 1 &&
    // envs->x_ctr[1] == 1 && envs->gbits == 0);
    NVTX_PUSH("GTO_ft_aopair_drv_bvk");
    GTO_ft_aopair_drv_bvk<prim, I, J, Fz>(out + out_off, ish, jsh, dims, expkL,
        cell_loc_bvk, Gv, b, gxyz, gs, nGv, iGv, envs, Ls,
        ovlp_mask + ovlp_mask_off, nimgs, nkpts, bvk_nimgs, nijg, naoj);
    NVTX_POP();
}

template <bool prim, int I, int J, int Fz>
static void GINT_ft_bvk(cuDoubleComplex *out, const int nkpts, const int comp,
    const int nimgs, const int bvk_nimgs, const int *cell_loc_bvk,
    const int8_t *ovlp_mask, double *env_loc, const double *Ls,
    const cuDoubleComplex *expkL, const int *shls_slice, const int *ao_loc,
    const double *sGv, const double *b, const int *sgxyz, const int3 &gs,
    const int nGv, int *atm, const int natm, int *bas, const int nbas) {
    NVTX_PUSH("GINT_ft_bvk");
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];

    size_t nth = (jsh1 - jsh0);
    nth *= (ish1 - ish0);
    nth *= nGv;
    int2 shlsi = {ish0, ish1};
    int2 shlsj = {jsh0, jsh1};
    GTO_ft_ovlp_cart_bvk<prim, I, J, Fz><<<(nth + 255) / 256, 256>>>(out, shlsi,
        shlsj, expkL, cell_loc_bvk, sGv, b, sgxyz, gs, nGv, atm, natm, bas,
        nbas, env_loc, Ls, ovlp_mask, ao_loc, nimgs, nkpts, bvk_nimgs);
    NVTX_POP();
}

template <bool prim, int I, int J>
void _PBC_ft_bvk(const int eval_gz, cuDoubleComplex *out, const int nkpts,
    const int comp, const int nimgs, const int bvk_nimgs,
    const int *cell_loc_bvk, const int8_t *ovlp_mask, double *env_loc,
    const double *Ls, const cuDoubleComplex *expkL, const int *shls_slice,
    const int *ao_loc, const double *sGv, const double *b, const int *sgxyz,
    const int *gs, const int nGv, int *atm, const int natm, int *bas,
    const int nbas) {
    int3 gs_ = {gs[0], gs[1], gs[2]};
    if (eval_gz != 0) {
        assert(sgxyz != NULL);
    }
    if (eval_gz == 0)
        GINT_ft_bvk<prim, I, J, 0>(out, nkpts, comp, nimgs, bvk_nimgs,
            cell_loc_bvk, ovlp_mask, env_loc, Ls, expkL, shls_slice, ao_loc,
            sGv, b, sgxyz, gs_, nGv, atm, natm, bas, nbas);
    else if (eval_gz == 1)
        GINT_ft_bvk<prim, I, J, 1>(out, nkpts, comp, nimgs, bvk_nimgs,
            cell_loc_bvk, ovlp_mask, env_loc, Ls, expkL, shls_slice, ao_loc,
            sGv, b, sgxyz, gs_, nGv, atm, natm, bas, nbas);
    else if (eval_gz == 2)
        GINT_ft_bvk<prim, I, J, 2>(out, nkpts, comp, nimgs, bvk_nimgs,
            cell_loc_bvk, ovlp_mask, env_loc, Ls, expkL, shls_slice, ao_loc,
            sGv, b, sgxyz, gs_, nGv, atm, natm, bas, nbas);
    else
        assert(false);
}
#pragma endregion
