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

// 108 * 2048 * 50% / 32 = 3456
#define BLOCKNUM 3456L
#define WARPSIZE 32
#define GRIDSIZE (BLOCKNUM * WARPSIZE)
#define MOD_MINU(x, d) ((x) >= (d) ? ((x) - (d)) : (x))
#define MOD_PLUS(x, d) ((x) < 0 ? ((x) + (d)) : (x))

template <int I, int J, int K, int L>
inline __device__ void GINTfill_int2e(ERITensor &eri, const GINTEnvVars &envs,
    const BasisProdCache &c_bpcache_ij, // auxbpcache
    const int *ao_loc_kl, const int primitive_ij, const double *env,
    const int *bas, const double diag_fac, const int ish, const int ksh,
    const int lsh, const double *rk, const double *rl,
    const int16_t *c_idx4c = NULL, const int igroup = 0, size_t index = 0) {
    constexpr int NROOTS = (I + J + K + L) / 2 + 1;
    constexpr int nfi = (I + 2) * (I + 1) / 2;
    constexpr int nfk = (K + 2) * (K + 1) / 2;
    constexpr int nfl = (L + 2) * (L + 1) / 2;
    constexpr int nf = nfi * nfk * nfl;

    const int *ao_loc_ij = c_bpcache_ij.ao_loc;

    double g[3 * (I + 1) * (K + 1) * (L + 1)];
    double gout[nf];
    for (int i = 0; i < nf; i++) {
        gout[i] = 0;
    }

    size_t const nprim_ij = envs.nprim_ij;
    size_t const nprim_kl = envs.nprim_kl;

    const int nbas_ij = c_bpcache_ij.nbas;
    const double *__restrict__ bas_x_ij = c_bpcache_ij.bas_coords;
    const double *__restrict__ bas_y_ij = bas_x_ij + nbas_ij;
    const double *__restrict__ bas_z_ij = bas_y_ij + nbas_ij;

    const double *ak, *al, *ck, *cl;
    ak = env + bas[PTR_EXP + ksh * BAS_SLOTS]; // offset
    al = env + bas[PTR_EXP + lsh * BAS_SLOTS];
    ck = env + bas[PTR_COEFF + ksh * BAS_SLOTS];
    cl = env + bas[PTR_COEFF + lsh * BAS_SLOTS];

    const double *__restrict__ a12_ij = c_bpcache_ij.a12;
    const double *__restrict__ e12_ij = c_bpcache_ij.e12;
    const double *__restrict__ x12_ij = c_bpcache_ij.x12;
    const double *__restrict__ y12_ij = c_bpcache_ij.y12;
    const double *__restrict__ z12_ij = c_bpcache_ij.z12;

    const int npl = bas[NPRIM_OF + lsh * BAS_SLOTS];
    const int lk = bas[ANG_OF + ksh * BAS_SLOTS];
    const int ll = bas[ANG_OF + lsh * BAS_SLOTS];

    const double xi = bas_x_ij[ish];
    const double yi = bas_y_ij[ish];
    const double zi = bas_z_ij[ish];
    const double xk = rk[0];
    const double yk = rk[1];
    const double zk = rk[2];
    const double xl = rl[0];
    const double yl = rl[1];
    const double zl = rl[2];

    const size_t prim_ij0 = primitive_ij + ish * nprim_ij;
    const size_t prim_ij1 = prim_ij0 + nprim_ij;

    const double omega2 = envs.omega * envs.omega;
    double norm = CINTcommon_fac_sp(lk) * CINTcommon_fac_sp(ll);
    if (ksh == lsh) {
        norm *= diag_fac;
    }
    const double dkl2 = square_dist(rk, rl);
    for (int ij = prim_ij0; ij < prim_ij1; ++ij) {
        const double aij = a12_ij[ij];
        const double eij = e12_ij[ij];
        const double xij = x12_ij[ij];
        const double yij = y12_ij[ij];
        const double zij = z12_ij[ij];
        for (int kl = 0; kl < nprim_kl; ++kl) {
            int kp = kl / npl;
            int lp = kl % npl;

            const double akl = ak[kp] + al[lp];
            const double ekl =
                norm * ck[kp] * cl[lp] * exp(-dkl2 * ak[kp] * al[lp] / akl);
            const double xkl = (ak[kp] * rk[0] + al[lp] * rl[0]) / akl;
            const double ykl = (ak[kp] * rk[1] + al[lp] * rl[1]) / akl;
            const double zkl = (ak[kp] * rk[2] + al[lp] * rl[2]) / akl;

            const double xijxkl = xij - xkl;
            const double yijykl = yij - ykl;
            const double zijzkl = zij - zkl;

            const double aijkl = aij + akl;
            const double a1 = aij * akl;
#pragma unroll
            for (int lr = 0; lr < 2; lr++) {
                double a0 = a1 / aijkl;
                double theta;
                if (lr) {
                    theta = omega2 / (omega2 + a0);
                    a0 *= theta;
                }

                const double x =
                    a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
                const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));

                double rw[NROOTS * 2];
                GINTrys_root<NROOTS>(x, rw);
                for (int irys = 0; irys < NROOTS; ++irys) {
                    const double u = rw[irys];
                    const double w = rw[irys + NROOTS];
                    if (lr)
                        rw[irys] /= rw[irys] + 1 - rw[irys] * theta;
                    double xkxl, ykyl, zkzl;
                    constexpr int nmax = I;
                    int mmax = L + K;
                    constexpr int klmin = K > L ? L : K;
                    constexpr int dm = I + 1;
                    constexpr int dl = (I + 1) * (K > L ? (K + 1) : (L + 1));
                    constexpr int dij = I + 1;

                    double *__restrict__ gx = g;
                    double *__restrict__ gy = g + (I + 1) * (L + 1) * (K + 1);
                    double *__restrict__ gz =
                        g + (I + 1) * (L + 1) * (K + 1) * 2;

                    int j, l, m, n, off;
                    double tmpb0;
                    double s0x, s1x, s2x, t0x, t1x;
                    double s0y, s1y, s2y, t0y, t1y;
                    double s0z, s1z, s2z, t0z, t1z;
                    double u2, tmp1, tmp2, tmp3, tmp4;
                    double b00, b10, b01, c00x, c00y, c00z, c0px, c0py, c0pz;

                    gx[0] = envs.fac;
                    gy[0] = fac;
                    gz[0] = rw[irys + NROOTS];

                    u2 = a0 * rw[irys];
                    tmp4 = .5 / (u2 * aijkl + a1);
                    b00 = u2 * tmp4;
                    tmp1 = 2 * b00;
                    tmp2 = tmp1 * akl;
                    b10 = b00 + tmp4 * akl;
                    c00x = xij - xi - tmp2 * xijxkl;
                    c00y = yij - yi - tmp2 * yijykl;
                    c00z = zij - zi - tmp2 * zijzkl;

                    // (0,0,0) -> (i,0,0)
                    if (nmax > 0) {
                        s0x = gx[0];
                        s0y = gy[0];
                        s0z = gz[0];
                        s1x = c00x * s0x;
                        s1y = c00y * s0y;
                        s1z = c00z * s0z;
                        gx[1] = s1x;
                        gy[1] = s1y;
                        gz[1] = s1z;
#pragma unroll
                        for (n = 1; n < nmax; ++n) {
                            s2x = c00x * s1x + n * b10 * s0x;
                            s2y = c00y * s1y + n * b10 * s0y;
                            s2z = c00z * s1z + n * b10 * s0z;
                            gx[n + 1] = s2x;
                            gy[n + 1] = s2y;
                            gz[n + 1] = s2z;
                            s0x = s1x;
                            s0y = s1y;
                            s0z = s1z;
                            s1x = s2x;
                            s1y = s2y;
                            s1z = s2z;
                        }
                    }

                    if (mmax > 0) {
                        tmp3 = tmp1 * aij;
                        b01 = b00 + tmp4 * aij;
                        c0px = xkl - xk + tmp3 * xijxkl;
                        c0py = ykl - yk + tmp3 * yijykl;
                        c0pz = zkl - zk + tmp3 * zijzkl;
                        s0x = gx[0];
                        s0y = gy[0];
                        s0z = gz[0];
                        s1x = c0px * s0x;
                        s1y = c0py * s0y;
                        s1z = c0pz * s0z;
                        gx[dm] = s1x;
                        gy[dm] = s1y;
                        gz[dm] = s1z;

// (0,0,0) -> (0,k,0)
#pragma unroll
                        for (m = 1; m < mmax; ++m) {
                            s2x = c0px * s1x + m * b01 * s0x;
                            s2y = c0py * s1y + m * b01 * s0y;
                            s2z = c0pz * s1z + m * b01 * s0z;
                            gx[(m + 1) * dm] = s2x;
                            gy[(m + 1) * dm] = s2y;
                            gz[(m + 1) * dm] = s2z;
                            s0x = s1x;
                            s0y = s1y;
                            s0z = s1z;
                            s1x = s2x;
                            s1y = s2y;
                            s1z = s2z;
                        }

                        if (nmax > 0) {
                            s0x = gx[1];
                            s0y = gy[1];
                            s0z = gz[1];
                            s1x = c0px * s0x + b00 * gx[0];
                            s1y = c0py * s0y + b00 * gy[0];
                            s1z = c0pz * s0z + b00 * gz[0];
                            gx[1 + dm] = s1x;
                            gy[1 + dm] = s1y;
                            gz[1 + dm] = s1z;
#pragma unroll
                            for (m = 1; m < mmax; ++m) {
                                s2x = c0px * s1x + m * b01 * s0x +
                                      b00 * gx[m * dm];
                                s2y = c0py * s1y + m * b01 * s0y +
                                      b00 * gy[m * dm];
                                s2z = c0pz * s1z + m * b01 * s0z +
                                      b00 * gz[m * dm];
                                gx[1 + (m + 1) * dm] = s2x;
                                gy[1 + (m + 1) * dm] = s2y;
                                gz[1 + (m + 1) * dm] = s2z;
                                s0x = s1x;
                                s0y = s1y;
                                s0z = s1z;
                                s1x = s2x;
                                s1y = s2y;
                                s1z = s2z;
                            }
                        }

#pragma unroll
                        for (m = 1; m <= mmax; ++m) {
                            off = m * dm;
                            j = off;
                            s0x = gx[j];
                            s0y = gy[j];
                            s0z = gz[j];
                            s1x = gx[j + 1];
                            s1y = gy[j + 1];
                            s1z = gz[j + 1];
                            tmpb0 = m * b00;
#pragma unroll
                            for (n = 1; n < nmax; ++n) {
                                s2x = c00x * s1x + n * b10 * s0x +
                                      tmpb0 * gx[j + n * 1 - dm];
                                s2y = c00y * s1y + n * b10 * s0y +
                                      tmpb0 * gy[j + n * 1 - dm];
                                s2z = c00z * s1z + n * b10 * s0z +
                                      tmpb0 * gz[j + n * 1 - dm];
                                gx[j + (n + 1) * 1] = s2x;
                                gy[j + (n + 1) * 1] = s2y;
                                gz[j + (n + 1) * 1] = s2z;
                                s0x = s1x;
                                s0y = s1y;
                                s0z = s1z;
                                s1x = s2x;
                                s1y = s2y;
                                s1z = s2z;
                            }
                        }
                    }

                    if (klmin > 0) {
                        xkxl = xk - xl;
                        ykyl = yk - yl;
                        zkzl = zk - zl;

// unrolling l
#pragma unroll
                        for (l = 0; l < klmin - 1; l += 2, mmax -= 2) {
                            off = l * dl;
#pragma unroll
                            for (n = off; n < off + dij; ++n) {
                                s0x = gx[n + mmax * dm - dm];
                                s0y = gy[n + mmax * dm - dm];
                                s0z = gz[n + mmax * dm - dm];
                                t1x = xkxl * s0x + gx[n + mmax * dm];
                                t1y = ykyl * s0y + gy[n + mmax * dm];
                                t1z = zkzl * s0z + gz[n + mmax * dm];
                                gx[dl + n + mmax * dm - dm] = t1x;
                                gy[dl + n + mmax * dm - dm] = t1y;
                                gz[dl + n + mmax * dm - dm] = t1z;
                                s1x = s0x;
                                s1y = s0y;
                                s1z = s0z;
#pragma unroll
                                for (int k = mmax - 2; k >= 0; k--) {
                                    s0x = gx[n + k * dm];
                                    s0y = gy[n + k * dm];
                                    s0z = gz[n + k * dm];
                                    t0x = xkxl * s0x + s1x;
                                    t0y = ykyl * s0y + s1y;
                                    t0z = zkzl * s0z + s1z;
                                    gx[dl + n + k * dm] = t0x;
                                    gy[dl + n + k * dm] = t0y;
                                    gz[dl + n + k * dm] = t0z;
                                    gx[dl + dl + n + k * dm] = xkxl * t0x + t1x;
                                    gy[dl + dl + n + k * dm] = ykyl * t0y + t1y;
                                    gz[dl + dl + n + k * dm] = zkzl * t0z + t1z;
                                    s1x = s0x;
                                    s1y = s0y;
                                    s1z = s0z;
                                    t1x = t0x;
                                    t1y = t0y;
                                    t1z = t0z;
                                }
                            }
                        }

                        if (l < klmin) {
                            off = l * dl;
#pragma unroll
                            for (n = off; n < off + dij; ++n) {
                                s1x = gx[n + mmax * dm];
                                s1y = gy[n + mmax * dm];
                                s1z = gz[n + mmax * dm];
#pragma unroll
                                for (int k = mmax - 1; k >= 0; k--) {
                                    s0x = gx[n + k * dm];
                                    s0y = gy[n + k * dm];
                                    s0z = gz[n + k * dm];
                                    gx[dl + n + k * dm] = xkxl * s0x + s1x;
                                    gy[dl + n + k * dm] = ykyl * s0y + s1y;
                                    gz[dl + n + k * dm] = zkzl * s0z + s1z;
                                    s1x = s0x;
                                    s1y = s0y;
                                    s1z = s0z;
                                }
                            }
                        }
                    }
                    GINTwrite_ints_s1<I, J, K, L>(gout, g, 1 - 2 * lr);
                }
            }
        }
    }
    GINTwrite_ints_s1<I, J, K, L>(
        eri, gout, ish, ksh, lsh, ao_loc_ij, ao_loc_kl);
}

template <int NROOTS, int UGSIZE>
inline __device__ void GINTfill_int2e(ERITensor &eri, const GINTEnvVars &envs,
    const BasisProdCache &c_bpcache_ij, // auxbpcache
    const int *ao_loc_kl, const int primitive_ij, const double *env,
    const int *bas, const double diag_fac, const int ish, const int ksh,
    const int lsh, const double *rk, const double *rl, const int16_t *c_idx4c,
    const int igroup, size_t index) {
    int *ao_loc_ij = c_bpcache_ij.ao_loc;

    double u, w;
    double g[UGSIZE];
    const size_t nprim_ij = envs.nprim_ij;
    const size_t nprim_kl = envs.nprim_kl;

    const int nbas_ij = c_bpcache_ij.nbas;
    const double *__restrict__ bas_x_ij = c_bpcache_ij.bas_coords;
    const double *__restrict__ bas_y_ij = bas_x_ij + nbas_ij;
    const double *__restrict__ bas_z_ij = bas_y_ij + nbas_ij;

    const double *ak, *al, *ck, *cl;
    ak = env + bas[PTR_EXP + ksh * BAS_SLOTS]; // offset
    al = env + bas[PTR_EXP + lsh * BAS_SLOTS];
    ck = env + bas[PTR_COEFF + ksh * BAS_SLOTS];
    cl = env + bas[PTR_COEFF + lsh * BAS_SLOTS];

    const double *__restrict__ a12_ij = c_bpcache_ij.a12;
    const double *__restrict__ e12_ij = c_bpcache_ij.e12;
    const double *__restrict__ x12_ij = c_bpcache_ij.x12;
    const double *__restrict__ y12_ij = c_bpcache_ij.y12;
    const double *__restrict__ z12_ij = c_bpcache_ij.z12;

    int npl = bas[NPRIM_OF + lsh * BAS_SLOTS];
    int lk = bas[ANG_OF + ksh * BAS_SLOTS];
    int ll = bas[ANG_OF + lsh * BAS_SLOTS];

    const double xi = bas_x_ij[ish];
    const double yi = bas_y_ij[ish];
    const double zi = bas_z_ij[ish];
    const double xk = rk[0];
    const double yk = rk[1];
    const double zk = rk[2];
    const double xl = rl[0];
    const double yl = rl[1];
    const double zl = rl[2];

    const size_t prim_ij0 = primitive_ij + ish * nprim_ij;

    const double omega2 = envs.omega * envs.omega;
    double norm = CINTcommon_fac_sp(lk) * CINTcommon_fac_sp(ll);
    if (ksh == lsh) {
        norm *= diag_fac;
    }
    const double dkl2 = square_dist(rk, rl);

    const int ij = index % nprim_ij + prim_ij0;
    index /= nprim_ij;
    const int kl = index % nprim_kl;
    index /= nprim_kl;
    const int iroot = index % NROOTS;
    const int lr = index / NROOTS;

    const double aij = a12_ij[ij];
    const double eij = e12_ij[ij];
    const double xij = x12_ij[ij];
    const double yij = y12_ij[ij];
    const double zij = z12_ij[ij];

    const int kp = kl / npl;
    const int lp = kl % npl;

    const double akl = ak[kp] + al[lp];
    const double ekl =
        norm * ck[kp] * cl[lp] * exp(-dkl2 * ak[kp] * al[lp] / akl);
    const double xkl = (ak[kp] * rk[0] + al[lp] * rl[0]) / akl;
    const double ykl = (ak[kp] * rk[1] + al[lp] * rl[1]) / akl;
    const double zkl = (ak[kp] * rk[2] + al[lp] * rl[2]) / akl;

    const double xijxkl = xij - xkl;
    const double yijykl = yij - ykl;
    const double zijzkl = zij - zkl;

    const double aijkl = aij + akl;
    const double a1 = aij * akl;

    double a0 = a1 / aijkl;
    double theta;
    if (lr) {
        theta = omega2 / (omega2 + a0);
        a0 *= theta;
    }

    const double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));

    GINTrys_root<NROOTS>(x, u, w, iroot);

    // double xixj, yiyj, zizj,
    double xkxl, ykyl, zkzl;
    const double xijxi = xij - xi;
    const double yijyi = yij - yi;
    const double zijzi = zij - zi;
    const double xklxk = xkl - xk;
    const double yklyk = ykl - yk;
    const double zklzk = zkl - zk;

    const int nmax = envs.i_l;
    int mmax = envs.k_l + envs.l_l;
    const int klmin = envs.klmin;
    const int dm = envs.stride_klmax / NROOTS; // (i_l+1) * (1)
    const int dn = 1;
    const int dl = envs.stride_klmin / NROOTS; // (i_l+1) * (1) * (k_l+1)
    const int dij = envs.g_size_ij / NROOTS;   // (i_l+1) * (1)

    double *__restrict__ gx = g;
    double *__restrict__ gy = g + envs.g_size / NROOTS;
    double *__restrict__ gz = g + envs.g_size / NROOTS * 2;

    // int i, k;
    int j, l, m, n, off;
    double tmpb0;
    double s0x, s1x, s2x, t0x, t1x;
    double s0y, s1y, s2y, t0y, t1y;
    double s0z, s1z, s2z, t0z, t1z;
    double u2, tmp1, tmp2, tmp3, tmp4;
    double b00, b10, b01, c00x, c00y, c00z, c0px, c0py, c0pz;
    if (lr) {
        /* u[:] = tau^2 / (1 - tau^2)
         * omega^2u^2 = a0 * tau^2 / (theta^-1 - tau^2)
         * transform u[:] to theta^-1 tau^2 / (theta^-1 - tau^2)
         * so the rest code can be reused.
         */
        u /= u + 1 - u * theta;
    }

    gx[0] = envs.fac;
    gy[0] = fac;
    gz[0] = w;

    u2 = a0 * u;
    tmp4 = .5 / (u2 * aijkl + a1);
    b00 = u2 * tmp4;
    tmp1 = 2 * b00;
    tmp2 = tmp1 * akl;
    b10 = b00 + tmp4 * akl;
    c00x = xijxi - tmp2 * xijxkl;
    c00y = yijyi - tmp2 * yijykl;
    c00z = zijzi - tmp2 * zijzkl;

    // (0,0,0) -> (i,0,0)
    if (nmax > 0) {
        s0x = gx[0];
        s0y = gy[0];
        s0z = gz[0];
        s1x = c00x * s0x;
        s1y = c00y * s0y;
        s1z = c00z * s0z;
        gx[dn] = s1x;
        gy[dn] = s1y;
        gz[dn] = s1z;
        for (n = 1; n < nmax; ++n) {
            s2x = c00x * s1x + n * b10 * s0x;
            s2y = c00y * s1y + n * b10 * s0y;
            s2z = c00z * s1z + n * b10 * s0z;
            gx[(n + 1) * dn] = s2x;
            gy[(n + 1) * dn] = s2y;
            gz[(n + 1) * dn] = s2z;
            s0x = s1x;
            s0y = s1y;
            s0z = s1z;
            s1x = s2x;
            s1y = s2y;
            s1z = s2z;
        }
    }

    if (mmax > 0) {
        tmp3 = tmp1 * aij;
        b01 = b00 + tmp4 * aij;
        c0px = xklxk + tmp3 * xijxkl;
        c0py = yklyk + tmp3 * yijykl;
        c0pz = zklzk + tmp3 * zijzkl;
        s0x = gx[0];
        s0y = gy[0];
        s0z = gz[0];
        s1x = c0px * s0x;
        s1y = c0py * s0y;
        s1z = c0pz * s0z;
        gx[dm] = s1x;
        gy[dm] = s1y;
        gz[dm] = s1z;
        // (0,0,0) -> (0,k,0)
        for (m = 1; m < mmax; ++m) {
            s2x = c0px * s1x + m * b01 * s0x;
            s2y = c0py * s1y + m * b01 * s0y;
            s2z = c0pz * s1z + m * b01 * s0z;
            gx[(m + 1) * dm] = s2x;
            gy[(m + 1) * dm] = s2y;
            gz[(m + 1) * dm] = s2z;
            s0x = s1x;
            s0y = s1y;
            s0z = s1z;
            s1x = s2x;
            s1y = s2y;
            s1z = s2z;
        }

        if (nmax > 0) {
            s0x = gx[dn];
            s0y = gy[dn];
            s0z = gz[dn];
            s1x = c0px * s0x + b00 * gx[0];
            s1y = c0py * s0y + b00 * gy[0];
            s1z = c0pz * s0z + b00 * gz[0];
            gx[dn + dm] = s1x;
            gy[dn + dm] = s1y;
            gz[dn + dm] = s1z;
            for (m = 1; m < mmax; ++m) {
                s2x = c0px * s1x + m * b01 * s0x + b00 * gx[m * dm];
                s2y = c0py * s1y + m * b01 * s0y + b00 * gy[m * dm];
                s2z = c0pz * s1z + m * b01 * s0z + b00 * gz[m * dm];
                gx[dn + (m + 1) * dm] = s2x;
                gy[dn + (m + 1) * dm] = s2y;
                gz[dn + (m + 1) * dm] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }

        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0x = gx[j];
            s0y = gy[j];
            s0z = gz[j];
            s1x = gx[j + dn];
            s1y = gy[j + dn];
            s1z = gz[j + dn];
            tmpb0 = m * b00;
            for (n = 1; n < nmax; ++n) {
                s2x = c00x * s1x + n * b10 * s0x + tmpb0 * gx[j + n * dn - dm];
                s2y = c00y * s1y + n * b10 * s0y + tmpb0 * gy[j + n * dn - dm];
                s2z = c00z * s1z + n * b10 * s0z + tmpb0 * gz[j + n * dn - dm];
                gx[j + (n + 1) * dn] = s2x;
                gy[j + (n + 1) * dn] = s2y;
                gz[j + (n + 1) * dn] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }
    }

    if (klmin > 0) {
        xkxl = xk - xl;
        ykyl = yk - yl;
        zkzl = zk - zl;

        // unrolling l
        for (l = 0; l < klmin - 1; l += 2, mmax -= 2) {
            off = l * dl;
            for (n = off; n < off + dij; ++n) {
                s0x = gx[n + mmax * dm - dm];
                s0y = gy[n + mmax * dm - dm];
                s0z = gz[n + mmax * dm - dm];
                t1x = xkxl * s0x + gx[n + mmax * dm];
                t1y = ykyl * s0y + gy[n + mmax * dm];
                t1z = zkzl * s0z + gz[n + mmax * dm];
                gx[dl + n + mmax * dm - dm] = t1x;
                gy[dl + n + mmax * dm - dm] = t1y;
                gz[dl + n + mmax * dm - dm] = t1z;
                s1x = s0x;
                s1y = s0y;
                s1z = s0z;
                for (int k = mmax - 2; k >= 0; k--) {
                    s0x = gx[n + k * dm];
                    s0y = gy[n + k * dm];
                    s0z = gz[n + k * dm];
                    t0x = xkxl * s0x + s1x;
                    t0y = ykyl * s0y + s1y;
                    t0z = zkzl * s0z + s1z;
                    gx[dl + n + k * dm] = t0x;
                    gy[dl + n + k * dm] = t0y;
                    gz[dl + n + k * dm] = t0z;
                    gx[dl + dl + n + k * dm] = xkxl * t0x + t1x;
                    gy[dl + dl + n + k * dm] = ykyl * t0y + t1y;
                    gz[dl + dl + n + k * dm] = zkzl * t0z + t1z;
                    s1x = s0x;
                    s1y = s0y;
                    s1z = s0z;
                    t1x = t0x;
                    t1y = t0y;
                    t1z = t0z;
                }
            }
        }

        if (l < klmin) {
            off = l * dl;
            for (n = off; n < off + dij; ++n) {
                s1x = gx[n + mmax * dm];
                s1y = gy[n + mmax * dm];
                s1z = gz[n + mmax * dm];
                for (int k = mmax - 1; k >= 0; k--) {
                    s0x = gx[n + k * dm];
                    s0y = gy[n + k * dm];
                    s0z = gz[n + k * dm];
                    gx[dl + n + k * dm] = xkxl * s0x + s1x;
                    gy[dl + n + k * dm] = ykyl * s0y + s1y;
                    gz[dl + n + k * dm] = zkzl * s0z + s1z;
                    s1x = s0x;
                    s1y = s0y;
                    s1z = s0z;
                }
            }
        }
    }
    GINTwrite_int3c_s1_g<NROOTS>(
        eri, g, ish, ksh, lsh, ao_loc_ij, ao_loc_kl, c_idx4c, !lr, igroup);
}

__device__ static void get_rc(double *rc, const double *ri, const double *rj,
    const double ei, const double ej) {
    const double eij = ei + ej;
    rc[0] = (ri[0] * ei + rj[0] * ej) / eij;
    rc[1] = (ri[1] * ei + rj[1] * ej) / eij;
    rc[2] = (ri[2] * ei + rj[2] * ej) / eij;
}

__device__ bool GINTprescreen_aexyz_ij_num(const int *d_refuniqshl_map,
    const double *d_uniq_dcut2s, const int *bas, const int *atm,
    const double *env, const double *Ls, const int prescreen_mask,
    const int ish, const int jsh, const int iL, const int jL) {
    if (prescreen_mask < 1)
        return true;
    const int iptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
    const int jptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
    const int Ish = d_refuniqshl_map[ish];
    const int Jsh = d_refuniqshl_map[jsh];
    const int IJsh = (Ish >= Jsh) ? (Ish * (Ish + 1) / 2 + Jsh)
                                  : (Jsh * (Jsh + 1) / 2 + Ish);
    const double dij2_cut = d_uniq_dcut2s[IJsh];
    double rij[6];
    rij[0] = env[iptrxyz + 0] + Ls[iL * 3 + 0];
    rij[1] = env[iptrxyz + 1] + Ls[iL * 3 + 1];
    rij[2] = env[iptrxyz + 2] + Ls[iL * 3 + 2];
    rij[3] = env[jptrxyz + 0] + Ls[jL * 3 + 0];
    rij[4] = env[jptrxyz + 1] + Ls[jL * 3 + 1];
    rij[5] = env[jptrxyz + 2] + Ls[jL * 3 + 2];
    const double dij2 = square_dist(rij, (rij + 3));
    return dij2 <= dij2_cut;
}

__global__ void GINTfill_prescreen_ij_num_kernel(unsigned long long &indij,
    const int pair_id0, const int pair_id1,
    const int *d_bas_pair2bra, const int *d_bas_pair2ket, const int iL0,
    const int iL1, const int jL0, const int jL1, const int *d_refuniqshl_map,
    const double *d_uniq_dcut2s, const int *d_bas, const int nbas,
    const int *d_atm, const double *d_env, const double *d_Ls,
    const int prescreen_mask) {
    __shared__ unsigned long long s_numij[32];
    const auto grid = cg::this_grid();
    const size_t rank = grid.thread_rank();
    const size_t gsize = grid.size();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);
    const int wid = warp.meta_group_rank();
    const int lane = warp.thread_rank();
    unsigned long long numij = 0;

    const size_t npair = pair_id1 - pair_id0;
    const size_t niL = iL1 - iL0;
    const size_t njL = jL1 - jL0;
    const size_t ntaskij = npair * niL * njL;
    const int nloopij = (ntaskij + gsize - 1) / gsize;

    for (int i = 0; i < nloopij; i++) {
        size_t idx = i * gsize + rank;
        if (idx < ntaskij) {
            const int jL = idx % njL + jL0;
            idx /= njL;
            const int iL = idx % niL + iL0;
            const int pair_id = idx / niL + pair_id0;
            const int ish = d_bas_pair2bra[pair_id];
            const int jsh = d_bas_pair2ket[pair_id];
            numij += GINTprescreen_aexyz_ij_num(d_refuniqshl_map, d_uniq_dcut2s,
                d_bas, d_atm, d_env, d_Ls, prescreen_mask, ish, jsh, iL, jL);
        }
    }
    warp.sync();
    const auto op = cg::plus<unsigned long long>();
    numij = cg::reduce(warp, numij, op);
    if (lane == 0)
        s_numij[wid] = numij;
    block.sync();
    if (wid == 0) {
        numij = s_numij[lane];
        numij = cg::reduce(warp, numij, op);
        if (lane == 0)
            atomicAdd(&indij, numij);
    }
}

__device__ bool GINTprescreen_aexyz_ij(const int *d_refuniqshl_map,
    const double *d_uniq_dcut2s, const int *bas, const int *atm,
    const double *env, const double *Ls, const int prescreen_mask,
    const int ish, const int jsh, const int iL, const int jL) {
    const int iptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
    const int jptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
    const int Ish = d_refuniqshl_map[ish];
    const int Jsh = d_refuniqshl_map[jsh];
    const int IJsh = (Ish >= Jsh) ? (Ish * (Ish + 1) / 2 + Jsh)
                                  : (Jsh * (Jsh + 1) / 2 + Ish);
    const double dij2_cut = d_uniq_dcut2s[IJsh];

    double rij[6];
    rij[0] = env[iptrxyz + 0] + Ls[iL * 3 + 0];
    rij[1] = env[iptrxyz + 1] + Ls[iL * 3 + 1];
    rij[2] = env[iptrxyz + 2] + Ls[iL * 3 + 2];
    rij[3] = env[jptrxyz + 0] + Ls[jL * 3 + 0];
    rij[4] = env[jptrxyz + 1] + Ls[jL * 3 + 1];
    rij[5] = env[jptrxyz + 2] + Ls[jL * 3 + 2];
    const double dij2 = square_dist(rij, (rij + 3));
    return dij2 <= dij2_cut || prescreen_mask < 1;
}

__global__ void GINTfill_prescreen_ij_kernel(unsigned long long &indij,
    size_t *idxij, const int pair_id0, const int pair_id1,
    const int *d_bas_pair2bra, const int *d_bas_pair2ket, const int iL0,
    const int iL1, const int jL0, const int jL1, const int *d_refuniqshl_map,
    const double *d_uniq_dcut2s, const int *d_bas, const int nbas,
    const int *d_atm, const double *d_env, const double *d_Ls,
    const int prescreen_mask) {
    constexpr int size = WARPSIZE + WARPSIZE;
    __shared__ size_t idxs[size];
    __shared__ int head, tail;
    size_t gind;
    head = 0;
    tail = 0;
    const auto grid = cg::this_grid();
    const size_t rank = grid.thread_rank();
    const size_t gsize = grid.size();
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int lane = warp.thread_rank();

    const size_t npair = pair_id1 - pair_id0;
    const size_t niL = iL1 - iL0;
    const size_t njL = jL1 - jL0;
    const size_t ntaskij = npair * niL * njL;
    const int nloopij = (ntaskij + gsize - 1) / gsize;

    for (int i = 0; i < nloopij; i++) {
        size_t idx = i * gsize + rank;
        if (idx < ntaskij) {
            const int jL = idx % njL + jL0;
            idx /= njL;
            const int iL = idx % niL + iL0;
            const int pair_id = idx / niL + pair_id0;
            const int ish = d_bas_pair2bra[pair_id];
            const int jsh = d_bas_pair2ket[pair_id];
            if (GINTprescreen_aexyz_ij(d_refuniqshl_map, d_uniq_dcut2s, d_bas,
                    d_atm, d_env, d_Ls, prescreen_mask, ish, jsh, iL, jL)) {
                const auto screen_threads = cg::coalesced_threads();
                const int ind =
                    MOD_MINU(tail + screen_threads.thread_rank(), size);
                idxs[ind] = i * gsize + rank;
                if (screen_threads.thread_rank() == 0) {
                    tail = MOD_MINU(tail + screen_threads.size(), size);
                }
            }
        }
        warp.sync();
        if (MOD_PLUS(tail - head, size) >= WARPSIZE) {
            const int ind = MOD_MINU(head + lane, size);
            if (lane == 0)
                gind = atomicAdd(&indij, WARPSIZE);
            gind = __shfl_sync(0xffffffff, gind, 0) + lane;
            idxij[gind] = idxs[ind];
            if (lane == 0) {
                head = MOD_MINU(head + WARPSIZE, size);
            }
            warp.sync();
        }
    }
    warp.sync();
    const int rem = MOD_PLUS(tail - head, size);
    if (lane == 0)
        gind = atomicAdd(&indij, rem);
    gind = __shfl_sync(0xffffffff, gind, 0) + lane;
    if (rem > 0) {
        if (lane < rem) {
            const int ind = MOD_MINU(head + lane, size);
            idxij[gind] = idxs[ind];
        }
    }
}

__device__ void GINTprescreen_aexyz_ij_cal(const int *d_refuniqshl_map,
    const int nbasauxuniq, const double *d_uniqexp, const double dcut_binsize,
    double *d_uniq_Rcut2s, const int *d_uniqshlpr_dij_loc, const int *bas,
    const int *atm, const double *env, const double *Ls, double *&uniq_Rcut2s_K,
    double *rc, double *rij, const int ish, const int jsh, const int iL,
    const int jL) {
    const int iptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
    const int jptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
    const int Ish = d_refuniqshl_map[ish];
    const int Jsh = d_refuniqshl_map[jsh];
    const int IJsh = (Ish >= Jsh) ? (Ish * (Ish + 1) / 2 + Jsh)
                                  : (Jsh * (Jsh + 1) / 2 + Ish);
    double ei = d_uniqexp[Ish];
    double ej = d_uniqexp[Jsh];
    rij[0] = env[iptrxyz + 0] + Ls[iL * 3 + 0];
    rij[1] = env[iptrxyz + 1] + Ls[iL * 3 + 1];
    rij[2] = env[iptrxyz + 2] + Ls[iL * 3 + 2];
    rij[3] = env[jptrxyz + 0] + Ls[jL * 3 + 0];
    rij[4] = env[jptrxyz + 1] + Ls[jL * 3 + 1];
    rij[5] = env[jptrxyz + 2] + Ls[jL * 3 + 2];
    get_rc(rc, rij, (rij + 3), ei, ej);

    const double dij2 = square_dist(rij, (rij + 3));
    const size_t idij = (size_t)(sqrt(dij2) / dcut_binsize);
    uniq_Rcut2s_K =
        d_uniq_Rcut2s + (d_uniqshlpr_dij_loc[IJsh] + idij) * nbasauxuniq;
}

__device__ bool GINTprescreen_aexyz_k(const int *d_auxuniqshl_map,
    const int *bas, const int nbas, const int *atm, const double *env,
    const int prescreen_mask, const double *uniq_Rcut2s_K, const double *rc,
    const int ksh) {
    if (prescreen_mask < 2)
        return true;
    const int Ksh = d_auxuniqshl_map[ksh];
    const double Rcut2 = uniq_Rcut2s_K[Ksh];
    const int kptrxyz =
        atm[PTR_COORD + bas[ATOM_OF + (ksh + nbas) * BAS_SLOTS] * ATM_SLOTS];
    const double *rk = (double *)(env + kptrxyz);
    const double Rijk2 = square_dist(rc, rk);
    return Rijk2 < Rcut2;
}

__global__ void GINTfill_prescreen_k_num_kernel(const size_t numij,
    size_t *numk, const size_t *idxij, const int pair_id0,
    const int *d_bas_pair2bra, const int *d_bas_pair2ket, const int iL0,
    const int iL1, const int jL0, const int jL1, const int ksh0, const int ksh1,
    const double *d_Ls, const int *d_refuniqshl_map, const int nbasauxuniq,
    const double *d_uniqexp, const double dcut_binsize, double *d_uniq_Rcut2s,
    const int *d_uniqshlpr_dij_loc, const int *d_auxuniqshl_map,
    const int *d_bas, const int nbas, const int *d_atm, const double *d_env,
    const int prescreen_mask) {
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int lane = warp.thread_rank();
    const auto op = cg::plus<int>();

    const size_t niL = iL1 - iL0;
    const size_t njL = jL1 - jL0;
    const auto bsize = gridDim.x;
    const int nloopij = (numij + bsize - 1) / bsize;
    const size_t nksh = ksh1 - ksh0;
    const int nloopk = (nksh + WARPSIZE - 1) / WARPSIZE;
    for (int i = 0; i < nloopij; i++) {
        const size_t ind = i * bsize + blockIdx.x;
        if (ind < numij) {
            size_t idx = idxij[ind];
            const int jL = idx % njL + jL0;
            idx /= njL;
            const int iL = idx % niL + iL0;
            const int pair_id = idx / niL + pair_id0;
            const int ish = d_bas_pair2bra[pair_id];
            const int jsh = d_bas_pair2ket[pair_id];

            double rc[3], rij[6];
            double *uniq_Rcut2s_K;
            GINTprescreen_aexyz_ij_cal(d_refuniqshl_map, nbasauxuniq, d_uniqexp,
                dcut_binsize, d_uniq_Rcut2s, d_uniqshlpr_dij_loc, d_bas, d_atm,
                d_env, d_Ls, uniq_Rcut2s_K, rc, rij, ish, jsh, iL, jL);
            int nk = 0;
            for (int j = 0; j < nloopk; j++) {
                const size_t ksh = j * WARPSIZE + lane + ksh0;
                if (ksh < ksh1)
                    nk += GINTprescreen_aexyz_k(d_auxuniqshl_map, d_bas, nbas,
                        d_atm, d_env, prescreen_mask, uniq_Rcut2s_K, rc, ksh);
            }
            nk = cg::reduce(warp, nk, op);
            if (lane == 0) {
                numk[ind] = nk;
            }
        }
    }
}

__global__ void GINTfill_prescreen_ijk_kernel(unsigned long long &indijk,
    const size_t numij, const size_t nijk, const size_t *idxij, int *shlijk,
    double *rij, const int pair_id0, const int *d_bas_pair2bra,
    const int *d_bas_pair2ket, const int iL0, const int iL1, const int jL0,
    const int jL1, const int ksh0, const int ksh1, const double *d_Ls,
    const int *d_refuniqshl_map, const int nbasauxuniq, const double *d_uniqexp,
    const double dcut_binsize, double *d_uniq_Rcut2s,
    const int *d_uniqshlpr_dij_loc, const int *d_auxuniqshl_map,
    const int *d_bas, const int nbas, const int *d_atm, const double *d_env,
    const int prescreen_mask) {
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int lane = warp.thread_rank();

    const int size = WARPSIZE + WARPSIZE;
    __shared__ int shls[size * 3];
    __shared__ double rs[size * 6];
    __shared__ int head, tail;
    head = 0;
    tail = 0;

    const size_t niL = iL1 - iL0;
    const size_t njL = jL1 - jL0;
    const auto bsize = gridDim.x;
    const int nloopij = (numij + bsize - 1) / bsize;
    const size_t nksh = ksh1 - ksh0;
    const int nloopk = (nksh + WARPSIZE - 1) / WARPSIZE;
    unsigned long long gind;

    for (int i = 0; i < nloopij; i++) {
        const size_t ind = i * bsize + blockIdx.x;
        if (ind < numij) {
            size_t idx = idxij[ind];
            const int jL = idx % njL + jL0;
            idx /= njL;
            const int iL = idx % niL + iL0;
            const int pair_id = idx / niL + pair_id0;
            const int ish = d_bas_pair2bra[pair_id];
            const int jsh = d_bas_pair2ket[pair_id];

            double rc[3];
            double r[6];
            double *uniq_Rcut2s_K;
            GINTprescreen_aexyz_ij_cal(d_refuniqshl_map, nbasauxuniq, d_uniqexp,
                dcut_binsize, d_uniq_Rcut2s, d_uniqshlpr_dij_loc, d_bas, d_atm,
                d_env, d_Ls, uniq_Rcut2s_K, rc, r, ish, jsh, iL, jL);
            for (int j = 0; j < nloopk; j++) {
                const size_t ksh = j * WARPSIZE + lane + ksh0;
                if (ksh < ksh1) {
                    if (GINTprescreen_aexyz_k(d_auxuniqshl_map, d_bas, nbas,
                            d_atm, d_env, prescreen_mask, uniq_Rcut2s_K, rc,
                            ksh)) {
                        const auto screen_threads = cg::coalesced_threads();
                        const int ind =
                            MOD_MINU(tail + screen_threads.thread_rank(), size);
                        shls[ind] = ish;
                        shls[ind + size] = jsh;
                        shls[ind + 2 * size] = ksh;
#pragma unroll
                        for (int p = 0; p < 6; p++) {
                            rs[ind + p * size] = r[p];
                        }
                        if (screen_threads.thread_rank() == 0) {
                            tail = MOD_MINU(tail + screen_threads.size(), size);
                        }
                    }
                }
                warp.sync();
                if (MOD_PLUS(tail - head, size) >= WARPSIZE) {
                    const int ind = MOD_MINU(head + lane, size);
                    if (lane == 0) {
                        gind = atomicAdd(&indijk, WARPSIZE);
                    }
                    gind = __shfl_sync(0xffffffff, gind, 0) + lane;
                    shlijk[gind] = shls[ind];
                    shlijk[gind + nijk] = shls[ind + size];
                    shlijk[gind + 2 * nijk] = shls[ind + 2 * size];
#pragma unroll
                    for (int p = 0; p < 6; p++) {
                        rij[gind + p * nijk] = rs[ind + p * size];
                    }
                    if (lane == 0) {
                        head = MOD_MINU(head + WARPSIZE, size);
                    }
                    warp.sync();
                }
            }
        }
    }
    warp.sync();
    const int rem = MOD_PLUS(tail - head, size);
    if (rem > 0) {
        if (lane == 0) {
            gind = atomicAdd(&indijk, rem);
        }
        gind = __shfl_sync(0xffffffff, gind, 0) + lane;
        if (lane < rem) {
            const int ind = MOD_MINU(head + lane, size);
            shlijk[gind] = shls[ind];
            shlijk[gind + nijk] = shls[ind + size];
            shlijk[gind + 2 * nijk] = shls[ind + 2 * size];
#pragma unroll
            for (int p = 0; p < 6; p++) {
                rij[gind + p * nijk] = rs[ind + p * size];
            }
        }
    }
}

template <int I, int J, int K, int L>
__global__ void GINTfill_prescreen_int2e_kernel(const GINTEnvVars envs,
    const size_t nijk, const int *shlijk, const double *rij,
    ERITensor eritensor, const BasisProdCache auxbpcache, const int *ao_loc,
    const int primitive_ij, const int nksh, const double diag_fac,
    const int *d_bas, const int nbas, const int *d_atm, const double *d_env,
    const int prescreen_mask, const int16_t *d_idx4c) {

    const auto grid = cg::this_grid();
    const auto gsize = grid.size();
    const auto rank = grid.thread_rank();

    for (size_t idx = rank; idx < nijk; idx += gsize) {
        const int ish = shlijk[idx];
        const int jsh = shlijk[idx + nijk];
        const int ksh = shlijk[idx + 2 * nijk];
        double r[6];
#pragma unroll
        for (int p = 0; p < 6; p++) {
            r[p] = rij[idx + p * nijk];
        }
        GINTfill_int2e<I, J, K, L>(eritensor, envs, auxbpcache, ao_loc,
            primitive_ij, d_env, d_bas, diag_fac, ksh, ish, jsh, r, r + 3,
            d_idx4c, 0, 0);
    }
}

template <int NROOTS, int UGSIZE>
__launch_bounds__(32, 32) __global__ void GINTfill_prescreen_int2e_kernel(
    const GINTEnvVars envs, const size_t nijk, const int *shlijk,
    const double *rij, ERITensor eritensor, const BasisProdCache auxbpcache,
    const int *ao_loc, const int primitive_ij, const int nksh,
    const double diag_fac, const int *d_bas, const int nbas, const int *d_atm,
    const double *d_env, const int prescreen_mask, const int16_t *d_idx4c) {
    const int nindex = NROOTS * 2 * envs.nprim_ij * envs.nprim_kl;

    const auto grid = cg::this_grid();
    const auto rank = grid.thread_rank();
    const size_t idx = rank / nindex;
    if (idx >= nijk)
        return;
    const int ish = shlijk[idx];
    const int jsh = shlijk[idx + nijk];
    const int ksh = shlijk[idx + 2 * nijk];
    double r[6];
#pragma unroll
    for (int p = 0; p < 6; p++) {
        r[p] = rij[idx + p * nijk];
    }
    GINTfill_int2e<NROOTS, UGSIZE>(eritensor, envs, auxbpcache, ao_loc,
        primitive_ij, d_env, d_bas, diag_fac, ksh, ish, jsh, r, r + 3, d_idx4c,
        nindex, rank % nindex);
}
