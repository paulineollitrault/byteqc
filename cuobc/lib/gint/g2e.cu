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
#include <math.h>
#include <assert.h>
#include "g2e.h"
#include "rys_roots.cuh"

template <int NROOTS>
__device__ void GINTg0_2e_2d4d(GINTEnvVars &envs, BasisProdCache &bpcache,
    double *__restrict__ g, double u, double w, double norm, int ish, int jsh,
    int ksh, int lsh, int prim_ij, int prim_kl, int iroot) {
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double aij = a12[prim_ij];
    double akl = a12[prim_kl];
    double eij = e12[prim_ij];
    double ekl = e12[prim_kl];
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double fac = eij * ekl / (sqrt(aijkl) * a1);

    int nbas = bpcache.nbas;
    int nmax = envs.i_l + envs.j_l;
    int mmax = envs.k_l + envs.l_l;
    int ijmin = envs.ijmin;
    int klmin = envs.klmin;
    int dm = envs.stride_klmax / NROOTS;
    int dn = envs.stride_ijmax / NROOTS;
    int di = envs.stride_ijmax / NROOTS;
    int dj = envs.stride_ijmin / NROOTS;
    int dk = envs.stride_klmax / NROOTS;
    int dl = envs.stride_klmin / NROOTS;
    int dij = envs.g_size_ij / NROOTS;
    int i, k;
    int j, l, m, n, off;
    double tmpb0;
    double u2, tmp1, tmp2, tmp3, tmp4;
    double b00, b10, b01;
    int tmax;

    u2 = a0 * u;
    tmp4 = .5 / (u2 * aijkl + a1);
    b00 = u2 * tmp4;
    tmp1 = 2 * b00;
    tmp2 = tmp1 * akl;
    b10 = b00 + tmp4 * akl;
    {
        double *__restrict__ x12 = bpcache.x12;
        double *__restrict__ gx = g;
        double xij = x12[prim_ij];
        double xkl = x12[prim_kl];
        double xijxkl = xij - xkl;
        double *__restrict__ bas_x = bpcache.bas_coords;
        double xixj, xkxl;
        double xi = bas_x[ish];
        double xk = bas_x[ksh];
        double xijxi = xij - xi;
        double xklxk = xkl - xk;
        double s0x, s1x, s2x, t0x, t1x;
        double c00x = xijxi - tmp2 * xijxkl;
        double c0px;
        gx[0] = norm;

        if (nmax > 0) {
            s0x = gx[0];
            s1x = c00x * s0x;
            gx[dn] = s1x;
            for (n = 1; n < nmax; ++n) {
                s2x = c00x * s1x + n * b10 * s0x;
                gx[(n + 1) * dn] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
        if (mmax > 0) {
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0px = xklxk + tmp3 * xijxkl;
            s0x = gx[0];
            s1x = c0px * s0x;
            gx[dm] = s1x;
            for (m = 1; m < mmax; ++m) {
                s2x = c0px * s1x + m * b01 * s0x;
                gx[(m + 1) * dm] = s2x;
                s0x = s1x;
                s1x = s2x;
            }

            if (nmax > 0) {
                s0x = gx[dn];
                s1x = c0px * s0x + b00 * gx[0];
                gx[dn + dm] = s1x;
                for (m = 1; m < mmax; ++m) {
                    s2x = c0px * s1x + m * b01 * s0x + b00 * gx[m * dm];
                    gx[dn + (m + 1) * dm] = s2x;
                    s0x = s1x;
                    s1x = s2x;
                }
            }
        }
        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0x = gx[j];
            s1x = gx[j + dn];
            tmpb0 = m * b00;
            for (n = 1; n < nmax; ++n) {
                s2x = c00x * s1x + n * b10 * s0x + tmpb0 * gx[j + n * dn - dm];
                gx[j + (n + 1) * dn] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
        if (ijmin > 0) {
            xixj = xi - bas_x[jsh];

            // unrolling j
            tmax = nmax;
            for (j = 0; j < ijmin - 1; j += 2, tmax -= 2) {
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s0x = gx[n + tmax * di - di];
                    t1x = xixj * s0x + gx[n + tmax * di];
                    gx[dj + n + tmax * di - di] = t1x;
                    s1x = s0x;
                    for (i = tmax - 2; i >= 0; i--) {
                        s0x = gx[n + i * di];
                        t0x = xixj * s0x + s1x;
                        gx[dj + n + i * di] = t0x;
                        gx[dj + dj + n + i * di] = xixj * t0x + t1x;
                        s1x = s0x;
                        t1x = t0x;
                    }
                }
            }

            if (j < ijmin) {
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s1x = gx[n + tmax * di];
                    for (i = tmax - 1; i >= 0; i--) {
                        s0x = gx[n + i * di];
                        gx[dj + n + i * di] = xixj * s0x + s1x;
                        s1x = s0x;
                    }
                }
            }
        }
        if (klmin > 0) {
            xkxl = xk - bas_x[lsh];

            tmax = mmax;
            // unrolling l
            for (l = 0; l < klmin - 1; l += 2, tmax -= 2) {
                off = l * dl;
                for (n = off; n < off + dij; ++n) {
                    s0x = gx[n + tmax * dk - dk];
                    t1x = xkxl * s0x + gx[n + tmax * dk];
                    gx[dl + n + tmax * dk - dk] = t1x;
                    s1x = s0x;
                    for (k = tmax - 2; k >= 0; k--) {
                        s0x = gx[n + k * dk];
                        t0x = xkxl * s0x + s1x;
                        gx[dl + n + k * dk] = t0x;
                        gx[dl + dl + n + k * dk] = xkxl * t0x + t1x;
                        s1x = s0x;
                        t1x = t0x;
                    }
                }
            }

            if (l < klmin) {
                off = l * dl;
                for (n = off; n < off + dij; ++n) {
                    s1x = gx[n + tmax * dk];
                    for (k = tmax - 1; k >= 0; k--) {
                        s0x = gx[n + k * dk];
                        gx[dl + n + k * dk] = xkxl * s0x + s1x;
                        s1x = s0x;
                    }
                }
            }
        }
    }
    {
        double *__restrict__ y12 = bpcache.y12;
        double *__restrict__ gy = g + envs.g_size / NROOTS;
        double yij = y12[prim_ij];
        double ykl = y12[prim_kl];
        double yijykl = yij - ykl;
        double *__restrict__ bas_y = bpcache.bas_coords + nbas;
        double yiyj, ykyl;
        double yi = bas_y[ish];
        double yk = bas_y[ksh];
        double yijyi = yij - yi;
        double yklyk = ykl - yk;
        double s0y, s1y, s2y, t0y, t1y;
        double c00y = yijyi - tmp2 * yijykl;
        double c0py;
        gy[0] = fac;

        if (nmax > 0) {
            s0y = gy[0];
            s1y = c00y * s0y;
            gy[dn] = s1y;
            for (n = 1; n < nmax; ++n) {
                s2y = c00y * s1y + n * b10 * s0y;
                gy[(n + 1) * dn] = s2y;
                s0y = s1y;
                s1y = s2y;
            }
        }
        if (mmax > 0) {
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0py = yklyk + tmp3 * yijykl;
            s0y = gy[0];
            s1y = c0py * s0y;
            gy[dm] = s1y;
            for (m = 1; m < mmax; ++m) {
                s2y = c0py * s1y + m * b01 * s0y;
                gy[(m + 1) * dm] = s2y;
                s0y = s1y;
                s1y = s2y;
            }

            if (nmax > 0) {
                s0y = gy[dn];
                s1y = c0py * s0y + b00 * gy[0];
                gy[dn + dm] = s1y;
                for (m = 1; m < mmax; ++m) {
                    s2y = c0py * s1y + m * b01 * s0y + b00 * gy[m * dm];
                    gy[dn + (m + 1) * dm] = s2y;
                    s0y = s1y;
                    s1y = s2y;
                }
            }
        }
        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0y = gy[j];
            s1y = gy[j + dn];
            tmpb0 = m * b00;
            for (n = 1; n < nmax; ++n) {
                s2y = c00y * s1y + n * b10 * s0y + tmpb0 * gy[j + n * dn - dm];
                gy[j + (n + 1) * dn] = s2y;
                s0y = s1y;
                s1y = s2y;
            }
        }
        if (ijmin > 0) {
            yiyj = yi - bas_y[jsh];

            tmax = nmax;
            // unrolling j
            for (j = 0; j < ijmin - 1; j += 2, tmax -= 2) {
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s0y = gy[n + tmax * di - di];
                    t1y = yiyj * s0y + gy[n + tmax * di];
                    gy[dj + n + tmax * di - di] = t1y;
                    s1y = s0y;
                    for (i = tmax - 2; i >= 0; i--) {
                        s0y = gy[n + i * di];
                        t0y = yiyj * s0y + s1y;
                        gy[dj + n + i * di] = t0y;
                        gy[dj + dj + n + i * di] = yiyj * t0y + t1y;
                        s1y = s0y;
                        t1y = t0y;
                    }
                }
            }

            if (j < ijmin) {
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s1y = gy[n + tmax * di];
                    for (i = tmax - 1; i >= 0; i--) {
                        s0y = gy[n + i * di];
                        gy[dj + n + i * di] = yiyj * s0y + s1y;
                        s1y = s0y;
                    }
                }
            }
        }
        if (klmin > 0) {
            ykyl = yk - bas_y[lsh];

            tmax = mmax;
            // unrolling l
            for (l = 0; l < klmin - 1; l += 2, tmax -= 2) {
                off = l * dl;
                for (n = off; n < off + dij; ++n) {
                    s0y = gy[n + tmax * dk - dk];
                    t1y = ykyl * s0y + gy[n + tmax * dk];
                    gy[dl + n + tmax * dk - dk] = t1y;
                    s1y = s0y;
                    for (k = tmax - 2; k >= 0; k--) {
                        s0y = gy[n + k * dk];
                        t0y = ykyl * s0y + s1y;
                        gy[dl + n + k * dk] = t0y;
                        gy[dl + dl + n + k * dk] = ykyl * t0y + t1y;
                        s1y = s0y;
                        t1y = t0y;
                    }
                }
            }

            if (l < klmin) {
                off = l * dl;
                for (n = off; n < off + dij; ++n) {
                    s1y = gy[n + tmax * dk];
                    for (k = tmax - 1; k >= 0; k--) {
                        s0y = gy[n + k * dk];
                        gy[dl + n + k * dk] = ykyl * s0y + s1y;
                        s1y = s0y;
                    }
                }
            }
        }
    }
    {
        double *__restrict__ z12 = bpcache.z12;
        double *__restrict__ gz = g + 2 * envs.g_size / NROOTS;
        double zij = z12[prim_ij];
        double zkl = z12[prim_kl];
        double zijzkl = zij - zkl;
        double *__restrict__ bas_z = bpcache.bas_coords + 2 * nbas;
        double zizj, zkzl;
        double zi = bas_z[ish];
        double zk = bas_z[ksh];
        double zijzi = zij - zi;
        double zklzk = zkl - zk;
        double s0z, s1z, s2z, t0z, t1z;
        double c00z = zijzi - tmp2 * zijzkl;
        double c0pz;
        gz[0] = w;

        if (nmax > 0) {
            s0z = gz[0];
            s1z = c00z * s0z;
            gz[dn] = s1z;
            for (n = 1; n < nmax; ++n) {
                s2z = c00z * s1z + n * b10 * s0z;
                gz[(n + 1) * dn] = s2z;
                s0z = s1z;
                s1z = s2z;
            }
        }
        if (mmax > 0) {
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0pz = zklzk + tmp3 * zijzkl;
            s0z = gz[0];
            s1z = c0pz * s0z;
            gz[dm] = s1z;
            for (m = 1; m < mmax; ++m) {
                s2z = c0pz * s1z + m * b01 * s0z;
                gz[(m + 1) * dm] = s2z;
                s0z = s1z;
                s1z = s2z;
            }

            if (nmax > 0) {
                s0z = gz[dn];
                s1z = c0pz * s0z + b00 * gz[0];
                gz[dn + dm] = s1z;
                for (m = 1; m < mmax; ++m) {
                    s2z = c0pz * s1z + m * b01 * s0z + b00 * gz[m * dm];
                    gz[dn + (m + 1) * dm] = s2z;
                    s0z = s1z;
                    s1z = s2z;
                }
            }
        }
        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0z = gz[j];
            s1z = gz[j + dn];
            tmpb0 = m * b00;
            for (n = 1; n < nmax; ++n) {
                s2z = c00z * s1z + n * b10 * s0z + tmpb0 * gz[j + n * dn - dm];
                gz[j + (n + 1) * dn] = s2z;
                s0z = s1z;
                s1z = s2z;
            }
        }
        if (ijmin > 0) {
            zizj = zi - bas_z[jsh];

            // unrolling j
            for (j = 0; j < ijmin - 1; j += 2, nmax -= 2) {
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s0z = gz[n + nmax * di - di];
                    t1z = zizj * s0z + gz[n + nmax * di];
                    gz[dj + n + nmax * di - di] = t1z;
                    s1z = s0z;
                    for (i = nmax - 2; i >= 0; i--) {
                        s0z = gz[n + i * di];
                        t0z = zizj * s0z + s1z;
                        gz[dj + n + i * di] = t0z;
                        gz[dj + dj + n + i * di] = zizj * t0z + t1z;
                        s1z = s0z;
                        t1z = t0z;
                    }
                }
            }

            if (j < ijmin) {
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s1z = gz[n + nmax * di];
                    for (i = nmax - 1; i >= 0; i--) {
                        s0z = gz[n + i * di];
                        gz[dj + n + i * di] = zizj * s0z + s1z;
                        s1z = s0z;
                    }
                }
            }
        }
        if (klmin > 0) {
            zkzl = zk - bas_z[lsh];

            // unrolling l
            for (l = 0; l < klmin - 1; l += 2, mmax -= 2) {
                off = l * dl;
                for (n = off; n < off + dij; ++n) {
                    s0z = gz[n + mmax * dk - dk];
                    t1z = zkzl * s0z + gz[n + mmax * dk];
                    gz[dl + n + mmax * dk - dk] = t1z;
                    s1z = s0z;
                    for (k = mmax - 2; k >= 0; k--) {
                        s0z = gz[n + k * dk];
                        t0z = zkzl * s0z + s1z;
                        gz[dl + n + k * dk] = t0z;
                        gz[dl + dl + n + k * dk] = zkzl * t0z + t1z;
                        s1z = s0z;
                        t1z = t0z;
                    }
                }
            }

            if (l < klmin) {
                off = l * dl;
                for (n = off; n < off + dij; ++n) {
                    s1z = gz[n + mmax * dk];
                    for (k = mmax - 1; k >= 0; k--) {
                        s0z = gz[n + k * dk];
                        gz[dl + n + k * dk] = zkzl * s0z + s1z;
                        s1z = s0z;
                    }
                }
            }
        }
    }
}

template <int I, int J, int K, int L>
__device__ void GINTg0_2e_2d4d(GINTEnvVars &envs, BasisProdCache &bpcache,
    double *__restrict__ g, double u, double w, double norm, int ish, int jsh,
    int ksh, int lsh, int prim_ij, int prim_kl, int iroot) {
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double aij = a12[prim_ij];
    double akl = a12[prim_kl];
    double eij = e12[prim_ij];
    double ekl = e12[prim_kl];
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double fac = eij * ekl / (sqrt(aijkl) * a1);

    int nbas = bpcache.nbas;
    constexpr int nmax = I + J;
    constexpr int mmax = K + L;
    constexpr int ijmin = I >= J ? J : I;
    constexpr int klmin = K >= L ? L : K;
    constexpr int di = 1;
    constexpr int dj = I >= J ? I + 1 : J + 1;
    constexpr int dk = (J + 1) * (I + 1);
    constexpr int dl = dk * (K >= L ? (K + 1) : (L + 1));
    constexpr int dm = dk;
    constexpr int dn = di;
    constexpr int dij = dk;
    int i, k;
    int j, l, m, n, off;
    double tmpb0;
    double u2, tmp1, tmp2, tmp3, tmp4;
    double b00, b10, b01;
    int tmax;

    u2 = a0 * u;
    tmp4 = .5 / (u2 * aijkl + a1);
    b00 = u2 * tmp4;
    tmp1 = 2 * b00;
    tmp2 = tmp1 * akl;
    b10 = b00 + tmp4 * akl;
    {
        double *__restrict__ x12 = bpcache.x12;
        double *__restrict__ gx = g;
        double xij = x12[prim_ij];
        double xkl = x12[prim_kl];
        double xijxkl = xij - xkl;
        double *__restrict__ bas_x = bpcache.bas_coords;
        double xixj, xkxl;
        double xi = bas_x[ish];
        double xk = bas_x[ksh];
        double xijxi = xij - xi;
        double xklxk = xkl - xk;
        double s0x, s1x, s2x, t0x, t1x;
        double c00x = xijxi - tmp2 * xijxkl;
        double c0px;
        gx[0] = norm;

        if constexpr (nmax > 0) {
            s0x = gx[0];
            s1x = c00x * s0x;
            gx[dn] = s1x;
#pragma unroll
            for (n = 1; n < nmax; ++n) {
                s2x = c00x * s1x + n * b10 * s0x;
                gx[(n + 1) * dn] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
        if constexpr (mmax > 0) {
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0px = xklxk + tmp3 * xijxkl;
            s0x = gx[0];
            s1x = c0px * s0x;
            gx[dm] = s1x;
#pragma unroll
            for (m = 1; m < mmax; ++m) {
                s2x = c0px * s1x + m * b01 * s0x;
                gx[(m + 1) * dm] = s2x;
                s0x = s1x;
                s1x = s2x;
            }

            if constexpr (nmax > 0) {
                s0x = gx[dn];
                s1x = c0px * s0x + b00 * gx[0];
                gx[dn + dm] = s1x;
#pragma unroll
                for (m = 1; m < mmax; ++m) {
                    s2x = c0px * s1x + m * b01 * s0x + b00 * gx[m * dm];
                    gx[dn + (m + 1) * dm] = s2x;
                    s0x = s1x;
                    s1x = s2x;
                }
            }
        }
#pragma unroll
        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0x = gx[j];
            s1x = gx[j + dn];
            tmpb0 = m * b00;
#pragma unroll
            for (n = 1; n < nmax; ++n) {
                s2x = c00x * s1x + n * b10 * s0x + tmpb0 * gx[j + n * dn - dm];
                gx[j + (n + 1) * dn] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
        if constexpr (ijmin > 0) {
            xixj = xi - bas_x[jsh];

            // unrolling j
            tmax = nmax;
#pragma unroll
            for (j = 0; j < ijmin - 1; j += 2, tmax -= 2) {
#pragma unroll
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s0x = gx[n + tmax * di - di];
                    t1x = xixj * s0x + gx[n + tmax * di];
                    gx[dj + n + tmax * di - di] = t1x;
                    s1x = s0x;
                    for (i = tmax - 2; i >= 0; i--) {
                        s0x = gx[n + i * di];
                        t0x = xixj * s0x + s1x;
                        gx[dj + n + i * di] = t0x;
                        gx[dj + dj + n + i * di] = xixj * t0x + t1x;
                        s1x = s0x;
                        t1x = t0x;
                    }
                }
            }

            if constexpr (ijmin % 2 == 1) {
#pragma unroll
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s1x = gx[n + tmax * di];
#pragma unroll
                    for (i = tmax - 1; i >= 0; i--) {
                        s0x = gx[n + i * di];
                        gx[dj + n + i * di] = xixj * s0x + s1x;
                        s1x = s0x;
                    }
                }
            }
        }
        if constexpr (klmin > 0) {
            xkxl = xk - bas_x[lsh];

            tmax = mmax;
// unrolling l
#pragma unroll
            for (l = 0; l < klmin - 1; l += 2, tmax -= 2) {
                off = l * dl;
#pragma unroll
                for (n = off; n < off + dij; ++n) {
                    s0x = gx[n + tmax * dk - dk];
                    t1x = xkxl * s0x + gx[n + tmax * dk];
                    gx[dl + n + tmax * dk - dk] = t1x;
                    s1x = s0x;
#pragma unroll
                    for (k = tmax - 2; k >= 0; k--) {
                        s0x = gx[n + k * dk];
                        t0x = xkxl * s0x + s1x;
                        gx[dl + n + k * dk] = t0x;
                        gx[dl + dl + n + k * dk] = xkxl * t0x + t1x;
                        s1x = s0x;
                        t1x = t0x;
                    }
                }
            }

            if (klmin % 2 == 1) {
                off = l * dl;
#pragma unroll
                for (n = off; n < off + dij; ++n) {
                    s1x = gx[n + tmax * dk];
#pragma unroll
                    for (k = tmax - 1; k >= 0; k--) {
                        s0x = gx[n + k * dk];
                        gx[dl + n + k * dk] = xkxl * s0x + s1x;
                        s1x = s0x;
                    }
                }
            }
        }
    }
    {
        double *__restrict__ y12 = bpcache.y12;
        double *__restrict__ gy = g + (I + 1) * (J + 1) * (K + 1) * (L + 1);
        double yij = y12[prim_ij];
        double ykl = y12[prim_kl];
        double yijykl = yij - ykl;
        double *__restrict__ bas_y = bpcache.bas_coords + nbas;
        double yiyj, ykyl;
        double yi = bas_y[ish];
        double yk = bas_y[ksh];
        double yijyi = yij - yi;
        double yklyk = ykl - yk;
        double s0y, s1y, s2y, t0y, t1y;
        double c00y = yijyi - tmp2 * yijykl;
        double c0py;
        gy[0] = fac;

        if constexpr (nmax > 0) {
            s0y = gy[0];
            s1y = c00y * s0y;
            gy[dn] = s1y;
#pragma unroll
            for (n = 1; n < nmax; ++n) {
                s2y = c00y * s1y + n * b10 * s0y;
                gy[(n + 1) * dn] = s2y;
                s0y = s1y;
                s1y = s2y;
            }
        }
        if constexpr (mmax > 0) {
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0py = yklyk + tmp3 * yijykl;
            s0y = gy[0];
            s1y = c0py * s0y;
            gy[dm] = s1y;
#pragma unroll
            for (m = 1; m < mmax; ++m) {
                s2y = c0py * s1y + m * b01 * s0y;
                gy[(m + 1) * dm] = s2y;
                s0y = s1y;
                s1y = s2y;
            }

            if constexpr (nmax > 0) {
                s0y = gy[dn];
                s1y = c0py * s0y + b00 * gy[0];
                gy[dn + dm] = s1y;
#pragma unroll
                for (m = 1; m < mmax; ++m) {
                    s2y = c0py * s1y + m * b01 * s0y + b00 * gy[m * dm];
                    gy[dn + (m + 1) * dm] = s2y;
                    s0y = s1y;
                    s1y = s2y;
                }
            }
        }
#pragma unroll
        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0y = gy[j];
            s1y = gy[j + dn];
            tmpb0 = m * b00;
#pragma unroll
            for (n = 1; n < nmax; ++n) {
                s2y = c00y * s1y + n * b10 * s0y + tmpb0 * gy[j + n * dn - dm];
                gy[j + (n + 1) * dn] = s2y;
                s0y = s1y;
                s1y = s2y;
            }
        }
        if constexpr (ijmin > 0) {
            yiyj = yi - bas_y[jsh];

            tmax = nmax;
// unrolling j
#pragma unroll
            for (j = 0; j < ijmin - 1; j += 2, tmax -= 2) {
#pragma unroll
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s0y = gy[n + tmax * di - di];
                    t1y = yiyj * s0y + gy[n + tmax * di];
                    gy[dj + n + tmax * di - di] = t1y;
                    s1y = s0y;
                    for (i = tmax - 2; i >= 0; i--) {
                        s0y = gy[n + i * di];
                        t0y = yiyj * s0y + s1y;
                        gy[dj + n + i * di] = t0y;
                        gy[dj + dj + n + i * di] = yiyj * t0y + t1y;
                        s1y = s0y;
                        t1y = t0y;
                    }
                }
            }

            if constexpr (ijmin % 2 == 1) {
#pragma unroll
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s1y = gy[n + tmax * di];
#pragma unroll
                    for (i = tmax - 1; i >= 0; i--) {
                        s0y = gy[n + i * di];
                        gy[dj + n + i * di] = yiyj * s0y + s1y;
                        s1y = s0y;
                    }
                }
            }
        }
        if constexpr (klmin > 0) {
            ykyl = yk - bas_y[lsh];

            tmax = mmax;
// unrolling l
#pragma unroll
            for (l = 0; l < klmin - 1; l += 2, tmax -= 2) {
                off = l * dl;
#pragma unroll
                for (n = off; n < off + dij; ++n) {
                    s0y = gy[n + tmax * dk - dk];
                    t1y = ykyl * s0y + gy[n + tmax * dk];
                    gy[dl + n + tmax * dk - dk] = t1y;
                    s1y = s0y;
#pragma unroll
                    for (k = tmax - 2; k >= 0; k--) {
                        s0y = gy[n + k * dk];
                        t0y = ykyl * s0y + s1y;
                        gy[dl + n + k * dk] = t0y;
                        gy[dl + dl + n + k * dk] = ykyl * t0y + t1y;
                        s1y = s0y;
                        t1y = t0y;
                    }
                }
            }

            if constexpr (klmin % 2 == 1) {
                off = l * dl;
#pragma unroll
                for (n = off; n < off + dij; ++n) {
                    s1y = gy[n + tmax * dk];
#pragma unroll
                    for (k = tmax - 1; k >= 0; k--) {
                        s0y = gy[n + k * dk];
                        gy[dl + n + k * dk] = ykyl * s0y + s1y;
                        s1y = s0y;
                    }
                }
            }
        }
    }
    {
        double *__restrict__ z12 = bpcache.z12;
        double *__restrict__ gz =
            g + 2 * (I + 1) * (J + 1) * (K + 1) * (L + 1);
        double zij = z12[prim_ij];
        double zkl = z12[prim_kl];
        double zijzkl = zij - zkl;
        double *__restrict__ bas_z = bpcache.bas_coords + 2 * nbas;
        double zizj, zkzl;
        double zi = bas_z[ish];
        double zk = bas_z[ksh];
        double zijzi = zij - zi;
        double zklzk = zkl - zk;
        double s0z, s1z, s2z, t0z, t1z;
        double c00z = zijzi - tmp2 * zijzkl;
        double c0pz;
        gz[0] = w;

        if constexpr (nmax > 0) {
            s0z = gz[0];
            s1z = c00z * s0z;
            gz[dn] = s1z;
#pragma unroll
            for (n = 1; n < nmax; ++n) {
                s2z = c00z * s1z + n * b10 * s0z;
                gz[(n + 1) * dn] = s2z;
                s0z = s1z;
                s1z = s2z;
            }
        }
        if constexpr (mmax > 0) {
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0pz = zklzk + tmp3 * zijzkl;
            s0z = gz[0];
            s1z = c0pz * s0z;
            gz[dm] = s1z;
#pragma unroll
            for (m = 1; m < mmax; ++m) {
                s2z = c0pz * s1z + m * b01 * s0z;
                gz[(m + 1) * dm] = s2z;
                s0z = s1z;
                s1z = s2z;
            }

            if constexpr (nmax > 0) {
                s0z = gz[dn];
                s1z = c0pz * s0z + b00 * gz[0];
                gz[dn + dm] = s1z;
#pragma unroll
                for (m = 1; m < mmax; ++m) {
                    s2z = c0pz * s1z + m * b01 * s0z + b00 * gz[m * dm];
                    gz[dn + (m + 1) * dm] = s2z;
                    s0z = s1z;
                    s1z = s2z;
                }
            }
        }
#pragma unroll
        for (m = 1; m <= mmax; ++m) {
            off = m * dm;
            j = off;
            s0z = gz[j];
            s1z = gz[j + dn];
            tmpb0 = m * b00;
#pragma unroll
            for (n = 1; n < nmax; ++n) {
                s2z = c00z * s1z + n * b10 * s0z + tmpb0 * gz[j + n * dn - dm];
                gz[j + (n + 1) * dn] = s2z;
                s0z = s1z;
                s1z = s2z;
            }
        }
        if constexpr (ijmin > 0) {
            zizj = zi - bas_z[jsh];

            tmax = nmax;
// unrolling j
#pragma unroll
            for (j = 0; j < ijmin - 1; j += 2, tmax -= 2) {
#pragma unroll
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s0z = gz[n + tmax * di - di];
                    t1z = zizj * s0z + gz[n + tmax * di];
                    gz[dj + n + tmax * di - di] = t1z;
                    s1z = s0z;
#pragma unroll
                    for (i = tmax - 2; i >= 0; i--) {
                        s0z = gz[n + i * di];
                        t0z = zizj * s0z + s1z;
                        gz[dj + n + i * di] = t0z;
                        gz[dj + dj + n + i * di] = zizj * t0z + t1z;
                        s1z = s0z;
                        t1z = t0z;
                    }
                }
            }

            if constexpr (ijmin % 2 == 1) {
#pragma unroll
                for (k = 0; k <= mmax; ++k) {
                    off = k * dk + j * dj;
                    n = off;
                    s1z = gz[n + tmax * di];
#pragma unroll
                    for (i = tmax - 1; i >= 0; i--) {
                        s0z = gz[n + i * di];
                        gz[dj + n + i * di] = zizj * s0z + s1z;
                        s1z = s0z;
                    }
                }
            }
        }
        if constexpr (klmin > 0) {
            zkzl = zk - bas_z[lsh];

            tmax = mmax;
// unrolling l
#pragma unroll
            for (l = 0; l < klmin - 1; l += 2, tmax -= 2) {
                off = l * dl;
#pragma unroll
                for (n = off; n < off + dij; ++n) {
                    s0z = gz[n + tmax * dk - dk];
                    t1z = zkzl * s0z + gz[n + tmax * dk];
                    gz[dl + n + tmax * dk - dk] = t1z;
                    s1z = s0z;
#pragma unroll
                    for (k = tmax - 2; k >= 0; k--) {
                        s0z = gz[n + k * dk];
                        t0z = zkzl * s0z + s1z;
                        gz[dl + n + k * dk] = t0z;
                        gz[dl + dl + n + k * dk] = zkzl * t0z + t1z;
                        s1z = s0z;
                        t1z = t0z;
                    }
                }
            }

            if constexpr (klmin % 2 == 1) {
                off = l * dl;
#pragma unroll
                for (n = off; n < off + dij; ++n) {
                    s1z = gz[n + tmax * dk];
#pragma unroll
                    for (k = tmax - 1; k >= 0; k--) {
                        s0z = gz[n + k * dk];
                        gz[dl + n + k * dk] = zkzl * s0z + s1z;
                        s1z = s0z;
                    }
                }
            }
        }
    }
}

template <int NROOTS, int UGSIZE>
__global__ static void GINTfill_int2e_kernel(ERITensor eri,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    igroup *= NROOTS;
    int iroot = task_ij % NROOTS;
    task_ij /= NROOTS;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];

    double u, w;
    double g[UGSIZE];

    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;

    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    GINTrys_root<NROOTS>(x, u, w, iroot);
    GINTg0_2e_2d4d<NROOTS>(envs, bpcache, g, u, w, norm, as_ish, as_jsh,
        as_ksh, as_lsh, ij, kl, iroot);
    GINTwrite_ints_s2_g<NROOTS>(
        envs, bpcache, eri, g, ish, jsh, ksh, lsh, iroot, igroup);
}

template <int I, int J, int K, int L>
__global__ static void GINTfill_int2e_kernel(ERITensor eri,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    constexpr int NROOTS = (I + J + K + L) / 2 + 1;
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    igroup *= NROOTS;
    int iroot = task_ij % NROOTS;
    task_ij /= NROOTS;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];

    double u, w;
    double g[3 * (I + 1) * (J + 1) * (K + 1) * (L + 1)];

    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;

    int as_ish, as_jsh, as_ksh, as_lsh;
    if constexpr (I >= J) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if constexpr (K >= L) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    GINTrys_root<NROOTS>(x, u, w, iroot);
    GINTg0_2e_2d4d<I, J, K, L>(envs, bpcache, g, u, w, norm, as_ish, as_jsh,
        as_ksh, as_lsh, ij, kl, iroot);
    GINTwrite_ints_s2_g<I, J, K, L>(
        envs, bpcache, eri, g, ish, jsh, ksh, lsh, iroot, igroup);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel0000(ERITensor eri,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    double gout0 = 0;
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);
    if (x > 3.e-10) {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        fac *= fmt0;
    }
    gout0 += fac;

    size_t istride = eri.stride_i;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int *ao_loc = bpcache.ao_loc;
    int i0 = ao_loc[ish] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh] - eri.ao_offsets_k;
    int l0 = ao_loc[lsh] - eri.ao_offsets_l;
    double *peri =
        eri.data + l0 * lstride + k0 * kstride + j0 * jstride + i0 * istride;
    if constexpr (T) {
        auto reduce = SegReduce<double>(igroup);
        gout0 = reduce(gout0);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri, gout0);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel0010(ERITensor eri,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);
    double root0, weight0;
    if (x < 3.e-10) {
        root0 = 0.5;
        weight0 = 1.;
    } else {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        weight0 = fmt0;
        double e = exp(-x);
        double b = .5 / x;
        double fmt1 = b * (fmt0 - e);
        root0 = fmt1 / (fmt0 - fmt1);
    }
    double u2 = a0 * root0;
    double tmp4 = .5 / (u2 * aijkl + a1);
    double b00 = u2 * tmp4;
    double tmp1 = 2 * b00;
    double tmp3 = tmp1 * aij;
    double c0px = xkl - xk + tmp3 * xijxkl;
    double c0py = ykl - yk + tmp3 * yijykl;
    double c0pz = zkl - zk + tmp3 * zijzkl;
    double g_0 = 1;
    double g_1 = c0px;
    double g_2 = 1;
    double g_3 = c0py;
    double g_4 = weight0 * fac;
    double g_5 = c0pz * g_4;
    gout0 += g_1 * g_2 * g_4;
    gout1 += g_0 * g_3 * g_4;
    gout2 += g_0 * g_2 * g_5;

    size_t istride = eri.stride_i;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int *ao_loc = bpcache.ao_loc;
    int i0 = ao_loc[ish] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh] - eri.ao_offsets_k;
    int l0 = ao_loc[lsh] - eri.ao_offsets_l;
    double *peri =
        eri.data + l0 * lstride + k0 * kstride + j0 * jstride + i0 * istride;
    if constexpr (T) {
        auto reduce = SegReduce<double>(igroup);
        gout0 = reduce(gout0);
        gout1 = reduce(gout1);
        gout2 = reduce(gout2);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * kstride, gout1);
    atomicAdd(peri + 2 * kstride, gout2);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel1000(ERITensor eri,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = eij * ekl / (sqrt(aijkl) * a1);
    double root0, weight0;
    if (x < 3.e-10) {
        root0 = 0.5;
        weight0 = 1.;
    } else {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        weight0 = fmt0;
        double e = exp(-x);
        double b = .5 / x;
        double fmt1 = b * (fmt0 - e);
        root0 = fmt1 / (fmt0 - fmt1);
    }
    double u2 = a0 * root0;
    double tmp2 = akl * u2 / (u2 * aijkl + a1);
    ;
    double c00x = xij - xi - tmp2 * xijxkl;
    double c00y = yij - yi - tmp2 * yijykl;
    double c00z = zij - zi - tmp2 * zijzkl;
    double g_0 = 1;
    double g_1 = c00x;
    double g_2 = 1;
    double g_3 = c00y;
    double g_4 = norm * fac * weight0;
    double g_5 = g_4 * c00z;
    gout0 += g_1 * g_2 * g_4;
    gout1 += g_0 * g_3 * g_4;
    gout2 += g_0 * g_2 * g_5;

    size_t istride = eri.stride_i;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int *ao_loc = bpcache.ao_loc;
    int i0 = ao_loc[ish] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh] - eri.ao_offsets_k;
    int l0 = ao_loc[lsh] - eri.ao_offsets_l;
    double *peri =
        eri.data + l0 * lstride + k0 * kstride + j0 * jstride + i0 * istride;
    if constexpr (T) {
        auto reduce = SegReduce<double>(igroup);
        gout0 = reduce(gout0);
        gout1 = reduce(gout1);
        gout2 = reduce(gout2);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
}
