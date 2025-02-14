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
#include "gint/gint.h"
#include "gint/reduce.cu"

template <int NROOTS>
__device__ static void GINTkernel_getjk(GINTEnvVars &envs,
    BasisProdCache &bpcache, JKMatrix &jk, double *__restrict__ g, int ish,
    int jsh, int ksh, int lsh, int igroup) {
    int *ao_loc = bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int i1 = ao_loc[ish + 1];
    int j0 = ao_loc[jsh];
    int j1 = ao_loc[jsh + 1];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];

    int nao = jk.nao;
    int i, j, k, l, i_dm, ix, iy, jx, jy, kx, ky, lx, ly;

    int n_dm = jk.n_dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double *__restrict__ dm = jk.dm;

    auto reduce = SegReduce<double>(igroup);
    int il = envs.i_l;
    int jl = envs.j_l;
    int kl = envs.k_l;
    int ll = envs.l_l;
    int si = il >= jl ? 1 : il + 1;
    int sj = il >= jl ? il + 1 : 1;
    int sk = (kl >= ll ? 1 : kl + 1) * envs.g_size_ij / NROOTS;
    int sl = (kl >= ll ? kl + 1 : 1) * envs.g_size_ij / NROOTS;
    int idx = si * il + sj * jl + sk * kl + sl * ll;
    int idy = envs.g_size / NROOTS;
    int idz = 2 * envs.g_size / NROOTS;

    if (vk == NULL) {
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            for (l = l0, lx = ll; lx >= 0; lx--) {
                for (ly = ll - lx; ly >= 0; ly--, l++) {
                    for (k = k0, kx = kl; kx >= 0; kx--) {
                        for (ky = kl - kx; ky >= 0; ky--, k++) {
                            double v_kl = 0;
                            double d_kl = dm[k + nao * l];
                            for (j = j0, jx = jl; jx >= 0; jx--) {
                                for (jy = jl - jx; jy >= 0; jy--, j++) {
                                    for (i = i0, ix = il; ix >= 0; ix--) {
                                        for (iy = il - ix; iy >= 0;
                                             iy--, i++) {
                                            double s =
                                                g[idx] * g[idy] * g[idz];
                                            reduce(s * d_kl, vj + i + nao * j);
                                            v_kl += s * dm[i + j * nao];

                                            idy -= si;
                                            idz += si;
                                        }
                                        idy += si * (il - ix + 2);
                                        idz -= si * (il - ix + 1);
                                        idx -= si;
                                    }
                                    idx += si * (il + 1);
                                    idy -= si * (il + 1);

                                    idy -= sj;
                                    idz += sj;
                                }
                                idy += sj * (jl - jx + 2);
                                idz -= sj * (jl - jx + 1);
                                idx -= sj;
                            }
                            reduce(v_kl, vj + k + nao * l);

                            idx += sj * (jl + 1);
                            idy -= sj * (jl + 1);

                            idy -= sk;
                            idz += sk;
                        }
                        idy += sk * (kl - kx + 2);
                        idz -= sk * (kl - kx + 1);
                        idx -= sk;
                    }
                    idx += sk * (kl + 1);
                    idy -= sk * (kl + 1);

                    idy -= sl;
                    idz += sl;
                }
                idy += sl * (ll - lx + 2);
                idz -= sl * (ll - lx + 1);
                idx -= sl;
            }
            dm += nao * nao;
            vj += nao * nao;
        }
        return;
    }

    if (vj == NULL) {
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            for (l = l0, lx = ll; lx >= 0; lx--) {
                for (ly = ll - lx; ly >= 0; ly--, l++) {
                    for (k = k0, kx = kl; kx >= 0; kx--) {
                        for (ky = kl - kx; ky >= 0; ky--, k++) {
                            double gout[GPU_CART_MAX * GPU_CART_MAX];
                            for (j = j0, jx = jl; jx >= 0; jx--) {
                                for (jy = jl - jx; jy >= 0; jy--, j++) {
                                    int jp = j - j0;
                                    for (i = i0, ix = il; ix >= 0; ix--) {
                                        for (iy = il - ix; iy >= 0;
                                             iy--, i++) {
                                            int ip = i - i0;
                                            double s =
                                                g[idx] * g[idy] * g[idz];
                                            gout[ip + GPU_CART_MAX * jp] = s;
                                            idy -= si;
                                            idz += si;
                                        }
                                        idy += si * (il - ix + 2);
                                        idz -= si * (il - ix + 1);
                                        idx -= si;
                                    }
                                    idx += si * (il + 1);
                                    idy -= si * (il + 1);

                                    idy -= sj;
                                    idz += sj;
                                }
                                idy += sj * (jl - jx + 2);
                                idz -= sj * (jl - jx + 1);
                                idx -= sj;
                            }

                            double v_ik[GPU_CART_MAX];
                            double d_ik[GPU_CART_MAX];
                            double v_il[GPU_CART_MAX];
                            double d_il[GPU_CART_MAX];
                            for (i = 0; i < i1 - i0; ++i) {
                                v_il[i] = 0.0;
                                d_il[i] = dm[i + i0 + l * nao];
                                v_ik[i] = 0.0;
                                d_ik[i] = dm[i + i0 + k * nao];
                            }

                            for (j = j0; j < j1; ++j) {
                                int jp = j - j0;
                                double v_jk = 0.0;
                                double v_jl = 0.0;
                                double d_jk = dm[j + nao * k];
                                double d_jl = dm[j + nao * l];
                                for (i = i0; i < i1; ++i) {
                                    int ip = i - i0;
                                    double s = gout[ip + GPU_CART_MAX * jp];
                                    v_il[ip] += s * d_jk;
                                    v_ik[ip] += s * d_jl;

                                    v_jl += s * d_ik[ip];
                                    v_jk += s * d_il[ip];
                                }
                                reduce(v_jk, vk + j + nao * k);
                                reduce(v_jl, vk + j + nao * l);
                            }
                            for (i = 0; i < i1 - i0; i++) {
                                reduce(v_ik[i], vk + i + i0 + nao * k);
                                reduce(v_il[i], vk + i + i0 + nao * l);
                            }

                            idx += sj * (jl + 1);
                            idy -= sj * (jl + 1);

                            idy -= sk;
                            idz += sk;
                        }
                        idy += sk * (kl - kx + 2);
                        idz -= sk * (kl - kx + 1);
                        idx -= sk;
                    }
                    idx += sk * (kl + 1);
                    idy -= sk * (kl + 1);

                    idy -= sl;
                    idz += sl;
                }
                idy += sl * (ll - lx + 2);
                idz -= sl * (ll - lx + 1);
                idx -= sl;
            }
            dm += nao * nao;
            vk += nao * nao;
        }
        return;
    }

    double v_il[GPU_CART_MAX];
    double v_ik[GPU_CART_MAX];

    double d_ik[GPU_CART_MAX];
    double d_il[GPU_CART_MAX];

    // vj != NULL and vk != NULL
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (l = l0, lx = ll; lx >= 0; lx--) {
            for (ly = ll - lx; ly >= 0; ly--, l++) {
                for (i = 0; i < i1 - i0; ++i) {
                    v_il[i] = 0.0;
                    d_il[i] = dm[i + i0 + l * nao];
                }
                for (k = k0, kx = kl; kx >= 0; kx--) {
                    for (ky = kl - kx; ky >= 0; ky--, k++) {
                        for (i = 0; i < i1 - i0; ++i) {
                            v_ik[i] = 0.0;
                            d_ik[i] = dm[i + i0 + k * nao];
                        }
                        double v_kl = 0;
                        double d_kl = dm[k + nao * l];
                        // for (n = 0, j = j0; j < j1; ++j) {
                        for (j = j0, jx = jl; jx >= 0; jx--) {
                            for (jy = jl - jx; jy >= 0; jy--, j++) {
                                double v_jk = 0.0;
                                double v_jl = 0.0;
                                double d_jk = dm[j + nao * k];
                                double d_jl = dm[j + nao * l];
                                for (i = i0, ix = il; ix >= 0; ix--) {
                                    for (iy = il - ix; iy >= 0; iy--, i++) {
                                        int ip = i - i0;
                                        double s = g[idx] * g[idy] * g[idz];
                                        double v_ij = s * d_kl;
                                        reduce(v_ij, vj + i + nao * j);

                                        v_il[ip] += s * d_jk;
                                        v_ik[ip] += s * d_jl;

                                        v_jl += s * d_ik[ip];
                                        v_jk += s * d_il[ip];

                                        v_kl += s * dm[i + j * nao];

                                        idy -= si;
                                        idz += si;
                                    }
                                    idy += si * (il - ix + 2);
                                    idz -= si * (il - ix + 1);
                                    idx -= si;
                                }
                                reduce(v_jk, vk + j + nao * k);
                                reduce(v_jl, vk + j + nao * l);

                                idx += si * (il + 1);
                                idy -= si * (il + 1);

                                idy -= sj;
                                idz += sj;
                            }
                            idy += sj * (jl - jx + 2);
                            idz -= sj * (jl - jx + 1);
                            idx -= sj;
                        }
                        for (i = 0; i < i1 - i0; i++) {
                            reduce(v_ik[i], vk + i + i0 + nao * k);
                        }
                        reduce(v_kl, vj + k + nao * l);
                        idx += sj * (jl + 1);
                        idy -= sj * (jl + 1);

                        idy -= sk;
                        idz += sk;
                    }
                    idy += sk * (kl - kx + 2);
                    idz -= sk * (kl - kx + 1);
                    idx -= sk;
                }
                for (i = 0; i < i1 - i0; i++) {
                    reduce(v_il[i], vk + i + i0 + nao * l);
                }

                idx += sk * (kl + 1);
                idy -= sk * (kl + 1);

                idy -= sl;
                idz += sl;
            }
            idy += sl * (ll - lx + 2);
            idz -= sl * (ll - lx + 1);
            idx -= sl;
        }

        dm += nao * nao;
        vj += nao * nao;
        vk += nao * nao;
    }
}

template <int I, int J, int K, int L>
__device__ static void GINTkernel_getjk(GINTEnvVars &envs,
    BasisProdCache &bpcache, JKMatrix &jk, double *__restrict__ g, int ish,
    int jsh, int ksh, int lsh, int igroup) {
    int *ao_loc = bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];

    int nao = jk.nao;
    int i, j, k, l, i_dm, ix, iy, jx, jy, kx, ky, lx, ly;

    int n_dm = jk.n_dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double *__restrict__ dm = jk.dm;

    auto reduce = SegReduce<double>(igroup);
    constexpr int nfi = (I + 1) * (I + 2) / 2;
    constexpr int nfj = (J + 1) * (J + 2) / 2;
    constexpr int si = I >= J ? 1 : I + 1;
    constexpr int sj = I >= J ? I + 1 : 1;
    constexpr int g_size_ij = (I + 1) * (J + 1);
    constexpr int g_size = g_size_ij * (K + 1) * (L + 1);
    constexpr int sk = (K >= L ? 1 : K + 1) * g_size_ij;
    constexpr int sl = (K >= L ? K + 1 : 1) * g_size_ij;
    int idx = si * I + sj * J + sk * K + sl * L;
    int idy = g_size;
    int idz = 2 * g_size;

    if (vk == NULL) {
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
#pragma unroll
            for (l = l0, lx = L; lx >= 0; lx--) {
#pragma unroll
                for (ly = L - lx; ly >= 0; ly--, l++) {
#pragma unroll
                    for (k = k0, kx = K; kx >= 0; kx--) {
#pragma unroll
                        for (ky = K - kx; ky >= 0; ky--, k++) {
                            double v_kl = 0;
                            double d_kl = dm[k + nao * l];
#pragma unroll
                            for (j = j0, jx = J; jx >= 0; jx--) {
#pragma unroll
                                for (jy = J - jx; jy >= 0; jy--, j++) {
#pragma unroll
                                    for (i = i0, ix = I; ix >= 0; ix--) {
#pragma unroll
                                        for (iy = I - ix; iy >= 0; iy--, i++) {
                                            double s =
                                                g[idx] * g[idy] * g[idz];
                                            double v_ij = s * d_kl;
                                            reduce(v_ij, vj + i + nao * j);
                                            v_kl += s * dm[i + j * nao];

                                            idy -= si;
                                            idz += si;
                                        }
                                        idy += si * (I - ix + 2);
                                        idz -= si * (I - ix + 1);
                                        idx -= si;
                                    }
                                    idx += si * (I + 1);
                                    idy -= si * (I + 1);

                                    idy -= sj;
                                    idz += sj;
                                }
                                idy += sj * (J - jx + 2);
                                idz -= sj * (J - jx + 1);
                                idx -= sj;
                            }
                            reduce(v_kl, vj + k + nao * l);

                            idx += sj * (J + 1);
                            idy -= sj * (J + 1);

                            idy -= sk;
                            idz += sk;
                        }
                        idy += sk * (K - kx + 2);
                        idz -= sk * (K - kx + 1);
                        idx -= sk;
                    }
                    idx += sk * (K + 1);
                    idy -= sk * (K + 1);

                    idy -= sl;
                    idz += sl;
                }
                idy += sl * (L - lx + 2);
                idz -= sl * (L - lx + 1);
                idx -= sl;
            }
            dm += nao * nao;
            vj += nao * nao;
        }
        return;
    }

    if (vj == NULL) {
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
#pragma unroll
            for (l = l0, lx = L; lx >= 0; lx--) {
#pragma unroll
                for (ly = L - lx; ly >= 0; ly--, l++) {
#pragma unroll
                    for (k = k0, kx = K; kx >= 0; kx--) {
#pragma unroll
                        for (ky = K - kx; ky >= 0; ky--, k++) {
                            double gout[nfi * nfj];
#pragma unroll
                            for (j = j0, jx = J; jx >= 0; jx--) {
#pragma unroll
                                for (jy = J - jx; jy >= 0; jy--, j++) {
                                    int jp = j - j0;
#pragma unroll
                                    for (i = i0, ix = I; ix >= 0; ix--) {
#pragma unroll
                                        for (iy = I - ix; iy >= 0; iy--, i++) {
                                            int ip = i - i0;
                                            double s =
                                                g[idx] * g[idy] * g[idz];
                                            gout[ip + ((I + 1) * (I + 2) / 2) *
                                                          jp] = s;
                                            idy -= si;
                                            idz += si;
                                        }
                                        idy += si * (I - ix + 2);
                                        idz -= si * (I - ix + 1);
                                        idx -= si;
                                    }
                                    idx += si * (I + 1);
                                    idy -= si * (I + 1);

                                    idy -= sj;
                                    idz += sj;
                                }
                                idy += sj * (J - jx + 2);
                                idz -= sj * (J - jx + 1);
                                idx -= sj;
                            }

                            double v_ik[nfi];
                            double d_ik[nfi];
                            double v_il[nfi];
                            double d_il[nfi];
#pragma unroll
                            for (i = 0; i < nfi; ++i) {
                                v_il[i] = 0.0;
                                d_il[i] = dm[i + i0 + l * nao];
                                v_ik[i] = 0.0;
                                d_ik[i] = dm[i + i0 + k * nao];
                            }
#pragma unroll
                            for (int jp = 0; jp < nfj; ++jp) {
                                j = jp + j0;
                                double v_jk = 0.0;
                                double v_jl = 0.0;
                                double d_jk = dm[j + nao * k];
                                double d_jl = dm[j + nao * l];
#pragma unroll
                                for (int ip = 0; ip < nfi; ++ip) {
                                    i = ip + i0;
                                    double s = gout[ip + nfi * jp];
                                    v_il[ip] += s * d_jk;
                                    v_ik[ip] += s * d_jl;

                                    v_jl += s * d_ik[ip];
                                    v_jk += s * d_il[ip];
                                }
                                reduce(v_jk, vk + j + nao * k);
                                reduce(v_jl, vk + j + nao * l);
                            }
#pragma unroll
                            for (int ip = 0; ip < nfi; ip++) {
                                i = ip + i0;
                                reduce(v_ik[i], vk + i + i0 + nao * k);
                                reduce(v_il[i], vk + i + i0 + nao * l);
                            }

                            idx += sj * (J + 1);
                            idy -= sj * (J + 1);

                            idy -= sk;
                            idz += sk;
                        }
                        idy += sk * (K - kx + 2);
                        idz -= sk * (K - kx + 1);
                        idx -= sk;
                    }
                    idx += sk * (K + 1);
                    idy -= sk * (K + 1);

                    idy -= sl;
                    idz += sl;
                }
                idy += sl * (L - lx + 2);
                idz -= sl * (L - lx + 1);
                idx -= sl;
            }
            dm += nao * nao;
            vk += nao * nao;
        }
        return;
    }

    double v_il[nfi];
    double v_ik[nfi];

    double d_ik[nfi];
    double d_il[nfi];

    // vj != NULL and vk != NULL
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
#pragma unroll
        for (l = l0, lx = L; lx >= 0; lx--) {
#pragma unroll
            for (ly = L - lx; ly >= 0; ly--, l++) {
#pragma unroll
                for (i = 0; i < nfi; ++i) {
                    v_il[i] = 0.0;
                    d_il[i] = dm[i + i0 + l * nao];
                }
#pragma unroll
                for (k = k0, kx = K; kx >= 0; kx--) {
#pragma unroll
                    for (ky = K - kx; ky >= 0; ky--, k++) {
#pragma unroll
                        for (i = 0; i < nfi; ++i) {
                            v_ik[i] = 0.0;
                            d_ik[i] = dm[i + i0 + k * nao];
                        }
                        double v_kl = 0;
                        double d_kl = dm[k + nao * l];
#pragma unroll
                        for (j = j0, jx = J; jx >= 0; jx--) {
#pragma unroll
                            for (jy = J - jx; jy >= 0; jy--, j++) {
                                double v_jk = 0.0;
                                double v_jl = 0.0;
                                double d_jk = dm[j + nao * k];
                                double d_jl = dm[j + nao * l];
#pragma unroll
                                for (i = i0, ix = I; ix >= 0; ix--) {
#pragma unroll
                                    for (iy = I - ix; iy >= 0; iy--, i++) {
                                        int ip = i - i0;
                                        double s = g[idx] * g[idy] * g[idz];
                                        double v_ij = s * d_kl;
                                        reduce(v_ij, vj + i + nao * j);

                                        v_il[ip] += s * d_jk;
                                        v_ik[ip] += s * d_jl;

                                        v_jl += s * d_ik[ip];
                                        v_jk += s * d_il[ip];

                                        v_kl += s * dm[i + j * nao];

                                        idy -= si;
                                        idz += si;
                                    }
                                    idy += si * (I - ix + 2);
                                    idz -= si * (I - ix + 1);
                                    idx -= si;
                                }
                                reduce(v_jk, vk + j + nao * k);
                                reduce(v_jl, vk + j + nao * l);

                                idx += si * (I + 1);
                                idy -= si * (I + 1);

                                idy -= sj;
                                idz += sj;
                            }
                            idy += sj * (J - jx + 2);
                            idz -= sj * (J - jx + 1);
                            idx -= sj;
                        }
#pragma unroll
                        for (i = 0; i < nfi; i++) {
                            reduce(v_ik[i], vk + i + i0 + nao * k);
                        }
                        reduce(v_kl, vj + k + nao * l);
                        idx += sj * (J + 1);
                        idy -= sj * (J + 1);

                        idy -= sk;
                        idz += sk;
                    }
                    idy += sk * (K - kx + 2);
                    idz -= sk * (K - kx + 1);
                    idx -= sk;
                }
#pragma unroll
                for (i = 0; i < nfi; i++) {
                    reduce(v_il[i], vk + i + i0 + nao * l);
                }

                idx += sk * (K + 1);
                idy -= sk * (K + 1);

                idy -= sl;
                idz += sl;
            }
            idy += sl * (L - lx + 2);
            idz -= sl * (L - lx + 1);
            idx -= sl;
        }

        dm += nao * nao;
        vj += nao * nao;
        vk += nao * nao;
    }
}
