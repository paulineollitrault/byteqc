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
#include "gint.h"
#include "reduce.cu"

template <int NROOTS>
__device__ void GINTwrite_ints_s2_g(GINTEnvVars &envs, BasisProdCache &bpcache,
    ERITensor eri, double *__restrict__ g, int ish, int jsh, int ksh, int lsh,
    int iroot, int igroup) {
    int *ao_loc = bpcache.ao_loc;
    size_t istride = eri.stride_i;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int ix, iy, jx, jy, kx, ky, lx, ly;
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
    double *peri = eri.data;
    peri += (ao_loc[ish] - eri.ao_offsets_i) * istride;
    peri += (ao_loc[jsh] - eri.ao_offsets_j) * jstride;
    peri += (ao_loc[ksh] - eri.ao_offsets_k) * kstride;
    peri += (ao_loc[lsh] - eri.ao_offsets_l) * lstride;

    for (lx = ll; lx >= 0; lx--) {
        for (ly = ll - lx; ly >= 0; ly--) {
            for (kx = kl; kx >= 0; kx--) {
                for (ky = kl - kx; ky >= 0; ky--) {
                    for (jx = jl; jx >= 0; jx--) {
                        for (jy = jl - jx; jy >= 0; jy--) {
                            for (ix = il; ix >= 0; ix--) {
                                for (iy = il - ix; iy >= 0; iy--) {
                                    reduce(g[idx] * g[idy] * g[idz], peri);
                                    idy -= si;
                                    idz += si;
                                    peri += istride;
                                }
                                idy += si * (il - ix + 2);
                                idz -= si * (il - ix + 1);
                                idx -= si;
                            }
                            idx += si * (il + 1);
                            idy -= si * (il + 1);
                            peri -= istride * envs.nfi;

                            idy -= sj;
                            idz += sj;
                            peri += jstride;
                        }
                        idy += sj * (jl - jx + 2);
                        idz -= sj * (jl - jx + 1);
                        idx -= sj;
                    }
                    idx += sj * (jl + 1);
                    idy -= sj * (jl + 1);
                    peri -= jstride * envs.nfj;

                    idy -= sk;
                    idz += sk;
                    peri += kstride;
                }
                idy += sk * (kl - kx + 2);
                idz -= sk * (kl - kx + 1);
                idx -= sk;
            }
            idx += sk * (kl + 1);
            idy -= sk * (kl + 1);
            peri -= kstride * envs.nfk;

            idy -= sl;
            idz += sl;
            peri += lstride;
        }
        idy += sl * (ll - lx + 2);
        idz -= sl * (ll - lx + 1);
        idx -= sl;
    }
}

template <int I, int J, int K, int L>
__device__ void GINTwrite_ints_s2_g(GINTEnvVars &envs, BasisProdCache &bpcache,
    ERITensor eri, double *__restrict__ g, int ish, int jsh, int ksh, int lsh,
    int iroot, int igroup) {
    int *ao_loc = bpcache.ao_loc;
    size_t istride = eri.stride_i;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int ix, iy, jx, jy, kx, ky, lx, ly;
    auto reduce = SegReduce<double>(igroup);
    constexpr int si = I >= J ? 1 : I + 1;
    constexpr int sj = I >= J ? I + 1 : 1;
    constexpr int g_size_ij = (I + 1) * (J + 1);
    constexpr int g_size = g_size_ij * (K + 1) * (L + 1);
    constexpr int sk = (K >= L ? 1 : K + 1) * g_size_ij;
    constexpr int sl = (K >= L ? K + 1 : 1) * g_size_ij;
    int idx = si * I + sj * J + sk * K + sl * L;
    int idy = g_size;
    int idz = 2 * g_size;
    double *peri = eri.data;
    peri += (ao_loc[ish] - eri.ao_offsets_i) * istride;
    peri += (ao_loc[jsh] - eri.ao_offsets_j) * jstride;
    peri += (ao_loc[ksh] - eri.ao_offsets_k) * kstride;
    peri += (ao_loc[lsh] - eri.ao_offsets_l) * lstride;

#pragma unroll
    for (lx = L; lx >= 0; lx--) {
#pragma unroll
        for (ly = L - lx; ly >= 0; ly--) {
#pragma unroll
            for (kx = K; kx >= 0; kx--) {
#pragma unroll
                for (ky = K - kx; ky >= 0; ky--) {
#pragma unroll
                    for (jx = J; jx >= 0; jx--) {
#pragma unroll
                        for (jy = J - jx; jy >= 0; jy--) {
#pragma unroll
                            for (ix = I; ix >= 0; ix--) {
#pragma unroll
                                for (iy = I - ix; iy >= 0; iy--) {
                                    reduce(g[idx] * g[idy] * g[idz], peri);
                                    idy -= si;
                                    idz += si;
                                    peri += istride;
                                }
                                idy += si * (I - ix + 2);
                                idz -= si * (I - ix + 1);
                                idx -= si;
                            }
                            idx += si * (I + 1);
                            idy -= si * (I + 1);
                            peri -= istride * envs.nfi;

                            idy -= sj;
                            idz += sj;
                            peri += jstride;
                        }
                        idy += sj * (J - jx + 2);
                        idz -= sj * (J - jx + 1);
                        idx -= sj;
                    }
                    idx += sj * (J + 1);
                    idy -= sj * (J + 1);
                    peri -= jstride * envs.nfj;

                    idy -= sk;
                    idz += sk;
                    peri += kstride;
                }
                idy += sk * (K - kx + 2);
                idz -= sk * (K - kx + 1);
                idx -= sk;
            }
            idx += sk * (K + 1);
            idy -= sk * (K + 1);
            peri -= kstride * envs.nfk;

            idy -= sl;
            idz += sl;
            peri += lstride;
        }
        idy += sl * (L - lx + 2);
        idz -= sl * (L - lx + 1);
        idx -= sl;
    }
}
