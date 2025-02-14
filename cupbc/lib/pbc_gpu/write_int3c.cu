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

template <int I, int J, int K, int L>
__device__ inline void GINTwrite_ints_s1(
    double *__restrict__ gout, const double *__restrict__ g, const int flag) {
    double s;
    int ix, iy, kx, ky, lx, ly;
    constexpr int il = I;
    constexpr int kl = K;
    constexpr int ll = L;
    constexpr int g_size_ij = I + 1;
    constexpr int g_size = (I + 1) * (K + 1) * (L + 1);
    constexpr int sk = (kl >= ll ? 1 : kl + 1) * g_size_ij;
    constexpr int sl = (kl >= ll ? kl + 1 : 1) * g_size_ij;
    int idx = il + sk * kl + sl * ll;
    int idy = g_size;
    int idz = 2 * g_size;

    int n = 0;
#pragma unroll
    for (lx = ll; lx >= 0; lx--) {
#pragma unroll
        for (ly = ll - lx; ly >= 0; ly--) {
#pragma unroll
            for (kx = kl; kx >= 0; kx--) {
#pragma unroll
                for (ky = kl - kx; ky >= 0; ky--) {
#pragma unroll
                    for (ix = il; ix >= 0; ix--) {
#pragma unroll
                        for (iy = il - ix; iy >= 0; iy--) {
                            s = g[idx] * g[idy] * g[idz] * flag;
                            gout[n] += s;
                            n++;
                            idy--;
                            idz++;
                        }
                        idy += il - ix + 2;
                        idz -= il - ix + 1;
                        idx--;
                    }
                    idx += il + 1;
                    idy -= il + 1;

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
}

template <int I, int J, int K, int L>
__device__ inline void GINTwrite_ints_s1(ERITensor eri,
    const double *__restrict__ gout, const int ish, const int ksh,
    const int lsh, const int *__restrict__ ao_loc_ij,
    const int *__restrict__ ao_loc_kl) {
    constexpr int nfi = (I + 2) * (I + 1) / 2;
    constexpr int nfk = (K + 2) * (K + 1) / 2;
    constexpr int nfl = (L + 2) * (L + 1) / 2;

    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    const int i0 = ao_loc_ij[ish];

    const int k0 = ao_loc_kl[ksh] - eri.ao_offsets_k;
    const int l0 = ao_loc_kl[lsh] - eri.ao_offsets_l;
    double *__restrict__ eri_ij = eri.data;
    const size_t eri_offset = l0 * lstride + k0 * kstride + i0;

    int n = 0;
    eri_ij += eri_offset;
#pragma unroll
    for (int l = 0; l < nfl; l++) {
#pragma unroll
        for (int k = 0; k < nfk; k++) {
            int ptr_offset = (l * lstride + k * kstride);
#pragma unroll
            for (int i = 0; i < nfi; i++, n++) {
                atomicAdd(eri_ij + ptr_offset + i, gout[n]);
            }
        }
    }
}

template <int NROOTS>
__device__ void GINTwrite_int3c_s1_g(ERITensor eri, double *__restrict__ g,
    const int ish, const int ksh, const int lsh, const int *__restrict__ ao_loc_ij,
    const int *__restrict__ ao_loc_kl, const int16_t *c_idx4c,
    const bool issr, const int igroup) {
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    const int i0 = ao_loc_ij[ish];
    const int i1 = ao_loc_ij[ish + 1];
    const int k0 = ao_loc_kl[ksh] - eri.ao_offsets_k;
    const int k1 = ao_loc_kl[ksh + 1] - eri.ao_offsets_k;
    const int l0 = ao_loc_kl[lsh] - eri.ao_offsets_l;
    const int l1 = ao_loc_kl[lsh + 1] - eri.ao_offsets_l;
    const int flag = issr ? 1 : -1;
    int k, l, n, i;

    const int16_t *idx = c_idx4c;
    const int16_t *idy = idx + 1;
    const int16_t *idz = idx + 2;
    int ix, iy, iz;
    auto reduce = SegReduce<double>(igroup);
    double *__restrict__ peri;
    for (n = 0, l = l0; l < l1; ++l) {
        for (k = k0; k < k1; ++k) {
            peri = eri.data + l * lstride + k * kstride;
            for (i = i0; i < i1; ++i, ++n) {
                const int id = n * 3;
                ix = idx[id];
                iy = idy[id];
                iz = idz[id];
                reduce(g[ix] * g[iy] * g[iz] * flag, peri + i);
            }
        }
    }
}
