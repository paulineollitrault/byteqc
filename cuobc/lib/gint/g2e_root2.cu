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

template <bool T>
__global__ static void GINTfill_int2e_kernel0011(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c0px;
        double g_2 = c0px + xkxl;
        double g_3 = c0px * (c0px + xkxl) + b01;
        double g_4 = 1;
        double g_5 = c0py;
        double g_6 = c0py + ykyl;
        double g_7 = c0py * (c0py + ykyl) + b01;
        double g_8 = weight0 * fac;
        double g_9 = c0pz * g_8;
        double g_10 = g_8 * (c0pz + zkzl);
        double g_11 = b01 * g_8 + c0pz * g_9 + zkzl * g_9;
        gout0 += g_3 * g_4 * g_8;
        gout1 += g_2 * g_5 * g_8;
        gout2 += g_2 * g_4 * g_9;
        gout3 += g_1 * g_6 * g_8;
        gout4 += g_0 * g_7 * g_8;
        gout5 += g_0 * g_6 * g_9;
        gout6 += g_1 * g_4 * g_10;
        gout7 += g_0 * g_5 * g_10;
        gout8 += g_0 * g_4 * g_11;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * kstride, gout1);
    atomicAdd(peri + 2 * kstride, gout2);
    atomicAdd(peri + 1 * lstride, gout3);
    atomicAdd(peri + 1 * kstride + 1 * lstride, gout4);
    atomicAdd(peri + 2 * kstride + 1 * lstride, gout5);
    atomicAdd(peri + 2 * lstride, gout6);
    atomicAdd(peri + 1 * kstride + 2 * lstride, gout7);
    atomicAdd(peri + 2 * kstride + 2 * lstride, gout8);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel0020(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c0px;
        double g_2 = c0px * c0px + b01;
        double g_3 = 1;
        double g_4 = c0py;
        double g_5 = c0py * c0py + b01;
        double g_6 = weight0 * fac;
        double g_7 = c0pz * g_6;
        double g_8 = b01 * g_6 + c0pz * g_7;
        gout0 += g_2 * g_3 * g_6;
        gout1 += g_1 * g_4 * g_6;
        gout2 += g_1 * g_3 * g_7;
        gout3 += g_0 * g_5 * g_6;
        gout4 += g_0 * g_4 * g_7;
        gout5 += g_0 * g_3 * g_8;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * kstride, gout1);
    atomicAdd(peri + 2 * kstride, gout2);
    atomicAdd(peri + 3 * kstride, gout3);
    atomicAdd(peri + 4 * kstride, gout4);
    atomicAdd(peri + 5 * kstride, gout5);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel0021(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c0px;
        double g_2 = c0px * c0px + b01;
        double g_3 = c0px + xkxl;
        double g_4 = c0px * (c0px + xkxl) + b01;
        double g_5 = c0px * (2 * b01 + g_2) + xkxl * g_2;
        double g_6 = 1;
        double g_7 = c0py;
        double g_8 = c0py * c0py + b01;
        double g_9 = c0py + ykyl;
        double g_10 = c0py * (c0py + ykyl) + b01;
        double g_11 = c0py * (2 * b01 + g_8) + ykyl * g_8;
        double g_12 = weight0 * fac;
        double g_13 = c0pz * g_12;
        double g_14 = b01 * g_12 + c0pz * g_13;
        double g_15 = g_12 * (c0pz + zkzl);
        double g_16 = b01 * g_12 + c0pz * g_13 + zkzl * g_13;
        double g_17 = 2 * b01 * g_13 + c0pz * g_14 + zkzl * g_14;
        gout0 += g_5 * g_6 * g_12;
        gout1 += g_4 * g_7 * g_12;
        gout2 += g_4 * g_6 * g_13;
        gout3 += g_3 * g_8 * g_12;
        gout4 += g_3 * g_7 * g_13;
        gout5 += g_3 * g_6 * g_14;
        gout6 += g_2 * g_9 * g_12;
        gout7 += g_1 * g_10 * g_12;
        gout8 += g_1 * g_9 * g_13;
        gout9 += g_0 * g_11 * g_12;
        gout10 += g_0 * g_10 * g_13;
        gout11 += g_0 * g_9 * g_14;
        gout12 += g_2 * g_6 * g_15;
        gout13 += g_1 * g_7 * g_15;
        gout14 += g_1 * g_6 * g_16;
        gout15 += g_0 * g_8 * g_15;
        gout16 += g_0 * g_7 * g_16;
        gout17 += g_0 * g_6 * g_17;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        gout10 = reduce(gout10);
        gout11 = reduce(gout11);
        gout12 = reduce(gout12);
        gout13 = reduce(gout13);
        gout14 = reduce(gout14);
        gout15 = reduce(gout15);
        gout16 = reduce(gout16);
        gout17 = reduce(gout17);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * kstride, gout1);
    atomicAdd(peri + 2 * kstride, gout2);
    atomicAdd(peri + 3 * kstride, gout3);
    atomicAdd(peri + 4 * kstride, gout4);
    atomicAdd(peri + 5 * kstride, gout5);
    atomicAdd(peri + 1 * lstride, gout6);
    atomicAdd(peri + 1 * kstride + 1 * lstride, gout7);
    atomicAdd(peri + 2 * kstride + 1 * lstride, gout8);
    atomicAdd(peri + 3 * kstride + 1 * lstride, gout9);
    atomicAdd(peri + 4 * kstride + 1 * lstride, gout10);
    atomicAdd(peri + 5 * kstride + 1 * lstride, gout11);
    atomicAdd(peri + 2 * lstride, gout12);
    atomicAdd(peri + 1 * kstride + 2 * lstride, gout13);
    atomicAdd(peri + 2 * kstride + 2 * lstride, gout14);
    atomicAdd(peri + 3 * kstride + 2 * lstride, gout15);
    atomicAdd(peri + 4 * kstride + 2 * lstride, gout16);
    atomicAdd(peri + 5 * kstride + 2 * lstride, gout17);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel0030(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c0px;
        double g_2 = c0px * c0px + b01;
        double g_3 = c0px * (2 * b01 + g_2);
        double g_4 = 1;
        double g_5 = c0py;
        double g_6 = c0py * c0py + b01;
        double g_7 = c0py * (2 * b01 + g_6);
        double g_8 = weight0 * fac;
        double g_9 = c0pz * g_8;
        double g_10 = b01 * g_8 + c0pz * g_9;
        double g_11 = 2 * b01 * g_9 + c0pz * g_10;
        gout0 += g_3 * g_4 * g_8;
        gout1 += g_2 * g_5 * g_8;
        gout2 += g_2 * g_4 * g_9;
        gout3 += g_1 * g_6 * g_8;
        gout4 += g_1 * g_5 * g_9;
        gout5 += g_1 * g_4 * g_10;
        gout6 += g_0 * g_7 * g_8;
        gout7 += g_0 * g_6 * g_9;
        gout8 += g_0 * g_5 * g_10;
        gout9 += g_0 * g_4 * g_11;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * kstride, gout1);
    atomicAdd(peri + 2 * kstride, gout2);
    atomicAdd(peri + 3 * kstride, gout3);
    atomicAdd(peri + 4 * kstride, gout4);
    atomicAdd(peri + 5 * kstride, gout5);
    atomicAdd(peri + 6 * kstride, gout6);
    atomicAdd(peri + 7 * kstride, gout7);
    atomicAdd(peri + 8 * kstride, gout8);
    atomicAdd(peri + 9 * kstride, gout9);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel1010(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c0px;
        double g_3 = c0px * c00x + b00;
        double g_4 = 1;
        double g_5 = c00y;
        double g_6 = c0py;
        double g_7 = c0py * c00y + b00;
        double g_8 = weight0 * fac;
        double g_9 = c00z * g_8;
        double g_10 = c0pz * g_8;
        double g_11 = b00 * g_8 + c0pz * g_9;
        gout0 += g_3 * g_4 * g_8;
        gout1 += g_2 * g_5 * g_8;
        gout2 += g_2 * g_4 * g_9;
        gout3 += g_1 * g_6 * g_8;
        gout4 += g_0 * g_7 * g_8;
        gout5 += g_0 * g_6 * g_9;
        gout6 += g_1 * g_4 * g_10;
        gout7 += g_0 * g_5 * g_10;
        gout8 += g_0 * g_4 * g_11;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 1 * kstride, gout3);
    atomicAdd(peri + 1 * istride + 1 * kstride, gout4);
    atomicAdd(peri + 2 * istride + 1 * kstride, gout5);
    atomicAdd(peri + 2 * kstride, gout6);
    atomicAdd(peri + 1 * istride + 2 * kstride, gout7);
    atomicAdd(peri + 2 * istride + 2 * kstride, gout8);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel1011(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c0px;
        double g_3 = c0px * c00x + b00;
        double g_4 = c0px + xkxl;
        double g_5 = c00x * (c0px + xkxl) + b00;
        double g_6 = c0px * (c0px + xkxl) + b01;
        double g_7 = b00 * c0px + b01 * c00x + c0px * g_3 + xkxl * g_3;
        double g_8 = 1;
        double g_9 = c00y;
        double g_10 = c0py;
        double g_11 = c0py * c00y + b00;
        double g_12 = c0py + ykyl;
        double g_13 = c00y * (c0py + ykyl) + b00;
        double g_14 = c0py * (c0py + ykyl) + b01;
        double g_15 = b00 * c0py + b01 * c00y + c0py * g_11 + ykyl * g_11;
        double g_16 = weight0 * fac;
        double g_17 = c00z * g_16;
        double g_18 = c0pz * g_16;
        double g_19 = b00 * g_16 + c0pz * g_17;
        double g_20 = g_16 * (c0pz + zkzl);
        double g_21 = b00 * g_16 + c0pz * g_17 + zkzl * g_17;
        double g_22 = b01 * g_16 + c0pz * g_18 + zkzl * g_18;
        double g_23 = b00 * g_18 + b01 * g_17 + c0pz * g_19 + zkzl * g_19;
        gout0 += g_7 * g_8 * g_16;
        gout1 += g_6 * g_9 * g_16;
        gout2 += g_6 * g_8 * g_17;
        gout3 += g_5 * g_10 * g_16;
        gout4 += g_4 * g_11 * g_16;
        gout5 += g_4 * g_10 * g_17;
        gout6 += g_5 * g_8 * g_18;
        gout7 += g_4 * g_9 * g_18;
        gout8 += g_4 * g_8 * g_19;
        gout9 += g_3 * g_12 * g_16;
        gout10 += g_2 * g_13 * g_16;
        gout11 += g_2 * g_12 * g_17;
        gout12 += g_1 * g_14 * g_16;
        gout13 += g_0 * g_15 * g_16;
        gout14 += g_0 * g_14 * g_17;
        gout15 += g_1 * g_12 * g_18;
        gout16 += g_0 * g_13 * g_18;
        gout17 += g_0 * g_12 * g_19;
        gout18 += g_3 * g_8 * g_20;
        gout19 += g_2 * g_9 * g_20;
        gout20 += g_2 * g_8 * g_21;
        gout21 += g_1 * g_10 * g_20;
        gout22 += g_0 * g_11 * g_20;
        gout23 += g_0 * g_10 * g_21;
        gout24 += g_1 * g_8 * g_22;
        gout25 += g_0 * g_9 * g_22;
        gout26 += g_0 * g_8 * g_23;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        gout10 = reduce(gout10);
        gout11 = reduce(gout11);
        gout12 = reduce(gout12);
        gout13 = reduce(gout13);
        gout14 = reduce(gout14);
        gout15 = reduce(gout15);
        gout16 = reduce(gout16);
        gout17 = reduce(gout17);
        gout18 = reduce(gout18);
        gout19 = reduce(gout19);
        gout20 = reduce(gout20);
        gout21 = reduce(gout21);
        gout22 = reduce(gout22);
        gout23 = reduce(gout23);
        gout24 = reduce(gout24);
        gout25 = reduce(gout25);
        gout26 = reduce(gout26);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 1 * kstride, gout3);
    atomicAdd(peri + 1 * istride + 1 * kstride, gout4);
    atomicAdd(peri + 2 * istride + 1 * kstride, gout5);
    atomicAdd(peri + 2 * kstride, gout6);
    atomicAdd(peri + 1 * istride + 2 * kstride, gout7);
    atomicAdd(peri + 2 * istride + 2 * kstride, gout8);
    atomicAdd(peri + 1 * lstride, gout9);
    atomicAdd(peri + 1 * istride + 1 * lstride, gout10);
    atomicAdd(peri + 2 * istride + 1 * lstride, gout11);
    atomicAdd(peri + 1 * kstride + 1 * lstride, gout12);
    atomicAdd(peri + 1 * istride + 1 * kstride + 1 * lstride, gout13);
    atomicAdd(peri + 2 * istride + 1 * kstride + 1 * lstride, gout14);
    atomicAdd(peri + 2 * kstride + 1 * lstride, gout15);
    atomicAdd(peri + 1 * istride + 2 * kstride + 1 * lstride, gout16);
    atomicAdd(peri + 2 * istride + 2 * kstride + 1 * lstride, gout17);
    atomicAdd(peri + 2 * lstride, gout18);
    atomicAdd(peri + 1 * istride + 2 * lstride, gout19);
    atomicAdd(peri + 2 * istride + 2 * lstride, gout20);
    atomicAdd(peri + 1 * kstride + 2 * lstride, gout21);
    atomicAdd(peri + 1 * istride + 1 * kstride + 2 * lstride, gout22);
    atomicAdd(peri + 2 * istride + 1 * kstride + 2 * lstride, gout23);
    atomicAdd(peri + 2 * kstride + 2 * lstride, gout24);
    atomicAdd(peri + 1 * istride + 2 * kstride + 2 * lstride, gout25);
    atomicAdd(peri + 2 * istride + 2 * kstride + 2 * lstride, gout26);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel1020(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c0px;
        double g_3 = c0px * c00x + b00;
        double g_4 = c0px * c0px + b01;
        double g_5 = b00 * c0px + b01 * c00x + c0px * g_3;
        double g_6 = 1;
        double g_7 = c00y;
        double g_8 = c0py;
        double g_9 = c0py * c00y + b00;
        double g_10 = c0py * c0py + b01;
        double g_11 = b00 * c0py + b01 * c00y + c0py * g_9;
        double g_12 = weight0 * fac;
        double g_13 = c00z * g_12;
        double g_14 = c0pz * g_12;
        double g_15 = b00 * g_12 + c0pz * g_13;
        double g_16 = b01 * g_12 + c0pz * g_14;
        double g_17 = b00 * g_14 + b01 * g_13 + c0pz * g_15;
        gout0 += g_5 * g_6 * g_12;
        gout1 += g_4 * g_7 * g_12;
        gout2 += g_4 * g_6 * g_13;
        gout3 += g_3 * g_8 * g_12;
        gout4 += g_2 * g_9 * g_12;
        gout5 += g_2 * g_8 * g_13;
        gout6 += g_3 * g_6 * g_14;
        gout7 += g_2 * g_7 * g_14;
        gout8 += g_2 * g_6 * g_15;
        gout9 += g_1 * g_10 * g_12;
        gout10 += g_0 * g_11 * g_12;
        gout11 += g_0 * g_10 * g_13;
        gout12 += g_1 * g_8 * g_14;
        gout13 += g_0 * g_9 * g_14;
        gout14 += g_0 * g_8 * g_15;
        gout15 += g_1 * g_6 * g_16;
        gout16 += g_0 * g_7 * g_16;
        gout17 += g_0 * g_6 * g_17;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        gout10 = reduce(gout10);
        gout11 = reduce(gout11);
        gout12 = reduce(gout12);
        gout13 = reduce(gout13);
        gout14 = reduce(gout14);
        gout15 = reduce(gout15);
        gout16 = reduce(gout16);
        gout17 = reduce(gout17);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 1 * kstride, gout3);
    atomicAdd(peri + 1 * istride + 1 * kstride, gout4);
    atomicAdd(peri + 2 * istride + 1 * kstride, gout5);
    atomicAdd(peri + 2 * kstride, gout6);
    atomicAdd(peri + 1 * istride + 2 * kstride, gout7);
    atomicAdd(peri + 2 * istride + 2 * kstride, gout8);
    atomicAdd(peri + 3 * kstride, gout9);
    atomicAdd(peri + 1 * istride + 3 * kstride, gout10);
    atomicAdd(peri + 2 * istride + 3 * kstride, gout11);
    atomicAdd(peri + 4 * kstride, gout12);
    atomicAdd(peri + 1 * istride + 4 * kstride, gout13);
    atomicAdd(peri + 2 * istride + 4 * kstride, gout14);
    atomicAdd(peri + 5 * kstride, gout15);
    atomicAdd(peri + 1 * istride + 5 * kstride, gout16);
    atomicAdd(peri + 2 * istride + 5 * kstride, gout17);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel1100(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x + xixj;
        double g_3 = c00x * (c00x + xixj) + b10;
        double g_4 = 1;
        double g_5 = c00y;
        double g_6 = c00y + yiyj;
        double g_7 = c00y * (c00y + yiyj) + b10;
        double g_8 = weight0 * fac;
        double g_9 = c00z * g_8;
        double g_10 = g_8 * (c00z + zizj);
        double g_11 = b10 * g_8 + c00z * g_9 + zizj * g_9;
        gout0 += g_3 * g_4 * g_8;
        gout1 += g_2 * g_5 * g_8;
        gout2 += g_2 * g_4 * g_9;
        gout3 += g_1 * g_6 * g_8;
        gout4 += g_0 * g_7 * g_8;
        gout5 += g_0 * g_6 * g_9;
        gout6 += g_1 * g_4 * g_10;
        gout7 += g_0 * g_5 * g_10;
        gout8 += g_0 * g_4 * g_11;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 1 * jstride, gout3);
    atomicAdd(peri + 1 * istride + 1 * jstride, gout4);
    atomicAdd(peri + 2 * istride + 1 * jstride, gout5);
    atomicAdd(peri + 2 * jstride, gout6);
    atomicAdd(peri + 1 * istride + 2 * jstride, gout7);
    atomicAdd(peri + 2 * istride + 2 * jstride, gout8);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel1110(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x + xixj;
        double g_3 = c00x * (c00x + xixj) + b10;
        double g_4 = c0px;
        double g_5 = c0px * c00x + b00;
        double g_6 = c0px * (c00x + xixj) + b00;
        double g_7 = b00 * c00x + b10 * c0px + c00x * g_5 + xixj * g_5;
        double g_8 = 1;
        double g_9 = c00y;
        double g_10 = c00y + yiyj;
        double g_11 = c00y * (c00y + yiyj) + b10;
        double g_12 = c0py;
        double g_13 = c0py * c00y + b00;
        double g_14 = c0py * (c00y + yiyj) + b00;
        double g_15 = b00 * c00y + b10 * c0py + c00y * g_13 + yiyj * g_13;
        double g_16 = weight0 * fac;
        double g_17 = c00z * g_16;
        double g_18 = g_16 * (c00z + zizj);
        double g_19 = b10 * g_16 + c00z * g_17 + zizj * g_17;
        double g_20 = c0pz * g_16;
        double g_21 = b00 * g_16 + c0pz * g_17;
        double g_22 = b00 * g_16 + c0pz * g_17 + zizj * g_20;
        double g_23 = b00 * g_17 + b10 * g_20 + c00z * g_21 + zizj * g_21;
        gout0 += g_7 * g_8 * g_16;
        gout1 += g_6 * g_9 * g_16;
        gout2 += g_6 * g_8 * g_17;
        gout3 += g_5 * g_10 * g_16;
        gout4 += g_4 * g_11 * g_16;
        gout5 += g_4 * g_10 * g_17;
        gout6 += g_5 * g_8 * g_18;
        gout7 += g_4 * g_9 * g_18;
        gout8 += g_4 * g_8 * g_19;
        gout9 += g_3 * g_12 * g_16;
        gout10 += g_2 * g_13 * g_16;
        gout11 += g_2 * g_12 * g_17;
        gout12 += g_1 * g_14 * g_16;
        gout13 += g_0 * g_15 * g_16;
        gout14 += g_0 * g_14 * g_17;
        gout15 += g_1 * g_12 * g_18;
        gout16 += g_0 * g_13 * g_18;
        gout17 += g_0 * g_12 * g_19;
        gout18 += g_3 * g_8 * g_20;
        gout19 += g_2 * g_9 * g_20;
        gout20 += g_2 * g_8 * g_21;
        gout21 += g_1 * g_10 * g_20;
        gout22 += g_0 * g_11 * g_20;
        gout23 += g_0 * g_10 * g_21;
        gout24 += g_1 * g_8 * g_22;
        gout25 += g_0 * g_9 * g_22;
        gout26 += g_0 * g_8 * g_23;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        gout10 = reduce(gout10);
        gout11 = reduce(gout11);
        gout12 = reduce(gout12);
        gout13 = reduce(gout13);
        gout14 = reduce(gout14);
        gout15 = reduce(gout15);
        gout16 = reduce(gout16);
        gout17 = reduce(gout17);
        gout18 = reduce(gout18);
        gout19 = reduce(gout19);
        gout20 = reduce(gout20);
        gout21 = reduce(gout21);
        gout22 = reduce(gout22);
        gout23 = reduce(gout23);
        gout24 = reduce(gout24);
        gout25 = reduce(gout25);
        gout26 = reduce(gout26);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 1 * jstride, gout3);
    atomicAdd(peri + 1 * istride + 1 * jstride, gout4);
    atomicAdd(peri + 2 * istride + 1 * jstride, gout5);
    atomicAdd(peri + 2 * jstride, gout6);
    atomicAdd(peri + 1 * istride + 2 * jstride, gout7);
    atomicAdd(peri + 2 * istride + 2 * jstride, gout8);
    atomicAdd(peri + 1 * kstride, gout9);
    atomicAdd(peri + 1 * istride + 1 * kstride, gout10);
    atomicAdd(peri + 2 * istride + 1 * kstride, gout11);
    atomicAdd(peri + 1 * jstride + 1 * kstride, gout12);
    atomicAdd(peri + 1 * istride + 1 * jstride + 1 * kstride, gout13);
    atomicAdd(peri + 2 * istride + 1 * jstride + 1 * kstride, gout14);
    atomicAdd(peri + 2 * jstride + 1 * kstride, gout15);
    atomicAdd(peri + 1 * istride + 2 * jstride + 1 * kstride, gout16);
    atomicAdd(peri + 2 * istride + 2 * jstride + 1 * kstride, gout17);
    atomicAdd(peri + 2 * kstride, gout18);
    atomicAdd(peri + 1 * istride + 2 * kstride, gout19);
    atomicAdd(peri + 2 * istride + 2 * kstride, gout20);
    atomicAdd(peri + 1 * jstride + 2 * kstride, gout21);
    atomicAdd(peri + 1 * istride + 1 * jstride + 2 * kstride, gout22);
    atomicAdd(peri + 2 * istride + 1 * jstride + 2 * kstride, gout23);
    atomicAdd(peri + 2 * jstride + 2 * kstride, gout24);
    atomicAdd(peri + 1 * istride + 2 * jstride + 2 * kstride, gout25);
    atomicAdd(peri + 2 * istride + 2 * jstride + 2 * kstride, gout26);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel2000(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
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
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = 1;
        double g_4 = c00y;
        double g_5 = c00y * c00y + b10;
        double g_6 = weight0 * fac;
        double g_7 = c00z * g_6;
        double g_8 = b10 * g_6 + c00z * g_7;
        gout0 += g_2 * g_3 * g_6;
        gout1 += g_1 * g_4 * g_6;
        gout2 += g_1 * g_3 * g_7;
        gout3 += g_0 * g_5 * g_6;
        gout4 += g_0 * g_4 * g_7;
        gout5 += g_0 * g_3 * g_8;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 3 * istride, gout3);
    atomicAdd(peri + 4 * istride, gout4);
    atomicAdd(peri + 5 * istride, gout5);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel2010(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c0px;
        double g_4 = c0px * c00x + b00;
        double g_5 = b00 * c00x + b10 * c0px + c00x * g_4;
        double g_6 = 1;
        double g_7 = c00y;
        double g_8 = c00y * c00y + b10;
        double g_9 = c0py;
        double g_10 = c0py * c00y + b00;
        double g_11 = b00 * c00y + b10 * c0py + c00y * g_10;
        double g_12 = weight0 * fac;
        double g_13 = c00z * g_12;
        double g_14 = b10 * g_12 + c00z * g_13;
        double g_15 = c0pz * g_12;
        double g_16 = b00 * g_12 + c0pz * g_13;
        double g_17 = b00 * g_13 + b10 * g_15 + c00z * g_16;
        gout0 += g_5 * g_6 * g_12;
        gout1 += g_4 * g_7 * g_12;
        gout2 += g_4 * g_6 * g_13;
        gout3 += g_3 * g_8 * g_12;
        gout4 += g_3 * g_7 * g_13;
        gout5 += g_3 * g_6 * g_14;
        gout6 += g_2 * g_9 * g_12;
        gout7 += g_1 * g_10 * g_12;
        gout8 += g_1 * g_9 * g_13;
        gout9 += g_0 * g_11 * g_12;
        gout10 += g_0 * g_10 * g_13;
        gout11 += g_0 * g_9 * g_14;
        gout12 += g_2 * g_6 * g_15;
        gout13 += g_1 * g_7 * g_15;
        gout14 += g_1 * g_6 * g_16;
        gout15 += g_0 * g_8 * g_15;
        gout16 += g_0 * g_7 * g_16;
        gout17 += g_0 * g_6 * g_17;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        gout10 = reduce(gout10);
        gout11 = reduce(gout11);
        gout12 = reduce(gout12);
        gout13 = reduce(gout13);
        gout14 = reduce(gout14);
        gout15 = reduce(gout15);
        gout16 = reduce(gout16);
        gout17 = reduce(gout17);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 3 * istride, gout3);
    atomicAdd(peri + 4 * istride, gout4);
    atomicAdd(peri + 5 * istride, gout5);
    atomicAdd(peri + 1 * kstride, gout6);
    atomicAdd(peri + 1 * istride + 1 * kstride, gout7);
    atomicAdd(peri + 2 * istride + 1 * kstride, gout8);
    atomicAdd(peri + 3 * istride + 1 * kstride, gout9);
    atomicAdd(peri + 4 * istride + 1 * kstride, gout10);
    atomicAdd(peri + 5 * istride + 1 * kstride, gout11);
    atomicAdd(peri + 2 * kstride, gout12);
    atomicAdd(peri + 1 * istride + 2 * kstride, gout13);
    atomicAdd(peri + 2 * istride + 2 * kstride, gout14);
    atomicAdd(peri + 3 * istride + 2 * kstride, gout15);
    atomicAdd(peri + 4 * istride + 2 * kstride, gout16);
    atomicAdd(peri + 5 * istride + 2 * kstride, gout17);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel2100(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
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

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x + xixj;
        double g_4 = c00x * (c00x + xixj) + b10;
        double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_6 = 1;
        double g_7 = c00y;
        double g_8 = c00y * c00y + b10;
        double g_9 = c00y + yiyj;
        double g_10 = c00y * (c00y + yiyj) + b10;
        double g_11 = c00y * (2 * b10 + g_8) + yiyj * g_8;
        double g_12 = weight0 * fac;
        double g_13 = c00z * g_12;
        double g_14 = b10 * g_12 + c00z * g_13;
        double g_15 = g_12 * (c00z + zizj);
        double g_16 = b10 * g_12 + c00z * g_13 + zizj * g_13;
        double g_17 = 2 * b10 * g_13 + c00z * g_14 + zizj * g_14;
        gout0 += g_5 * g_6 * g_12;
        gout1 += g_4 * g_7 * g_12;
        gout2 += g_4 * g_6 * g_13;
        gout3 += g_3 * g_8 * g_12;
        gout4 += g_3 * g_7 * g_13;
        gout5 += g_3 * g_6 * g_14;
        gout6 += g_2 * g_9 * g_12;
        gout7 += g_1 * g_10 * g_12;
        gout8 += g_1 * g_9 * g_13;
        gout9 += g_0 * g_11 * g_12;
        gout10 += g_0 * g_10 * g_13;
        gout11 += g_0 * g_9 * g_14;
        gout12 += g_2 * g_6 * g_15;
        gout13 += g_1 * g_7 * g_15;
        gout14 += g_1 * g_6 * g_16;
        gout15 += g_0 * g_8 * g_15;
        gout16 += g_0 * g_7 * g_16;
        gout17 += g_0 * g_6 * g_17;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        gout10 = reduce(gout10);
        gout11 = reduce(gout11);
        gout12 = reduce(gout12);
        gout13 = reduce(gout13);
        gout14 = reduce(gout14);
        gout15 = reduce(gout15);
        gout16 = reduce(gout16);
        gout17 = reduce(gout17);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 3 * istride, gout3);
    atomicAdd(peri + 4 * istride, gout4);
    atomicAdd(peri + 5 * istride, gout5);
    atomicAdd(peri + 1 * jstride, gout6);
    atomicAdd(peri + 1 * istride + 1 * jstride, gout7);
    atomicAdd(peri + 2 * istride + 1 * jstride, gout8);
    atomicAdd(peri + 3 * istride + 1 * jstride, gout9);
    atomicAdd(peri + 4 * istride + 1 * jstride, gout10);
    atomicAdd(peri + 5 * istride + 1 * jstride, gout11);
    atomicAdd(peri + 2 * jstride, gout12);
    atomicAdd(peri + 1 * istride + 2 * jstride, gout13);
    atomicAdd(peri + 2 * istride + 2 * jstride, gout14);
    atomicAdd(peri + 3 * istride + 2 * jstride, gout15);
    atomicAdd(peri + 4 * istride + 2 * jstride, gout16);
    atomicAdd(peri + 5 * istride + 2 * jstride, gout17);
}

template <bool T>
__global__ static void GINTfill_int2e_kernel3000(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
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
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = 1;
        double g_5 = c00y;
        double g_6 = c00y * c00y + b10;
        double g_7 = c00y * (2 * b10 + g_6);
        double g_8 = weight0 * fac;
        double g_9 = c00z * g_8;
        double g_10 = b10 * g_8 + c00z * g_9;
        double g_11 = 2 * b10 * g_9 + c00z * g_10;
        gout0 += g_3 * g_4 * g_8;
        gout1 += g_2 * g_5 * g_8;
        gout2 += g_2 * g_4 * g_9;
        gout3 += g_1 * g_6 * g_8;
        gout4 += g_1 * g_5 * g_9;
        gout5 += g_1 * g_4 * g_10;
        gout6 += g_0 * g_7 * g_8;
        gout7 += g_0 * g_6 * g_9;
        gout8 += g_0 * g_5 * g_10;
        gout9 += g_0 * g_4 * g_11;
    }

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
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        if (!reduce.first)
            return;
    }
    atomicAdd(peri + 0 * istride, gout0);
    atomicAdd(peri + 1 * istride, gout1);
    atomicAdd(peri + 2 * istride, gout2);
    atomicAdd(peri + 3 * istride, gout3);
    atomicAdd(peri + 4 * istride, gout4);
    atomicAdd(peri + 5 * istride, gout5);
    atomicAdd(peri + 6 * istride, gout6);
    atomicAdd(peri + 7 * istride, gout7);
    atomicAdd(peri + 8 * istride, gout8);
    atomicAdd(peri + 9 * istride, gout9);
}

template <bool T, int BLOCKDIM = 64, int WARPSIZE = 32>
__global__ static void GINTfill_int2e_kernel3000_smem(ERITensor eri,
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
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
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
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[4];
    double root0, weight0;
    GINTrys_root<2>(x, rw);
    int irys;
    for (irys = 0; irys < 2; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 2];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = 1;
        double g_5 = c00y;
        double g_6 = c00y * c00y + b10;
        double g_7 = c00y * (2 * b10 + g_6);
        double g_8 = weight0 * fac;
        double g_9 = c00z * g_8;
        double g_10 = b10 * g_8 + c00z * g_9;
        double g_11 = 2 * b10 * g_9 + c00z * g_10;
        gout0 += g_3 * g_4 * g_8;
        gout1 += g_2 * g_5 * g_8;
        gout2 += g_2 * g_4 * g_9;
        gout3 += g_1 * g_6 * g_8;
        gout4 += g_1 * g_5 * g_9;
        gout5 += g_1 * g_4 * g_10;
        gout6 += g_0 * g_7 * g_8;
        gout7 += g_0 * g_6 * g_9;
        gout8 += g_0 * g_5 * g_10;
        gout9 += g_0 * g_4 * g_11;
    }

    size_t istride = eri.stride_i;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int *ao_loc = bpcache.ao_loc;
    int i0 = ao_loc[ish] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh] - eri.ao_offsets_k;
    int l0 = ao_loc[lsh] - eri.ao_offsets_l;
    size_t eri_offset =
        l0 * lstride + k0 * kstride + j0 * jstride + i0 * istride;
    double *peri = eri.data;
    if constexpr (T) {
        auto reduce = SegReduce<double>(igroup);
        gout0 = reduce(gout0);
        gout1 = reduce(gout1);
        gout2 = reduce(gout2);
        gout3 = reduce(gout3);
        gout4 = reduce(gout4);
        gout5 = reduce(gout5);
        gout6 = reduce(gout6);
        gout7 = reduce(gout7);
        gout8 = reduce(gout8);
        gout9 = reduce(gout9);
        if (!reduce.first)
            return;
    }

    __shared__ double gout_smem[BLOCKDIM][11];
    int bid = threadIdx.x;
    int warp_id = threadIdx.x / WARPSIZE;
    int lane_id = threadIdx.x - warp_id * WARPSIZE;
    gout_smem[bid][0] = gout0;
    gout_smem[bid][1] = gout1;
    gout_smem[bid][2] = gout2;
    gout_smem[bid][3] = gout3;
    gout_smem[bid][4] = gout4;
    gout_smem[bid][5] = gout5;
    gout_smem[bid][6] = gout6;
    gout_smem[bid][7] = gout7;
    gout_smem[bid][8] = gout8;
    gout_smem[bid][9] = gout9;

    for (int i = 0; i < WARPSIZE; i++) {
        size_t ptr_offset;
        ptr_offset = __shfl_sync(0xffffffff, eri_offset, i, WARPSIZE);
        if (lane_id < 10) {
            atomicAdd(peri + ptr_offset + lane_id * istride,
                gout_smem[i + warp_id * WARPSIZE][lane_id]);
        }
    }
}
