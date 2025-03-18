/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

Licensed under the Apache License, Version 2.0 (the "License")
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https: // www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

template <bool, int, int>
void _PBC_ft_latsum(const int, cuDoubleComplex *, const int, const int,
    const int, double *, const double *, const cuDoubleComplex *, const int *,
    const int *, const double *, const double *, const int *, const int *,
    const int, int *, const int, int *, const int);

#define FUNC_LATSUM(F, I, J)                                                   \
    extern template void _PBC_ft_latsum<F, I, J>(const int, cuDoubleComplex *, \
        const int, const int, const int, double *, const double *,             \
        const cuDoubleComplex *, const int *, const int *, const double *,     \
        const double *, const int *, const int *, const int, int *, const int, \
        int *, const int);

#define FUNC_LATSUM_F(I, J) FUNC_LATSUM(true, I, J) FUNC_LATSUM(false, I, J)

#define FUNC_LATSUM_F_0(I) FUNC_LATSUM_F(I, 0)
#define FUNC_LATSUM_F_1(I) FUNC_LATSUM_F(I, 1) FUNC_LATSUM_F_0(I)
#define FUNC_LATSUM_F_2(I) FUNC_LATSUM_F(I, 2) FUNC_LATSUM_F_1(I)
#define FUNC_LATSUM_F_3(I) FUNC_LATSUM_F(I, 3) FUNC_LATSUM_F_2(I)
#define FUNC_LATSUM_F_4(I) FUNC_LATSUM_F(I, 4) FUNC_LATSUM_F_3(I)
#define FUNC_LATSUM_F_5(I) FUNC_LATSUM_F(I, 5) FUNC_LATSUM_F_4(I)

#define FUNC_LATSUM_F_J(I) FUNC_LATSUM_F_##I(I)

FUNC_LATSUM_F_J(0)
FUNC_LATSUM_F_J(1)
FUNC_LATSUM_F_J(2)
FUNC_LATSUM_F_J(3)
FUNC_LATSUM_F_J(4)
FUNC_LATSUM_F_J(5)
