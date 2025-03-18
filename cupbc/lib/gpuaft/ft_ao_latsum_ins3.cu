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

#include <cuComplex.h>
#include <assert.h>
#include "ft_ao.h"

#include "ft_ao_template.cu"

template void _PBC_ft_latsum<false, 4, 4>(const int, cuDoubleComplex *,
    const int, const int, const int, double *, const double *,
    const cuDoubleComplex *, const int *, const int *, const double *,
    const double *, const int *, const int *, const int, int *, const int,
    int *, const int);
template void _PBC_ft_latsum<false, 5, 4>(const int, cuDoubleComplex *,
    const int, const int, const int, double *, const double *,
    const cuDoubleComplex *, const int *, const int *, const double *,
    const double *, const int *, const int *, const int, int *, const int,
    int *, const int);
