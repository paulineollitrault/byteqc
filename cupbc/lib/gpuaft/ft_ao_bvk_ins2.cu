/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ByteQC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <cuComplex.h>
#include <assert.h>
#include "ft_ao.h"

#include "ft_ao_template.cu"

template void _PBC_ft_bvk<true, 5, 5>(const int, cuDoubleComplex *, const int,
    const int, const int, const int, const int *, const int8_t *, double *,
    const double *, const cuDoubleComplex *, const int *, const int *,
    const double *, const double *, const int *, const int *, const int, int *,
    const int, int *, const int);
