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

#include "gint.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif
void GINTinit_EnvVars(
    GINTEnvVars *envs, ContractionProdType *cp_ij, ContractionProdType *cp_kl);

void GINTinit_contraction_types(BasisProdCache *bpcache, int *bas_pair2shls,
    int *bas_pairs_locs, int ncptype, int *atm, int natm, int *bas, int nbas,
    double *env);
void GINTsort_bas_coordinates(
    double *bas_coords, int *atm, int natm, int *bas, int nbas, double *env);
void GINTinit_aexyz(double *aexyz, BasisProdCache *bpcache, double diag_fac,
    int *atm, int natm, int *bas, int nbas, double *env);
#ifdef __cplusplus
}
#endif
