/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf,
https://github.com/hongzhouye/pyscf/tree/rsdf_direct),
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
    Author: Hong-Zhou Ye <hzyechem@gmail.com>

*/

#include "gint.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif
void GINT_get_3c2e_Rcuts(int nijd, int nbasaux, double *Rcuts, double *dijs,
    double *Qijs, int *dijs_ij, int *ls, double *es, double *cs, int *lks,
    double *eks, double *cks, double omega, double precision, int estimator);
#ifdef __cplusplus
}
#endif
