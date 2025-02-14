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

#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <assert.h>
#include "cuda_alloc.cuh"
#include "ft_ao.h"

#include "ft_ao_latsum_ext.h"

template <bool prim>
void PBC_ft_latsum(const int eval_gz, cuDoubleComplex *out, void *buf,
    size_t bufsize, int nkpts, int comp, int nimgs, double *d_Ls,
    cuDoubleComplex *d_expkL, int *shls_slice, int *d_ao_loc, double *d_Gv,
    double *d_b, int *d_gxyz, int *gs, int nGv, int *atm, int natm, int *bas,
    int nbas, double *env) {
    int nenv = PBCsizeof_env(shls_slice, atm, natm, bas, nbas, env);
    nenv = MAX(nenv, PBCsizeof_env(shls_slice + 2, atm, natm, bas, nbas, env));
    DEVICE_INIT_MEMPOOL(int, d_bas, bas, 2 * 8 * nbas, buf, bufsize);
    DEVICE_INIT_MEMPOOL(int, d_atm, atm, 2 * 6 * natm, buf, bufsize);
    // DEVICE_INIT_MEMPOOL(int, d_ao_loc, ao_loc, 2 * nbas + 1, buf, bufsize);
    DEVICE_INIT_MEMPOOL(double, d_env, env, nenv, buf, bufsize);
    const int li = bas(ANG_OF, shls_slice[0]);
    const int lj = bas(ANG_OF, shls_slice[2]);
    const int l = (li << 3) + lj;
    // printf("l: %d li:%d, ji:%d\n", l, li, lj);
    switch (l) {
    case 0:
        _PBC_ft_latsum<prim, 0, 0>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 8:
        _PBC_ft_latsum<prim, 1, 0>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 9:
        _PBC_ft_latsum<prim, 1, 1>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 16:
        _PBC_ft_latsum<prim, 2, 0>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 17:
        _PBC_ft_latsum<prim, 2, 1>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 18:
        _PBC_ft_latsum<prim, 2, 2>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 24:
        _PBC_ft_latsum<prim, 3, 0>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 25:
        _PBC_ft_latsum<prim, 3, 1>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 26:
        _PBC_ft_latsum<prim, 3, 2>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 27:
        _PBC_ft_latsum<prim, 3, 3>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 32:
        _PBC_ft_latsum<prim, 4, 0>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 33:
        _PBC_ft_latsum<prim, 4, 1>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 34:
        _PBC_ft_latsum<prim, 4, 2>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 35:
        _PBC_ft_latsum<prim, 4, 3>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 36:
        _PBC_ft_latsum<prim, 4, 4>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 40:
        _PBC_ft_latsum<prim, 5, 0>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 41:
        _PBC_ft_latsum<prim, 5, 1>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 42:
        _PBC_ft_latsum<prim, 5, 2>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 43:
        _PBC_ft_latsum<prim, 5, 3>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 44:
        _PBC_ft_latsum<prim, 5, 4>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    case 45:
        _PBC_ft_latsum<prim, 5, 5>(eval_gz, out, nkpts, comp, nimgs, d_env,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            d_atm, natm, d_bas, nbas);
        break;
    default:
        printf("Unexpected i:%d and j:%d!\n", li, lj);
        assert(false);
    }
}
void PBC_ft_latsum_drv(const int eval_gz, cuDoubleComplex *out, void *d_buf,
    size_t bufsize, int nkpts, int comp, int nimgs, double *d_Ls,
    cuDoubleComplex *d_expkL, int *shls_slice, int *d_ao_loc, double *d_Gv,
    double *d_b, int *d_gxyz, int *gs, int nGv, int *atm, int natm, int *bas,
    int nbas, double *env) {
    NVTX_PUSH("Drv&Fill latsum");
    const int i_prim = bas(NPRIM_OF, shls_slice[0]);
    const int j_prim = bas(NPRIM_OF, shls_slice[2]);
    if (i_prim * j_prim < 3)
        PBC_ft_latsum<true>(eval_gz, out, d_buf, bufsize, nkpts, comp, nimgs,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            atm, natm, bas, nbas, env);
    else
        PBC_ft_latsum<false>(eval_gz, out, d_buf, bufsize, nkpts, comp, nimgs,
            d_Ls, d_expkL, shls_slice, d_ao_loc, d_Gv, d_b, d_gxyz, gs, nGv,
            atm, natm, bas, nbas, env);
    NVTX_POP();
}
