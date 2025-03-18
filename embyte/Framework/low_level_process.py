# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ByteQC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import h5py
import os
import time
from byteqc import lib
from byteqc.embyte.Localization import iao_pao_localization
from byteqc.embyte.Tools.tool_lib import fix_orbital_sign
from functools import reduce
import numpy
from byteqc.embyte.ERI import eri_trans
from byteqc.cuobc import scf
import pyscf
from pyscf import df
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


class low_level_info:
    '''
    Collect all the low level information from the mean field calculation.
    Commonly the mean field comes from the converged HF check file which is provided by pyscf.
    The low-level calculation would not be done in SIE workflow.
    '''

    def __init__(self, mol,
                 mf,
                 LG,
                 aux_basis=None,
                 jk_file=None,
                 with_eri=False,
                 oei=None,
                 local_orb_path=None,
                 ):

        logger = LG.logger
        if aux_basis is None:
            self.auxmol = df.addons.make_auxmol(mol, df.make_auxbasis(mol))
        else:
            self.auxmol = df.addons.make_auxmol(mol, aux_basis)

        self.mo_energy = mf.mo_energy.copy()
        self.mol_full = mol

        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

        blksize = int((lib.gpu_avail_bytes() / (8 * 2)) ** (1 / 3))

        if with_eri is False:
            self.j2c = os.path.join(LG.filepath + 'j2c')
            lib.free_all_blocks()
            with h5py.File(self.j2c, 'w') as f:
                logger.info("--- start to get j2c")
                j2c, self.nL_un, self.nL = eri_trans.get_j2c(
                    logger, mol=self.mol_full, auxmol=self.auxmol)
                f.create_dataset('j2c', data=j2c, dtype='f8')
                del j2c
                logger.info(f'--- Save j2c to the disk {self.j2c}')
        else:

            self.j2c = None

        lib.free_all_blocks()

        t_localized = time.time()

        if local_orb_path == 'lowdin':
            logger.info("----------- Using meta_lowdin")
            self.AOLO = cupy.asarray(
                pyscf.lo.orth.orth_ao(
                    mf, method='meta_lowdin'))
        elif local_orb_path is not None:
            logger.info("----------- Using load AOLO from file")
            self.AOLO = cupy.asarray(numpy.load(local_orb_path))
        else:
            logger.info("----------- Using IAO+PAO localizer")
            self.AOLO = fix_orbital_sign(iao_pao_localization(mol, mf))

        logger.info(
            '----------- localization_function time cost is %s' %
            (time.time() - t_localized))

        self.low_scf_energy = mf.e_tot
        mf_mo_coeff = cupy.asarray(mf.mo_coeff)
        low_scf_dm = reduce(
            cupy.dot,
            (
                mf_mo_coeff,
                cupy.diag(cupy.asarray(mf.mo_occ)),
                mf_mo_coeff.T
            )
        )

        self.onerdm_low_ao = low_scf_dm.get()
        if jk_file is None:
            blksize = int((lib.gpu_avail_bytes() / (8 * 2)) ** (1 / 4))
            vhfopt = scf.hf._VHFOpt(mol, 'int2e')
            vhfopt.build(group_size=blksize)
            t_get_JK = time.time()
            j, k = scf.hf.get_jk(mol, low_scf_dm.get(), vhfopt=vhfopt)
            logger.info(
                '----------- get_jk time cost is %s' %
                (time.time() - t_get_JK))
            del vhfopt
            lib.free_all_blocks()
            low_scf_twoint = cupy.asarray(j - 0.5 * k)
        else:
            logger.info('----------- load JK from %s' % jk_file)
            low_scf_twoint = cupy.asarray(numpy.load(jk_file))

        if oei is None:
            self.low_scf_fock = cupy.asarray(
                mf.get_hcore(mol) + low_scf_twoint.get())
        else:
            assert isinstance(oei, str), f'The oei type is not right, expect str but {type(oei)} provided'
            self.low_scf_fock = cupy.asarray(
                numpy.load(oei) + low_scf_twoint.get())

        self.ao_ovlp = cupy.asarray(mf.get_ovlp(mol))

        if not numpy.isclose(reduce(numpy.dot, (self.AOLO.T,
                                                self.ao_ovlp, self.AOLO)).sum(), mol.nao):
            logger.info(
                f'+++ localized orbitals may not orthogonal!')

        self.LOMO = reduce(
            cupy.dot, (self.AOLO.T, self.ao_ovlp, mf_mo_coeff))
        self.LOMO = cupy.asarray(fix_orbital_sign(self.LOMO.get()))
        self.onerdm_low = reduce(
            cupy.dot,
            (self.LOMO,
             cupy.diag(
                 cupy.asarray(
                     mf.mo_occ)),
                self.LOMO.T)).get()

        self.core_constant_energy = mf.mol.energy_nuc()
        self.oei_LO = reduce(
            cupy.dot,
            (self.AOLO.T,
             self.low_scf_fock
             - low_scf_twoint,
             self.AOLO)).get()
        self.fock_LO = reduce(
            cupy.dot,
            (self.AOLO.T,
             self.low_scf_fock,
             self.AOLO)).get()
        self.low_scf_fock = self.low_scf_fock.get()
        self.MOLO = reduce(cupy.dot, (mf_mo_coeff.T, self.ao_ovlp, self.AOLO))
        self.ao_ovlp = self.ao_ovlp.get()
        self.LOMO = self.LOMO.get()
        self.AOMO = mf_mo_coeff.get()
        del mf_mo_coeff

        self.num_occ = int(mol.nelectron / 2)

        self.MOLO = self.MOLO.get()

        del mf
        try:
            del self.auxmol.stdout
        except:
            pass
