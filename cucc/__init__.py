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

from pyscf import scf
from byteqc.cucc import ccsd
from byteqc.cucc import dfccsd
from pyscf.lib import param


def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None, gpulim=None,
         cpulim=None, pool=None, path=param.TMPDIR, mem_ratio=0.7):
    if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.ghf.GHF):
        AssertionError('Not implement')
    else:
        return RCCSD(mf, frozen, mo_coeff, mo_occ, gpulim, cpulim,
                     pool=pool, path=path, mem_ratio=mem_ratio)


def RCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None, gpulim=None,
          cpulim=None, pool=None, path=param.TMPDIR, mem_ratio=0.7):
    if getattr(mf, 'with_df', None):
        return dfccsd.RCCSD(mf, frozen, mo_coeff, mo_occ, gpulim,
                            cpulim, pool=pool, path=path, mem_ratio=mem_ratio)
    else:
        return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ, gpulim,
                         cpulim, pool=pool, path=path, mem_ratio=mem_ratio)
