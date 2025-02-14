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

from pyscf.pbc import scf
from byteqc.cupbc import scf as pscf
import sys
from pyscf.lib import logger
from byteqc.cupbc.test.getconfig import getconfig
from pyscf.pbc import gto

atom, basis, a, kmesh, omega, pseudo, name, _ = getconfig()
cell = gto.Cell(atom=atom, basis=basis, a=a, pseudo=pseudo)
cell.verbose = 0
cell.build()

print("nbas:%s nao:%d" % (cell.nbas, cell.nao))
scaled_center = None
# scaled_center = np.random.rand(3)
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
log = logger.Logger(cell.stdout, 6)
log.info("kmesh= %s", kmesh)
log.info("kpts = %s", kpts)
max_cycle = 0


def run_rsdf(scf, max_cycle=0, verbose=3, **kargs):
    mf = scf.KRHF(cell, kpts=kpts).rs_density_fit(**kargs)
    mf.verbose = verbose
    mf.max_memory = 700000
    mf.with_df.max_memory = 700000
    mf.max_cycle = max_cycle
    mf.with_df.omega = omega
    mf.with_df.direct = True
    mf.with_df.ksym = 's1'
    mf.with_df.use_bvk = [True, True]
    # mf.with_df.build(with_j3c=True)
    mf.kernel()
    return mf.e_tot


RSDF_gpu_Enabled = True
RSDF_cpu_Enabled = True
if len(sys.argv) >= 2:
    if (sys.argv[1]) == 'cpu':
        RSDF_gpu_Enabled = False
    if (sys.argv[1]) == 'gpu':
        RSDF_cpu_Enabled = False


if (RSDF_gpu_Enabled):
    egpu = run_rsdf(pscf, max_cycle)
if (RSDF_cpu_Enabled):
    ecpu = run_rsdf(scf, max_cycle)

if (RSDF_gpu_Enabled):
    log.info("HF/GPU-RSDF energy   : % .10f", egpu)
if (RSDF_cpu_Enabled):
    log.info("HF/CPU-RSDF energy   : % .10f", ecpu)
