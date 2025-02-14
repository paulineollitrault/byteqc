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
import numpy
import sys
from pyscf.lib import logger
from byteqc.cupbctest.getconfig import getconfig
from pyscf.pbc import gto
from byteqc.cupbc.df.rsdf_direct_helper import get_kptij_lst
from byteqc.cupbc import scf

atom, basis, a, kmesh, omega, pseudo, name, _ = getconfig()
cell = gto.Cell(atom=atom, basis=basis, a=a, pseudo=pseudo)
cell.verbose = 0
cell.build()

print("nbas:%s nao:%d" % (cell.nbas, cell.nao))
scaled_center = None
# scaled_center = np.random.rand(3)
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
kptij_lst = get_kptij_lst(kpts, ksym='s1')
log = logger.Logger(cell.stdout, 6)
log.info("kmesh= %s", kmesh)
log.info("kpts = %s", kpts)
max_cycle = 0

mf = scf.KRHF(cell, kpts=kpts).rs_density_fit()
mf.verbose = 5
mf.max_memory = 700000
mf.with_df.max_memory = 700000
mf.max_cycle = max_cycle
mf.with_df.omega = omega
mf.with_df.direct = True
mf.with_df.ksym = 's1'
mf.with_df.use_bvk = [True, True]
mf.with_df.build(with_j3c=False)
mydf = mf.with_df
ENABLE_CPU = ENABLE_GPU = True
if len(sys.argv) > 1:
    if (sys.argv[1]) == 'cpu':
        ENABLE_GPU = False
    if (sys.argv[1]) == 'gpu':
        ENABLE_CPU = False

if ENABLE_CPU:
    mf.with_df.dump_eri('gpuj3c.dat')
if ENABLE_GPU:
    mydf._make_j3c(mydf.cell, mydf.auxcell, kptij_lst, 'cpuj3c.dat')
fc = h5py.File('cpuj3c.dat', 'r')
fg = h5py.File('gpuj3c.dat', 'r')
print('j3c-kptij is same:', numpy.allclose(fc['j3c-kptij'], fg['j3c-kptij']))
allflag = True
for k in fc['j3c'].keys():
    nstep = len(fc['j3c'][k].keys())
    cpu = numpy.concatenate([fc['j3c/%s/%d' % (k, i)][:]
                            for i in range(nstep)], axis=1)
    nstep = len(fg['j3c'][k].keys())
    gpu = numpy.concatenate([fg['j3c/%s/%d' % (k, i)][:]
                            for i in range(nstep)], axis=1)
    flag = numpy.allclose(cpu, gpu)
    if not flag:
        print('j3c[%s] is not same!!!' % (k,), abs(cpu - gpu).max())
        allflag = False
    else:
        print('j3c[%s] is same' % (k,), abs(cpu - gpu).max())
if allflag:
    print('j3c is same!')

fc.close()
fg.close()
