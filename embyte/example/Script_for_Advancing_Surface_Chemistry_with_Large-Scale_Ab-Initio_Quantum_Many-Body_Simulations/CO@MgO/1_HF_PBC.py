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

from pathlib import Path
from pyscf.pbc import gto as pbcgto
import numpy
import os


def MgO_CO_L4(basis_set, mol_type):
    direct_crood = numpy.asarray(
        [
            [0.500000000, 0.500000000, 0.647738350],
            [0.500000000, 0.500000000, 0.691904590],
            [0.000000000, 0.750000000, 0.308095410],
            [0.250000000, 0.000000000, 0.308095410],
            [0.000000000, 0.000000000, 0.389365055],
            [0.250000000, 0.750000000, 0.389365055],
            [0.000000000, 0.749939268, 0.471142648],
            [0.250060732, 0.000000000, 0.471142648],
            [0.000000000, 0.000000000, 0.552192338],
            [0.249858327, 0.750141673, 0.552047952],
            [0.000000000, 0.250000000, 0.308095410],
            [0.250000000, 0.500000000, 0.308095410],
            [0.000000000, 0.500000000, 0.389365055],
            [0.250000000, 0.250000000, 0.389365055],
            [0.000000000, 0.250060732, 0.471142648],
            [0.250378379, 0.500000000, 0.471254605],
            [0.000000000, 0.500000000, 0.552085521],
            [0.249858327, 0.249858327, 0.552047952],
            [0.500000000, 0.750000000, 0.308095410],
            [0.750000000, 0.000000000, 0.308095410],
            [0.500000000, 0.000000000, 0.389365055],
            [0.750000000, 0.750000000, 0.389365055],
            [0.500000000, 0.749621621, 0.471254605],
            [0.749939268, 0.000000000, 0.471142648],
            [0.500000000, 0.000000000, 0.552085521],
            [0.750141673, 0.750141673, 0.552047952],
            [0.500000000, 0.250000000, 0.308095410],
            [0.750000000, 0.500000000, 0.308095410],
            [0.500000000, 0.500000000, 0.389365055],
            [0.750000000, 0.250000000, 0.389365055],
            [0.500000000, 0.250378379, 0.471254605],
            [0.749621621, 0.500000000, 0.471254605],
            [0.500000000, 0.500000000, 0.553729877],
            [0.750141673, 0.249858327, 0.552047952],
            [0.000000000, 0.750000000, 0.389365055],
            [0.250000000, 0.000000000, 0.389365055],
            [0.000000000, 0.000000000, 0.308095410],
            [0.250000000, 0.750000000, 0.308095410],
            [0.000000000, 0.750218533, 0.554027498],
            [0.249781467, 0.000000000, 0.554027498],
            [0.000000000, 0.000000000, 0.470918096],
            [0.249945770, 0.750054230, 0.470958532],
            [0.000000000, 0.250000000, 0.389365055],
            [0.250000000, 0.500000000, 0.389365055],
            [0.000000000, 0.500000000, 0.308095410],
            [0.250000000, 0.250000000, 0.308095410],
            [0.000000000, 0.249781467, 0.554027498],
            [0.248822710, 0.500000000, 0.553948320],
            [0.000000000, 0.500000000, 0.470879301],
            [0.249945770, 0.249945770, 0.470958532],
            [0.500000000, 0.750000000, 0.389365055],
            [0.750000000, 0.000000000, 0.389365055],
            [0.500000000, 0.000000000, 0.308095410],
            [0.750000000, 0.750000000, 0.308095410],
            [0.500000000, 0.751177290, 0.553948320],
            [0.750218533, 0.000000000, 0.554027498],
            [0.500000000, 0.000000000, 0.470879301],
            [0.750054230, 0.750054230, 0.470958532],
            [0.500000000, 0.250000000, 0.389365055],
            [0.750000000, 0.500000000, 0.389365055],
            [0.500000000, 0.500000000, 0.308095410],
            [0.750000000, 0.250000000, 0.308095410],
            [0.500000000, 0.248822710, 0.553948320],
            [0.751177290, 0.500000000, 0.553948320],
            [0.500000000, 0.500000000, 0.471264325],
            [0.750054230, 0.249945770, 0.470958532],
        ]
    )

    abc_params = numpy.asarray(
        [
            [8.440984654, 0.000000000, 0.000000000],
            [0.000000000, 8.440984654, 0.000000000],
            [0.000000000, 0.000000000, 25.965982408],
        ]
    )

    symbols = {
        0: {'C1': 'C', 'O1': 'O', 'Mg': 'ghost-Mg', 'O': 'ghost-O'},
        1: {'C1': 'ghost-C', 'O1': 'ghost-O', 'Mg': 'Mg', 'O': 'O'},
        2: {'C1': 'C', 'O1': 'O', 'Mg': 'Mg', 'O': 'O'},
    }

    a = abc_params[0]
    b = abc_params[1]
    c = abc_params[2]

    cell = pbcgto.Cell()
    cell.unit = 'A'

    atom_list = []
    for atm_ind in range(direct_crood.shape[0]):
        crood = direct_crood[atm_ind]
        crood = crood[0] * a + crood[1] * b + crood[2] * c
        if atm_ind == 0:
            atom_list.append([symbols[mol_type]['C1'], crood])
        elif atm_ind == 1:
            atom_list.append([symbols[mol_type]['O1'], crood])
        elif atm_ind < 34:
            atom_list.append([symbols[mol_type]['Mg'], crood])
        else:
            atom_list.append([symbols[mol_type]['O'], crood])

    cell.atom = atom_list
    cell.basis = basis_set
    cell.a = abc_params

    cell.verbose = 6
    cell.exp_to_discard = 0.1
    cell.build()

    return cell


if __name__ == '__main__':

    basis = 'aug-cc-pVDZ'
    # basis = 'aug-cc-pVTZ'
    # basis = 'aug-cc-pVQZ'

    mol_type = 2
    # mol_type = 1
    # mol_type = 0

    cell = MgO_CO_L4(basis, mol_type)

    current = Path(__file__).resolve().parent
    if mol_type == 2:
        logdir = os.path.join(current, f'{basis}/Full')
    elif mol_type == 1:
        logdir = os.path.join(current, f'{basis}/ghost-CO')
    elif mol_type == 0:
        logdir = os.path.join(current, f'{basis}/ghost-MgO')
    else:
        assert False

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    JK_file = os.path.join(logdir, 'JK_file.npy')
    oei_file = os.path.join(logdir, 'oei.npy')
    logfile = os.path.join(logdir, 'HF.log')

    cell.output = logfile
    cell.max_memory = 900e3
    cell.build()
    from pyscf.pbc import scf as pbcscf

    mf = pbcscf.RHF(cell, exxdiv=None).density_fit(auxbasis='weigend+etb')
    mf.with_df._cderi_to_save = os.path.join(logdir, 'cderi.h5')

    mf.chkfile = chkfile
    mf.diis_space = 12

    mf.kernel()

    j, k = mf.get_jk()
    veff = j - 0.5 * k
    numpy.save(JK_file, veff)

    oei = mf.get_hcore()
    numpy.save(oei_file, oei)
