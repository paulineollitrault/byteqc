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
from pyscf import gto, scf
from byteqc import cump2
import os


def CO2_on_CPO_27_Mg(basis_set, mol_type):
    current = Path(__file__).resolve().parent
    mol_path = os.path.join(current, f'CO2+6B_cluster.xyz')

    mol_ref = gto.M()
    mol_ref.atom = mol_path

    mol_ref.basis = basis_set
    mol_ref.verbose = 6
    mol_ref.build()

    mol = mol_ref.copy()

    if mol_type == 1:
        mol.atom = [(mol_ref.atom_pure_symbol(atom_ind),
                     mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                    else ('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                          mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                    for atom_ind in range(mol_ref.natm)]
    elif mol_type == 0:
        mol.atom = [('ghost-' + mol_ref.atom_pure_symbol(atom_ind),
                     mol_ref.atom_coord(atom_ind, 'angstrom').tolist()) if atom_ind > 2
                    else (mol_ref.atom_pure_symbol(atom_ind),
                          mol_ref.atom_coord(atom_ind, 'angstrom').tolist())
                    for atom_ind in range(mol_ref.natm)]
    elif mol_type == 2:
        pass
    else:
        assert False

    if 'ccecp' in basis_set:
        mol.ecp = 'ccecp'

    mol.build()

    return mol


if __name__ == '__main__':
    basis = 'aug-cc-pVDZ'
    # basis = 'aug-cc-pVTZ'

    mol_type = 2
    # mol_type = 1
    # mol_type = 0

    mol = CO2_on_CPO_27_Mg(basis, mol_type)

    current = Path(__file__).resolve().parent
    if mol_type == 2:
        logdir = os.path.join(current, f'{basis}/Full')
    elif mol_type == 1:
        logdir = os.path.join(current, f'{basis}/ghost-CO2')
    elif mol_type == 0:
        logdir = os.path.join(current, f'{basis}/ghost-MOF')
    else:
        assert False

    chkfile = os.path.join(logdir, 'HF_chkfile.chk')
    JK_file = os.path.join(logdir, 'JK_file.npy')
    assert os.path.exists(chkfile) and os.path.exists(JK_file)

    logfile = os.path.join(logdir, 'MP2.log')
    mol.output = logfile
    mol.build()
    mf = scf.RHF(mol)
    mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))
    e_corr = cump2.DFMP2(mol, mf, auxbasis='weigend+etb', with_rdm1=False)
