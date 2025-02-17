# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC includes code adapted from libDMET (https://github.com/gkclab/libdmet_preview)
# which are licensed under the GPL-3.0 license.
# The original copyright:
#     A library of density matrix embedding theory (DMET) for lattice models and realistic solids.
#     Copyright (C) 2022 The libDMET Developers.

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

from functools import reduce
from pyscf import gto
import numpy
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def mdot(*args):
    """
    Reduced matrix dot.
    """
    return reduce(cupy.dot, args)


def orth_cano(c, s, tol=1e-12):

    return cupy.dot(c, _cano(mdot(c.T, s, c), tol=tol))


def _cano(s, tol=1e-12):
    e, v = cupy.linalg.eigh(s)
    idx = e > tol
    return v[:, idx] / numpy.sqrt(e[idx])


def _vec_lowdin(c, s=1):
    """
    Lowdin orth for the metric c.T*s*c and get x, then c*x
    """
    return cupy.dot(c, _lowdin(mdot(c.T, s, c)))


def _lowdin(s, tol=1e-14):
    """
    New basis is |mu> c^{lowdin}_{mu i}
    """
    e, v = cupy.linalg.eigh(s)
    idx = e > tol
    return cupy.dot(v[:, idx] / cupy.sqrt(e[idx]), v[:, idx].T)


def is_ghost_atom(symb_or_chg):
    if isinstance(symb_or_chg, (int, numpy.integer)):
        return symb_or_chg == 0
    elif 'GHOST' in symb_or_chg.upper():
        return True
    else:
        return symb_or_chg[0] == 'X' and symb_or_chg[:2].upper() != 'XE'


def reference_mol(mol, minao='minao', tol=1e-12):
    '''Create a molecule which uses reference minimal basis'''
    pmol = mol.copy()
    atoms = [atom for atom in gto.format_atom(pmol.atom, unit=1)]
    # remove ghost atoms
    pmol.atom = [atom for atom in atoms if not is_ghost_atom(atom[0])]
    if len(pmol.atom) != len(atoms):
        print(mol, 'Ghost atoms found in system. '
              'Current IAO does not support ghost atoms. '
              'They are removed from IAO reference basis.')
    if getattr(pmol, 'rcut', None) is not None:
        pmol.rcut = None
    pmol.build(False, False, basis=minao)
    return pmol


def reference_mol_get_mask(mol):
    '''Create a molecule which uses reference minimal basis'''
    pmol = mol.copy()
    atoms = [atom for atom in gto.format_atom(pmol.atom, unit=1)]
    # remove ghost atoms
    mask = [i for i, atom in enumerate(atoms) if not is_ghost_atom(atom[0])]
    # mask = numpy.asarray([False] * len(atoms))

    return mask


def iao_localization(mol, mf, minao='minao', tol=1e-12):
    '''
    GPU accelerated IAO+PAO localization.
    '''

    orbocc = cupy.asarray(mf.mo_coeff, dtype=cupy.float64)[
        :, cupy.asarray(mf.mo_occ) > 0]
    # IAO part
    pmol = reference_mol(mol, minao)

    if getattr(mol, 'pbc_intor', None):  # cell object has pbc_intor method
        from pyscf.pbc import gto as pbcgto
        s1 = cupy.asarray(mol.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
        s2 = cupy.asarray(pmol.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
        s12 = cupy.asarray(
            pbcgto.cell.intor_cross(
                'int1e_ovlp',
                mol,
                pmol,
                kpts=None))
    else:

        s1 = cupy.asarray(mol.intor_symmetric('int1e_ovlp'), dtype=cupy.float64)
        s2 = cupy.asarray(pmol.intor_symmetric('int1e_ovlp'), dtype=cupy.float64)
        s12 = cupy.asarray(
            gto.mole.intor_cross(
                'int1e_ovlp',
                mol,
                pmol),
            dtype=cupy.float64)

    s21 = s12.T

    s1cd = cupy.linalg.cholesky(s1)
    s2cd = cupy.linalg.cholesky(s2)
    p12 = cupy.linalg.solve(s1cd.T, cupy.linalg.solve(s1cd, s12))

    C = orbocc
    ctild = cupy.linalg.solve(s2cd.T, cupy.linalg.solve(s2cd, cupy.dot(s21, C)))
    ctild = cupy.linalg.solve(s1cd.T, cupy.linalg.solve(s1cd, cupy.dot(s12, ctild)))
    ctild = orth_cano(ctild, s1, tol=tol)
    ccs1 = mdot(C, C.T, s1)
    ccs2 = mdot(ctild, ctild.T, s1)

    c = (p12 + mdot(ccs1, ccs2, p12) * 2 - cupy.dot(ccs1, p12) - cupy.dot(ccs2, p12))
    s = s1
    S = s

    c_occ = _vec_lowdin(c, S)
    print('occ shape:', c_occ.shape)
    if c_occ.shape[0] == c_occ.shape[1]:
        return c_occ

    C_ao_iao = c_occ
    S = s
    B1_labels = numpy.asarray(mol.ao_labels())
    pmol_tmp = mol.copy()
    pmol_tmp.basis = minao
    pmol_tmp.build()
    B2_labels = numpy.asarray(
        [r for r in pmol_tmp.ao_labels() if 'GHOST' not in r])
    notin_B2 = numpy.isin(B1_labels, B2_labels, invert=True)
    virt_idx = numpy.where(notin_B2)[0].tolist()
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_idx)
    assert nB2 + nvirt == nB1

    CCdS = mdot(C_ao_iao, C_ao_iao.T, S)

    c_vir = (cupy.eye(nB1) - CCdS)[:, virt_idx]
    c_vir = _vec_lowdin(c_vir, s)
    print('vir shape:', c_vir.shape)

    refer_mol = pmol

    mol_atom_orb_num = numpy.array(mol.ao_labels(fmt=False))
    mol_atom_orb_num = numpy.array(mol_atom_orb_num[:, 0], dtype=int)
    mol_atom_orb_num = numpy.bincount(mol_atom_orb_num)

    refer_mol_atom_orb_num = numpy.array(refer_mol.ao_labels(fmt=False))
    refer_mol_atom_orb_num = numpy.array(refer_mol_atom_orb_num[:, 0], dtype=int)
    refer_mol_atom_orb_num = numpy.bincount(refer_mol_atom_orb_num)

    if refer_mol_atom_orb_num.shape != mol_atom_orb_num.shape:
        refer_mol_atom_orb_num_temp = numpy.asarray(
            [0] * mol_atom_orb_num.shape[0])
        refer_mol_atom_orb_num_temp[reference_mol_get_mask(
            mol)] = refer_mol_atom_orb_num
        refer_mol_atom_orb_num = refer_mol_atom_orb_num_temp
        pass

    vir_atom_orb_num = mol_atom_orb_num - refer_mol_atom_orb_num
    num_occ_cumsum = numpy.cumsum(refer_mol_atom_orb_num)
    num_vir_cumsum = numpy.cumsum(vir_atom_orb_num)

    c_occ_split = cupy.split(c_occ, num_occ_cumsum, axis=1)[:-1]
    c_vir_split = cupy.split(c_vir, num_vir_cumsum, axis=1)[:-1]

    c_iao_parts = [
        cupy.hstack(
            (occ, vir)) for occ, vir in zip(
            c_occ_split, c_vir_split)]

    c_iao = cupy.hstack(c_iao_parts)

    return c_iao
