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

ij = []
for i in 0:5
    for j in 0:i
        push!(ij, (i,j))
    end
end
# timebvk = [2.899,3.356,3.254,3.098,3.764,5.38,3.221,4.422,7.151,14.108,3.497,5.65,10.619,23.108,46.684,3.902,7.015,15.312,35.609,72.99,115.02]
# rank = 1:42
# timebvk = [timebvk[cld(i,2)]/2 for i in 1:42]
# perm = sortperm(timebvk)
# rank = rank[perm]
# timebvk = timebvk[perm]
# print(timebvk)
# print(rank)
slicebvk = [[42],[41],[30,40],[29,39],[27,28,37,38],[17,18,19,20,24,25,26,33,34,35,36],[1,2,3,4,7,8,5,6,13,14,21,22,9,10,31,32,15,16,11,12,23]]
for n in 1:length(slicebvk)
    open("/mnt/bn/jemiry-nas/code/GPU_RSDF/pyscf/lib/gpuaft/ft_ao_bvk_ins$n.cu", "w") do f
        print(f, "#include <cuComplex.h>\n#include <assert.h>\n#include \"ft_ao.h\"\n\n#include \"ft_ao_template.cu\"\n\n")
        for ind in slicebvk[n]
            i, j = ij[cld(ind, 2)]
            b = ind % 2 == 1 ? "true" : "false"
            print(f,"template void _PBC_ft_bvk<$b, $i, $j>(const int, cuDoubleComplex*, const int, const int, const int, const int, const int*, const int8_t*, double*, const double*, const cuDoubleComplex*, const int*, const int*, const double*, const double*, const int*, const int*, const int, int*, const int, int*, const int);\n")
        end
    end
end
open("/mnt/bn/jemiry-nas/code/GPU_RSDF/pyscf/lib/gpuaft/ft_ao_bvk_ext.h", "w") do f
    print(f,"template<bool, int, int> void _PBC_ft_bvk(const int, cuDoubleComplex*, const int, const int, const int, const int, const int*, const int8_t*, double*, const double*, const cuDoubleComplex*, const int*, const int*, const double*, const double*, const int*, const int*, const int, int*, const int, int*, const int);\n")
    for n in 1:length(ij)
        i, j = ij[n]
        for b in ["true", "false"]
            print(f,"extern template void _PBC_ft_bvk<$b, $i, $j>(const int, cuDoubleComplex*, const int, const int, const int, const int, const int*, const int8_t*, double*, const double*, const cuDoubleComplex*, const int*, const int*, const double*, const double*, const int*, const int*, const int, int*, const int, int*, const int);\n")
        end
    end
end

slicelat=[[42],[41],[30,40],[29,39],[27,28,37,38],[17,18,19,20,25,26,33,34,35,36],[1,2,7,8,13,14,5,6,3,4,21,22,9,10,31,32,15,16,11,12,23,24]]
for n in 1:length(slicelat)
    open("/mnt/bn/jemiry-nas/code/GPU_RSDF/pyscf/lib/gpuaft/ft_ao_latsum_ins$n.cu", "w") do f
        print(f, "#include <cuComplex.h>\n#include <assert.h>\n#include \"ft_ao.h\"\n\n#include \"ft_ao_template.cu\"\n\n")
        for ind in slicelat[n]
            i, j = ij[cld(ind, 2)]
            b = ind % 2 == 1 ? "true" : "false"
            print(f,"template void _PBC_ft_latsum<$b, $i, $j>(const int, cuDoubleComplex*, const int, const int, const int, double*, const double*, const cuDoubleComplex*, const int*, const int*, const double*, const double*, const int*, const int*, const int, int*, const int, int*, const int);\n")
        end
    end
end
open("/mnt/bn/jemiry-nas/code/GPU_RSDF/pyscf/lib/gpuaft/ft_ao_latsum_ext.h", "w") do f
    print(f,"template<bool, int, int> void _PBC_ft_latsum(const int, cuDoubleComplex*, const int, const int, const int, double*, const double*, const cuDoubleComplex*, const int*, const int*, const double*, const double*, const int*, const int*, const int, int*, const int, int*, const int);\n")
    for n in 1:length(ij)
        i, j = ij[n]
        for b in ["true", "false"]
            print(f,"extern template void _PBC_ft_latsum<$b, $i, $j>(const int, cuDoubleComplex*, const int, const int, const int, double*, const double*, const cuDoubleComplex*, const int*, const int*, const double*, const double*, const int*, const int*, const int, int*, const int, int*, const int);\n")
        end
    end
end
for i in 1:length(slicebvk)
    print("ft_ao_bvk_ins$i.cu ")
end
for i in 1:length(slicelat)
    print("ft_ao_latsum_ins$i.cu ")
end
