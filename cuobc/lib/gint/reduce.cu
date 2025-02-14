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

#ifndef SEGREDUCE
#define SEGREDUCE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define _WARPSIZE 32
#define WARP2POWER 5

template <typename T> struct SegReduce {
    uint32_t l;
    uint32_t maxsum;
    bool first;
    cg::plus<T> op;
    cg::coalesced_group active;

    __device__ SegReduce(long inl)
        : l(inl), op(cg::plus<T>()), active(cg::coalesced_threads()) {
        uint32_t lane = threadIdx.x & (_WARPSIZE - 1);
        size_t thid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t thid2 = (thid / inl) * inl + inl;
        first = lane == 0 || (thid2 - thid == inl);
        thid = ((thid >> WARP2POWER) << WARP2POWER) + _WARPSIZE;
        maxsum = ((min(thid, thid2) - 1) & (_WARPSIZE - 1)) - lane + 1;
    }
    __device__ __forceinline__ T operator()(T val) {
        for (int i = 1; i < l; i <<= 1) {
            T x = active.shfl_down(val, i);
            if (i < maxsum)
                val = op(val, x);
        }
        return val;
    }
    __device__ __forceinline__ void operator()(T val, T *ind) {
        for (int i = 1; i < l; i <<= 1) {
            T x = active.shfl_down(val, i);
            if (i < maxsum)
                val = op(val, x);
        }
        if (first)
            atomicAdd(ind, val);
    }
};
#endif
