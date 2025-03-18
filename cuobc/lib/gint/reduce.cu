/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

Licensed under the Apache License, Version 2.0 (the "License")
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https: // www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
