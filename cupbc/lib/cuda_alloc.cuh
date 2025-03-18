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

#define ALIGN_256(n) ((n + 255) / 256 * 256)

#define MALLOC_HOST(type, var, size) \
    type *var; \
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&var), sizeof(type) * (size)))

#define MALLOC(type, var, size) \
    type *var; \
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&var), sizeof(type) * (size)))

#define MALLOC_ALIGN_MSTREAM(type, var, size, n_stream) \
    type *var; \
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&var), ALIGN_256(sizeof(type) * (size)) * n_stream))

#define FREE(var) \
    if(var != NULL) \
        CUDA_CHECK(cudaFree(var))

#define FREE_HOST(var) \
    if(var != NULL) \
        CUDA_CHECK(cudaFreeHost(var))

#define MEMSET(addr, val, size) \
    CUDA_CHECK(cudaMemset(addr, val, size))

#define DEVICE_INIT(type, dst, src, size) \
    MALLOC(type, dst, size); \
    CUDA_CHECK(cudaMemcpy(dst, src, sizeof(type) * (size), cudaMemcpyHostToDevice))

////////////////////////////////////////////////////////////////////////////////
// Async version
////////////////////////////////////////////////////////////////////////////////

#define MALLOC_ASYNC(type, var, size, stream) \
    type *var; \
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&var), sizeof(type) * (size), stream))

#define MALLOC_ALIGN_MSTREAM_ASYNC(type, var, size, n_stream, stream) \
    type *var; \
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&var), ALIGN_256(sizeof(type) * (size)) * n_stream, stream))

#define FREE_ASYNC(var, stream) \
    if (var != NULL); \
    CUDA_CHECK(cudaFreeAsync(var, stream))

#define MEMSET_ASYNC(addr, val, size, stream) \
    CUDA_CHECK(cudaMemsetAsync(addr, val, size, stream))

#define DEVICE_INIT_ASYNC(type, dst, src, size, stream) \
    MALLOC(type, dst, size); \
    CUDA_CHECK(cudaMemcpyAsync(dst, src, sizeof(type) * (size), cudaMemcpyHostToDevice, stream))

////////////////////////////////////////////////////////////////////////////////
// Mempool version
////////////////////////////////////////////////////////////////////////////////
#define MALLOC_ALIGN_MEMPOOL(type, var, size, buf_ptr, avail_buf_size) \
    type *var; \
    var = (type *)buf_ptr; \
    {unsigned long align_size = ALIGN_256(sizeof(type) * size); \
    buf_ptr = (void *)((char *)buf_ptr + align_size); \
    avail_buf_size = avail_buf_size - align_size;\
    if ((long)avail_buf_size < 0)\
    {fprintf(stderr, "avail_bufsize is not available for %s in %s line %d: %ld\n", #var, __FILE__, __LINE__, (long)avail_buf_size);}}

#define MALLOC_ALIGN_MSTREAM_MEMPOOL(type, var, size, n_stream, buf_ptr, avail_buf_size) type *var; \
    var = (type *)buf_ptr; \
    {unsigned long align_size = ALIGN_256(sizeof(type) * (size)) * n_stream; \
    buf_ptr = (void *)((char *)buf_ptr + align_size); \
    avail_buf_size = avail_buf_size - align_size;\
    if ((long)avail_buf_size < 0)\
    {fprintf(stderr, "avail_bufsize is not available for %s in %s line %d: %ld\n", #var, __FILE__, __LINE__, (long)avail_buf_size);}}

#define DEVICE_INIT_MEMPOOL(type, dst, src, size, buf_ptr, avail_buf_size) \
    MALLOC_ALIGN_MEMPOOL(type, dst, size, buf_ptr, avail_buf_size); \
    CUDA_CHECK(cudaMemcpy(dst, src, sizeof(type) * (size), cudaMemcpyHostToDevice))
